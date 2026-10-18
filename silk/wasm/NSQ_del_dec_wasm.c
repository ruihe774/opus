/* Copyright (c) 2026 Xiph.Org Foundation */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <wasm_simd128.h>
#include "main.h"
#include "stack_alloc.h"

/* The delayed-decision states are processed in parallel, one state per
   32-bit lane, so up to four states are supported. Unused lanes are
   computed along but masked out of all decisions. Everything is bit-exact
   with silk_NSQ_del_dec_c(), including the wrap-around and saturation
   behaviour of the C macros. */
#define WASM_DEL_DEC_STATES 4

#if QUANT_LEVEL_ADJUST_Q10 != 80
#error "silk_noise_shape_quantizer_del_dec_wasm() assumes QUANT_LEVEL_ADJUST_Q10 == 80"
#endif

typedef struct {
    opus_int32 sLPC_Q14[ MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH ][ WASM_DEL_DEC_STATES ];
    opus_int32 RandState[ DECISION_DELAY ][     WASM_DEL_DEC_STATES ];
    opus_int32 Q_Q10[     DECISION_DELAY ][     WASM_DEL_DEC_STATES ];
    opus_int32 Xq_Q14[    DECISION_DELAY ][     WASM_DEL_DEC_STATES ];
    opus_int32 Pred_Q15[  DECISION_DELAY ][     WASM_DEL_DEC_STATES ];
    opus_int32 Shape_Q14[ DECISION_DELAY ][     WASM_DEL_DEC_STATES ];
    opus_int32 sAR2_Q14[ MAX_SHAPE_LPC_ORDER ][ WASM_DEL_DEC_STATES ];
    opus_int32 LF_AR_Q14[ WASM_DEL_DEC_STATES ];
    opus_int32 Diff_Q14[  WASM_DEL_DEC_STATES ];
    opus_int32 Seed[      WASM_DEL_DEC_STATES ];
    opus_int32 SeedInit[  WASM_DEL_DEC_STATES ];
    opus_int32 RD_Q10[    WASM_DEL_DEC_STATES ];
} NSQ_del_decs_wasm_struct;

#define LD( p )     wasm_v128_load( p )
#define ST( p, v )  wasm_v128_store( p, v )

/* mask ? a : b, for masks whose lanes are all ones or all zeros. Under that
   condition relaxed_laneselect gives the same result on every
   implementation. */
static OPUS_INLINE v128_t silk_select_wasm( v128_t a, v128_t b, v128_t mask )
{
#if defined(__wasm_relaxed_simd__)
    return wasm_i32x4_relaxed_laneselect( a, b, mask );
#else
    return wasm_v128_bitselect( a, b, mask );
#endif
}

/* Byte shuffle with a run-time selector whose indices are all in 0..15, for
   which relaxed_swizzle is fully specified. */
static OPUS_INLINE v128_t silk_swizzle_wasm( v128_t a, v128_t sel )
{
#if defined(__wasm_relaxed_simd__)
    return wasm_i8x16_relaxed_swizzle( a, sel );
#else
    return wasm_i8x16_swizzle( a, sel );
#endif
}

/* silk_SMULWB() on each lane: ( a * (opus_int16)b ) >> 16, with b_sx holding
   the sign-extended 16-bit coefficient. Split a into its signed high and
   unsigned low halves; both partial products fit in 32 bits. */
static OPUS_INLINE v128_t silk_SMULWB_wasm( v128_t a, v128_t b_sx )
{
    v128_t hi = wasm_i32x4_mul( wasm_i32x4_shr( a, 16 ), b_sx );
    v128_t lo = wasm_i32x4_mul( wasm_v128_and( a, wasm_i32x4_const_splat( 0xFFFF ) ), b_sx );
    return wasm_i32x4_add( hi, wasm_i32x4_shr( lo, 16 ) );
}

/* Same as silk_SMULWB_wasm(), with shorter latency for the serial allpass
   chain: i32x4.dot_i16x8_s multiplies the 16-bit halves directly.
   b_hi = b << 16 and b_lo = b & 0xFFFF place the coefficient next to the
   high or low half of each lane. The low half is taken as signed by the dot
   product, which is corrected by adding b when its top bit is set. */
static OPUS_INLINE v128_t silk_SMULWB_dot_wasm( v128_t a, v128_t b_hi, v128_t b_lo, v128_t b_sx )
{
    v128_t hi  = wasm_i32x4_dot_i16x8( a, b_hi );
    v128_t lo  = wasm_i32x4_shr( wasm_i32x4_dot_i16x8( a, b_lo ), 16 );
    v128_t adj = wasm_v128_and( wasm_i32x4_shr( wasm_i32x4_shl( a, 16 ), 31 ), b_sx );
    return wasm_i32x4_add( wasm_i32x4_add( hi, lo ), adj );
}

/* silk_SMULWW() on each lane: ( a * b ) >> 16, truncated to 32 bits. */
static OPUS_INLINE v128_t silk_SMULWW_wasm( v128_t a, v128_t b )
{
    v128_t lo = wasm_i64x2_shr( wasm_i64x2_extmul_low_i32x4( a, b ), 16 );
    v128_t hi = wasm_i64x2_shr( wasm_i64x2_extmul_high_i32x4( a, b ), 16 );
    return wasm_i32x4_shuffle( lo, hi, 0, 2, 4, 6 );
}

/* silk_ADD_SAT32() and silk_SUB_SAT32() */
static OPUS_INLINE v128_t silk_ADD_SAT32_wasm( v128_t a, v128_t b )
{
    v128_t r   = wasm_i32x4_add( a, b );
    v128_t ovf = wasm_i32x4_shr( wasm_v128_and( wasm_v128_xor( a, r ), wasm_v128_xor( b, r ) ), 31 );
    v128_t sat = wasm_v128_xor( wasm_i32x4_shr( a, 31 ), wasm_i32x4_const_splat( silk_int32_MAX ) );
    return silk_select_wasm( sat, r, ovf );
}

static OPUS_INLINE v128_t silk_SUB_SAT32_wasm( v128_t a, v128_t b )
{
    v128_t r   = wasm_i32x4_sub( a, b );
    v128_t ovf = wasm_i32x4_shr( wasm_v128_and( wasm_v128_xor( a, r ), wasm_v128_xor( a, b ) ), 31 );
    v128_t sat = wasm_v128_xor( wasm_i32x4_shr( a, 31 ), wasm_i32x4_const_splat( silk_int32_MAX ) );
    return silk_select_wasm( sat, r, ovf );
}

/* silk_RSHIFT_ROUND() for shift > 1, which cannot overflow */
#define silk_RSHIFT_ROUND_wasm( a, shift ) \
    wasm_i32x4_shr( wasm_i32x4_add( wasm_i32x4_shr( a, (shift) - 1 ), wasm_i32x4_const_splat( 1 ) ), 1 )

/* Index of the first lane holding the minimum (maximum) of num, which is
   what the strict comparisons of the C search loops find. Lanes set in mask
   (unused states) are ignored; lane 0 is always in use. */
static OPUS_INLINE opus_int silk_first_min_wasm( v128_t num, v128_t mask, opus_int32 *val )
{
    v128_t m;
    num = silk_select_wasm( wasm_i32x4_const_splat( silk_int32_MAX ), num, mask );
    m = wasm_i32x4_min( num, wasm_i32x4_shuffle( num, num, 2, 3, 0, 1 ) );
    m = wasm_i32x4_min( m, wasm_i32x4_shuffle( m, m, 1, 0, 3, 2 ) );
    *val = wasm_i32x4_extract_lane( m, 0 );
    return __builtin_ctz( wasm_i32x4_bitmask( wasm_i32x4_eq( num, m ) ) );
}

static OPUS_INLINE opus_int silk_first_max_wasm( v128_t num, v128_t mask, opus_int32 *val )
{
    v128_t m;
    num = silk_select_wasm( wasm_i32x4_const_splat( silk_int32_MIN ), num, mask );
    m = wasm_i32x4_max( num, wasm_i32x4_shuffle( num, num, 2, 3, 0, 1 ) );
    m = wasm_i32x4_max( m, wasm_i32x4_shuffle( m, m, 1, 0, 3, 2 ) );
    *val = wasm_i32x4_extract_lane( m, 0 );
    return __builtin_ctz( wasm_i32x4_bitmask( wasm_i32x4_eq( num, m ) ) );
}

static OPUS_INLINE void silk_nsq_del_dec_scale_states_wasm(
    const silk_encoder_state *psEncC,               /* I    Encoder State                       */
    silk_nsq_state      *NSQ,                       /* I/O  NSQ state                           */
    NSQ_del_decs_wasm_struct *psDelDec,             /* I/O  Delayed decision states             */
    const opus_int16    x16[],                      /* I    Input                               */
    opus_int32          x_sc_Q10[],                 /* O    Input scaled with 1/Gain in Q10     */
    const opus_int16    sLTP[],                     /* I    Re-whitened LTP state in Q0         */
    opus_int32          sLTP_Q15[],                 /* O    LTP state matching scaled input     */
    opus_int            subfr,                      /* I    Subframe number                     */
    const opus_int      LTP_scale_Q14,              /* I    LTP state scaling                   */
    const opus_int32    Gains_Q16[ MAX_NB_SUBFR ],  /* I                                        */
    const opus_int      pitchL[ MAX_NB_SUBFR ],     /* I    Pitch lag                           */
    const opus_int      signal_type,                /* I    Signal type                         */
    const opus_int      decisionDelay               /* I    Decision delay                      */
);

/******************************************/
/* Noise shape quantizer for one subframe */
/******************************************/
static OPUS_INLINE void silk_noise_shape_quantizer_del_dec_wasm(
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                           */
    NSQ_del_decs_wasm_struct *psDelDec,         /* I/O  Delayed decision states             */
    opus_int            signalType,             /* I    Signal type                         */
    const opus_int32    x_Q10[],                /* I                                        */
    opus_int8           pulses[],               /* O                                        */
    opus_int16          xq[],                   /* O                                        */
    opus_int32          sLTP_Q15[],             /* I/O  LTP filter state                    */
    opus_int32          delayedGain_Q10[],      /* I/O  Gain delay buffer                   */
    const opus_int16    a_Q12[],                /* I    Short term prediction coefs         */
    const opus_int16    b_Q14[],                /* I    Long term prediction coefs          */
    const opus_int16    AR_shp_Q13[],           /* I    Noise shaping coefs                 */
    opus_int            lag,                    /* I    Pitch lag                           */
    opus_int32          HarmShapeFIRPacked_Q14, /* I                                        */
    opus_int            Tilt_Q14,               /* I    Spectral tilt                       */
    opus_int32          LF_shp_Q14,             /* I                                        */
    opus_int32          Gain_Q16,               /* I                                        */
    opus_int            Lambda_Q10,             /* I                                        */
    opus_int            offset_Q10,             /* I                                        */
    opus_int            length,                 /* I    Input length                        */
    opus_int            subfr,                  /* I    Subframe number                     */
    opus_int            shapingLPCOrder,        /* I    Shaping LPC filter order            */
    opus_int            predictLPCOrder,        /* I    Prediction filter order             */
    opus_int            warping_Q16,            /* I                                        */
    v128_t              MaskDelDec,             /* I    All ones in the lanes of unused states */
    opus_int            *smpl_buf_idx,          /* I/O  Index to newest samples in buffers  */
    opus_int            decisionDelay           /* I                                        */
)
{
    opus_int     i, j, Winner_ind, RDmin_ind, RDmax_ind, last_smple_idx;
    opus_int32   LTP_pred_Q14, n_LTP_Q14, Gain_Q10, RDmin_Q10, RDmax_Q10;
    opus_int32   *pred_lag_ptr, *shp_lag_ptr;
    v128_t       a_Q12_sx[ MAX_LPC_ORDER ];
    v128_t       AR_shp_sx[ MAX_SHAPE_LPC_ORDER ];
    v128_t       warp_hi, warp_lo, warp_sx, Tilt_sx, LF_shp_lo_sx, LF_shp_hi_sx, Lambda_lo;
    v128_t       Seed, LF_AR_Q14, Diff_Q14, RD_Q10;
    const v128_t lane_idx = wasm_i32x4_make( 0, 1, 2, 3 );
    const v128_t byte_idx = wasm_i8x16_make( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
    const opus_int warped = ( (opus_int16)warping_Q16 != 0 );

    celt_assert( ( shapingLPCOrder & 1 ) == 0 );   /* check that order is even */
    celt_assert( shapingLPCOrder >= 2 && shapingLPCOrder <= MAX_SHAPE_LPC_ORDER );
    celt_assert( predictLPCOrder == 10 || predictLPCOrder == 16 );

    shp_lag_ptr  = &NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2 ];
    pred_lag_ptr = &sLTP_Q15[ NSQ->sLTP_buf_idx - lag + LTP_ORDER / 2 ];
    Gain_Q10     = silk_RSHIFT( Gain_Q16, 6 );

    for( j = 0; j < predictLPCOrder; j++ ) {
        a_Q12_sx[ j ] = wasm_i32x4_splat( a_Q12[ j ] );
    }
    for( j = 0; j < shapingLPCOrder; j++ ) {
        AR_shp_sx[ j ] = wasm_i32x4_splat( AR_shp_Q13[ j ] );
    }
    warp_hi      = wasm_i32x4_splat( (opus_int32)( (opus_uint32)warping_Q16 << 16 ) );
    warp_lo      = wasm_i32x4_splat( warping_Q16 & 0xFFFF );
    warp_sx      = wasm_i32x4_splat( (opus_int16)warping_Q16 );
    Tilt_sx      = wasm_i32x4_splat( (opus_int16)Tilt_Q14 );
    LF_shp_lo_sx = wasm_i32x4_splat( (opus_int16)LF_shp_Q14 );
    LF_shp_hi_sx = wasm_i32x4_splat( silk_RSHIFT( LF_shp_Q14, 16 ) );
    /* Lambda in the low half of each lane, zero in the high half, so that
       i32x4.dot_i16x8_s computes silk_SMULBB( x, Lambda_Q10 ) */
    Lambda_lo    = wasm_i32x4_splat( Lambda_Q10 & 0xFFFF );

    Seed      = LD( psDelDec->Seed );
    LF_AR_Q14 = LD( psDelDec->LF_AR_Q14 );
    Diff_Q14  = LD( psDelDec->Diff_Q14 );
    RD_Q10    = LD( psDelDec->RD_Q10 );

    for( i = 0; i < length; i++ ) {
        v128_t LPC_pred_Q14, n_AR_Q14, n_LF_Q14, tmp1, tmp2, sign, r_Q10;
        v128_t q1_Q0, q1_Q10, q2_Q10, rd1_Q10, rd2_Q10, mask;
        v128_t SS0_Q_Q10, SS0_RD_Q10, SS0_xq_Q14, SS0_LF_AR_Q14, SS0_Diff_Q14, SS0_sLTP_shp_Q14, SS0_LPC_exc_Q14;
        v128_t SS1_Q_Q10, SS1_RD_Q10, SS1_xq_Q14, SS1_LF_AR_Q14, SS1_Diff_Q14, SS1_sLTP_shp_Q14, SS1_LPC_exc_Q14;
        v128_t x_Q14, exc_Q14;
        const opus_int32 (*psLPC_Q14)[ WASM_DEL_DEC_STATES ];

        /* Perform common calculations used in all states */

        /* Long-term prediction */
        if( signalType == TYPE_VOICED ) {
            /* Unrolled loop */
            /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
            LTP_pred_Q14 = 2;
            LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[  0 ], b_Q14[ 0 ] );
            LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[ -1 ], b_Q14[ 1 ] );
            LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[ -2 ], b_Q14[ 2 ] );
            LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[ -3 ], b_Q14[ 3 ] );
            LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[ -4 ], b_Q14[ 4 ] );
            LTP_pred_Q14 = silk_LSHIFT( LTP_pred_Q14, 1 );                          /* Q13 -> Q14 */
            pred_lag_ptr++;
        } else {
            LTP_pred_Q14 = 0;
        }

        /* Long-term shaping */
        if( lag > 0 ) {
            /* Symmetric, packed FIR coefficients */
            n_LTP_Q14 = silk_SMULWB( silk_ADD_SAT32( shp_lag_ptr[ 0 ], shp_lag_ptr[ -2 ] ), HarmShapeFIRPacked_Q14 );
            n_LTP_Q14 = silk_SMLAWT( n_LTP_Q14, shp_lag_ptr[ -1 ], HarmShapeFIRPacked_Q14 );
            n_LTP_Q14 = silk_SUB_LSHIFT32( LTP_pred_Q14, n_LTP_Q14, 2 );            /* Q12 -> Q14 */
            shp_lag_ptr++;
        } else {
            n_LTP_Q14 = 0;
        }

        /* Generate dither */
        Seed = wasm_i32x4_add( wasm_i32x4_mul( Seed, wasm_i32x4_const_splat( RAND_MULTIPLIER ) ),
                               wasm_i32x4_const_splat( RAND_INCREMENT ) );

        /* Short-term prediction */
        psLPC_Q14 = &psDelDec->sLPC_Q14[ NSQ_LPC_BUF_LENGTH - 1 + i ];
        /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
        tmp1 = wasm_i32x4_add( wasm_i32x4_splat( silk_RSHIFT( predictLPCOrder, 1 ) ),
                               silk_SMULWB_wasm( LD( psLPC_Q14[  0 ] ), a_Q12_sx[ 0 ] ) );
        tmp2 = silk_SMULWB_wasm( LD( psLPC_Q14[ -1 ] ), a_Q12_sx[ 1 ] );
        tmp1 = wasm_i32x4_add( tmp1, silk_SMULWB_wasm( LD( psLPC_Q14[ -2 ] ), a_Q12_sx[ 2 ] ) );
        tmp2 = wasm_i32x4_add( tmp2, silk_SMULWB_wasm( LD( psLPC_Q14[ -3 ] ), a_Q12_sx[ 3 ] ) );
        tmp1 = wasm_i32x4_add( tmp1, silk_SMULWB_wasm( LD( psLPC_Q14[ -4 ] ), a_Q12_sx[ 4 ] ) );
        tmp2 = wasm_i32x4_add( tmp2, silk_SMULWB_wasm( LD( psLPC_Q14[ -5 ] ), a_Q12_sx[ 5 ] ) );
        tmp1 = wasm_i32x4_add( tmp1, silk_SMULWB_wasm( LD( psLPC_Q14[ -6 ] ), a_Q12_sx[ 6 ] ) );
        tmp2 = wasm_i32x4_add( tmp2, silk_SMULWB_wasm( LD( psLPC_Q14[ -7 ] ), a_Q12_sx[ 7 ] ) );
        tmp1 = wasm_i32x4_add( tmp1, silk_SMULWB_wasm( LD( psLPC_Q14[ -8 ] ), a_Q12_sx[ 8 ] ) );
        tmp2 = wasm_i32x4_add( tmp2, silk_SMULWB_wasm( LD( psLPC_Q14[ -9 ] ), a_Q12_sx[ 9 ] ) );
        if( predictLPCOrder == 16 ) {
            tmp1 = wasm_i32x4_add( tmp1, silk_SMULWB_wasm( LD( psLPC_Q14[ -10 ] ), a_Q12_sx[ 10 ] ) );
            tmp2 = wasm_i32x4_add( tmp2, silk_SMULWB_wasm( LD( psLPC_Q14[ -11 ] ), a_Q12_sx[ 11 ] ) );
            tmp1 = wasm_i32x4_add( tmp1, silk_SMULWB_wasm( LD( psLPC_Q14[ -12 ] ), a_Q12_sx[ 12 ] ) );
            tmp2 = wasm_i32x4_add( tmp2, silk_SMULWB_wasm( LD( psLPC_Q14[ -13 ] ), a_Q12_sx[ 13 ] ) );
            tmp1 = wasm_i32x4_add( tmp1, silk_SMULWB_wasm( LD( psLPC_Q14[ -14 ] ), a_Q12_sx[ 14 ] ) );
            tmp2 = wasm_i32x4_add( tmp2, silk_SMULWB_wasm( LD( psLPC_Q14[ -15 ] ), a_Q12_sx[ 15 ] ) );
        }
        LPC_pred_Q14 = wasm_i32x4_shl( wasm_i32x4_add( tmp1, tmp2 ), 4 );    /* Q10 -> Q14 */

        /* Noise shape feedback */
        if( warped ) {
            v128_t s0, s1;
            /* Output of lowpass section */
            s0   = LD( psDelDec->sAR2_Q14[ 0 ] );
            tmp2 = wasm_i32x4_add( Diff_Q14, silk_SMULWB_dot_wasm( s0, warp_hi, warp_lo, warp_sx ) );
            /* Output of allpass section */
            s1   = LD( psDelDec->sAR2_Q14[ 1 ] );
            tmp1 = wasm_i32x4_add( s0, silk_SMULWB_dot_wasm( wasm_i32x4_sub( s1, tmp2 ), warp_hi, warp_lo, warp_sx ) );
            ST( psDelDec->sAR2_Q14[ 0 ], tmp2 );
            n_AR_Q14 = wasm_i32x4_add( wasm_i32x4_splat( silk_RSHIFT( shapingLPCOrder, 1 ) ),
                                       silk_SMULWB_wasm( tmp2, AR_shp_sx[ 0 ] ) );
            /* Loop over allpass sections */
            for( j = 2; j < shapingLPCOrder; j += 2 ) {
                /* Output of allpass section */
                s0   = LD( psDelDec->sAR2_Q14[ j + 0 ] );
                tmp2 = wasm_i32x4_add( s1, silk_SMULWB_dot_wasm( wasm_i32x4_sub( s0, tmp1 ), warp_hi, warp_lo, warp_sx ) );
                ST( psDelDec->sAR2_Q14[ j - 1 ], tmp1 );
                n_AR_Q14 = wasm_i32x4_add( n_AR_Q14, silk_SMULWB_wasm( tmp1, AR_shp_sx[ j - 1 ] ) );
                /* Output of allpass section */
                s1   = LD( psDelDec->sAR2_Q14[ j + 1 ] );
                tmp1 = wasm_i32x4_add( s0, silk_SMULWB_dot_wasm( wasm_i32x4_sub( s1, tmp2 ), warp_hi, warp_lo, warp_sx ) );
                ST( psDelDec->sAR2_Q14[ j + 0 ], tmp2 );
                n_AR_Q14 = wasm_i32x4_add( n_AR_Q14, silk_SMULWB_wasm( tmp2, AR_shp_sx[ j ] ) );
            }
            ST( psDelDec->sAR2_Q14[ shapingLPCOrder - 1 ], tmp1 );
            n_AR_Q14 = wasm_i32x4_add( n_AR_Q14, silk_SMULWB_wasm( tmp1, AR_shp_sx[ shapingLPCOrder - 1 ] ) );
        } else {
            /* Without warping the allpass sections reduce to a delay line */
            tmp1 = Diff_Q14;
            n_AR_Q14 = wasm_i32x4_splat( silk_RSHIFT( shapingLPCOrder, 1 ) );
            tmp2 = wasm_i32x4_const_splat( 0 );
            for( j = 0; j < shapingLPCOrder; j += 2 ) {
                v128_t s0, s1;
                s0 = LD( psDelDec->sAR2_Q14[ j + 0 ] );
                s1 = LD( psDelDec->sAR2_Q14[ j + 1 ] );
                ST( psDelDec->sAR2_Q14[ j + 0 ], tmp1 );
                ST( psDelDec->sAR2_Q14[ j + 1 ], s0 );
                n_AR_Q14 = wasm_i32x4_add( n_AR_Q14, silk_SMULWB_wasm( tmp1, AR_shp_sx[ j + 0 ] ) );
                tmp2     = wasm_i32x4_add( tmp2,     silk_SMULWB_wasm( s0,   AR_shp_sx[ j + 1 ] ) );
                tmp1 = s1;
            }
            n_AR_Q14 = wasm_i32x4_add( n_AR_Q14, tmp2 );
        }

        n_AR_Q14 = wasm_i32x4_shl( n_AR_Q14, 1 );                                          /* Q11 -> Q12 */
        n_AR_Q14 = wasm_i32x4_add( n_AR_Q14, silk_SMULWB_wasm( LF_AR_Q14, Tilt_sx ) );     /* Q12 */
        n_AR_Q14 = wasm_i32x4_shl( n_AR_Q14, 2 );                                          /* Q12 -> Q14 */

        n_LF_Q14 = silk_SMULWB_wasm( LD( psDelDec->Shape_Q14[ *smpl_buf_idx ] ), LF_shp_lo_sx ); /* Q12 */
        n_LF_Q14 = wasm_i32x4_add( n_LF_Q14, silk_SMULWB_wasm( LF_AR_Q14, LF_shp_hi_sx ) );        /* Q12 */
        n_LF_Q14 = wasm_i32x4_shl( n_LF_Q14, 2 );                                                  /* Q12 -> Q14 */

        /* Input minus prediction plus noise feedback                       */
        /* r = x[ i ] - LTP_pred - LPC_pred + n_AR + n_Tilt + n_LF + n_LTP  */
        tmp1 = silk_ADD_SAT32_wasm( n_AR_Q14, n_LF_Q14 );                                  /* Q14 */
        tmp2 = wasm_i32x4_add( wasm_i32x4_splat( n_LTP_Q14 ), LPC_pred_Q14 );              /* Q13 */
        tmp1 = silk_SUB_SAT32_wasm( tmp2, tmp1 );                                          /* Q13 */
        tmp1 = silk_RSHIFT_ROUND_wasm( tmp1, 4 );                                          /* Q10 */

        r_Q10 = wasm_i32x4_sub( wasm_i32x4_splat( x_Q10[ i ] ), tmp1 );                    /* residual error Q10 */

        /* Flip sign depending on dither */
        sign  = wasm_i32x4_shr( Seed, 31 );
        r_Q10 = wasm_i32x4_sub( wasm_v128_xor( r_Q10, sign ), sign );
        r_Q10 = wasm_i32x4_max( wasm_i32x4_min( r_Q10, wasm_i32x4_const_splat( 30 << 10 ) ),
                                wasm_i32x4_const_splat( -( 31 << 10 ) ) );

        /* Find two quantization level candidates and measure their rate-distortion */
        q1_Q10 = wasm_i32x4_sub( r_Q10, wasm_i32x4_splat( offset_Q10 ) );
        q1_Q0  = wasm_i32x4_shr( q1_Q10, 10 );
        if( Lambda_Q10 > 2048 ) {
            /* For aggressive RDO, the bias becomes more than one pulse. */
            const opus_int32 rdo_offset = Lambda_Q10/2 - 512;
            v128_t gt = wasm_i32x4_gt( q1_Q10, wasm_i32x4_splat(  rdo_offset ) );
            v128_t lt = wasm_i32x4_lt( q1_Q10, wasm_i32x4_splat( -rdo_offset ) );
            tmp1  = wasm_i32x4_shr( wasm_i32x4_sub( q1_Q10, wasm_i32x4_splat( rdo_offset ) ), 10 );
            tmp2  = wasm_i32x4_shr( wasm_i32x4_add( q1_Q10, wasm_i32x4_splat( rdo_offset ) ), 10 );
            q1_Q0 = silk_select_wasm( tmp1, silk_select_wasm( tmp2, wasm_i32x4_shr( q1_Q10, 31 ), lt ), gt );
        }
        /* q1_Q10 = q1_Q0 * 1024 + offset_Q10, moved QUANT_LEVEL_ADJUST_Q10 towards zero */
        tmp1   = wasm_i32x4_max( wasm_i32x4_min( q1_Q0, wasm_i32x4_const_splat( 1 ) ), wasm_i32x4_const_splat( -1 ) );
        tmp1   = wasm_i32x4_add( wasm_i32x4_shl( tmp1, 6 ), wasm_i32x4_shl( tmp1, 4 ) ); /* * QUANT_LEVEL_ADJUST_Q10 */
        q1_Q10 = wasm_i32x4_add( wasm_i32x4_sub( wasm_i32x4_shl( q1_Q0, 10 ), tmp1 ), wasm_i32x4_splat( offset_Q10 ) );
        /* q2_Q10 is one level above, closer when q1_Q0 is 0 or -1 */
        mask   = wasm_i32x4_eq( wasm_i32x4_add( q1_Q0, wasm_u32x4_shr( q1_Q0, 31 ) ), wasm_i32x4_const_splat( 0 ) );
        q2_Q10 = wasm_i32x4_add( q1_Q10, wasm_i32x4_sub( wasm_i32x4_const_splat( 1024 ),
                                 wasm_v128_and( mask, wasm_i32x4_const_splat( QUANT_LEVEL_ADJUST_Q10 ) ) ) );
        /* The rate term is Lambda times the magnitude of the level: q1_Q10 and q2_Q10
           are negative exactly when the C code negates them, since 0 < offset_Q10 < 944. */
        rd1_Q10 = wasm_i32x4_dot_i16x8( wasm_i32x4_abs( q1_Q10 ), Lambda_lo );
        rd2_Q10 = wasm_i32x4_dot_i16x8( wasm_i32x4_abs( q2_Q10 ), Lambda_lo );
        tmp1    = wasm_i32x4_sub( r_Q10, q1_Q10 );
        rd1_Q10 = wasm_i32x4_add( rd1_Q10, wasm_i32x4_dot_i16x8( tmp1, wasm_v128_and( tmp1, wasm_i32x4_const_splat( 0xFFFF ) ) ) );
        rd1_Q10 = wasm_i32x4_shr( rd1_Q10, 10 );
        tmp2    = wasm_i32x4_sub( r_Q10, q2_Q10 );
        rd2_Q10 = wasm_i32x4_add( rd2_Q10, wasm_i32x4_dot_i16x8( tmp2, wasm_v128_and( tmp2, wasm_i32x4_const_splat( 0xFFFF ) ) ) );
        rd2_Q10 = wasm_i32x4_shr( rd2_Q10, 10 );

        mask       = wasm_i32x4_lt( rd1_Q10, rd2_Q10 );
        SS0_RD_Q10 = wasm_i32x4_add( RD_Q10, wasm_i32x4_min( rd1_Q10, rd2_Q10 ) );
        SS1_RD_Q10 = wasm_i32x4_add( RD_Q10, wasm_i32x4_max( rd1_Q10, rd2_Q10 ) );
        SS0_Q_Q10  = silk_select_wasm( q1_Q10, q2_Q10, mask );
        SS1_Q_Q10  = silk_select_wasm( q2_Q10, q1_Q10, mask );

        x_Q14 = wasm_i32x4_splat( (opus_int32)( (opus_uint32)x_Q10[ i ] << 4 ) );

        /* Update states for best quantization */

        /* Quantized excitation */
        exc_Q14 = wasm_i32x4_shl( SS0_Q_Q10, 4 );
        exc_Q14 = wasm_i32x4_sub( wasm_v128_xor( exc_Q14, sign ), sign );

        /* Add predictions */
        SS0_LPC_exc_Q14  = wasm_i32x4_add( exc_Q14, wasm_i32x4_splat( LTP_pred_Q14 ) );
        SS0_xq_Q14       = wasm_i32x4_add( SS0_LPC_exc_Q14, LPC_pred_Q14 );

        /* Update states */
        SS0_Diff_Q14     = wasm_i32x4_sub( SS0_xq_Q14, x_Q14 );
        SS0_LF_AR_Q14    = wasm_i32x4_sub( SS0_Diff_Q14, n_AR_Q14 );
        SS0_sLTP_shp_Q14 = silk_SUB_SAT32_wasm( SS0_LF_AR_Q14, n_LF_Q14 );

        /* Update states for second best quantization */

        /* Quantized excitation */
        exc_Q14 = wasm_i32x4_shl( SS1_Q_Q10, 4 );
        exc_Q14 = wasm_i32x4_sub( wasm_v128_xor( exc_Q14, sign ), sign );

        /* Add predictions */
        SS1_LPC_exc_Q14  = wasm_i32x4_add( exc_Q14, wasm_i32x4_splat( LTP_pred_Q14 ) );
        SS1_xq_Q14       = wasm_i32x4_add( SS1_LPC_exc_Q14, LPC_pred_Q14 );

        /* Update states */
        SS1_Diff_Q14     = wasm_i32x4_sub( SS1_xq_Q14, x_Q14 );
        SS1_LF_AR_Q14    = wasm_i32x4_sub( SS1_Diff_Q14, n_AR_Q14 );
        SS1_sLTP_shp_Q14 = silk_SUB_SAT32_wasm( SS1_LF_AR_Q14, n_LF_Q14 );

        *smpl_buf_idx  = *smpl_buf_idx ? ( *smpl_buf_idx - 1 ) : ( DECISION_DELAY - 1 );
        last_smple_idx = *smpl_buf_idx + decisionDelay;
        if( last_smple_idx >= DECISION_DELAY ) last_smple_idx -= DECISION_DELAY;

        /* Find winner */
        Winner_ind = silk_first_min_wasm( SS0_RD_Q10, MaskDelDec, &RDmin_Q10 );

        /* Increase RD values of expired states */
        tmp1 = LD( psDelDec->RandState[ last_smple_idx ] );
        mask = wasm_i32x4_eq( tmp1, wasm_i32x4_splat( psDelDec->RandState[ last_smple_idx ][ Winner_ind ] ) );
        tmp1 = wasm_v128_andnot( wasm_i32x4_const_splat( silk_int32_MAX >> 4 ), mask );
        SS0_RD_Q10 = wasm_i32x4_add( SS0_RD_Q10, tmp1 );
        SS1_RD_Q10 = wasm_i32x4_add( SS1_RD_Q10, tmp1 );

        /* Find worst in first set and best in second set */
        RDmax_ind = silk_first_max_wasm( SS0_RD_Q10, MaskDelDec, &RDmax_Q10 );
        RDmin_ind = silk_first_min_wasm( SS1_RD_Q10, MaskDelDec, &RDmin_Q10 );

        /* Replace a state if best from second set outperforms worst in first set */
        if( RDmin_Q10 < RDmax_Q10 ) {
            /* Byte selector moving lane RDmin_ind into lane RDmax_ind */
            v128_t lane_mask = wasm_i32x4_eq( lane_idx, wasm_i32x4_splat( RDmax_ind ) );
            v128_t bcast     = wasm_i32x4_splat( RDmin_ind * 0x04040404 + 0x03020100 );
            v128_t sel       = silk_select_wasm( bcast, byte_idx, lane_mask );
            opus_int32 (*ptr)[ WASM_DEL_DEC_STATES ] = psDelDec->RandState;

            /* Only the last ( NSQ_LPC_BUF_LENGTH - 1 ) values of sLPC_Q14 are used again */
            for( j = i + 1; j < i + NSQ_LPC_BUF_LENGTH; j++ ) {
                ST( psDelDec->sLPC_Q14[ j ], silk_swizzle_wasm( LD( psDelDec->sLPC_Q14[ j ] ), sel ) );
            }
            for( j = 0; j < shapingLPCOrder; j++ ) {
                ST( psDelDec->sAR2_Q14[ j ], silk_swizzle_wasm( LD( psDelDec->sAR2_Q14[ j ] ), sel ) );
            }
            /* RandState, Q_Q10, Xq_Q14, Pred_Q15 and Shape_Q14 are mostly read
               as scalars later, so copy them lane by lane */
            for( j = 0; j < 5 * DECISION_DELAY; j++ ) {
                ptr[ j ][ RDmax_ind ] = ptr[ j ][ RDmin_ind ];
            }
            Seed = silk_swizzle_wasm( Seed, sel );
            ST( psDelDec->SeedInit, silk_swizzle_wasm( LD( psDelDec->SeedInit ), sel ) );
            /* LF_AR_Q14, Diff_Q14 and RD_Q10 are overwritten below */

            SS0_Q_Q10        = silk_select_wasm( silk_swizzle_wasm( SS1_Q_Q10,        bcast ), SS0_Q_Q10,        lane_mask );
            SS0_RD_Q10       = silk_select_wasm( silk_swizzle_wasm( SS1_RD_Q10,       bcast ), SS0_RD_Q10,       lane_mask );
            SS0_xq_Q14       = silk_select_wasm( silk_swizzle_wasm( SS1_xq_Q14,       bcast ), SS0_xq_Q14,       lane_mask );
            SS0_LF_AR_Q14    = silk_select_wasm( silk_swizzle_wasm( SS1_LF_AR_Q14,    bcast ), SS0_LF_AR_Q14,    lane_mask );
            SS0_Diff_Q14     = silk_select_wasm( silk_swizzle_wasm( SS1_Diff_Q14,     bcast ), SS0_Diff_Q14,     lane_mask );
            SS0_sLTP_shp_Q14 = silk_select_wasm( silk_swizzle_wasm( SS1_sLTP_shp_Q14, bcast ), SS0_sLTP_shp_Q14, lane_mask );
            SS0_LPC_exc_Q14  = silk_select_wasm( silk_swizzle_wasm( SS1_LPC_exc_Q14,  bcast ), SS0_LPC_exc_Q14,  lane_mask );
        }

        /* Write samples from winner to output and long-term filter states */
        if( subfr > 0 || i >= decisionDelay ) {
            pulses[  i - decisionDelay ] = (opus_int8)silk_RSHIFT_ROUND( psDelDec->Q_Q10[ last_smple_idx ][ Winner_ind ], 10 );
            xq[ i - decisionDelay ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND(
                silk_SMULWW( psDelDec->Xq_Q14[ last_smple_idx ][ Winner_ind ], delayedGain_Q10[ last_smple_idx ] ), 8 ) );
            NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - decisionDelay ] = psDelDec->Shape_Q14[ last_smple_idx ][ Winner_ind ];
            sLTP_Q15[          NSQ->sLTP_buf_idx     - decisionDelay ] = psDelDec->Pred_Q15[  last_smple_idx ][ Winner_ind ];
        }
        NSQ->sLTP_shp_buf_idx++;
        NSQ->sLTP_buf_idx++;

        /* Update states */
        LF_AR_Q14 = SS0_LF_AR_Q14;
        Diff_Q14  = SS0_Diff_Q14;
        RD_Q10    = SS0_RD_Q10;
        ST( psDelDec->sLPC_Q14[ NSQ_LPC_BUF_LENGTH + i ], SS0_xq_Q14 );
        ST( psDelDec->Xq_Q14[    *smpl_buf_idx ], SS0_xq_Q14 );
        ST( psDelDec->Q_Q10[     *smpl_buf_idx ], SS0_Q_Q10 );
        ST( psDelDec->Pred_Q15[  *smpl_buf_idx ], wasm_i32x4_shl( SS0_LPC_exc_Q14, 1 ) );
        ST( psDelDec->Shape_Q14[ *smpl_buf_idx ], SS0_sLTP_shp_Q14 );
        Seed = wasm_i32x4_add( Seed, silk_RSHIFT_ROUND_wasm( SS0_Q_Q10, 10 ) );
        ST( psDelDec->RandState[ *smpl_buf_idx ], Seed );
        delayedGain_Q10[ *smpl_buf_idx ] = Gain_Q10;
    }
    ST( psDelDec->Seed,      Seed );
    ST( psDelDec->LF_AR_Q14, LF_AR_Q14 );
    ST( psDelDec->Diff_Q14,  Diff_Q14 );
    ST( psDelDec->RD_Q10,    RD_Q10 );

    /* Update LPC states */
    silk_memcpy( psDelDec->sLPC_Q14[ 0 ], psDelDec->sLPC_Q14[ length ], NSQ_LPC_BUF_LENGTH * sizeof( psDelDec->sLPC_Q14[ 0 ] ) );
}

/* Copy the final part of the signals from the winner state to the output and
   long-term filter states. */
static OPUS_INLINE void silk_copy_winner_state_wasm(
    const NSQ_del_decs_wasm_struct *psDelDec,
    silk_nsq_state      *NSQ,
    opus_int8           pulses[],
    opus_int16          pxq[],
    opus_int            smpl_buf_idx,
    opus_int            decisionDelay,
    opus_int            Winner_ind,
    opus_int32          gain,
    opus_int            shift
)
{
    opus_int i, last_smple_idx;

    last_smple_idx = smpl_buf_idx + decisionDelay;
    for( i = 0; i < decisionDelay; i++ ) {
        last_smple_idx = ( last_smple_idx - 1 ) % DECISION_DELAY;
        if( last_smple_idx < 0 ) last_smple_idx += DECISION_DELAY;
        pulses[ i - decisionDelay ] = (opus_int8)silk_RSHIFT_ROUND( psDelDec->Q_Q10[ last_smple_idx ][ Winner_ind ], 10 );
        pxq[ i - decisionDelay ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND(
            silk_SMULWW( psDelDec->Xq_Q14[ last_smple_idx ][ Winner_ind ], gain ), shift ) );
        NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - decisionDelay + i ] = psDelDec->Shape_Q14[ last_smple_idx ][ Winner_ind ];
    }
}

void silk_NSQ_del_dec_wasm(
    const silk_encoder_state    *psEncC,                                      /* I    Encoder State                   */
    silk_nsq_state              *NSQ,                                         /* I/O  NSQ state                       */
    SideInfoIndices             *psIndices,                                   /* I/O  Quantization Indices            */
    const opus_int16            x16[],                                        /* I    Input                           */
    opus_int8                   pulses[],                                     /* O    Quantized pulse signal          */
    const opus_int16            *PredCoef_Q12,                                /* I    Short term prediction coefs     */
    const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],      /* I    Long term prediction coefs      */
    const opus_int16            AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I    Noise shaping coefs             */
    const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],            /* I    Long term shaping coefs         */
    const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                     /* I    Spectral tilt                   */
    const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                   /* I    Low frequency shaping coefs     */
    const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                    /* I    Quantization step sizes         */
    const opus_int              pitchL[ MAX_NB_SUBFR ],                       /* I    Pitch lags                      */
    const opus_int              Lambda_Q10,                                   /* I    Rate/distortion tradeoff        */
    const opus_int              LTP_scale_Q14                                 /* I    LTP state scaling               */
)
{
    opus_int            i, k, lag, start_idx, LSF_interpolation_flag, Winner_ind, subfr;
    opus_int            smpl_buf_idx, decisionDelay;
    const opus_int16    *A_Q12, *B_Q14, *AR_shp_Q13;
    opus_int16          *pxq;
    VARDECL( opus_int32, sLTP_Q15 );
    VARDECL( opus_int16, sLTP );
    opus_int32          HarmShapeFIRPacked_Q14;
    opus_int            offset_Q10;
    opus_int32          RDmin_Q10, Gain_Q10;
    opus_int32          x_sc_Q10[ MAX_SUB_FRAME_LENGTH ];
    opus_int32          delayedGain_Q10[ DECISION_DELAY ];
    NSQ_del_decs_wasm_struct psDelDec;
    v128_t              MaskDelDec, t;
    SAVE_STACK;

    if( psEncC->nStatesDelayedDecision > WASM_DEL_DEC_STATES ) {
        silk_NSQ_del_dec_c( psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, HarmShapeGain_Q14,
            Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14 );
        RESTORE_STACK;
        return;
    }

    /* Lanes of states beyond nStatesDelayedDecision never take part in decisions */
    MaskDelDec = wasm_i32x4_gt( wasm_i32x4_make( 0, 1, 2, 3 ), wasm_i32x4_splat( psEncC->nStatesDelayedDecision - 1 ) );

    /* Set unvoiced lag to the previous one, overwrite later for voiced */
    lag = NSQ->lagPrev;

    silk_assert( NSQ->prev_gain_Q16 != 0 );

    /* Initialize delayed decision states */
    silk_memset( &psDelDec, 0, sizeof( psDelDec ) );
    t = wasm_v128_and( wasm_i32x4_add( wasm_i32x4_make( 0, 1, 2, 3 ), wasm_i32x4_splat( psIndices->Seed ) ),
                       wasm_i32x4_const_splat( 3 ) );
    ST( psDelDec.Seed, t );
    ST( psDelDec.SeedInit, t );
    ST( psDelDec.LF_AR_Q14, wasm_i32x4_splat( NSQ->sLF_AR_shp_Q14 ) );
    ST( psDelDec.Diff_Q14, wasm_i32x4_splat( NSQ->sDiff_shp_Q14 ) );
    ST( psDelDec.Shape_Q14[ 0 ], wasm_i32x4_splat( NSQ->sLTP_shp_Q14[ psEncC->ltp_mem_length - 1 ] ) );
    for( i = 0; i < NSQ_LPC_BUF_LENGTH; i++ ) {
        ST( psDelDec.sLPC_Q14[ i ], wasm_i32x4_splat( NSQ->sLPC_Q14[ i ] ) );
    }
    for( i = 0; i < MAX_SHAPE_LPC_ORDER; i++ ) {
        ST( psDelDec.sAR2_Q14[ i ], wasm_i32x4_splat( NSQ->sAR2_Q14[ i ] ) );
    }

    offset_Q10   = silk_Quantization_Offsets_Q10[ psIndices->signalType >> 1 ][ psIndices->quantOffsetType ];
    smpl_buf_idx = 0; /* index of oldest samples */

    decisionDelay = silk_min_int( DECISION_DELAY, psEncC->subfr_length );

    /* For voiced frames limit the decision delay to lower than the pitch lag */
    if( psIndices->signalType == TYPE_VOICED ) {
        for( k = 0; k < psEncC->nb_subfr; k++ ) {
            decisionDelay = silk_min_int( decisionDelay, pitchL[ k ] - LTP_ORDER / 2 - 1 );
        }
    } else {
        if( lag > 0 ) {
            decisionDelay = silk_min_int( decisionDelay, lag - LTP_ORDER / 2 - 1 );
        }
    }

    if( psIndices->NLSFInterpCoef_Q2 == 4 ) {
        LSF_interpolation_flag = 0;
    } else {
        LSF_interpolation_flag = 1;
    }

    ALLOC( sLTP_Q15, psEncC->ltp_mem_length + psEncC->frame_length, opus_int32 );
    ALLOC( sLTP, psEncC->ltp_mem_length + psEncC->frame_length, opus_int16 );
    /* Set up pointers to start of sub frame */
    pxq                   = &NSQ->xq[ psEncC->ltp_mem_length ];
    NSQ->sLTP_shp_buf_idx = psEncC->ltp_mem_length;
    NSQ->sLTP_buf_idx     = psEncC->ltp_mem_length;
    subfr = 0;
    for( k = 0; k < psEncC->nb_subfr; k++ ) {
        A_Q12      = &PredCoef_Q12[ ( ( k >> 1 ) | ( 1 - LSF_interpolation_flag ) ) * MAX_LPC_ORDER ];
        B_Q14      = &LTPCoef_Q14[ k * LTP_ORDER           ];
        AR_shp_Q13 = &AR_Q13[     k * MAX_SHAPE_LPC_ORDER ];

        /* Noise shape parameters */
        silk_assert( HarmShapeGain_Q14[ k ] >= 0 );
        HarmShapeFIRPacked_Q14  =                          silk_RSHIFT( HarmShapeGain_Q14[ k ], 2 );
        HarmShapeFIRPacked_Q14 |= silk_LSHIFT( (opus_int32)silk_RSHIFT( HarmShapeGain_Q14[ k ], 1 ), 16 );

        NSQ->rewhite_flag = 0;
        if( psIndices->signalType == TYPE_VOICED ) {
            /* Voiced */
            lag = pitchL[ k ];

            /* Re-whitening */
            if( ( k & ( 3 - silk_LSHIFT( LSF_interpolation_flag, 1 ) ) ) == 0 ) {
                if( k == 2 ) {
                    /* RESET DELAYED DECISIONS */
                    /* Find winner */
                    t = LD( psDelDec.RD_Q10 );
                    Winner_ind = silk_first_min_wasm( t, MaskDelDec, &RDmin_Q10 );
                    t = wasm_i32x4_add( t, wasm_v128_andnot( wasm_i32x4_const_splat( silk_int32_MAX >> 4 ),
                            wasm_i32x4_eq( wasm_i32x4_make( 0, 1, 2, 3 ), wasm_i32x4_splat( Winner_ind ) ) ) );
                    ST( psDelDec.RD_Q10, t );

                    /* Copy final part of signals from winner state to output and long-term filter states */
                    silk_copy_winner_state_wasm( &psDelDec, NSQ, pulses, pxq, smpl_buf_idx, decisionDelay,
                        Winner_ind, Gains_Q16[ 1 ], 14 );

                    subfr = 0;
                }

                /* Rewhiten with new A coefs */
                start_idx = psEncC->ltp_mem_length - lag - psEncC->predictLPCOrder - LTP_ORDER / 2;
                celt_assert( start_idx > 0 );

                silk_LPC_analysis_filter( &sLTP[ start_idx ], &NSQ->xq[ start_idx + k * psEncC->subfr_length ],
                    A_Q12, psEncC->ltp_mem_length - start_idx, psEncC->predictLPCOrder, psEncC->arch );

                NSQ->sLTP_buf_idx = psEncC->ltp_mem_length;
                NSQ->rewhite_flag = 1;
            }
        }

        silk_nsq_del_dec_scale_states_wasm( psEncC, NSQ, &psDelDec, x16, x_sc_Q10, sLTP, sLTP_Q15, k,
            LTP_scale_Q14, Gains_Q16, pitchL, psIndices->signalType, decisionDelay );

        silk_noise_shape_quantizer_del_dec_wasm( NSQ, &psDelDec, psIndices->signalType, x_sc_Q10, pulses, pxq, sLTP_Q15,
            delayedGain_Q10, A_Q12, B_Q14, AR_shp_Q13, lag, HarmShapeFIRPacked_Q14, Tilt_Q14[ k ], LF_shp_Q14[ k ],
            Gains_Q16[ k ], Lambda_Q10, offset_Q10, psEncC->subfr_length, subfr++, psEncC->shapingLPCOrder,
            psEncC->predictLPCOrder, psEncC->warping_Q16, MaskDelDec, &smpl_buf_idx, decisionDelay );

        x16    += psEncC->subfr_length;
        pulses += psEncC->subfr_length;
        pxq    += psEncC->subfr_length;
    }

    /* Find winner */
    Winner_ind = silk_first_min_wasm( LD( psDelDec.RD_Q10 ), MaskDelDec, &RDmin_Q10 );

    /* Copy final part of signals from winner state to output and long-term filter states */
    psIndices->Seed = psDelDec.SeedInit[ Winner_ind ];
    Gain_Q10 = silk_RSHIFT32( Gains_Q16[ psEncC->nb_subfr - 1 ], 6 );
    silk_copy_winner_state_wasm( &psDelDec, NSQ, pulses, pxq, smpl_buf_idx, decisionDelay, Winner_ind, Gain_Q10, 8 );
    for( i = 0; i < NSQ_LPC_BUF_LENGTH; i++ ) {
        NSQ->sLPC_Q14[ i ] = psDelDec.sLPC_Q14[ psEncC->subfr_length + i ][ Winner_ind ];
    }
    for( i = 0; i < MAX_SHAPE_LPC_ORDER; i++ ) {
        NSQ->sAR2_Q14[ i ] = psDelDec.sAR2_Q14[ i ][ Winner_ind ];
    }

    /* Update states */
    NSQ->sLF_AR_shp_Q14 = psDelDec.LF_AR_Q14[ Winner_ind ];
    NSQ->sDiff_shp_Q14  = psDelDec.Diff_Q14[ Winner_ind ];
    NSQ->lagPrev        = pitchL[ psEncC->nb_subfr - 1 ];

    /* Save quantized speech signal */
    silk_memmove( NSQ->xq,           &NSQ->xq[           psEncC->frame_length ], psEncC->ltp_mem_length * sizeof( opus_int16 ) );
    silk_memmove( NSQ->sLTP_shp_Q14, &NSQ->sLTP_shp_Q14[ psEncC->frame_length ], psEncC->ltp_mem_length * sizeof( opus_int32 ) );
    RESTORE_STACK;
}

static OPUS_INLINE void silk_nsq_del_dec_scale_states_wasm(
    const silk_encoder_state *psEncC,               /* I    Encoder State                       */
    silk_nsq_state      *NSQ,                       /* I/O  NSQ state                           */
    NSQ_del_decs_wasm_struct *psDelDec,             /* I/O  Delayed decision states             */
    const opus_int16    x16[],                      /* I    Input                               */
    opus_int32          x_sc_Q10[],                 /* O    Input scaled with 1/Gain in Q10     */
    const opus_int16    sLTP[],                     /* I    Re-whitened LTP state in Q0         */
    opus_int32          sLTP_Q15[],                 /* O    LTP state matching scaled input     */
    opus_int            subfr,                      /* I    Subframe number                     */
    const opus_int      LTP_scale_Q14,              /* I    LTP state scaling                   */
    const opus_int32    Gains_Q16[ MAX_NB_SUBFR ],  /* I                                        */
    const opus_int      pitchL[ MAX_NB_SUBFR ],     /* I    Pitch lag                           */
    const opus_int      signal_type,                /* I    Signal type                         */
    const opus_int      decisionDelay               /* I    Decision delay                      */
)
{
    opus_int            i, lag;
    opus_int32          gain_adj_Q16, inv_gain_Q31, inv_gain_Q26;
    v128_t              g;

    lag          = pitchL[ subfr ];
    inv_gain_Q31 = silk_INVERSE32_varQ( silk_max( Gains_Q16[ subfr ], 1 ), 47 );
    silk_assert( inv_gain_Q31 != 0 );

    /* Scale input */
    inv_gain_Q26 = silk_RSHIFT_ROUND( inv_gain_Q31, 5 );
    g = wasm_i32x4_splat( inv_gain_Q26 );
    for( i = 0; i < psEncC->subfr_length - 3; i += 4 ) {
        ST( &x_sc_Q10[ i ], silk_SMULWW_wasm( wasm_i32x4_load16x4( &x16[ i ] ), g ) );
    }
    for( ; i < psEncC->subfr_length; i++ ) {
        x_sc_Q10[ i ] = silk_SMULWW( x16[ i ], inv_gain_Q26 );
    }

    /* After rewhitening the LTP state is un-scaled, so scale with inv_gain_Q16 */
    if( NSQ->rewhite_flag ) {
        if( subfr == 0 ) {
            /* Do LTP downscaling */
            inv_gain_Q31 = silk_LSHIFT( silk_SMULWB( inv_gain_Q31, LTP_scale_Q14 ), 2 );
        }
        g = wasm_i32x4_splat( inv_gain_Q31 );
        for( i = NSQ->sLTP_buf_idx - lag - LTP_ORDER / 2; i < NSQ->sLTP_buf_idx - 3; i += 4 ) {
            silk_assert( i < MAX_FRAME_LENGTH );
            ST( &sLTP_Q15[ i ], silk_SMULWB_wasm( g, wasm_i32x4_load16x4( &sLTP[ i ] ) ) );
        }
        for( ; i < NSQ->sLTP_buf_idx; i++ ) {
            sLTP_Q15[ i ] = silk_SMULWB( inv_gain_Q31, sLTP[ i ] );
        }
    }

    /* Adjust for changing gain */
    if( Gains_Q16[ subfr ] != NSQ->prev_gain_Q16 ) {
        gain_adj_Q16 =  silk_DIV32_varQ( NSQ->prev_gain_Q16, Gains_Q16[ subfr ], 16 );
        g = wasm_i32x4_splat( gain_adj_Q16 );

        /* Scale long-term shaping state */
        for( i = NSQ->sLTP_shp_buf_idx - psEncC->ltp_mem_length; i < NSQ->sLTP_shp_buf_idx - 3; i += 4 ) {
            ST( &NSQ->sLTP_shp_Q14[ i ], silk_SMULWW_wasm( LD( &NSQ->sLTP_shp_Q14[ i ] ), g ) );
        }
        for( ; i < NSQ->sLTP_shp_buf_idx; i++ ) {
            NSQ->sLTP_shp_Q14[ i ] = silk_SMULWW( gain_adj_Q16, NSQ->sLTP_shp_Q14[ i ] );
        }

        /* Scale long-term prediction state */
        if( signal_type == TYPE_VOICED && NSQ->rewhite_flag == 0 ) {
            for( i = NSQ->sLTP_buf_idx - lag - LTP_ORDER / 2; i < NSQ->sLTP_buf_idx - decisionDelay - 3; i += 4 ) {
                ST( &sLTP_Q15[ i ], silk_SMULWW_wasm( LD( &sLTP_Q15[ i ] ), g ) );
            }
            for( ; i < NSQ->sLTP_buf_idx - decisionDelay; i++ ) {
                sLTP_Q15[ i ] = silk_SMULWW( gain_adj_Q16, sLTP_Q15[ i ] );
            }
        }

        /* Scale scalar states */
        ST( psDelDec->LF_AR_Q14, silk_SMULWW_wasm( LD( psDelDec->LF_AR_Q14 ), g ) );
        ST( psDelDec->Diff_Q14,  silk_SMULWW_wasm( LD( psDelDec->Diff_Q14 ),  g ) );

        /* Scale short-term prediction and shaping states */
        for( i = 0; i < NSQ_LPC_BUF_LENGTH; i++ ) {
            ST( psDelDec->sLPC_Q14[ i ], silk_SMULWW_wasm( LD( psDelDec->sLPC_Q14[ i ] ), g ) );
        }
        for( i = 0; i < MAX_SHAPE_LPC_ORDER; i++ ) {
            ST( psDelDec->sAR2_Q14[ i ], silk_SMULWW_wasm( LD( psDelDec->sAR2_Q14[ i ] ), g ) );
        }
        for( i = 0; i < DECISION_DELAY; i++ ) {
            ST( psDelDec->Pred_Q15[  i ], silk_SMULWW_wasm( LD( psDelDec->Pred_Q15[  i ] ), g ) );
            ST( psDelDec->Shape_Q14[ i ], silk_SMULWW_wasm( LD( psDelDec->Shape_Q14[ i ] ), g ) );
        }

        /* Save inverse gain */
        NSQ->prev_gain_Q16 = Gains_Q16[ subfr ];
    }
}
