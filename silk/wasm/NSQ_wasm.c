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

/* The quantizer runs one sample at a time, so the filters are vectorized
   along their taps. The short-term prediction and noise shaping filter
   states are kept in registers and shifted by one lane per sample, which
   avoids reloading values that were just stored. The newest tap of both
   filters, which depends on the previous sample, is computed separately in
   scalar code, so that the vector products of the older taps stay off the
   critical path from one sample to the next. Bit-exact with silk_NSQ_c(). */

/* silk_SMULWB() on each lane: ( a * (opus_int16)b ) >> 16, with b_sx holding
   the sign-extended 16-bit coefficient. Split a into its signed high and
   unsigned low halves; both partial products fit in 32 bits. */
static OPUS_INLINE v128_t silk_SMULWB_wasm( v128_t a, v128_t b_sx )
{
    v128_t hi = wasm_i32x4_mul( wasm_i32x4_shr( a, 16 ), b_sx );
    v128_t lo = wasm_i32x4_mul( wasm_v128_and( a, wasm_i32x4_const_splat( 0xFFFF ) ), b_sx );
    return wasm_i32x4_add( hi, wasm_i32x4_shr( lo, 16 ) );
}

/* silk_SMULWW() on each lane: ( a * b ) >> 16, truncated to 32 bits. */
static OPUS_INLINE v128_t silk_SMULWW_wasm( v128_t a, v128_t b )
{
    v128_t lo = wasm_i64x2_shr( wasm_i64x2_extmul_low_i32x4( a, b ), 16 );
    v128_t hi = wasm_i64x2_shr( wasm_i64x2_extmul_high_i32x4( a, b ), 16 );
    return wasm_i32x4_shuffle( lo, hi, 0, 2, 4, 6 );
}

static OPUS_INLINE void silk_nsq_scale_states_wasm(
    const silk_encoder_state *psEncC,           /* I    Encoder State                   */
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                       */
    const opus_int16    x16[],                  /* I    input                           */
    opus_int32          x_sc_Q10[],             /* O    input scaled with 1/Gain        */
    const opus_int16    sLTP[],                 /* I    re-whitened LTP state in Q0     */
    opus_int32          sLTP_Q15[],             /* O    LTP state matching scaled input */
    opus_int            subfr,                  /* I    subframe number                 */
    const opus_int      LTP_scale_Q14,          /* I                                    */
    const opus_int32    Gains_Q16[ MAX_NB_SUBFR ], /* I                                 */
    const opus_int      pitchL[ MAX_NB_SUBFR ], /* I    Pitch lag                       */
    const opus_int      signal_type             /* I    Signal type                     */
)
{
    opus_int   i, lag;
    opus_int32 gain_adj_Q16, inv_gain_Q31, inv_gain_Q26;
    v128_t     g;

    lag          = pitchL[ subfr ];
    inv_gain_Q31 = silk_INVERSE32_varQ( silk_max( Gains_Q16[ subfr ], 1 ), 47 );
    silk_assert( inv_gain_Q31 != 0 );

    /* Scale input */
    inv_gain_Q26 = silk_RSHIFT_ROUND( inv_gain_Q31, 5 );
    g = wasm_i32x4_splat( inv_gain_Q26 );
    for( i = 0; i < psEncC->subfr_length - 3; i += 4 ) {
        wasm_v128_store( &x_sc_Q10[ i ], silk_SMULWW_wasm( wasm_i32x4_load16x4( &x16[ i ] ), g ) );
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
            wasm_v128_store( &sLTP_Q15[ i ], silk_SMULWB_wasm( g, wasm_i32x4_load16x4( &sLTP[ i ] ) ) );
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
            wasm_v128_store( &NSQ->sLTP_shp_Q14[ i ], silk_SMULWW_wasm( wasm_v128_load( &NSQ->sLTP_shp_Q14[ i ] ), g ) );
        }
        for( ; i < NSQ->sLTP_shp_buf_idx; i++ ) {
            NSQ->sLTP_shp_Q14[ i ] = silk_SMULWW( gain_adj_Q16, NSQ->sLTP_shp_Q14[ i ] );
        }

        /* Scale long-term prediction state */
        if( signal_type == TYPE_VOICED && NSQ->rewhite_flag == 0 ) {
            for( i = NSQ->sLTP_buf_idx - lag - LTP_ORDER / 2; i < NSQ->sLTP_buf_idx - 3; i += 4 ) {
                wasm_v128_store( &sLTP_Q15[ i ], silk_SMULWW_wasm( wasm_v128_load( &sLTP_Q15[ i ] ), g ) );
            }
            for( ; i < NSQ->sLTP_buf_idx; i++ ) {
                sLTP_Q15[ i ] = silk_SMULWW( gain_adj_Q16, sLTP_Q15[ i ] );
            }
        }

        NSQ->sLF_AR_shp_Q14 = silk_SMULWW( gain_adj_Q16, NSQ->sLF_AR_shp_Q14 );
        NSQ->sDiff_shp_Q14 = silk_SMULWW( gain_adj_Q16, NSQ->sDiff_shp_Q14 );

        /* Scale short-term prediction and shaping states */
        for( i = 0; i < NSQ_LPC_BUF_LENGTH; i += 4 ) {
            wasm_v128_store( &NSQ->sLPC_Q14[ i ], silk_SMULWW_wasm( wasm_v128_load( &NSQ->sLPC_Q14[ i ] ), g ) );
        }
        for( i = 0; i < MAX_SHAPE_LPC_ORDER; i += 4 ) {
            wasm_v128_store( &NSQ->sAR2_Q14[ i ], silk_SMULWW_wasm( wasm_v128_load( &NSQ->sAR2_Q14[ i ] ), g ) );
        }

        /* Save inverse gain */
        NSQ->prev_gain_Q16 = Gains_Q16[ subfr ];
    }
}

/* Noise shape quantizer for one subframe. ar_vecs is a compile-time constant
   (4 or 6): the number of vectors that hold the noise shaping filter. */
static OPUS_INLINE void silk_noise_shape_quantizer_wasm(
    silk_nsq_state      *NSQ,                   /* I/O  NSQ state                       */
    opus_int            signalType,             /* I    Signal type                     */
    const opus_int32    x_sc_Q10[],             /* I                                    */
    opus_int8           pulses[],               /* O                                    */
    opus_int16          xq[],                   /* O                                    */
    opus_int32          sLTP_Q15[],             /* I/O  LTP state                       */
    const opus_int16    a_Q12[],                /* I    Short term prediction coefs     */
    const opus_int16    b_Q14[],                /* I    Long term prediction coefs      */
    const opus_int16    AR_shp_Q13[],           /* I    Noise shaping AR coefs          */
    opus_int            lag,                    /* I    Pitch lag                       */
    opus_int32          HarmShapeFIRPacked_Q14, /* I                                    */
    opus_int            Tilt_Q14,               /* I    Spectral tilt                   */
    opus_int32          LF_shp_Q14,             /* I                                    */
    opus_int32          Gain_Q16,               /* I                                    */
    opus_int            Lambda_Q10,             /* I                                    */
    opus_int            offset_Q10,             /* I                                    */
    opus_int            length,                 /* I    Input length                    */
    opus_int            shapingLPCOrder,        /* I    Noise shaping AR filter order   */
    opus_int            predictLPCOrder,        /* I    Prediction filter order         */
    const opus_int      ar_vecs                 /* I    Noise shaping vectors (4 or 6)  */
)
{
    opus_int     i;
    opus_int32   LTP_pred_Q13, LPC_pred_Q10, n_AR_Q12, n_LTP_Q13;
    opus_int32   n_LF_Q12, r_Q10, rr_Q10, q1_Q0, q1_Q10, q2_Q10, rd1_Q20, rd2_Q20;
    opus_int32   exc_Q14, LPC_exc_Q14, xq_Q14, Gain_Q10;
    opus_int32   tmp1, tmp2, sLF_AR_shp_Q14, sDiff_shp_Q14, xq_prev_Q14;
    opus_int32   *psLPC_Q14, *shp_lag_ptr, *pred_lag_ptr;
    opus_int32   coef[ MAX_SHAPE_LPC_ORDER ];
    v128_t       lpc0, lpc1, lpc2, lpc3, a0, a1, a2, a3;
    v128_t       ar0, ar1, ar2, ar3, ar4, ar5, c0, c1, c2, c3, c4, c5;
    v128_t       b_sx;

    celt_assert( predictLPCOrder == 10 || predictLPCOrder == 16 );
    celt_assert( shapingLPCOrder <= 4 * ar_vecs );

    shp_lag_ptr  = &NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2 ];
    pred_lag_ptr = &sLTP_Q15[ NSQ->sLTP_buf_idx - lag + LTP_ORDER / 2 ];
    Gain_Q10     = silk_RSHIFT( Gain_Q16, 6 );

    /* Set up short term AR state */
    psLPC_Q14 = &NSQ->sLPC_Q14[ NSQ_LPC_BUF_LENGTH - 1 ];

    /* Short-term prediction: lpc0..lpc3 = psLPC_Q14[ -16 .. -1 ] against a_Q12[ 16 .. 1 ],
       with zeros above predictLPCOrder; psLPC_Q14[ 0 ] is xq_prev_Q14 */
    coef[ 0 ] = 0;
    for( i = 1; i < MAX_LPC_ORDER; i++ ) {
        coef[ MAX_LPC_ORDER - i ] = i < predictLPCOrder ? a_Q12[ i ] : 0;
    }
    a0 = wasm_v128_load( &coef[ 0 ] );
    a1 = wasm_v128_load( &coef[ 4 ] );
    a2 = wasm_v128_load( &coef[ 8 ] );
    a3 = wasm_v128_load( &coef[ 12 ] );
    lpc0 = wasm_v128_load( &psLPC_Q14[ -15 ] );
    lpc1 = wasm_v128_load( &psLPC_Q14[ -11 ] );
    lpc2 = wasm_v128_load( &psLPC_Q14[ -7 ] );
    lpc3 = wasm_v128_load( &psLPC_Q14[ -3 ] );
    lpc3 = wasm_i32x4_shuffle( lpc2, lpc3, 3, 4, 5, 6 );
    lpc2 = wasm_i32x4_shuffle( lpc1, lpc2, 3, 4, 5, 6 );
    lpc1 = wasm_i32x4_shuffle( lpc0, lpc1, 3, 4, 5, 6 );
    lpc0 = wasm_i32x4_shuffle( wasm_i32x4_const_splat( 0 ), lpc0, 3, 4, 5, 6 );
    xq_prev_Q14 = psLPC_Q14[ 0 ];

    /* Noise shape feedback: the filter input is { sDiff_shp_Q14, sAR2_Q14[ 0 ], sAR2_Q14[ 1 ], ... }.
       ar0..ar5 hold sAR2_Q14[], against AR_shp_Q13[ 1 .. ] with zeros from shapingLPCOrder on. */
    for( i = 0; i < 4 * ar_vecs; i++ ) {
        coef[ i ] = i + 1 < shapingLPCOrder ? AR_shp_Q13[ i + 1 ] : 0;
    }
    c0 = wasm_v128_load( &coef[ 0 ] );
    c1 = wasm_v128_load( &coef[ 4 ] );
    c2 = wasm_v128_load( &coef[ 8 ] );
    c3 = wasm_v128_load( &coef[ 12 ] );
    ar0 = wasm_v128_load( &NSQ->sAR2_Q14[ 0 ] );
    ar1 = wasm_v128_load( &NSQ->sAR2_Q14[ 4 ] );
    ar2 = wasm_v128_load( &NSQ->sAR2_Q14[ 8 ] );
    ar3 = wasm_v128_load( &NSQ->sAR2_Q14[ 12 ] );
    if( ar_vecs > 4 ) {
        c4 = wasm_v128_load( &coef[ 16 ] );
        c5 = wasm_v128_load( &coef[ 20 ] );
        ar4 = wasm_v128_load( &NSQ->sAR2_Q14[ 16 ] );
        ar5 = wasm_v128_load( &NSQ->sAR2_Q14[ 20 ] );
    } else {
        c4 = c5 = ar4 = ar5 = wasm_i32x4_const_splat( 0 );
    }

    /* Long-term prediction: taps 1..4 in a vector, against pred_lag_ptr[ -4 .. -1 ] */
    b_sx = wasm_i32x4_make( b_Q14[ 4 ], b_Q14[ 3 ], b_Q14[ 2 ], b_Q14[ 1 ] );

    sDiff_shp_Q14  = NSQ->sDiff_shp_Q14;
    sLF_AR_shp_Q14 = NSQ->sLF_AR_shp_Q14;

    for( i = 0; i < length; i++ ) {
        v128_t lpc, ar, ltp, t0, t1;

        /* Generate dither */
        NSQ->rand_seed = silk_RAND( NSQ->rand_seed );

        /* Short-term prediction */
        lpc = wasm_i32x4_add( wasm_i32x4_add( silk_SMULWB_wasm( lpc0, a0 ), silk_SMULWB_wasm( lpc1, a1 ) ),
                              wasm_i32x4_add( silk_SMULWB_wasm( lpc2, a2 ), silk_SMULWB_wasm( lpc3, a3 ) ) );

        /* Long-term prediction */
        if( signalType == TYPE_VOICED ) {
            ltp = silk_SMULWB_wasm( wasm_v128_load( &pred_lag_ptr[ -4 ] ), b_sx );
        } else {
            ltp = wasm_i32x4_const_splat( 0 );
        }

        /* Noise shape feedback */
        celt_assert( ( shapingLPCOrder & 1 ) == 0 );   /* check that order is even */
        ar = wasm_i32x4_add( wasm_i32x4_add( silk_SMULWB_wasm( ar0, c0 ), silk_SMULWB_wasm( ar1, c1 ) ),
                             wasm_i32x4_add( silk_SMULWB_wasm( ar2, c2 ), silk_SMULWB_wasm( ar3, c3 ) ) );
        if( ar_vecs > 4 ) {
            ar = wasm_i32x4_add( ar, wasm_i32x4_add( silk_SMULWB_wasm( ar4, c4 ), silk_SMULWB_wasm( ar5, c5 ) ) );
        }

        /* Horizontal sums: lane 0 = LPC, lane 1 = AR, lane 2 = LTP */
        t0 = wasm_i32x4_add( wasm_i32x4_shuffle( lpc, ar, 0, 1, 4, 5 ), wasm_i32x4_shuffle( lpc, ar, 2, 3, 6, 7 ) );
        t1 = wasm_i32x4_add( ltp, wasm_i32x4_shuffle( ltp, ltp, 2, 3, 0, 1 ) );
        t0 = wasm_i32x4_add( wasm_i32x4_shuffle( t0, t1, 0, 2, 4, 6 ), wasm_i32x4_shuffle( t0, t1, 1, 3, 5, 7 ) );

        /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
        LPC_pred_Q10 = silk_ADD32_ovflw( silk_RSHIFT( predictLPCOrder, 1 ), wasm_i32x4_extract_lane( t0, 0 ) );
        LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, xq_prev_Q14, a_Q12[ 0 ] );
        if( signalType == TYPE_VOICED ) {
            LTP_pred_Q13 = silk_ADD32_ovflw( 2, wasm_i32x4_extract_lane( t0, 2 ) );
            LTP_pred_Q13 = silk_SMLAWB( LTP_pred_Q13, pred_lag_ptr[ 0 ], b_Q14[ 0 ] );
            pred_lag_ptr++;
        } else {
            LTP_pred_Q13 = 0;
        }
        n_AR_Q12 = silk_ADD32_ovflw( silk_RSHIFT( shapingLPCOrder, 1 ), wasm_i32x4_extract_lane( t0, 1 ) );
        n_AR_Q12 = silk_SMLAWB( n_AR_Q12, sDiff_shp_Q14, AR_shp_Q13[ 0 ] );
        /* Q11 -> Q12 */
        n_AR_Q12 = silk_LSHIFT32( n_AR_Q12, 1 );

        n_AR_Q12 = silk_SMLAWB( n_AR_Q12, sLF_AR_shp_Q14, Tilt_Q14 );

        n_LF_Q12 = silk_SMULWB( NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx - 1 ], LF_shp_Q14 );
        n_LF_Q12 = silk_SMLAWT( n_LF_Q12, sLF_AR_shp_Q14, LF_shp_Q14 );

        celt_assert( lag > 0 || signalType != TYPE_VOICED );

        /* Combine prediction and noise shaping signals */
        tmp1 = silk_SUB32_ovflw( silk_LSHIFT32( LPC_pred_Q10, 2 ), n_AR_Q12 );  /* Q12 */
        tmp1 = silk_SUB32_ovflw( tmp1, n_LF_Q12 );                              /* Q12 */
        if( lag > 0 ) {
            /* Symmetric, packed FIR coefficients */
            n_LTP_Q13 = silk_SMULWB( silk_ADD_SAT32( shp_lag_ptr[ 0 ], shp_lag_ptr[ -2 ] ), HarmShapeFIRPacked_Q14 );
            n_LTP_Q13 = silk_SMLAWT( n_LTP_Q13, shp_lag_ptr[ -1 ],                      HarmShapeFIRPacked_Q14 );
            n_LTP_Q13 = silk_LSHIFT( n_LTP_Q13, 1 );
            shp_lag_ptr++;

            tmp2 = silk_SUB32( LTP_pred_Q13, n_LTP_Q13 );                       /* Q13 */
            tmp1 = silk_ADD32_ovflw( tmp2, silk_LSHIFT32( tmp1, 1 ) );          /* Q13 */
            tmp1 = silk_RSHIFT_ROUND( tmp1, 3 );                                /* Q10 */
        } else {
            tmp1 = silk_RSHIFT_ROUND( tmp1, 2 );                                /* Q10 */
        }

        r_Q10 = silk_SUB32( x_sc_Q10[ i ], tmp1 );                              /* residual error Q10 */

        /* Flip sign depending on dither */
        if( NSQ->rand_seed < 0 ) {
            r_Q10 = -r_Q10;
        }
        r_Q10 = silk_LIMIT_32( r_Q10, -(31 << 10), 30 << 10 );

        /* Find two quantization level candidates and measure their rate-distortion */
        q1_Q10 = silk_SUB32( r_Q10, offset_Q10 );
        q1_Q0 = silk_RSHIFT( q1_Q10, 10 );
        if (Lambda_Q10 > 2048) {
            /* For aggressive RDO, the bias becomes more than one pulse. */
            int rdo_offset = Lambda_Q10/2 - 512;
            if (q1_Q10 > rdo_offset) {
                q1_Q0 = silk_RSHIFT( q1_Q10 - rdo_offset, 10 );
            } else if (q1_Q10 < -rdo_offset) {
                q1_Q0 = silk_RSHIFT( q1_Q10 + rdo_offset, 10 );
            } else if (q1_Q10 < 0) {
                q1_Q0 = -1;
            } else {
                q1_Q0 = 0;
            }
        }
        if( q1_Q0 > 0 ) {
            q1_Q10  = silk_SUB32( silk_LSHIFT( q1_Q0, 10 ), QUANT_LEVEL_ADJUST_Q10 );
            q1_Q10  = silk_ADD32( q1_Q10, offset_Q10 );
            q2_Q10  = silk_ADD32( q1_Q10, 1024 );
            rd1_Q20 = silk_SMULBB( q1_Q10, Lambda_Q10 );
            rd2_Q20 = silk_SMULBB( q2_Q10, Lambda_Q10 );
        } else if( q1_Q0 == 0 ) {
            q1_Q10  = offset_Q10;
            q2_Q10  = silk_ADD32( q1_Q10, 1024 - QUANT_LEVEL_ADJUST_Q10 );
            rd1_Q20 = silk_SMULBB( q1_Q10, Lambda_Q10 );
            rd2_Q20 = silk_SMULBB( q2_Q10, Lambda_Q10 );
        } else if( q1_Q0 == -1 ) {
            q2_Q10  = offset_Q10;
            q1_Q10  = silk_SUB32( q2_Q10, 1024 - QUANT_LEVEL_ADJUST_Q10 );
            rd1_Q20 = silk_SMULBB( -q1_Q10, Lambda_Q10 );
            rd2_Q20 = silk_SMULBB(  q2_Q10, Lambda_Q10 );
        } else {            /* Q1_Q0 < -1 */
            q1_Q10  = silk_ADD32( silk_LSHIFT( q1_Q0, 10 ), QUANT_LEVEL_ADJUST_Q10 );
            q1_Q10  = silk_ADD32( q1_Q10, offset_Q10 );
            q2_Q10  = silk_ADD32( q1_Q10, 1024 );
            rd1_Q20 = silk_SMULBB( -q1_Q10, Lambda_Q10 );
            rd2_Q20 = silk_SMULBB( -q2_Q10, Lambda_Q10 );
        }
        rr_Q10  = silk_SUB32( r_Q10, q1_Q10 );
        rd1_Q20 = silk_SMLABB( rd1_Q20, rr_Q10, rr_Q10 );
        rr_Q10  = silk_SUB32( r_Q10, q2_Q10 );
        rd2_Q20 = silk_SMLABB( rd2_Q20, rr_Q10, rr_Q10 );

        if( rd2_Q20 < rd1_Q20 ) {
            q1_Q10 = q2_Q10;
        }

        pulses[ i ] = (opus_int8)silk_RSHIFT_ROUND( q1_Q10, 10 );

        /* Excitation */
        exc_Q14 = silk_LSHIFT( q1_Q10, 4 );
        if ( NSQ->rand_seed < 0 ) {
           exc_Q14 = -exc_Q14;
        }

        /* Add predictions */
        LPC_exc_Q14 = silk_ADD_LSHIFT32( exc_Q14, LTP_pred_Q13, 1 );
        xq_Q14      = silk_ADD32_ovflw( LPC_exc_Q14, silk_LSHIFT32( LPC_pred_Q10, 4 ) );

        /* Update states */
        psLPC_Q14++;
        *psLPC_Q14 = xq_Q14;
        lpc0 = wasm_i32x4_shuffle( lpc0, lpc1, 1, 2, 3, 4 );
        lpc1 = wasm_i32x4_shuffle( lpc1, lpc2, 1, 2, 3, 4 );
        lpc2 = wasm_i32x4_shuffle( lpc2, lpc3, 1, 2, 3, 4 );
        lpc3 = wasm_i32x4_shuffle( lpc3, wasm_i32x4_splat( xq_prev_Q14 ), 1, 2, 3, 4 );
        xq_prev_Q14 = xq_Q14;
        if( ar_vecs > 4 ) {
            ar5 = wasm_i32x4_shuffle( ar4, ar5, 3, 4, 5, 6 );
            ar4 = wasm_i32x4_shuffle( ar3, ar4, 3, 4, 5, 6 );
        }
        ar3 = wasm_i32x4_shuffle( ar2, ar3, 3, 4, 5, 6 );
        ar2 = wasm_i32x4_shuffle( ar1, ar2, 3, 4, 5, 6 );
        ar1 = wasm_i32x4_shuffle( ar0, ar1, 3, 4, 5, 6 );
        ar0 = wasm_i32x4_shuffle( wasm_i32x4_splat( sDiff_shp_Q14 ), ar0, 3, 4, 5, 6 );
        sDiff_shp_Q14 = silk_SUB32_ovflw( xq_Q14, silk_LSHIFT32( x_sc_Q10[ i ], 4 ) );
        sLF_AR_shp_Q14 = silk_SUB32_ovflw( sDiff_shp_Q14, silk_LSHIFT32( n_AR_Q12, 2 ) );

        NSQ->sLTP_shp_Q14[ NSQ->sLTP_shp_buf_idx ] = silk_SUB32_ovflw( sLF_AR_shp_Q14, silk_LSHIFT32( n_LF_Q12, 2 ) );
        sLTP_Q15[ NSQ->sLTP_buf_idx ] = silk_LSHIFT( LPC_exc_Q14, 1 );
        NSQ->sLTP_shp_buf_idx++;
        NSQ->sLTP_buf_idx++;

        /* Make dither dependent on quantized signal */
        NSQ->rand_seed = silk_ADD32_ovflw( NSQ->rand_seed, pulses[ i ] );
    }
    NSQ->sDiff_shp_Q14  = sDiff_shp_Q14;
    NSQ->sLF_AR_shp_Q14 = sLF_AR_shp_Q14;

    /* Save the noise shaping state; entries at and above shapingLPCOrder are unchanged */
    wasm_v128_store( &coef[ 0 ],  ar0 );
    wasm_v128_store( &coef[ 4 ],  ar1 );
    wasm_v128_store( &coef[ 8 ],  ar2 );
    wasm_v128_store( &coef[ 12 ], ar3 );
    if( ar_vecs > 4 ) {
        wasm_v128_store( &coef[ 16 ], ar4 );
        wasm_v128_store( &coef[ 20 ], ar5 );
    }
    silk_memcpy( NSQ->sAR2_Q14, coef, shapingLPCOrder * sizeof( opus_int32 ) );

    /* Scale XQ back to normal level before saving */
    {
        const v128_t gain = wasm_i32x4_splat( Gain_Q10 );
        psLPC_Q14 = &NSQ->sLPC_Q14[ NSQ_LPC_BUF_LENGTH ];
        for( i = 0; i < length - 7; i += 8 ) {
            v128_t xq0 = silk_SMULWW_wasm( wasm_v128_load( &psLPC_Q14[ i + 0 ] ), gain );
            v128_t xq1 = silk_SMULWW_wasm( wasm_v128_load( &psLPC_Q14[ i + 4 ] ), gain );
            /* silk_RSHIFT_ROUND( xq, 8 ), saturated to 16 bits */
            xq0 = wasm_i32x4_shr( wasm_i32x4_add( wasm_i32x4_shr( xq0, 7 ), wasm_i32x4_const_splat( 1 ) ), 1 );
            xq1 = wasm_i32x4_shr( wasm_i32x4_add( wasm_i32x4_shr( xq1, 7 ), wasm_i32x4_const_splat( 1 ) ), 1 );
            wasm_v128_store( &xq[ i ], wasm_i16x8_narrow_i32x4( xq0, xq1 ) );
        }
        for( ; i < length; i++ ) {
            xq[ i ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( silk_SMULWW( psLPC_Q14[ i ], Gain_Q10 ), 8 ) );
        }
    }

    /* Update LPC synth buffer */
    silk_memcpy( NSQ->sLPC_Q14, &NSQ->sLPC_Q14[ length ], NSQ_LPC_BUF_LENGTH * sizeof( opus_int32 ) );
}

void silk_NSQ_wasm(
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
    opus_int            k, lag, start_idx, LSF_interpolation_flag;
    const opus_int16    *A_Q12, *B_Q14, *AR_shp_Q13;
    opus_int16          *pxq;
    VARDECL( opus_int32, sLTP_Q15 );
    VARDECL( opus_int16, sLTP );
    opus_int32          HarmShapeFIRPacked_Q14;
    opus_int            offset_Q10;
    VARDECL( opus_int32, x_sc_Q10 );
    SAVE_STACK;

    NSQ->rand_seed = psIndices->Seed;

    /* Set unvoiced lag to the previous one, overwrite later for voiced */
    lag = NSQ->lagPrev;

    silk_assert( NSQ->prev_gain_Q16 != 0 );

    offset_Q10 = silk_Quantization_Offsets_Q10[ psIndices->signalType >> 1 ][ psIndices->quantOffsetType ];

    if( psIndices->NLSFInterpCoef_Q2 == 4 ) {
        LSF_interpolation_flag = 0;
    } else {
        LSF_interpolation_flag = 1;
    }

    ALLOC( sLTP_Q15, psEncC->ltp_mem_length + psEncC->frame_length, opus_int32 );
    ALLOC( sLTP, psEncC->ltp_mem_length + psEncC->frame_length, opus_int16 );
    ALLOC( x_sc_Q10, psEncC->subfr_length, opus_int32 );
    /* Set up pointers to start of sub frame */
    NSQ->sLTP_shp_buf_idx = psEncC->ltp_mem_length;
    NSQ->sLTP_buf_idx     = psEncC->ltp_mem_length;
    pxq                   = &NSQ->xq[ psEncC->ltp_mem_length ];
    for( k = 0; k < psEncC->nb_subfr; k++ ) {
        A_Q12      = &PredCoef_Q12[ (( k >> 1 ) | ( 1 - LSF_interpolation_flag )) * MAX_LPC_ORDER ];
        B_Q14      = &LTPCoef_Q14[ k * LTP_ORDER ];
        AR_shp_Q13 = &AR_Q13[ k * MAX_SHAPE_LPC_ORDER ];

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
                /* Rewhiten with new A coefs */
                start_idx = psEncC->ltp_mem_length - lag - psEncC->predictLPCOrder - LTP_ORDER / 2;
                celt_assert( start_idx > 0 );

                silk_LPC_analysis_filter( &sLTP[ start_idx ], &NSQ->xq[ start_idx + k * psEncC->subfr_length ],
                    A_Q12, psEncC->ltp_mem_length - start_idx, psEncC->predictLPCOrder, psEncC->arch );

                NSQ->rewhite_flag = 1;
                NSQ->sLTP_buf_idx = psEncC->ltp_mem_length;
            }
        }

        silk_nsq_scale_states_wasm( psEncC, NSQ, x16, x_sc_Q10, sLTP, sLTP_Q15, k, LTP_scale_Q14, Gains_Q16, pitchL, psIndices->signalType );

        if( psEncC->shapingLPCOrder <= 16 ) {
            silk_noise_shape_quantizer_wasm( NSQ, psIndices->signalType, x_sc_Q10, pulses, pxq, sLTP_Q15, A_Q12, B_Q14,
                AR_shp_Q13, lag, HarmShapeFIRPacked_Q14, Tilt_Q14[ k ], LF_shp_Q14[ k ], Gains_Q16[ k ], Lambda_Q10,
                offset_Q10, psEncC->subfr_length, psEncC->shapingLPCOrder, psEncC->predictLPCOrder, 4 );
        } else {
            silk_noise_shape_quantizer_wasm( NSQ, psIndices->signalType, x_sc_Q10, pulses, pxq, sLTP_Q15, A_Q12, B_Q14,
                AR_shp_Q13, lag, HarmShapeFIRPacked_Q14, Tilt_Q14[ k ], LF_shp_Q14[ k ], Gains_Q16[ k ], Lambda_Q10,
                offset_Q10, psEncC->subfr_length, psEncC->shapingLPCOrder, psEncC->predictLPCOrder, 6 );
        }

        x16    += psEncC->subfr_length;
        pulses += psEncC->subfr_length;
        pxq    += psEncC->subfr_length;
    }

    /* Update lagPrev for next frame */
    NSQ->lagPrev = pitchL[ psEncC->nb_subfr - 1 ];

    /* Save quantized speech and noise shaping signals */
    silk_memmove( NSQ->xq,           &NSQ->xq[           psEncC->frame_length ], psEncC->ltp_mem_length * sizeof( opus_int16 ) );
    silk_memmove( NSQ->sLTP_shp_Q14, &NSQ->sLTP_shp_Q14[ psEncC->frame_length ], psEncC->ltp_mem_length * sizeof( opus_int32 ) );
    RESTORE_STACK;
}
