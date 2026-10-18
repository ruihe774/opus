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
#include "SigProc_FIX.h"
#include "define.h"

#define QA                          24
#define A_LIMIT                     SILK_FIX_CONST( 0.99975, QA )

#define MUL32_FRAC_Q(a32, b32, Q)   ((opus_int32)(silk_RSHIFT_ROUND64(silk_SMULL(a32, b32), Q)))

/* High 32 bits of the four 64-bit lanes of lo and hi. */
static OPUS_INLINE v128_t silk_high32_i64_wasm( const v128_t lo, const v128_t hi )
{
    return wasm_i32x4_shuffle( lo, hi, 1, 3, 5, 7 );
}

/* MUL32_FRAC_Q( a, b, 31 ) on each lane, for b != silk_int32_MIN.
   ( p + 2^30 ) >> 31 is taken as the upper half of 2 * p + 2^31, which cannot overflow,
   since 64-bit arithmetic right shifts are expensive with SIMD128. */
static OPUS_INLINE v128_t silk_MUL32_FRAC_Q31_wasm( const v128_t a, const v128_t b )
{
    const v128_t rnd = wasm_i64x2_splat( (opus_int64)1 << 31 );
    v128_t lo = wasm_i64x2_extmul_low_i32x4( a, b );
    v128_t hi = wasm_i64x2_extmul_high_i32x4( a, b );
    lo = wasm_i64x2_add( wasm_i64x2_add( lo, lo ), rnd );
    hi = wasm_i64x2_add( wasm_i64x2_add( hi, hi ), rnd );
    return silk_high32_i64_wasm( lo, hi );
}

/* silk_SUB_SAT32( a, b ) on each lane */
static OPUS_INLINE v128_t silk_SUB_SAT32_wasm( const v128_t a, const v128_t b )
{
    v128_t r, ovfl, sat;
    r    = wasm_i32x4_sub( a, b );
    /* All ones in lanes where a and b have different signs and the sign of the result differs from a */
    ovfl = wasm_i32x4_shr( wasm_v128_and( wasm_v128_xor( a, b ), wasm_v128_xor( a, r ) ), 31 );
    sat  = wasm_v128_xor( wasm_i32x4_shr( a, 31 ), wasm_i32x4_splat( silk_int32_MAX ) );
#if defined(__wasm_relaxed_simd__)
    /* ovfl is all ones or all zeros in each lane, so this is exact */
    return wasm_i32x4_relaxed_laneselect( sat, r, ovfl );
#else
    return wasm_v128_bitselect( sat, r, ovfl );
#endif
}

/* silk_RSHIFT_ROUND64( silk_SMULL( a, b ), shift ) on each lane, for 1 <= shift <= 31.
   *bad gets all ones in lanes where the result does not fit in 32 bits; otherwise
   ( p + 2^( shift - 1 ) ) << ( 32 - shift ) does not overflow, and its upper half is the result. */
static OPUS_INLINE v128_t silk_SMULL_RSHIFT_ROUND_wasm( const v128_t a, const v128_t b, const opus_int shift, v128_t *bad )
{
    const v128_t rnd = wasm_i64x2_splat( (opus_int64)1 << ( shift - 1 ) );
    const v128_t max = wasm_i64x2_splat( ( (opus_int64)1 << ( 31 + shift ) ) - 1 );
    const v128_t min = wasm_i64x2_splat( -( (opus_int64)1 << ( 31 + shift ) ) );
    v128_t lo, hi;
    /* |a * b| < 2^62, so adding the rounding offset cannot overflow */
    lo = wasm_i64x2_add( wasm_i64x2_extmul_low_i32x4( a, b ), rnd );
    hi = wasm_i64x2_add( wasm_i64x2_extmul_high_i32x4( a, b ), rnd );
    *bad = wasm_v128_or( *bad, wasm_v128_or( wasm_i64x2_gt( lo, max ), wasm_i64x2_lt( lo, min ) ) );
    *bad = wasm_v128_or( *bad, wasm_v128_or( wasm_i64x2_gt( hi, max ), wasm_i64x2_lt( hi, min ) ) );
    lo = wasm_i64x2_shl( lo, 32 - shift );
    hi = wasm_i64x2_shl( hi, 32 - shift );
    return silk_high32_i64_wasm( lo, hi );
}

/* Compute inverse of LPC prediction gain, and                          */
/* test if LPC coefficients are stable (all poles within unit circle)   */
static opus_int32 LPC_inverse_pred_gain_QA_wasm(            /* O   Returns inverse prediction gain in energy domain, Q30    */
    opus_int32           A_QA[ SILK_MAX_ORDER_LPC ],        /* I   Prediction coefficients                                  */
    const opus_int       order                              /* I   Prediction order                                         */
)
{
    opus_int   k, n, mult2Q;
    opus_int32 invGain_Q30, rc_Q31, rc_mult1_Q30, rc_mult2, tmp1, tmp2;

    invGain_Q30 = SILK_FIX_CONST( 1, 30 );
    for( k = order - 1; k > 0; k-- ) {
        v128_t rc_Q31_v, rc_mult2_v, bad;

        /* Check for stability */
        if( ( A_QA[ k ] > A_LIMIT ) || ( A_QA[ k ] < -A_LIMIT ) ) {
            return 0;
        }

        /* Set RC equal to negated AR coef */
        rc_Q31 = -silk_LSHIFT( A_QA[ k ], 31 - QA );

        /* rc_mult1_Q30 range: [ 1 : 2^30 ] */
        rc_mult1_Q30 = silk_SUB32( SILK_FIX_CONST( 1, 30 ), silk_SMMUL( rc_Q31, rc_Q31 ) );
        silk_assert( rc_mult1_Q30 > ( 1 << 15 ) );                   /* reduce A_LIMIT if fails */
        silk_assert( rc_mult1_Q30 <= ( 1 << 30 ) );

        /* Update inverse gain */
        /* invGain_Q30 range: [ 0 : 2^30 ] */
        invGain_Q30 = silk_LSHIFT( silk_SMMUL( invGain_Q30, rc_mult1_Q30 ), 2 );
        silk_assert( invGain_Q30 >= 0           );
        silk_assert( invGain_Q30 <= ( 1 << 30 ) );
        if( invGain_Q30 < SILK_FIX_CONST( 1.0f / MAX_PREDICTION_POWER_GAIN, 30 ) ) {
            return 0;
        }

        /* rc_mult2 range: [ 2^30 : silk_int32_MAX ] */
        mult2Q = 32 - silk_CLZ32( silk_abs( rc_mult1_Q30 ) );
        rc_mult2 = silk_INVERSE32_varQ( rc_mult1_Q30, mult2Q + 30 );

        /* Update AR coefficient */
        rc_Q31_v   = wasm_i32x4_splat( rc_Q31 );
        rc_mult2_v = wasm_i32x4_splat( rc_mult2 );
        bad        = wasm_i32x4_splat( 0 );
        for( n = 0; n < ( ( k + 1 ) >> 1 ) - 3; n += 4 ) {
            /* tmp1 = A_QA[ n + {0,1,2,3} ], tmp2 = A_QA[ k - n - 1 - {0,1,2,3} ].
               When k is odd, the middle coefficient may be in both vectors; both then
               compute and store the same value. */
            v128_t t1, t2, u1, u2;
            t1 = wasm_v128_load( &A_QA[ n ] );
            t2 = wasm_v128_load( &A_QA[ k - n - 4 ] );
            t2 = wasm_i32x4_shuffle( t2, t2, 3, 2, 1, 0 );
            u1 = silk_SUB_SAT32_wasm( t1, silk_MUL32_FRAC_Q31_wasm( t2, rc_Q31_v ) );
            u2 = silk_SUB_SAT32_wasm( t2, silk_MUL32_FRAC_Q31_wasm( t1, rc_Q31_v ) );
            u1 = silk_SMULL_RSHIFT_ROUND_wasm( u1, rc_mult2_v, mult2Q, &bad );
            u2 = silk_SMULL_RSHIFT_ROUND_wasm( u2, rc_mult2_v, mult2Q, &bad );
            wasm_v128_store( &A_QA[ n ], u1 );
            wasm_v128_store( &A_QA[ k - n - 4 ], wasm_i32x4_shuffle( u2, u2, 3, 2, 1, 0 ) );
        }
        /* The C code returns as soon as one coefficient overflows; the result is the same */
        if( wasm_v128_any_true( bad ) ) {
            return 0;
        }
        for( ; n < (k + 1) >> 1; n++ ) {
            opus_int64 tmp64;
            tmp1 = A_QA[ n ];
            tmp2 = A_QA[ k - n - 1 ];
            tmp64 = silk_RSHIFT_ROUND64( silk_SMULL( silk_SUB_SAT32(tmp1,
                  MUL32_FRAC_Q( tmp2, rc_Q31, 31 ) ), rc_mult2 ), mult2Q);
            if( tmp64 > silk_int32_MAX || tmp64 < silk_int32_MIN ) {
               return 0;
            }
            A_QA[ n ] = ( opus_int32 )tmp64;
            tmp64 = silk_RSHIFT_ROUND64( silk_SMULL( silk_SUB_SAT32(tmp2,
                  MUL32_FRAC_Q( tmp1, rc_Q31, 31 ) ), rc_mult2), mult2Q);
            if( tmp64 > silk_int32_MAX || tmp64 < silk_int32_MIN ) {
               return 0;
            }
            A_QA[ k - n - 1 ] = ( opus_int32 )tmp64;
        }
    }

    /* Check for stability */
    if( ( A_QA[ k ] > A_LIMIT ) || ( A_QA[ k ] < -A_LIMIT ) ) {
        return 0;
    }

    /* Set RC equal to negated AR coef */
    rc_Q31 = -silk_LSHIFT( A_QA[ 0 ], 31 - QA );

    /* Range: [ 1 : 2^30 ] */
    rc_mult1_Q30 = silk_SUB32( SILK_FIX_CONST( 1, 30 ), silk_SMMUL( rc_Q31, rc_Q31 ) );

    /* Update inverse gain */
    /* Range: [ 0 : 2^30 ] */
    invGain_Q30 = silk_LSHIFT( silk_SMMUL( invGain_Q30, rc_mult1_Q30 ), 2 );
    silk_assert( invGain_Q30 >= 0           );
    silk_assert( invGain_Q30 <= ( 1 << 30 ) );
    if( invGain_Q30 < SILK_FIX_CONST( 1.0f / MAX_PREDICTION_POWER_GAIN, 30 ) ) {
        return 0;
    }

    return invGain_Q30;
}

/* For input in Q12 domain */
opus_int32 silk_LPC_inverse_pred_gain_wasm(         /* O   Returns inverse prediction gain in energy domain, Q30        */
    const opus_int16            *A_Q12,             /* I   Prediction coefficients, Q12 [order]                         */
    const opus_int              order               /* I   Prediction order                                             */
)
{
    opus_int   k;
    opus_int32 Atmp_QA[ SILK_MAX_ORDER_LPC ];
    opus_int32 DC_resp;
    v128_t     dc;

    /* Increase Q domain of the AR coefficients */
    dc = wasm_i32x4_splat( 0 );
    for( k = 0; k < order - 7; k += 8 ) {
        v128_t a = wasm_v128_load( &A_Q12[ k ] );
        dc = wasm_i32x4_add( dc, wasm_i32x4_dot_i16x8( a, wasm_i16x8_splat( 1 ) ) );
        wasm_v128_store( &Atmp_QA[ k + 0 ], wasm_i32x4_shl( wasm_i32x4_extend_low_i16x8( a ), QA - 12 ) );
        wasm_v128_store( &Atmp_QA[ k + 4 ], wasm_i32x4_shl( wasm_i32x4_extend_high_i16x8( a ), QA - 12 ) );
    }
    dc = wasm_i32x4_add( dc, wasm_i32x4_shuffle( dc, dc, 2, 3, 0, 1 ) );
    dc = wasm_i32x4_add( dc, wasm_i32x4_shuffle( dc, dc, 1, 0, 3, 2 ) );
    DC_resp = wasm_i32x4_extract_lane( dc, 0 );
    for( ; k < order; k++ ) {
        DC_resp += (opus_int32)A_Q12[ k ];
        Atmp_QA[ k ] = silk_LSHIFT32( (opus_int32)A_Q12[ k ], QA - 12 );
    }
    /* If the DC is unstable, we don't even need to do the full calculations */
    if( DC_resp >= 4096 ) {
        return 0;
    }
    return LPC_inverse_pred_gain_QA_wasm( Atmp_QA, order );
}
