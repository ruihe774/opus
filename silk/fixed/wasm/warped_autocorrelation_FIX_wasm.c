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
#include "stack_alloc.h"
#include "main_FIX.h"

#ifdef FIXED_POINT

/* Number of allpass sections computed per pass */
#define NB_TAPS 8

/* With s_0( n ) = input[ n ] << QS and s_i( n ) the output of allpass section i - 1,
   the C code computes
       s_i( n )    = s_{i-1}( n - 1 ) + silk_SMULWB( s_i( n - 1 ) - s_{i-1}( n ), warping_Q16 )
       corr_QC[ i ] = sum_n ( s_i( n ) * s_0( n ) ) >> ( 2 * QS - QC ).
   Each pass handles NB_TAPS sections i0 + 1 ... i0 + NB_TAPS along a skewed wavefront:
   at step t, lane L holds s_{i0+NB_TAPS-L}( t - NB_TAPS + L ), so all the inputs of a step
   are in the previous two steps, and the output s_{i0+NB_TAPS}( n ) is stored as the
   input of the next pass. */
void silk_warped_autocorrelation_FIX_wasm(
          opus_int32                *corr,                                  /* O    Result [order + 1]                                                          */
          opus_int                  *scale,                                 /* O    Scaling of the correlation vector                                           */
    const opus_int16                *input,                                 /* I    Input data to correlate                                                     */
    const opus_int                  warping_Q16,                            /* I    Warping coefficient                                                         */
    const opus_int                  length,                                 /* I    Length of input                                                             */
    const opus_int                  order                                   /* I    Correlation order (even)                                                    */
)
{
    opus_int   n, i, j, t, i0, lsh;
    opus_int64 corr_QC[ MAX_SHAPE_LPC_ORDER + NB_TAPS + 1 ];
    opus_int32 *x_in, *in_QS, *out_QS;
    VARDECL( opus_int32, x_buf );
    VARDECL( opus_int32, state_buf );
    v128_t     acc, warping;
    const v128_t low_mask = wasm_i32x4_splat( ( 1 << ( QS - QC ) ) - 1 );
    SAVE_STACK;

    /* Order must be even */
    celt_assert( ( order & 1 ) == 0 );
    silk_assert( QS >= QC );
    silk_assert( order <= MAX_SHAPE_LPC_ORDER );

    /* x_buf:     NB_TAPS zeros, input[], NB_TAPS zeros (correlation window, Q0)
       state_buf: two buffers holding s_i( -NB_TAPS ... length + NB_TAPS - 1 ), where
                  s_i( n ) for n < 0 is never read and s_i( n ) for n >= length is zero.
                  The values beyond length do not affect the correlations, since they
                  are multiplied with zero input samples. */
    ALLOC( x_buf, length + 2 * NB_TAPS, opus_int32 );
    ALLOC( state_buf, 2 * ( length + 2 * NB_TAPS ), opus_int32 );
    x_in = x_buf + NB_TAPS;
    in_QS = state_buf + NB_TAPS;
    out_QS = state_buf + length + 3 * NB_TAPS;

    acc = wasm_i64x2_splat( 0 );
    for( n = 0; n < length - 7; n += 8 ) {
        v128_t x16 = wasm_v128_load( &input[ n ] );
        v128_t lo = wasm_i32x4_extend_low_i16x8( x16 );
        v128_t hi = wasm_i32x4_extend_high_i16x8( x16 );
        v128_t sq = wasm_i32x4_dot_i16x8( x16, x16 );   /* Sum of two squares, in [ 0, 2^31 ] as unsigned */
        acc = wasm_i64x2_add( acc, wasm_u64x2_extend_low_u32x4( sq ) );
        acc = wasm_i64x2_add( acc, wasm_u64x2_extend_high_u32x4( sq ) );
        wasm_v128_store( &x_in[ n ], lo );
        wasm_v128_store( &x_in[ n + 4 ], hi );
        wasm_v128_store( &in_QS[ n ], wasm_i32x4_shl( lo, QS ) );
        wasm_v128_store( &in_QS[ n + 4 ], wasm_i32x4_shl( hi, QS ) );
    }
    corr_QC[ 0 ] = wasm_i64x2_extract_lane( acc, 0 ) + wasm_i64x2_extract_lane( acc, 1 );
    for( ; n < length; n++ ) {
        x_in[ n ] = input[ n ];
        in_QS[ n ] = silk_LSHIFT32( (opus_int32)input[ n ], QS );
        corr_QC[ 0 ] += silk_SMULL( input[ n ], input[ n ] );
    }
    /* s_0( n ) * s_0( n ) >> ( 2 * QS - QC ) is exact */
    corr_QC[ 0 ] = silk_LSHIFT64( corr_QC[ 0 ], QC );
    for( n = 0; n < NB_TAPS; n++ ) {
        x_in[ n - NB_TAPS ] = 0;
        x_in[ length + n ] = 0;
        in_QS[ length + n ] = 0;
        out_QS[ length + n ] = 0;
    }

    warping = wasm_i32x4_splat( (opus_int16)warping_Q16 );
    for( i0 = 0; i0 < order; i0 += NB_TAPS ) {
        /* V: state wavefront at step t, W_prev: shifted wavefront of step t - 1 (see below) */
        v128_t V_lo, V_hi, W_prev_lo, W_prev_hi;
        /* ( s_i( n ) * s_0( n ) ) >> ( 2 * QS - QC ) == p >> ( QS - QC ), with p = s_i( n ) * input[ n ].
           Their sum is computed exactly as ( sum_n p - sum_n ( p & low_mask ) ) >> ( QS - QC ),
           which needs no 64-bit shifts; p & low_mask only depends on the low 32 bits of p. */
        v128_t acc64[ 4 ], acc_low_lo, acc_low_hi;
        opus_int32 *tmp;

        V_lo = V_hi = W_prev_lo = W_prev_hi = wasm_i32x4_splat( 0 );
        acc64[ 0 ] = acc64[ 1 ] = acc64[ 2 ] = acc64[ 3 ] = wasm_i64x2_splat( 0 );
        acc_low_lo = acc_low_hi = wasm_i32x4_splat( 0 );
        for( t = 0; t < length + NB_TAPS; t++ ) {
            v128_t x_lo, x_hi, W_lo, W_hi, D, D_hi, D_lo;

            /* Correlate lane L with input[ t - NB_TAPS + L ] */
            x_lo = wasm_v128_load( &x_in[ t - 8 ] );
            x_hi = wasm_v128_load( &x_in[ t - 4 ] );
            acc64[ 0 ] = wasm_i64x2_add( acc64[ 0 ], wasm_i64x2_extmul_low_i32x4( V_lo, x_lo ) );
            acc64[ 1 ] = wasm_i64x2_add( acc64[ 1 ], wasm_i64x2_extmul_high_i32x4( V_lo, x_lo ) );
            acc64[ 2 ] = wasm_i64x2_add( acc64[ 2 ], wasm_i64x2_extmul_low_i32x4( V_hi, x_hi ) );
            acc64[ 3 ] = wasm_i64x2_add( acc64[ 3 ], wasm_i64x2_extmul_high_i32x4( V_hi, x_hi ) );
            acc_low_lo = wasm_i32x4_add( acc_low_lo, wasm_v128_and( wasm_i32x4_mul( V_lo, x_lo ), low_mask ) );
            acc_low_hi = wasm_i32x4_add( acc_low_hi, wasm_v128_and( wasm_i32x4_mul( V_hi, x_hi ), low_mask ) );

            /* Output of the last section of this pass: s_{i0+8}( t - 8 ) */
            wasm_v128_store32_lane( &out_QS[ t - NB_TAPS ], V_lo, 0 );

            /* W( t ) = { V( t )[ 1 ... 7 ], s_{i0}( t ) } holds s_{i-1}( n ) for the lanes of V( t + 1 ) */
            W_lo = wasm_i32x4_shuffle( V_lo, V_hi, 1, 2, 3, 4 );
            W_hi = wasm_i32x4_shuffle( V_hi, wasm_v128_load32_zero( &in_QS[ t ] ), 1, 2, 3, 4 );

            /* V( t + 1 ) = W( t - 1 ) + silk_SMULWB( V( t ) - W( t ), warping_Q16 ) */
            D    = wasm_i32x4_sub( V_lo, W_lo );
            D_hi = wasm_i32x4_mul( wasm_i32x4_shr( D, 16 ), warping );
            D_lo = wasm_i32x4_mul( wasm_v128_and( D, wasm_i32x4_splat( 0x0000FFFF ) ), warping );
            V_lo = wasm_i32x4_add( W_prev_lo, wasm_i32x4_add( D_hi, wasm_i32x4_shr( D_lo, 16 ) ) );
            D    = wasm_i32x4_sub( V_hi, W_hi );
            D_hi = wasm_i32x4_mul( wasm_i32x4_shr( D, 16 ), warping );
            D_lo = wasm_i32x4_mul( wasm_v128_and( D, wasm_i32x4_splat( 0x0000FFFF ) ), warping );
            V_hi = wasm_i32x4_add( W_prev_hi, wasm_i32x4_add( D_hi, wasm_i32x4_shr( D_lo, 16 ) ) );

            W_prev_lo = W_lo;
            W_prev_hi = W_hi;
        }

        /* Lane L holds section i0 + NB_TAPS - L */
        {
            opus_int64 p[ NB_TAPS ];
            opus_int32 low[ NB_TAPS ];
            for( j = 0; j < 4; j++ ) {
                p[ 2 * j + 0 ] = wasm_i64x2_extract_lane( acc64[ j ], 0 );
                p[ 2 * j + 1 ] = wasm_i64x2_extract_lane( acc64[ j ], 1 );
            }
            wasm_v128_store( &low[ 0 ], acc_low_lo );
            wasm_v128_store( &low[ 4 ], acc_low_hi );
            for( j = 0; j < NB_TAPS; j++ ) {
                corr_QC[ i0 + NB_TAPS - j ] = silk_RSHIFT64( p[ j ] - low[ j ], QS - QC );
            }
        }

        /* The output of this pass is the input of the next one */
        tmp = in_QS;
        in_QS = out_QS;
        out_QS = tmp;
    }

    lsh = silk_CLZ64( corr_QC[ 0 ] ) - 35;
    lsh = silk_LIMIT( lsh, -12 - QC, 30 - QC );
    *scale = -( QC + lsh );
    silk_assert( *scale >= -30 && *scale <= 12 );
    if( lsh >= 0 ) {
        for( i = 0; i < order + 1; i++ ) {
            corr[ i ] = (opus_int32)silk_CHECK_FIT32( silk_LSHIFT64( corr_QC[ i ], lsh ) );
        }
    } else {
        for( i = 0; i < order + 1; i++ ) {
            corr[ i ] = (opus_int32)silk_CHECK_FIT32( silk_RSHIFT64( corr_QC[ i ], -lsh ) );
        }
    }
    silk_assert( corr_QC[ 0 ] >= 0 ); /* If breaking, decrease QC*/
    RESTORE_STACK;
}

#endif /* FIXED_POINT */
