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

/* silk_SMULWB() on each lane; the lanes of b hold 16-bit values. */
static OPUS_INLINE v128_t silk_SMULWB_wasm( v128_t a, v128_t b )
{
    v128_t hi = wasm_i32x4_mul( wasm_i32x4_shr( a, 16 ), b );
    v128_t lo = wasm_i32x4_mul( wasm_v128_and( a, wasm_i32x4_splat( 0x0000FFFF ) ), b );
    return wasm_i32x4_add( hi, wasm_i32x4_shr( lo, 16 ) );
}

/* Entropy constrained matrix-weighted VQ, hard-coded to 5-element vectors, for a single input data vector */
void silk_VQ_WMat_EC_wasm(
    opus_int8                   *ind,                           /* O    index of best codebook vector               */
    opus_int32                  *res_nrg_Q15,                   /* O    best residual energy                        */
    opus_int32                  *rate_dist_Q8,                  /* O    best total bitrate                          */
    opus_int                    *gain_Q7,                       /* O    sum of absolute LTP coefficients            */
    const opus_int32            *XX_Q17,                        /* I    correlation matrix                          */
    const opus_int32            *xX_Q17,                        /* I    correlation vector                          */
    const opus_int8             *cb_Q7,                         /* I    codebook                                    */
    const opus_uint8            *cb_gain_Q7,                    /* I    codebook effective gain                     */
    const opus_uint8            *cl_Q5,                         /* I    code length for each codebook vector        */
    const opus_int              subfr_len,                      /* I    number of samples per subframe              */
    const opus_int32            max_gain_Q7,                    /* I    maximum sum of absolute LTP coefficients    */
    const opus_int              L                               /* I    number of vectors in codebook               */
)
{
    opus_int   i, k, gain_tmp_Q7;
    const opus_int8 *cb_row_Q7;
    opus_int32 neg_xX_Q24[ 5 ];
    opus_int32 sum1_Q15, sum2_Q24;
    opus_int32 bits_res_Q8, bits_tot_Q8;
    opus_int32 sum1_Q15_buf[ 4 ];

    /* Negate and convert to new Q domain */
    neg_xX_Q24[ 0 ] = -silk_LSHIFT32( xX_Q17[ 0 ], 7 );
    neg_xX_Q24[ 1 ] = -silk_LSHIFT32( xX_Q17[ 1 ], 7 );
    neg_xX_Q24[ 2 ] = -silk_LSHIFT32( xX_Q17[ 2 ], 7 );
    neg_xX_Q24[ 3 ] = -silk_LSHIFT32( xX_Q17[ 3 ], 7 );
    neg_xX_Q24[ 4 ] = -silk_LSHIFT32( xX_Q17[ 4 ], 7 );

    /* Loop over codebook */
    *rate_dist_Q8 = silk_int32_MAX;
    *res_nrg_Q15 = silk_int32_MAX;
    cb_row_Q7 = cb_Q7;
    /* If things go really bad, at least *ind is set to something safe. */
    *ind = 0;
    for( k = 0; k < L; ) {
        opus_int n;
        if( k < L - 3 ) {
            /* Quantization error of four codebook vectors at once: lane i handles vector k + i.
               All sums wrap modulo 2^32 exactly like the 32-bit C arithmetic, so the order of the
               additions does not matter. */
            v128_t v0, v1, b0123, b4, t, cb0, cb1, cb2, cb3, cb4, sum2, sum1;

            /* Transpose the 4 x 5 codebook block (20 bytes) to one vector per coefficient */
            v0    = wasm_v128_load( cb_row_Q7 );
            v1    = wasm_v128_load( cb_row_Q7 + 4 );
            b0123 = wasm_i8x16_shuffle( v0, v1, 0, 5, 10, 15, 1, 6, 11, 28, 2, 7, 12, 29, 3, 8, 13, 30 );
            b4    = wasm_i8x16_shuffle( v0, v1, 4, 9, 14, 31, 4, 9, 14, 31, 4, 9, 14, 31, 4, 9, 14, 31 );
            t     = wasm_i16x8_extend_low_i8x16( b0123 );
            cb0   = wasm_i32x4_extend_low_i16x8( t );
            cb1   = wasm_i32x4_extend_high_i16x8( t );
            t     = wasm_i16x8_extend_high_i8x16( b0123 );
            cb2   = wasm_i32x4_extend_low_i16x8( t );
            cb3   = wasm_i32x4_extend_high_i16x8( t );
            cb4   = wasm_i32x4_extend_low_i16x8( wasm_i16x8_extend_low_i8x16( b4 ) );

            /* Weighted rate */
            /* Quantization error: 1 - 2 * xX * cb + cb' * XX * cb */
            sum1 = wasm_i32x4_splat( SILK_FIX_CONST( 1.001, 15 ) );

            /* first row of XX_Q17 */
            sum2 = wasm_i32x4_add( wasm_i32x4_splat( neg_xX_Q24[ 0 ] ), wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 1 ] ), cb1 ) );
            sum2 = wasm_i32x4_add( sum2, wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 2 ] ), cb2 ) );
            sum2 = wasm_i32x4_add( sum2, wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 3 ] ), cb3 ) );
            sum2 = wasm_i32x4_add( sum2, wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 4 ] ), cb4 ) );
            sum2 = wasm_i32x4_shl( sum2, 1 );
            sum2 = wasm_i32x4_add( sum2, wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 0 ] ), cb0 ) );
            sum1 = wasm_i32x4_add( sum1, silk_SMULWB_wasm( sum2, cb0 ) );

            /* second row of XX_Q17 */
            sum2 = wasm_i32x4_add( wasm_i32x4_splat( neg_xX_Q24[ 1 ] ), wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 7 ] ), cb2 ) );
            sum2 = wasm_i32x4_add( sum2, wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 8 ] ), cb3 ) );
            sum2 = wasm_i32x4_add( sum2, wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 9 ] ), cb4 ) );
            sum2 = wasm_i32x4_shl( sum2, 1 );
            sum2 = wasm_i32x4_add( sum2, wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 6 ] ), cb1 ) );
            sum1 = wasm_i32x4_add( sum1, silk_SMULWB_wasm( sum2, cb1 ) );

            /* third row of XX_Q17 */
            sum2 = wasm_i32x4_add( wasm_i32x4_splat( neg_xX_Q24[ 2 ] ), wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 13 ] ), cb3 ) );
            sum2 = wasm_i32x4_add( sum2, wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 14 ] ), cb4 ) );
            sum2 = wasm_i32x4_shl( sum2, 1 );
            sum2 = wasm_i32x4_add( sum2, wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 12 ] ), cb2 ) );
            sum1 = wasm_i32x4_add( sum1, silk_SMULWB_wasm( sum2, cb2 ) );

            /* fourth row of XX_Q17 */
            sum2 = wasm_i32x4_add( wasm_i32x4_splat( neg_xX_Q24[ 3 ] ), wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 19 ] ), cb4 ) );
            sum2 = wasm_i32x4_shl( sum2, 1 );
            sum2 = wasm_i32x4_add( sum2, wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 18 ] ), cb3 ) );
            sum1 = wasm_i32x4_add( sum1, silk_SMULWB_wasm( sum2, cb3 ) );

            /* last row of XX_Q17 */
            sum2 = wasm_i32x4_splat( silk_LSHIFT32( neg_xX_Q24[ 4 ], 1 ) );
            sum2 = wasm_i32x4_add( sum2, wasm_i32x4_mul( wasm_i32x4_splat( XX_Q17[ 24 ] ), cb4 ) );
            sum1 = wasm_i32x4_add( sum1, silk_SMULWB_wasm( sum2, cb4 ) );

            wasm_v128_store( sum1_Q15_buf, sum1 );
            n = 4;
        } else {
            /* Leftover codebook vectors, same as the C code */
            sum1_Q15 = SILK_FIX_CONST( 1.001, 15 );

            /* first row of XX_Q17 */
            sum2_Q24 = silk_MLA( neg_xX_Q24[ 0 ], XX_Q17[  1 ], cb_row_Q7[ 1 ] );
            sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  2 ], cb_row_Q7[ 2 ] );
            sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  3 ], cb_row_Q7[ 3 ] );
            sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  4 ], cb_row_Q7[ 4 ] );
            sum2_Q24 = silk_LSHIFT32( sum2_Q24, 1 );
            sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  0 ], cb_row_Q7[ 0 ] );
            sum1_Q15 = silk_SMLAWB( sum1_Q15,        sum2_Q24,  cb_row_Q7[ 0 ] );

            /* second row of XX_Q17 */
            sum2_Q24 = silk_MLA( neg_xX_Q24[ 1 ], XX_Q17[  7 ], cb_row_Q7[ 2 ] );
            sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  8 ], cb_row_Q7[ 3 ] );
            sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  9 ], cb_row_Q7[ 4 ] );
            sum2_Q24 = silk_LSHIFT32( sum2_Q24, 1 );
            sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  6 ], cb_row_Q7[ 1 ] );
            sum1_Q15 = silk_SMLAWB( sum1_Q15,        sum2_Q24,  cb_row_Q7[ 1 ] );

            /* third row of XX_Q17 */
            sum2_Q24 = silk_MLA( neg_xX_Q24[ 2 ], XX_Q17[ 13 ], cb_row_Q7[ 3 ] );
            sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[ 14 ], cb_row_Q7[ 4 ] );
            sum2_Q24 = silk_LSHIFT32( sum2_Q24, 1 );
            sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[ 12 ], cb_row_Q7[ 2 ] );
            sum1_Q15 = silk_SMLAWB( sum1_Q15,        sum2_Q24,  cb_row_Q7[ 2 ] );

            /* fourth row of XX_Q17 */
            sum2_Q24 = silk_MLA( neg_xX_Q24[ 3 ], XX_Q17[ 19 ], cb_row_Q7[ 4 ] );
            sum2_Q24 = silk_LSHIFT32( sum2_Q24, 1 );
            sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[ 18 ], cb_row_Q7[ 3 ] );
            sum1_Q15 = silk_SMLAWB( sum1_Q15,        sum2_Q24,  cb_row_Q7[ 3 ] );

            /* last row of XX_Q17 */
            sum2_Q24 = silk_LSHIFT32( neg_xX_Q24[ 4 ], 1 );
            sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[ 24 ], cb_row_Q7[ 4 ] );
            sum1_Q15 = silk_SMLAWB( sum1_Q15,        sum2_Q24,  cb_row_Q7[ 4 ] );

            sum1_Q15_buf[ 0 ] = sum1_Q15;
            n = 1;
        }

        /* find best, in codebook order */
        for( i = 0; i < n; i++, k++ ) {
            opus_int32 penalty;
            sum1_Q15 = sum1_Q15_buf[ i ];
            gain_tmp_Q7 = cb_gain_Q7[ k ];

            /* Penalty for too large gain */
            penalty = silk_LSHIFT32( silk_max( silk_SUB32( gain_tmp_Q7, max_gain_Q7 ), 0 ), 11 );

            if( sum1_Q15 >= 0 ) {
                /* Translate residual energy to bits using high-rate assumption (6 dB ==> 1 bit/sample) */
                bits_res_Q8 = silk_SMULBB( subfr_len, silk_lin2log( sum1_Q15 + penalty) - (15 << 7) );
                /* In the following line we reduce the codelength component by half ("-1"); seems to slightly improve quality */
                bits_tot_Q8 = silk_ADD_LSHIFT32( bits_res_Q8, cl_Q5[ k ], 3-1 );
                if( bits_tot_Q8 <= *rate_dist_Q8 ) {
                    *rate_dist_Q8 = bits_tot_Q8;
                    *res_nrg_Q15 = sum1_Q15 + penalty;
                    *ind = (opus_int8)k;
                    *gain_Q7 = gain_tmp_Q7;
                }
            }
        }

        /* Go to next cbk vector */
        cb_row_Q7 += n * LTP_ORDER;
    }
}
