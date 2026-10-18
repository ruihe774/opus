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
#include "tuning_parameters.h"
#include "pitch.h"

#ifdef FIXED_POINT

#define MAX_FRAME_SIZE              384             /* subfr_length * nb_subfr = ( 0.005 * 16000 + 16 ) * 4 = 384 */

#define QA                          25
#define N_BITS_HEAD_ROOM            3
#define MIN_RSHIFTS                 -16
#define MAX_RSHIFTS                 (32 - QA)

/* All the vectorized loops below only change the order of 32-bit additions, which
   wrap modulo 2^32 exactly like the C code, so the results are bit-exact. */

/* silk_SMMUL( a, b ) on each lane */
static OPUS_INLINE v128_t silk_SMMUL_wasm( const v128_t a, const v128_t b )
{
    return wasm_i32x4_shuffle( wasm_i64x2_extmul_low_i32x4( a, b ), wasm_i64x2_extmul_high_i32x4( a, b ), 1, 3, 5, 7 );
}

/* x[ 3 ], x[ 2 ], x[ 1 ], x[ 0 ] */
static OPUS_INLINE v128_t silk_load_rev32x4_wasm( const opus_int32 *x )
{
    v128_t v = wasm_v128_load( x );
    return wasm_i32x4_shuffle( v, v, 3, 2, 1, 0 );
}

static OPUS_INLINE opus_int32 silk_hsum32_wasm( v128_t v )
{
    v = wasm_i32x4_add( v, wasm_i32x4_shuffle( v, v, 2, 3, 0, 1 ) );
    v = wasm_i32x4_add( v, wasm_i32x4_shuffle( v, v, 1, 0, 3, 2 ) );
    return wasm_i32x4_extract_lane( v, 0 );
}

/* Compute reflection coefficients from input signal */
void silk_burg_modified_wasm(
    opus_int32                  *res_nrg,           /* O    Residual energy                                             */
    opus_int                    *res_nrg_Q,         /* O    Residual energy Q value                                     */
    opus_int32                  A_Q16[],            /* O    Prediction coefficients (length order)                      */
    const opus_int16            x[],                /* I    Input signal, length: nb_subfr * ( D + subfr_length )       */
    const opus_int32            minInvGain_Q30,     /* I    Inverse of max prediction gain                              */
    const opus_int              subfr_length,       /* I    Input signal subframe length (incl. D preceding samples)    */
    const opus_int              nb_subfr,           /* I    Number of subframes stacked in x                            */
    const opus_int              D,                  /* I    Order                                                       */
    int                         arch                /* I    Run-time architecture                                       */
)
{
    opus_int         k, n, s, lz, rshifts, reached_max_gain;
    opus_int32       C0, num, nrg, rc_Q31, invGain_Q30, Atmp_QA, Atmp1, tmp1, tmp2, x1, x2;
    const opus_int16 *x_ptr;
    opus_int32       C_first_row[ SILK_MAX_ORDER_LPC ];
    opus_int32       C_last_row[  SILK_MAX_ORDER_LPC ];
    opus_int32       Af_QA[       SILK_MAX_ORDER_LPC ];
    opus_int32       CAf[ SILK_MAX_ORDER_LPC + 1 ];
    opus_int32       CAb[ SILK_MAX_ORDER_LPC + 1 ];
    opus_int32       xcorr[ SILK_MAX_ORDER_LPC ];
    opus_int64       C0_64;
    /* Scratch for the vectorized loops */
    opus_int32       Atmp1_buf[   SILK_MAX_ORDER_LPC ];
    opus_int32       lsh_mul[     SILK_MAX_ORDER_LPC ];
    opus_int         lsh_buf[     SILK_MAX_ORDER_LPC ];
    opus_int32       Af_new[      SILK_MAX_ORDER_LPC ];
    opus_int32       CAf_new[ SILK_MAX_ORDER_LPC + 1 ];
    opus_int32       CAb_new[ SILK_MAX_ORDER_LPC + 1 ];

    celt_assert( subfr_length * nb_subfr <= MAX_FRAME_SIZE );

    /* Compute autocorrelations, added over subframes.
       Make sure the noise corresponds to at least a unit pulse. */
    C0_64 = silk_inner_prod16_wasm( x, x, subfr_length*nb_subfr ) + 1;
    lz = silk_CLZ64(C0_64);
    rshifts = 32 + 1 + N_BITS_HEAD_ROOM - lz;
    if (rshifts > MAX_RSHIFTS) rshifts = MAX_RSHIFTS;
    if (rshifts < MIN_RSHIFTS) rshifts = MIN_RSHIFTS;

    if (rshifts > 0) {
        C0 = (opus_int32)silk_RSHIFT64(C0_64, rshifts );
    } else {
        C0 = silk_LSHIFT32((opus_int32)C0_64, -rshifts );
    }

    CAb[ 0 ] = CAf[ 0 ] = C0 + silk_SMMUL( SILK_FIX_CONST( FIND_LPC_COND_FAC, 32 ), C0 ) + 1;                                /* Q(-rshifts) */
    silk_memset( C_first_row, 0, SILK_MAX_ORDER_LPC * sizeof( opus_int32 ) );
    if( rshifts > 0 ) {
        for( s = 0; s < nb_subfr; s++ ) {
            x_ptr = x + s * subfr_length;
            for( n = 1; n < D + 1; n++ ) {
                C_first_row[ n - 1 ] += (opus_int32)silk_RSHIFT64(
                    silk_inner_prod16_wasm( x_ptr, x_ptr + n, subfr_length - n ), rshifts );
            }
        }
    } else {
        for( s = 0; s < nb_subfr; s++ ) {
            int i;
            opus_int32 d;
            x_ptr = x + s * subfr_length;
            celt_pitch_xcorr(x_ptr, x_ptr + 1, xcorr, subfr_length - D, D, arch );
            for( n = 1; n < D + 1; n++ ) {
               for ( i = n + subfr_length - D, d = 0; i < subfr_length; i++ )
                  d = MAC16_16( d, x_ptr[ i ], x_ptr[ i - n ] );
               xcorr[ n - 1 ] += d;
            }
            for( n = 1; n < D + 1; n++ ) {
                C_first_row[ n - 1 ] += silk_LSHIFT32( xcorr[ n - 1 ], -rshifts );
            }
        }
    }
    silk_memcpy( C_last_row, C_first_row, SILK_MAX_ORDER_LPC * sizeof( opus_int32 ) );

    /* Initialize */
    CAb[ 0 ] = CAf[ 0 ] = C0 + silk_SMMUL( SILK_FIX_CONST( FIND_LPC_COND_FAC, 32 ), C0 ) + 1;                                /* Q(-rshifts) */

    invGain_Q30 = (opus_int32)1 << 30;
    reached_max_gain = 0;
    for( n = 0; n < D; n++ ) {
        /* Update first row of correlation matrix (without first element) */
        /* Update last row of correlation matrix (without last element, stored in reversed order) */
        /* Update C * Af */
        /* Update C * flipud(Af) (stored in reversed order) */
        /* Plain C: this part gains nothing from SIMD128 over what the compiler generates */
        if( rshifts > -2 ) {
            for( s = 0; s < nb_subfr; s++ ) {
                x_ptr = x + s * subfr_length;
                x1  = -silk_LSHIFT32( (opus_int32)x_ptr[ n ],                    16 - rshifts );        /* Q(16-rshifts) */
                x2  = -silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n - 1 ], 16 - rshifts );        /* Q(16-rshifts) */
                tmp1 = silk_LSHIFT32( (opus_int32)x_ptr[ n ],                    QA - 16 );             /* Q(QA-16) */
                tmp2 = silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n - 1 ], QA - 16 );             /* Q(QA-16) */
                for( k = 0; k < n; k++ ) {
                    C_first_row[ k ] = silk_SMLAWB( C_first_row[ k ], x1, x_ptr[ n - k - 1 ]            ); /* Q( -rshifts ) */
                    C_last_row[ k ]  = silk_SMLAWB( C_last_row[ k ],  x2, x_ptr[ subfr_length - n + k ] ); /* Q( -rshifts ) */
                    Atmp_QA = Af_QA[ k ];
                    tmp1 = silk_SMLAWB( tmp1, Atmp_QA, x_ptr[ n - k - 1 ]            );                 /* Q(QA-16) */
                    tmp2 = silk_SMLAWB( tmp2, Atmp_QA, x_ptr[ subfr_length - n + k ] );                 /* Q(QA-16) */
                }
                tmp1 = silk_LSHIFT32( -tmp1, 32 - QA - rshifts );                                       /* Q(16-rshifts) */
                tmp2 = silk_LSHIFT32( -tmp2, 32 - QA - rshifts );                                       /* Q(16-rshifts) */
                for( k = 0; k <= n; k++ ) {
                    CAf[ k ] = silk_SMLAWB( CAf[ k ], tmp1, x_ptr[ n - k ]                    );        /* Q( -rshift ) */
                    CAb[ k ] = silk_SMLAWB( CAb[ k ], tmp2, x_ptr[ subfr_length - n + k - 1 ] );        /* Q( -rshift ) */
                }
            }
        } else {
            for( s = 0; s < nb_subfr; s++ ) {
                x_ptr = x + s * subfr_length;
                x1  = -silk_LSHIFT32( (opus_int32)x_ptr[ n ],                    -rshifts );            /* Q( -rshifts ) */
                x2  = -silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n - 1 ], -rshifts );            /* Q( -rshifts ) */
                tmp1 = silk_LSHIFT32( (opus_int32)x_ptr[ n ],                    17 );                  /* Q17 */
                tmp2 = silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n - 1 ], 17 );                  /* Q17 */
                for( k = 0; k < n; k++ ) {
                    C_first_row[ k ] = silk_MLA( C_first_row[ k ], x1, x_ptr[ n - k - 1 ]            ); /* Q( -rshifts ) */
                    C_last_row[ k ]  = silk_MLA( C_last_row[ k ],  x2, x_ptr[ subfr_length - n + k ] ); /* Q( -rshifts ) */
                    Atmp1 = silk_RSHIFT_ROUND( Af_QA[ k ], QA - 17 );                                   /* Q17 */
                    /* We sometimes get overflows in the multiplications (even beyond +/- 2^32),
                       but they cancel each other and the real result seems to always fit in a 32-bit
                       signed integer. This was determined experimentally, not theoretically (unfortunately). */
                    tmp1 = silk_MLA_ovflw( tmp1, x_ptr[ n - k - 1 ],            Atmp1 );                      /* Q17 */
                    tmp2 = silk_MLA_ovflw( tmp2, x_ptr[ subfr_length - n + k ], Atmp1 );                      /* Q17 */
                }
                tmp1 = -tmp1;                                                                           /* Q17 */
                tmp2 = -tmp2;                                                                           /* Q17 */
                for( k = 0; k <= n; k++ ) {
                    CAf[ k ] = silk_SMLAWW( CAf[ k ], tmp1,
                        silk_LSHIFT32( (opus_int32)x_ptr[ n - k ], -rshifts - 1 ) );                    /* Q( -rshift ) */
                    CAb[ k ] = silk_SMLAWW( CAb[ k ], tmp2,
                        silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n + k - 1 ], -rshifts - 1 ) ); /* Q( -rshift ) */
                }
            }
        }

        /* Calculate nominator and denominator for the next order reflection (parcor) coefficient */
        tmp1 = C_first_row[ n ];                                                                        /* Q( -rshifts ) */
        tmp2 = C_last_row[ n ];                                                                         /* Q( -rshifts ) */
        num  = 0;                                                                                       /* Q( -rshifts ) */
        nrg  = silk_ADD32( CAb[ 0 ], CAf[ 0 ] );                                                        /* Q( 1-rshifts ) */
        for( k = 0; k < n; k++ ) {
            Atmp_QA = Af_QA[ k ];
            lz = silk_CLZ32( silk_abs( Atmp_QA ) ) - 1;
            lz = silk_min( 32 - QA, lz );
            Atmp1_buf[ k ] = silk_LSHIFT32( Atmp_QA, lz );                                              /* Q( QA + lz ) */
            lsh_buf[ k ] = 32 - QA - lz;
            /* silk_ADD_LSHIFT32( a, b, lsh_buf ) == a + b * lsh_mul (mod 2^32) */
            lsh_mul[ k ] = silk_LSHIFT32( 1, lsh_buf[ k ] );
        }
        {
            v128_t T1, T2, N, E;
            T1 = T2 = N = E = wasm_i32x4_splat( 0 );
            for( k = 0; k < n - 3; k += 4 ) {
                v128_t A, M, t;
                A = wasm_v128_load( &Atmp1_buf[ k ] );
                M = wasm_v128_load( &lsh_mul[ k ] );
                t = silk_SMMUL_wasm( silk_load_rev32x4_wasm( &C_last_row[ n - k - 4 ] ), A );
                T1 = wasm_i32x4_add( T1, wasm_i32x4_mul( t, M ) );
                t = silk_SMMUL_wasm( silk_load_rev32x4_wasm( &C_first_row[ n - k - 4 ] ), A );
                T2 = wasm_i32x4_add( T2, wasm_i32x4_mul( t, M ) );
                t = silk_SMMUL_wasm( silk_load_rev32x4_wasm( &CAb[ n - k - 3 ] ), A );
                N  = wasm_i32x4_add( N,  wasm_i32x4_mul( t, M ) );
                t = wasm_i32x4_add( wasm_v128_load( &CAb[ k + 1 ] ), wasm_v128_load( &CAf[ k + 1 ] ) );
                t = silk_SMMUL_wasm( t, A );
                E  = wasm_i32x4_add( E,  wasm_i32x4_mul( t, M ) );
            }
            tmp1 += silk_hsum32_wasm( T1 );
            tmp2 += silk_hsum32_wasm( T2 );
            num  += silk_hsum32_wasm( N );
            nrg  += silk_hsum32_wasm( E );
        }
        for( ; k < n; k++ ) {
            Atmp1 = Atmp1_buf[ k ];                                                                     /* Q( QA + lz ) */
            tmp1 = silk_ADD32( tmp1, silk_LSHIFT32( silk_SMMUL( C_last_row[  n - k - 1 ], Atmp1 ), lsh_buf[ k ] ) );  /* Q( -rshifts ) */
            tmp2 = silk_ADD32( tmp2, silk_LSHIFT32( silk_SMMUL( C_first_row[ n - k - 1 ], Atmp1 ), lsh_buf[ k ] ) );  /* Q( -rshifts ) */
            num  = silk_ADD32( num,  silk_LSHIFT32( silk_SMMUL( CAb[ n - k ],             Atmp1 ), lsh_buf[ k ] ) );  /* Q( -rshifts ) */
            nrg  = silk_ADD32( nrg,  silk_LSHIFT32( silk_SMMUL( silk_ADD32( CAb[ k + 1 ], CAf[ k + 1 ] ),
                                                                                Atmp1 ), lsh_buf[ k ] ) );    /* Q( 1-rshifts ) */
        }
        CAf[ n + 1 ] = tmp1;                                                                            /* Q( -rshifts ) */
        CAb[ n + 1 ] = tmp2;                                                                            /* Q( -rshifts ) */
        num = silk_ADD32( num, tmp2 );                                                                  /* Q( -rshifts ) */
        num = silk_LSHIFT32( -num, 1 );                                                                 /* Q( 1-rshifts ) */

        /* Calculate the next order reflection (parcor) coefficient */
        if( silk_abs( num ) < nrg ) {
            rc_Q31 = silk_DIV32_varQ( num, nrg, 31 );
        } else {
            rc_Q31 = ( num > 0 ) ? silk_int32_MAX : silk_int32_MIN;
        }

        /* Update inverse prediction gain */
        tmp1 = ( (opus_int32)1 << 30 ) - silk_SMMUL( rc_Q31, rc_Q31 );
        tmp1 = silk_LSHIFT( silk_SMMUL( invGain_Q30, tmp1 ), 2 );
        if( tmp1 <= minInvGain_Q30 ) {
            /* Max prediction gain exceeded; set reflection coefficient such that max prediction gain is exactly hit */
            tmp2 = ( (opus_int32)1 << 30 ) - silk_DIV32_varQ( minInvGain_Q30, invGain_Q30, 30 );            /* Q30 */
            rc_Q31 = silk_SQRT_APPROX( tmp2 );                                                  /* Q15 */
            if( rc_Q31 > 0 ) {
                /* Newton-Raphson iteration */
                rc_Q31 = silk_RSHIFT32( rc_Q31 + silk_DIV32( tmp2, rc_Q31 ), 1 );                       /* Q15 */
                rc_Q31 = silk_LSHIFT32( rc_Q31, 16 );                                                   /* Q31 */
                if( num < 0 ) {
                    /* Ensure adjusted reflection coefficients has the original sign */
                    rc_Q31 = -rc_Q31;
                }
            }
            invGain_Q30 = minInvGain_Q30;
            reached_max_gain = 1;
        } else {
            invGain_Q30 = tmp1;
        }

        /* Update the AR coefficients: Af_QA[ k ] += SMMUL( Af_QA[ n - k - 1 ], rc_Q31 ) << 1, all from the old values */
        {
            const v128_t rc = wasm_i32x4_splat( rc_Q31 );
            for( k = 0; k < n - 3; k += 4 ) {
                v128_t t = silk_SMMUL_wasm( silk_load_rev32x4_wasm( &Af_QA[ n - k - 4 ] ), rc );
                wasm_v128_store( &Af_new[ k ], wasm_i32x4_add( wasm_v128_load( &Af_QA[ k ] ), wasm_i32x4_shl( t, 1 ) ) );
            }
            for( ; k < n; k++ ) {
                Af_new[ k ] = silk_ADD_LSHIFT32( Af_QA[ k ], silk_SMMUL( Af_QA[ n - k - 1 ], rc_Q31 ), 1 );  /* QA */
            }
            silk_memcpy( Af_QA, Af_new, n * sizeof( opus_int32 ) );
        }
        Af_QA[ n ] = silk_RSHIFT32( rc_Q31, 31 - QA );                                          /* QA */

        if( reached_max_gain ) {
            /* Reached max prediction gain; set remaining coefficients to zero and exit loop */
            for( k = n + 1; k < D; k++ ) {
                Af_QA[ k ] = 0;
            }
            break;
        }

        /* Update C * Af and C * Ab, all from the old values:
           CAf[ k ] += SMMUL( CAb[ n + 1 - k ], rc_Q31 ) << 1; CAb[ k ] += SMMUL( CAf[ n + 1 - k ], rc_Q31 ) << 1 */
        {
            const v128_t rc = wasm_i32x4_splat( rc_Q31 );
            for( k = 0; k < n - 1; k += 4 ) {
                v128_t t;
                t = silk_SMMUL_wasm( silk_load_rev32x4_wasm( &CAb[ n - k - 2 ] ), rc );
                wasm_v128_store( &CAf_new[ k ], wasm_i32x4_add( wasm_v128_load( &CAf[ k ] ), wasm_i32x4_shl( t, 1 ) ) );
                t = silk_SMMUL_wasm( silk_load_rev32x4_wasm( &CAf[ n - k - 2 ] ), rc );
                wasm_v128_store( &CAb_new[ k ], wasm_i32x4_add( wasm_v128_load( &CAb[ k ] ), wasm_i32x4_shl( t, 1 ) ) );
            }
            for( ; k <= n + 1; k++ ) {
                CAf_new[ k ] = silk_ADD_LSHIFT32( CAf[ k ], silk_SMMUL( CAb[ n - k + 1 ], rc_Q31 ), 1 );  /* Q( -rshifts ) */
                CAb_new[ k ] = silk_ADD_LSHIFT32( CAb[ k ], silk_SMMUL( CAf[ n - k + 1 ], rc_Q31 ), 1 );  /* Q( -rshifts ) */
            }
            silk_memcpy( CAf, CAf_new, ( n + 2 ) * sizeof( opus_int32 ) );
            silk_memcpy( CAb, CAb_new, ( n + 2 ) * sizeof( opus_int32 ) );
        }
    }

    if( reached_max_gain ) {
        for( k = 0; k < D; k++ ) {
            /* Scale coefficients */
            A_Q16[ k ] = -silk_RSHIFT_ROUND( Af_QA[ k ], QA - 16 );
        }
        /* Subtract energy of preceding samples from C0 */
        if( rshifts > 0 ) {
            for( s = 0; s < nb_subfr; s++ ) {
                x_ptr = x + s * subfr_length;
                C0 -= (opus_int32)silk_RSHIFT64( silk_inner_prod16_wasm( x_ptr, x_ptr, D ), rshifts );
            }
        } else {
            for( s = 0; s < nb_subfr; s++ ) {
                x_ptr = x + s * subfr_length;
                C0 -= silk_LSHIFT32( silk_inner_prod_aligned( x_ptr, x_ptr, D, arch ), -rshifts );
            }
        }
        /* Approximate residual energy */
        *res_nrg = silk_LSHIFT( silk_SMMUL( invGain_Q30, C0 ), 2 );
        *res_nrg_Q = -rshifts;
    } else {
        /* Return residual energy */
        nrg  = CAf[ 0 ];                                                                            /* Q( -rshifts ) */
        tmp1 = (opus_int32)1 << 16;                                                                             /* Q16 */
        for( k = 0; k < D; k++ ) {
            Atmp1 = silk_RSHIFT_ROUND( Af_QA[ k ], QA - 16 );                                       /* Q16 */
            nrg  = silk_SMLAWW( nrg, CAf[ k + 1 ], Atmp1 );                                         /* Q( -rshifts ) */
            tmp1 = silk_SMLAWW( tmp1, Atmp1, Atmp1 );                                               /* Q16 */
            A_Q16[ k ] = -Atmp1;
        }
        *res_nrg = silk_SMLAWW( nrg, silk_SMMUL( SILK_FIX_CONST( FIND_LPC_COND_FAC, 32 ), C0 ), -tmp1 );/* Q( -rshifts ) */
        *res_nrg_Q = -rshifts;
    }
}

#endif /* FIXED_POINT */
