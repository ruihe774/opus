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

/* Weighting factors for tilt measure */
static const opus_int32 tiltWeights[ VAD_N_BANDS ] = { 30000, 6000, -12000, -12000 };

/* Sum of ( x[ i ] >> 3 )^2. Each square is at most 2^24, so (as in the C code)
   the 32-bit accumulation cannot overflow for len <= 128. */
static OPUS_INLINE opus_int32 silk_VAD_energy_wasm( const opus_int16 *x, opus_int len )
{
    opus_int   i;
    opus_int32 sum;
    v128_t     acc, v;

    acc = wasm_i32x4_splat( 0 );
    for( i = 0; i < len - 7; i += 8 ) {
        v   = wasm_i16x8_shr( wasm_v128_load( &x[ i ] ), 3 );
        acc = wasm_i32x4_add( acc, wasm_i32x4_dot_i16x8( v, v ) );
    }
    if( i < len - 3 ) {
        /* Upper four lanes are zero and contribute nothing */
        v   = wasm_i16x8_shr( wasm_v128_load64_zero( &x[ i ] ), 3 );
        acc = wasm_i32x4_add( acc, wasm_i32x4_dot_i16x8( v, v ) );
        i += 4;
    }
    acc = wasm_i32x4_add( acc, wasm_i32x4_shuffle( acc, acc, 2, 3, 0, 1 ) );
    acc = wasm_i32x4_add( acc, wasm_i32x4_shuffle( acc, acc, 1, 0, 3, 2 ) );
    sum = wasm_i32x4_extract_lane( acc, 0 );

    for( ; i < len; i++ ) {
        opus_int32 x_tmp = silk_RSHIFT( x[ i ], 3 );
        sum = silk_SMLABB( sum, x_tmp, x_tmp );
    }
    return sum;
}

/***************************************/
/* Get the speech activity level in Q8 */
/***************************************/
opus_int silk_VAD_GetSA_Q8_wasm(                    /* O    Return value, 0 if success                  */
    silk_encoder_state          *psEncC,            /* I/O  Encoder state                               */
    const opus_int16            pIn[]               /* I    PCM input                                   */
)
{
    opus_int   SA_Q15, pSNR_dB_Q7, input_tilt;
    opus_int   decimated_framelength1, decimated_framelength2;
    opus_int   decimated_framelength;
    opus_int   dec_subframe_length, dec_subframe_offset, SNR_Q7, i, b, s;
    opus_int32 sumSquared, smooth_coef_Q16;
    opus_int16 HPstateTmp;
    VARDECL( opus_int16, X );
    opus_int32 Xnrg[ VAD_N_BANDS ];
    opus_int32 NrgToNoiseRatio_Q8[ VAD_N_BANDS ];
    opus_int32 speech_nrg;
    opus_int   X_offset[ VAD_N_BANDS ];
    opus_int   ret = 0;
    silk_VAD_state *psSilk_VAD = &psEncC->sVAD;

    SAVE_STACK;

    /* Safety checks */
    silk_assert( VAD_N_BANDS == 4 );
    celt_assert( MAX_FRAME_LENGTH >= psEncC->frame_length );
    celt_assert( psEncC->frame_length <= 512 );
    celt_assert( psEncC->frame_length == 8 * silk_RSHIFT( psEncC->frame_length, 3 ) );

    /***********************/
    /* Filter and Decimate */
    /***********************/
    decimated_framelength1 = silk_RSHIFT( psEncC->frame_length, 1 );
    decimated_framelength2 = silk_RSHIFT( psEncC->frame_length, 2 );
    decimated_framelength = silk_RSHIFT( psEncC->frame_length, 3 );
    /* Decimate into 4 bands:
       0       L      3L       L              3L                             5L
               -      --       -              --                             --
               8       8       2               4                              4

       [0-1 kHz| temp. |1-2 kHz|    2-4 kHz    |            4-8 kHz           |

       They're arranged to allow the minimal ( frame_length / 4 ) extra
       scratch space during the downsampling process */
    X_offset[ 0 ] = 0;
    X_offset[ 1 ] = decimated_framelength + decimated_framelength2;
    X_offset[ 2 ] = X_offset[ 1 ] + decimated_framelength;
    X_offset[ 3 ] = X_offset[ 2 ] + decimated_framelength2;
    ALLOC( X, X_offset[ 3 ] + decimated_framelength1, opus_int16 );

    /* 0-8 kHz to 0-4 kHz and 4-8 kHz */
    silk_ana_filt_bank_1( pIn, &psSilk_VAD->AnaState[  0 ],
        X, &X[ X_offset[ 3 ] ], psEncC->frame_length );

    /* 0-4 kHz to 0-2 kHz and 2-4 kHz */
    silk_ana_filt_bank_1( X, &psSilk_VAD->AnaState1[ 0 ],
        X, &X[ X_offset[ 2 ] ], decimated_framelength1 );

    /* 0-2 kHz to 0-1 kHz and 1-2 kHz */
    silk_ana_filt_bank_1( X, &psSilk_VAD->AnaState2[ 0 ],
        X, &X[ X_offset[ 1 ] ], decimated_framelength2 );

    /*********************************************/
    /* HP filter on lowest band (differentiator) */
    /*********************************************/
    /* X[ i ] = ( X[ i ] >> 1 ) - ( X[ i - 1 ] >> 1 ), processed from the end so
       that every load still sees the unmodified input. The difference of two
       halved 16-bit values always fits in 16 bits. */
    HPstateTmp = silk_RSHIFT( X[ decimated_framelength - 1 ], 1 );
    for( i = decimated_framelength - 8; i > 0; i -= 8 ) {
        v128_t cur  = wasm_i16x8_shr( wasm_v128_load( &X[ i ] ), 1 );
        v128_t prev = wasm_i16x8_shr( wasm_v128_load( &X[ i - 1 ] ), 1 );
        wasm_v128_store( &X[ i ], wasm_i16x8_sub( cur, prev ) );
    }
    for( i += 7; i > 0; i-- ) {
        X[ i ] = silk_RSHIFT( X[ i ], 1 ) - silk_RSHIFT( X[ i - 1 ], 1 );
    }
    X[ 0 ] = silk_RSHIFT( X[ 0 ], 1 ) - psSilk_VAD->HPstate;
    psSilk_VAD->HPstate = HPstateTmp;

    /*************************************/
    /* Calculate the energy in each band */
    /*************************************/
    for( b = 0; b < VAD_N_BANDS; b++ ) {
        /* Find the decimated framelength in the non-uniformly divided bands */
        decimated_framelength = silk_RSHIFT( psEncC->frame_length, silk_min_int( VAD_N_BANDS - b, VAD_N_BANDS - 1 ) );

        /* Split length into subframe lengths */
        dec_subframe_length = silk_RSHIFT( decimated_framelength, VAD_INTERNAL_SUBFRAMES_LOG2 );
        dec_subframe_offset = 0;

        /* Compute energy per sub-frame */
        /* initialize with summed energy of last subframe */
        Xnrg[ b ] = psSilk_VAD->XnrgSubfr[ b ];
        sumSquared = 0;
        for( s = 0; s < VAD_INTERNAL_SUBFRAMES; s++ ) {
            sumSquared = silk_VAD_energy_wasm( &X[ X_offset[ b ] + dec_subframe_offset ], dec_subframe_length );

            /* Safety check */
            silk_assert( sumSquared >= 0 );

            /* Add/saturate summed energy of current subframe */
            if( s < VAD_INTERNAL_SUBFRAMES - 1 ) {
                Xnrg[ b ] = silk_ADD_POS_SAT32( Xnrg[ b ], sumSquared );
            } else {
                /* Look-ahead subframe */
                Xnrg[ b ] = silk_ADD_POS_SAT32( Xnrg[ b ], silk_RSHIFT( sumSquared, 1 ) );
            }

            dec_subframe_offset += dec_subframe_length;
        }
        psSilk_VAD->XnrgSubfr[ b ] = sumSquared;
    }

    /********************/
    /* Noise estimation */
    /********************/
    silk_VAD_GetNoiseLevels( &Xnrg[ 0 ], psSilk_VAD );

    /***********************************************/
    /* Signal-plus-noise to noise ratio estimation */
    /***********************************************/
    sumSquared = 0;
    input_tilt = 0;
    for( b = 0; b < VAD_N_BANDS; b++ ) {
        speech_nrg = Xnrg[ b ] - psSilk_VAD->NL[ b ];
        if( speech_nrg > 0 ) {
            /* Divide, with sufficient resolution */
            if( ( Xnrg[ b ] & 0xFF800000 ) == 0 ) {
                NrgToNoiseRatio_Q8[ b ] = silk_DIV32( silk_LSHIFT( Xnrg[ b ], 8 ), psSilk_VAD->NL[ b ] + 1 );
            } else {
                NrgToNoiseRatio_Q8[ b ] = silk_DIV32( Xnrg[ b ], silk_RSHIFT( psSilk_VAD->NL[ b ], 8 ) + 1 );
            }

            /* Convert to log domain */
            SNR_Q7 = silk_lin2log( NrgToNoiseRatio_Q8[ b ] ) - 8 * 128;

            /* Sum-of-squares */
            sumSquared = silk_SMLABB( sumSquared, SNR_Q7, SNR_Q7 );          /* Q14 */

            /* Tilt measure */
            if( speech_nrg < ( (opus_int32)1 << 20 ) ) {
                /* Scale down SNR value for small subband speech energies */
                SNR_Q7 = silk_SMULWB( silk_LSHIFT( silk_SQRT_APPROX( speech_nrg ), 6 ), SNR_Q7 );
            }
            input_tilt = silk_SMLAWB( input_tilt, tiltWeights[ b ], SNR_Q7 );
        } else {
            NrgToNoiseRatio_Q8[ b ] = 256;
        }
    }

    /* Mean-of-squares */
    sumSquared = silk_DIV32_16( sumSquared, VAD_N_BANDS ); /* Q14 */

    /* Root-mean-square approximation, scale to dBs, and write to output pointer */
    pSNR_dB_Q7 = (opus_int16)( 3 * silk_SQRT_APPROX( sumSquared ) ); /* Q7 */

    /*********************************/
    /* Speech Probability Estimation */
    /*********************************/
    SA_Q15 = silk_sigm_Q15( silk_SMULWB( VAD_SNR_FACTOR_Q16, pSNR_dB_Q7 ) - VAD_NEGATIVE_OFFSET_Q5 );

    /**************************/
    /* Frequency Tilt Measure */
    /**************************/
    psEncC->input_tilt_Q15 = silk_LSHIFT( silk_sigm_Q15( input_tilt ) - 16384, 1 );

    /**************************************************/
    /* Scale the sigmoid output based on power levels */
    /**************************************************/
    speech_nrg = 0;
    for( b = 0; b < VAD_N_BANDS; b++ ) {
        /* Accumulate signal-without-noise energies, higher frequency bands have more weight */
        speech_nrg += ( b + 1 ) * silk_RSHIFT( Xnrg[ b ] - psSilk_VAD->NL[ b ], 4 );
    }

    if( psEncC->frame_length == 20 * psEncC->fs_kHz ) {
        speech_nrg = silk_RSHIFT32( speech_nrg, 1 );
    }
    /* Power scaling */
    if( speech_nrg <= 0 ) {
        SA_Q15 = silk_RSHIFT( SA_Q15, 1 );
    } else if( speech_nrg < 16384 ) {
        speech_nrg = silk_LSHIFT32( speech_nrg, 16 );

        /* square-root */
        speech_nrg = silk_SQRT_APPROX( speech_nrg );
        SA_Q15 = silk_SMULWB( 32768 + speech_nrg, SA_Q15 );
    }

    /* Copy the resulting speech activity in Q8 */
    psEncC->speech_activity_Q8 = silk_min_int( silk_RSHIFT( SA_Q15, 7 ), silk_uint8_MAX );

    /***********************************/
    /* Energy Level and SNR estimation */
    /***********************************/
    /* Smoothing coefficient */
    smooth_coef_Q16 = silk_SMULWB( VAD_SNR_SMOOTH_COEF_Q18, silk_SMULWB( (opus_int32)SA_Q15, SA_Q15 ) );

    if( psEncC->frame_length == 10 * psEncC->fs_kHz ) {
        smooth_coef_Q16 >>= 1;
    }

    for( b = 0; b < VAD_N_BANDS; b++ ) {
        /* compute smoothed energy-to-noise ratio per band */
        psSilk_VAD->NrgRatioSmth_Q8[ b ] = silk_SMLAWB( psSilk_VAD->NrgRatioSmth_Q8[ b ],
            NrgToNoiseRatio_Q8[ b ] - psSilk_VAD->NrgRatioSmth_Q8[ b ], smooth_coef_Q16 );

        /* signal to noise ratio in dB per band */
        SNR_Q7 = 3 * ( silk_lin2log( psSilk_VAD->NrgRatioSmth_Q8[b] ) - 8 * 128 );
        /* quality = sigmoid( 0.25 * ( SNR_dB - 16 ) ); */
        psEncC->input_quality_bands_Q15[ b ] = silk_sigm_Q15( silk_RSHIFT( SNR_Q7 - 16 * 128, 4 ) );
    }

    RESTORE_STACK;
    return( ret );
}
