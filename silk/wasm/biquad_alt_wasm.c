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

/* silk_SMULWB( a, b ) on each lane, with a split into its upper (signed) and lower
   (unsigned) 16 bits beforehand, and b holding 16-bit values. This is exactly
   ( a * b ) >> 16 computed with 64-bit precision, without needing 64-bit lanes. */
static OPUS_INLINE v128_t silk_SMULWB_split_wasm( const v128_t a_hi, const v128_t a_lo, const v128_t b )
{
    return wasm_i32x4_add( wasm_i32x4_mul( a_hi, b ), wasm_i32x4_shr( wasm_i32x4_mul( a_lo, b ), 16 ) );
}

/* One time step for both channels.
   State lanes: S{0,1,2,3} = { S[ 0 ], S[ 2 ], S[ 1 ], S[ 3 ] }.
   t0_dup = silk_SMULWB( B_Q28[ 0 ], in{0,1,0,1} ),
   t12    = silk_SMULWB( B_Q28[ {1,1,2,2} ], in{0,1,0,1} ).
   Returns out32_Q14{0,1,0,1}. */
static OPUS_INLINE v128_t silk_biquad_alt_stride2_step_wasm( v128_t *S, const v128_t t0_dup, const v128_t t12,
                                                             const v128_t A_L, const v128_t A_U )
{
    v128_t out32_Q14, o_hi, o_lo, t_L, t_U;

    /* out32_Q14{0,1} = silk_LSHIFT( silk_SMLAWB( S{0,2}, B_Q28[ 0 ], in{0,1} ), 2 ) */
    out32_Q14 = wasm_i32x4_shuffle( *S, *S, 0, 1, 0, 1 );
    out32_Q14 = wasm_i32x4_shl( wasm_i32x4_add( out32_Q14, t0_dup ), 2 );
    o_hi = wasm_i32x4_shr( out32_Q14, 16 );
    o_lo = wasm_v128_and( out32_Q14, wasm_i32x4_splat( 0x0000FFFF ) );

    /* silk_RSHIFT_ROUND( silk_SMULWB( out32_Q14, A{0,0,1,1}_L_Q28 ), 14 ) */
    t_L = silk_SMULWB_split_wasm( o_hi, o_lo, A_L );
    t_L = wasm_i32x4_shr( wasm_i32x4_add( wasm_i32x4_shr( t_L, 13 ), wasm_i32x4_splat( 1 ) ), 1 );
    /* silk_SMULWB( out32_Q14, A{0,0,1,1}_U_Q28 ) */
    t_U = silk_SMULWB_split_wasm( o_hi, o_lo, A_U );

    /* S{0,2} = S{1,3} + ...; S{1,3} = ... */
    *S = wasm_i32x4_shuffle( *S, wasm_i32x4_splat( 0 ), 2, 3, 4, 4 );
    *S = wasm_i32x4_add( wasm_i32x4_add( *S, t_L ), wasm_i32x4_add( t_U, t12 ) );
    return out32_Q14;
}

void silk_biquad_alt_stride2_wasm(
    const opus_int16            *in,                /* I     input signal                                               */
    const opus_int32            *B_Q28,             /* I     MA coefficients [3]                                        */
    const opus_int32            *A_Q28,             /* I     AR coefficients [2]                                        */
    opus_int32                  *S,                 /* I/O   State vector [4]                                           */
    opus_int16                  *out,               /* O     output signal                                              */
    const opus_int32            len                 /* I     signal length (must be even)                               */
)
{
    /* DIRECT FORM II TRANSPOSED (uses 2 element state vector) */
    opus_int   k;
    opus_int32 A0_U_Q28, A0_L_Q28, A1_U_Q28, A1_L_Q28;
    v128_t     A_L, A_U, B0_hi, B0_lo, B1_hi, B1_lo, B2_hi, B2_lo, S_v;
    const v128_t offset = wasm_i32x4_splat( ( 1 << 14 ) - 1 );

    /* Negate A_Q28 values and split in two parts */
    A0_L_Q28 = ( -A_Q28[ 0 ] ) & 0x00003FFF;        /* lower part */
    A0_U_Q28 = silk_RSHIFT( -A_Q28[ 0 ], 14 );      /* upper part */
    A1_L_Q28 = ( -A_Q28[ 1 ] ) & 0x00003FFF;        /* lower part */
    A1_U_Q28 = silk_RSHIFT( -A_Q28[ 1 ], 14 );      /* upper part */

    /* silk_SMLAWB() only uses the lower 16 bits of its last argument */
    A_L = wasm_i32x4_make( A0_L_Q28, A0_L_Q28, A1_L_Q28, A1_L_Q28 );
    A_U = wasm_i32x4_make( (opus_int16)A0_U_Q28, (opus_int16)A0_U_Q28, (opus_int16)A1_U_Q28, (opus_int16)A1_U_Q28 );

    /* The MA part does not depend on the state: B_Q28[] are split for silk_SMULWB_split_wasm() */
    B0_hi = wasm_i32x4_splat( silk_RSHIFT( B_Q28[ 0 ], 16 ) );
    B0_lo = wasm_i32x4_splat( B_Q28[ 0 ] & 0x0000FFFF );
    B1_hi = wasm_i32x4_splat( silk_RSHIFT( B_Q28[ 1 ], 16 ) );
    B1_lo = wasm_i32x4_splat( B_Q28[ 1 ] & 0x0000FFFF );
    B2_hi = wasm_i32x4_splat( silk_RSHIFT( B_Q28[ 2 ], 16 ) );
    B2_lo = wasm_i32x4_splat( B_Q28[ 2 ] & 0x0000FFFF );

    S_v = wasm_i32x4_make( S[ 0 ], S[ 2 ], S[ 1 ], S[ 3 ] );

    for( k = 0; k < len - 3; k += 4 ) {
        v128_t in16, in32[ 2 ], o[ 4 ], out_lo, out_hi;
        int j;

        in16 = wasm_v128_load( &in[ 2 * k ] );
        in32[ 0 ] = wasm_i32x4_extend_low_i16x8( in16 );    /* in{0,1} for samples k, k + 1     */
        in32[ 1 ] = wasm_i32x4_extend_high_i16x8( in16 );   /* in{0,1} for samples k + 2, k + 3 */
        for( j = 0; j < 2; j++ ) {
            v128_t t0, t1, t2;
            t0 = silk_SMULWB_split_wasm( B0_hi, B0_lo, in32[ j ] );
            t1 = silk_SMULWB_split_wasm( B1_hi, B1_lo, in32[ j ] );
            t2 = silk_SMULWB_split_wasm( B2_hi, B2_lo, in32[ j ] );
            o[ 2 * j + 0 ] = silk_biquad_alt_stride2_step_wasm( &S_v, wasm_i32x4_shuffle( t0, t0, 0, 1, 0, 1 ),
                wasm_i32x4_shuffle( t1, t2, 0, 1, 4, 5 ), A_L, A_U );
            o[ 2 * j + 1 ] = silk_biquad_alt_stride2_step_wasm( &S_v, wasm_i32x4_shuffle( t0, t0, 2, 3, 2, 3 ),
                wasm_i32x4_shuffle( t1, t2, 2, 3, 6, 7 ), A_L, A_U );
        }

        /* Scale back to Q0 and saturate */
        out_lo = wasm_i32x4_shuffle( o[ 0 ], o[ 1 ], 0, 1, 4, 5 );
        out_hi = wasm_i32x4_shuffle( o[ 2 ], o[ 3 ], 0, 1, 4, 5 );
        out_lo = wasm_i32x4_shr( wasm_i32x4_add( out_lo, offset ), 14 );
        out_hi = wasm_i32x4_shr( wasm_i32x4_add( out_hi, offset ), 14 );
        wasm_v128_store( &out[ 2 * k ], wasm_i16x8_narrow_i32x4( out_lo, out_hi ) );
    }

    /* Process leftover. */
    for( ; k < len; k++ ) {
        v128_t in32, t0, t1, t2, o;

        in32 = wasm_i32x4_make( in[ 2 * k + 0 ], in[ 2 * k + 1 ], in[ 2 * k + 0 ], in[ 2 * k + 1 ] );
        t0 = silk_SMULWB_split_wasm( B0_hi, B0_lo, in32 );
        t1 = silk_SMULWB_split_wasm( B1_hi, B1_lo, in32 );
        t2 = silk_SMULWB_split_wasm( B2_hi, B2_lo, in32 );
        o = silk_biquad_alt_stride2_step_wasm( &S_v, t0, wasm_i32x4_shuffle( t1, t2, 0, 1, 4, 5 ), A_L, A_U );

        /* Scale back to Q0 and saturate */
        o = wasm_i32x4_shr( wasm_i32x4_add( o, offset ), 14 );
        o = wasm_i16x8_narrow_i32x4( o, o );
        out[ 2 * k + 0 ] = wasm_i16x8_extract_lane( o, 0 );
        out[ 2 * k + 1 ] = wasm_i16x8_extract_lane( o, 1 );
    }

    S[ 0 ] = wasm_i32x4_extract_lane( S_v, 0 );
    S[ 1 ] = wasm_i32x4_extract_lane( S_v, 2 );
    S[ 2 ] = wasm_i32x4_extract_lane( S_v, 1 );
    S[ 3 ] = wasm_i32x4_extract_lane( S_v, 3 );
}
