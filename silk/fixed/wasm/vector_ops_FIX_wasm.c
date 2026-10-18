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

#ifdef FIXED_POINT

/* i32x4.dot_i16x8_s adds two 16x16-bit products per lane. The true sum lies in
   [ -2^31 + 2^16, 2^31 ], so it only wraps for 2 * (-32768)^2. Adding DOT_BIAS
   maps that range onto [ 0, 2^32 - 2^16 ], which is exact as an unsigned 32-bit
   value and can be zero-extended into 64-bit accumulators. */
#define DOT_BIAS 0x7FFF0000

opus_int64 silk_inner_prod16_wasm(
    const opus_int16            *inVec1,            /*    I input vector 1                                              */
    const opus_int16            *inVec2,            /*    I input vector 2                                              */
    const opus_int              len                 /*    I vector lengths                                              */
)
{
    opus_int   i, nb_lanes;
    opus_int64 sum;
    v128_t     acc0, acc1, d0, d1;
    const v128_t bias = wasm_i32x4_splat( DOT_BIAS );

    acc0 = acc1 = wasm_i64x2_splat( 0 );
    nb_lanes = 0;
    for( i = 0; i < len - 15; i += 16 ) {
        d0 = wasm_i32x4_dot_i16x8( wasm_v128_load( &inVec1[ i ] ), wasm_v128_load( &inVec2[ i ] ) );
        d1 = wasm_i32x4_dot_i16x8( wasm_v128_load( &inVec1[ i + 8 ] ), wasm_v128_load( &inVec2[ i + 8 ] ) );
        d0 = wasm_i32x4_add( d0, bias );
        d1 = wasm_i32x4_add( d1, bias );
        acc0 = wasm_i64x2_add( acc0, wasm_u64x2_extend_low_u32x4( d0 ) );
        acc1 = wasm_i64x2_add( acc1, wasm_u64x2_extend_high_u32x4( d0 ) );
        acc0 = wasm_i64x2_add( acc0, wasm_u64x2_extend_low_u32x4( d1 ) );
        acc1 = wasm_i64x2_add( acc1, wasm_u64x2_extend_high_u32x4( d1 ) );
        nb_lanes += 8;
    }
    if( i < len - 7 ) {
        d0 = wasm_i32x4_dot_i16x8( wasm_v128_load( &inVec1[ i ] ), wasm_v128_load( &inVec2[ i ] ) );
        d0 = wasm_i32x4_add( d0, bias );
        acc0 = wasm_i64x2_add( acc0, wasm_u64x2_extend_low_u32x4( d0 ) );
        acc1 = wasm_i64x2_add( acc1, wasm_u64x2_extend_high_u32x4( d0 ) );
        nb_lanes += 4;
        i += 8;
    }
    if( i < len - 3 ) {
        /* Only the two lower lanes hold products */
        d0 = wasm_i32x4_dot_i16x8( wasm_v128_load64_zero( &inVec1[ i ] ), wasm_v128_load64_zero( &inVec2[ i ] ) );
        d0 = wasm_i32x4_add( d0, bias );
        acc0 = wasm_i64x2_add( acc0, wasm_u64x2_extend_low_u32x4( d0 ) );
        nb_lanes += 2;
        i += 4;
    }
    acc0 = wasm_i64x2_add( acc0, acc1 );
    sum = wasm_i64x2_extract_lane( acc0, 0 ) + wasm_i64x2_extract_lane( acc0, 1 );
    sum -= (opus_int64)nb_lanes * DOT_BIAS;

    for( ; i < len; i++ ) {
        sum = silk_SMLALBB( sum, inVec1[ i ], inVec2[ i ] );
    }

    return sum;
}

#endif /* FIXED_POINT */
