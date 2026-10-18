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
#include "SigProc_FLP.h"

#ifndef FIXED_POINT

/* acc + a * b in double precision. The product of two floats is exact in double,
   so a fused multiply-add gives exactly the same result as a separate multiply and add. */
#if defined(__wasm_relaxed_simd__)
# define MADD_F64X2(a, b, acc) wasm_f64x2_relaxed_madd( a, b, acc )
#else
# define MADD_F64X2(a, b, acc) wasm_f64x2_add( acc, wasm_f64x2_mul( a, b ) )
#endif

/* Promotes the upper two floats of a vector */
#define PROMOTE_HIGH_F32X4(x) wasm_f64x2_promote_low_f32x4( wasm_i32x4_shuffle( x, x, 2, 3, 2, 3 ) )

/* inner product of two silk_float arrays, with result as double */
double silk_inner_product_FLP_wasm(
    const silk_float    *data1,
    const silk_float    *data2,
    opus_int            dataSize
)
{
    opus_int i;
    double   result;
    v128_t   acc0, acc1, acc2, acc3, x, y;

    acc0 = acc1 = acc2 = acc3 = wasm_f64x2_splat( 0.0 );
    for( i = 0; i < dataSize - 7; i += 8 ) {
        x = wasm_v128_load( &data1[ i ] );
        y = wasm_v128_load( &data2[ i ] );
        acc0 = MADD_F64X2( wasm_f64x2_promote_low_f32x4( x ), wasm_f64x2_promote_low_f32x4( y ), acc0 );
        acc1 = MADD_F64X2( PROMOTE_HIGH_F32X4( x ), PROMOTE_HIGH_F32X4( y ), acc1 );
        x = wasm_v128_load( &data1[ i + 4 ] );
        y = wasm_v128_load( &data2[ i + 4 ] );
        acc2 = MADD_F64X2( wasm_f64x2_promote_low_f32x4( x ), wasm_f64x2_promote_low_f32x4( y ), acc2 );
        acc3 = MADD_F64X2( PROMOTE_HIGH_F32X4( x ), PROMOTE_HIGH_F32X4( y ), acc3 );
    }
    if( i < dataSize - 3 ) {
        x = wasm_v128_load( &data1[ i ] );
        y = wasm_v128_load( &data2[ i ] );
        acc0 = MADD_F64X2( wasm_f64x2_promote_low_f32x4( x ), wasm_f64x2_promote_low_f32x4( y ), acc0 );
        acc1 = MADD_F64X2( PROMOTE_HIGH_F32X4( x ), PROMOTE_HIGH_F32X4( y ), acc1 );
        i += 4;
    }
    acc0 = wasm_f64x2_add( wasm_f64x2_add( acc0, acc1 ), wasm_f64x2_add( acc2, acc3 ) );
    result = wasm_f64x2_extract_lane( acc0, 0 ) + wasm_f64x2_extract_lane( acc0, 1 );

    /* add any remaining products */
    for( ; i < dataSize; i++ ) {
        result += data1[ i ] * (double)data2[ i ];
    }

    return result;
}

#endif /* FIXED_POINT */
