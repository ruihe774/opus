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

#include "mathops.h"
#include "float_cast.h"

#if !defined(DISABLE_FLOAT_API)

#include <wasm_simd128.h>

/* Same as FLOAT2INT16() on four values, bit-exact including NaN (which
   gives -32768) and halfway cases (rounded to even, like float2int()).
   f32x4.pmax/pmin are the C ternaries MAX32()/MIN32() with the same
   operand order, so they treat NaN the same way.
   The clamped value is rounded and converted by adding 1.5*2^23: the sum
   is rounded to an integer (to nearest, ties to even) and that integer
   ends up in the low mantissa bits. This is exact for |x| < 2^22 and is
   cheaper than f32x4.nearest + i32x4.trunc_sat_f32x4 (the saturating
   conversion needs a long instruction sequence on x86), and as cheap as
   their relaxed counterparts. */
static OPUS_INLINE v128_t float2int16x4(v128_t x)
{
   x = wasm_f32x4_mul(x, wasm_f32x4_splat(CELT_SIG_SCALE));
   x = wasm_f32x4_pmax(wasm_f32x4_splat(-32768.f), x);
   x = wasm_f32x4_pmin(wasm_f32x4_splat(32767.f), x);
   x = wasm_f32x4_add(x, wasm_f32x4_splat(12582912.f));
   return wasm_i32x4_sub(x, wasm_i32x4_splat(0x4B400000));
}

void celt_float2int16_wasm(const float * OPUS_RESTRICT in,
      short * OPUS_RESTRICT out, int cnt)
{
   int i;
   for (i=0;i<cnt-15;i+=16)
   {
      v128_t a, b, c, d;
      a = float2int16x4(wasm_v128_load(&in[i]));
      b = float2int16x4(wasm_v128_load(&in[i+4]));
      c = float2int16x4(wasm_v128_load(&in[i+8]));
      d = float2int16x4(wasm_v128_load(&in[i+12]));
      wasm_v128_store(&out[i], wasm_i16x8_narrow_i32x4(a, b));
      wasm_v128_store(&out[i+8], wasm_i16x8_narrow_i32x4(c, d));
   }
   if (i<cnt-7)
   {
      v128_t a, b;
      a = float2int16x4(wasm_v128_load(&in[i]));
      b = float2int16x4(wasm_v128_load(&in[i+4]));
      wasm_v128_store(&out[i], wasm_i16x8_narrow_i32x4(a, b));
      i += 8;
   }
   for (;i<cnt;i++)
   {
      out[i] = FLOAT2INT16(in[i]);
   }
}

int opus_limit2_checkwithin1_wasm(float *samples, int cnt)
{
   const float hardclipMin = -2.0f;
   const float hardclipMax = 2.0f;
   int i;
   int blockedSize;
   int exceeding1 = 0;
   opus_int32 absmax;
   v128_t max0, max1;

   if (cnt <= 0)
   {
      return 1;
   }

   /* First pass: only find the largest magnitude. With the sign bit
      cleared, the bit patterns of floats are ordered like signed integers,
      with +inf and then NaN above every finite value, so an integer max
      (unlike f32x4.pmax) cannot skip a NaN, and is cheaper than f32x4.max.
      A NaN is reported as exceeding 1 (the C version never claims all
      samples are within [-1,1]). */
   blockedSize = cnt & ~7;
   max0 = max1 = wasm_i32x4_splat(0);
   for (i=0;i<blockedSize;i+=8)
   {
      max0 = wasm_i32x4_max(max0, wasm_v128_and(wasm_v128_load(&samples[i]),
            wasm_i32x4_splat(0x7fffffff)));
      max1 = wasm_i32x4_max(max1, wasm_v128_and(wasm_v128_load(&samples[i+4]),
            wasm_i32x4_splat(0x7fffffff)));
   }
   max0 = wasm_i32x4_max(max0, max1);
   max0 = wasm_i32x4_max(max0, wasm_i64x2_shuffle(max0, max0, 1, 0));
   max0 = wasm_i32x4_max(max0, wasm_i32x4_shuffle(max0, max0, 1, 0, 3, 2));
   absmax = wasm_i32x4_extract_lane(max0, 0);

   if (absmax > 0x40000000 /* 2.f */)
   {
      /* Clamp like FMIN(2, FMAX(-2, x)), which leaves NaN unchanged. */
      const v128_t hardclipMinReg = wasm_f32x4_splat(hardclipMin);
      const v128_t hardclipMaxReg = wasm_f32x4_splat(hardclipMax);
      for (i=0;i<blockedSize;i+=4)
      {
         v128_t x = wasm_v128_load(&samples[i]);
         x = wasm_f32x4_pmax(x, hardclipMinReg);
         x = wasm_f32x4_pmin(x, hardclipMaxReg);
         wasm_v128_store(&samples[i], x);
      }
   }
   exceeding1 = absmax > 0x3f800000 /* 1.f */;

   for (i=blockedSize;i<cnt;i++)
   {
      const float origVal = samples[i];
      float clippedVal = origVal;
      clippedVal = FMAX(hardclipMin, clippedVal);
      clippedVal = FMIN(hardclipMax, clippedVal);
      samples[i] = clippedVal;

      exceeding1 |= !(origVal <= 1.0f && origVal >= -1.0f);
   }

   return !exceeding1;
}

#endif /* !DISABLE_FLOAT_API */
