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

#include "celt_lpc.h"
#include "stack_alloc.h"
#include "pitch.h"

#if defined(FIXED_POINT)

#include <wasm_simd128.h>

/* Sums the lanes of a, b, c and d: returns {sum(a), sum(b), sum(c), sum(d)}. */
static OPUS_INLINE v128_t wasm_i32x4_hadd4(v128_t a, v128_t b, v128_t c,
      v128_t d)
{
   v128_t ab, cd;
   ab = wasm_i32x4_add(wasm_i32x4_shuffle(a, b, 0, 4, 1, 5),
                       wasm_i32x4_shuffle(a, b, 2, 6, 3, 7));
   cd = wasm_i32x4_add(wasm_i32x4_shuffle(c, d, 0, 4, 1, 5),
                       wasm_i32x4_shuffle(c, d, 2, 6, 3, 7));
   return wasm_i32x4_add(wasm_i64x2_shuffle(ab, cd, 0, 2),
                         wasm_i64x2_shuffle(ab, cd, 1, 3));
}

/* Applies the same rounding as the C code to four filter outputs:
   SROUND16(SHL32(x, SIG_SHIFT) + sum, SIG_SHIFT), with every step done in
   the same order and with the same 32-bit wrap-around, so the result is
   bit-exact. SROUND16() saturates to +/-32767, hence the max() before the
   (saturating) narrowing. */
static OPUS_INLINE v128_t fir_round4(v128_t sum, const opus_val16 *x)
{
   sum = wasm_i32x4_add(sum,
         wasm_i32x4_shl(wasm_i32x4_load16x4(x), SIG_SHIFT));
   sum = wasm_i32x4_add(sum, wasm_i32x4_splat(1<<(SIG_SHIFT-1)));
   sum = wasm_i32x4_shr(sum, SIG_SHIFT);
   return wasm_i32x4_max(sum, wasm_i32x4_splat(-32767));
}

void celt_fir_wasm(const opus_val16 *x,
         const opus_val16 *num,
         opus_val16 *y,
         int N,
         int ord,
         int arch)
{
   int i,j;
   VARDECL(opus_val16, rnum);
   SAVE_STACK;
   (void)arch;
   celt_assert(x != y);
   ALLOC(rnum, ord, opus_val16);
   for(i=0;i<ord;i++)
      rnum[i] = num[ord-i-1];
   /* Eight outputs at a time: each vector of taps is loaded once for eight
      dot products. */
   for (i=0;i<N-7;i+=8)
   {
      const opus_val16 *xp = x+i-ord;
      v128_t sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, lo, hi;
      sum0 = sum1 = sum2 = sum3 = wasm_i32x4_splat(0);
      sum4 = sum5 = sum6 = sum7 = wasm_i32x4_splat(0);
      for (j=0;j<ord-7;j+=8)
      {
         v128_t nj = wasm_v128_load(rnum+j);
         sum0 = wasm_i32x4_add(sum0, wasm_i32x4_dot_i16x8(nj,
               wasm_v128_load(xp+j)));
         sum1 = wasm_i32x4_add(sum1, wasm_i32x4_dot_i16x8(nj,
               wasm_v128_load(xp+j+1)));
         sum2 = wasm_i32x4_add(sum2, wasm_i32x4_dot_i16x8(nj,
               wasm_v128_load(xp+j+2)));
         sum3 = wasm_i32x4_add(sum3, wasm_i32x4_dot_i16x8(nj,
               wasm_v128_load(xp+j+3)));
         sum4 = wasm_i32x4_add(sum4, wasm_i32x4_dot_i16x8(nj,
               wasm_v128_load(xp+j+4)));
         sum5 = wasm_i32x4_add(sum5, wasm_i32x4_dot_i16x8(nj,
               wasm_v128_load(xp+j+5)));
         sum6 = wasm_i32x4_add(sum6, wasm_i32x4_dot_i16x8(nj,
               wasm_v128_load(xp+j+6)));
         sum7 = wasm_i32x4_add(sum7, wasm_i32x4_dot_i16x8(nj,
               wasm_v128_load(xp+j+7)));
      }
      lo = wasm_i32x4_hadd4(sum0, sum1, sum2, sum3);
      hi = wasm_i32x4_hadd4(sum4, sum5, sum6, sum7);
      for (;j<ord;j++)
      {
         v128_t nj = wasm_i32x4_splat(rnum[j]);
         lo = wasm_i32x4_add(lo, wasm_i32x4_mul(nj,
               wasm_i32x4_load16x4(xp+j)));
         hi = wasm_i32x4_add(hi, wasm_i32x4_mul(nj,
               wasm_i32x4_load16x4(xp+j+4)));
      }
      lo = fir_round4(lo, x+i);
      hi = fir_round4(hi, x+i+4);
      wasm_v128_store(y+i, wasm_i16x8_narrow_i32x4(lo, hi));
   }
   if (i<N-3)
   {
      opus_val32 sum[4] = {0, 0, 0, 0};
      v128_t s;
      xcorr_kernel(rnum, x+i-ord, sum, ord, arch);
      s = fir_round4(wasm_v128_load(sum), x+i);
      wasm_v128_store64_lane(y+i, wasm_i16x8_narrow_i32x4(s, s), 0);
      i += 4;
   }
   for (;i<N;i++)
   {
      opus_val32 sum = SHL32(EXTEND32(x[i]), SIG_SHIFT);
      for (j=0;j<ord;j++)
         sum = MAC16_16(sum,rnum[j],x[i+j-ord]);
      y[i] = SROUND16(sum, SIG_SHIFT);
   }
   RESTORE_STACK;
}

#endif /* FIXED_POINT */
