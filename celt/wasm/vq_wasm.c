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

#include "stack_alloc.h"
#include "mathops.h"
#include "vq.h"

#if !defined(FIXED_POINT)

#include <wasm_simd128.h>

/* With relaxed SIMD, use the single-instruction lane select (e.g. blendvps
   on x86). Every mask passed to it is a comparison result, i.e. all-ones or
   all-zeros per lane, for which relaxed_laneselect is fully defined. */
#if defined(__wasm_relaxed_simd__)
# define WASM_LANESELECT(a, b, m) wasm_i32x4_relaxed_laneselect(a, b, m)
#else
# define WASM_LANESELECT(a, b, m) wasm_v128_bitselect(a, b, m)
#endif

static OPUS_INLINE float wasm_f32x4_hsum(v128_t a)
{
   a = wasm_f32x4_add(a, wasm_i64x2_shuffle(a, a, 1, 0));
   a = wasm_f32x4_add(a, wasm_i32x4_shuffle(a, a, 1, 0, 3, 2));
   return wasm_f32x4_extract_lane(a, 0);
}

static OPUS_INLINE int wasm_i32x4_hsum(v128_t a)
{
   a = wasm_i32x4_add(a, wasm_i64x2_shuffle(a, a, 1, 0));
   a = wasm_i32x4_add(a, wasm_i32x4_shuffle(a, a, 1, 0, 3, 2));
   return wasm_i32x4_extract_lane(a, 0);
}

opus_val16 op_pvq_search_wasm(celt_norm *_X, int *iy, int K, int N, int arch)
{
   int i, j;
   int pulsesLeft;
   float xy, yy;
   VARDECL(celt_norm, y);
   VARDECL(celt_norm, X);
   VARDECL(int, signy);
   v128_t sums;
   SAVE_STACK;

   (void)arch;
   /* X and y are padded with (at least) 8 entries for the search loop. */
   ALLOC(y, N+8, celt_norm);
   ALLOC(X, N+8, celt_norm);
   ALLOC(signy, N+3, int);

   /* Copy the last partial vector of _X into a zero-padded X; the full
      vectors are read from _X directly below. */
   j = N&~3;
   wasm_v128_store(&X[j], wasm_f32x4_splat(0));
   wasm_v128_store(&X[j+4], wasm_f32x4_splat(0));
   for (i=j;i<N;i++)
      X[i] = _X[i];
   sums = wasm_f32x4_splat(0);
   for (j=0;j<N;j+=4)
   {
      v128_t x4, s4;
      x4 = wasm_v128_load(j+4 <= N ? &_X[j] : &X[j]);
      s4 = wasm_f32x4_lt(x4, wasm_f32x4_splat(0));
      /* Get rid of the sign */
      x4 = wasm_f32x4_abs(x4);
      sums = wasm_f32x4_add(sums, x4);
      /* Clear y and iy in case we don't do the projection. */
      wasm_v128_store(&y[j], wasm_f32x4_splat(0));
      wasm_v128_store(&iy[j], wasm_i32x4_splat(0));
      wasm_v128_store(&X[j], x4);
      wasm_v128_store(&signy[j], s4);
   }

   xy = yy = 0;

   pulsesLeft = K;

   /* Do a pre-search by projecting on the pyramid */
   if (K > (N>>1))
   {
      v128_t pulses_sum, xy4, yy4, rcp4;
      opus_val32 sum = wasm_f32x4_hsum(sums);
      /* If X is too small, just replace it with a pulse at 0 */
      /* Prevents infinities and NaNs from causing too many pulses
         to be allocated. 64 is an approximation of infinity here. */
      if (!(sum > EPSILON && sum < 64))
      {
         X[0] = QCONST16(1.f,14);
         j=1; do
            X[j]=0;
         while (++j<N);
         sum = QCONST16(1.f,14);
      }
      /* Using K+e with e < 1 guarantees we cannot get more than K pulses. */
      rcp4 = wasm_f32x4_splat(EXTRACT16(MULT16_32_Q16(K+0.8f, celt_rcp(sum))));
      xy4 = yy4 = wasm_f32x4_splat(0);
      pulses_sum = wasm_i32x4_splat(0);
      for (j=0;j<N;j+=4)
      {
         v128_t x4, y4, iy4;
         x4 = wasm_v128_load(&X[j]);
         /* x4 >= 0 here, so truncation is the same as the floor() in the C
            code. x4*rcp4 is finite, not NaN and at most K+.8, so the
            relaxed truncation is exact too. */
#if defined(__wasm_relaxed_simd__)
         iy4 = wasm_i32x4_relaxed_trunc_f32x4(wasm_f32x4_mul(x4, rcp4));
#else
         iy4 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_mul(x4, rcp4));
#endif
         pulses_sum = wasm_i32x4_add(pulses_sum, iy4);
         wasm_v128_store(&iy[j], iy4);
         y4 = wasm_f32x4_convert_i32x4(iy4);
         xy4 = wasm_f32x4_add(xy4, wasm_f32x4_mul(x4, y4));
         yy4 = wasm_f32x4_add(yy4, wasm_f32x4_mul(y4, y4));
         /* double the y[] vector so we don't have to do it in the search loop. */
         wasm_v128_store(&y[j], wasm_f32x4_add(y4, y4));
      }
      pulsesLeft -= wasm_i32x4_hsum(pulses_sum);
      xy = wasm_f32x4_hsum(xy4);
      yy = wasm_f32x4_hsum(yy4);
   }
   /* The padding lanes get X=0 and a huge y, so their score is either 0
      (never selected, as only scores strictly above 0 are) or far below
      the score of any real position. */
   wasm_v128_store(&y[N], wasm_f32x4_splat(1e15f));
   wasm_v128_store(&y[N+4], wasm_f32x4_splat(1e15f));
   celt_sig_assert(pulsesLeft>=0);

   /* This should never happen, but just in case it does (e.g. on silence)
      we fill the first bin with pulses. */
   if (pulsesLeft > N+3)
   {
      opus_val16 tmp = (opus_val16)pulsesLeft;
      yy = MAC16_16(yy, tmp, tmp);
      yy = MAC16_16(yy, tmp, y[0]);
      iy[0] += pulsesLeft;
      pulsesLeft=0;
   }

   for (i=0;i<pulsesLeft;i++)
   {
      int best_id;
      v128_t xy4, yy4, best0, best1, pos0, pos1, max, count;
      /* The squared magnitude term gets added anyway, so we might as well
         add it outside the loop */
      yy = ADD16(yy, 1);
      xy4 = wasm_f32x4_splat(xy);
      yy4 = wasm_f32x4_splat(yy);
      /* Two independent sets of per-lane maxima, to halve the length of the
         loop-carried dependency chain. */
      best0 = best1 = wasm_f32x4_splat(0);
      pos0 = pos1 = wasm_i32x4_splat(0);
      count = wasm_i32x4_make(0, 1, 2, 3);
      for (j=0;j<N;j+=8)
      {
         v128_t rxy0, ryy0, r0, rxy1, ryy1, r1;
         /* Temporary sums of the new pulse(s) */
         rxy0 = wasm_f32x4_add(wasm_v128_load(&X[j]), xy4);
         rxy1 = wasm_f32x4_add(wasm_v128_load(&X[j+4]), xy4);
         /* We're multiplying y[j] by two so we don't have to do it here */
         ryy0 = wasm_f32x4_add(wasm_v128_load(&y[j]), yy4);
         ryy1 = wasm_f32x4_add(wasm_v128_load(&y[j+4]), yy4);
         /* We maximise Rxy/sqrt(Ryy), i.e. Rxy^2/Ryy since Rxy >= 0.
            WebAssembly has no reciprocal (square root) estimate, and one
            division is cheaper than a square root plus a division. */
         r0 = wasm_f32x4_div(wasm_f32x4_mul(rxy0, rxy0), ryy0);
         r1 = wasm_f32x4_div(wasm_f32x4_mul(rxy1, rxy1), ryy1);
         /* Keep the first index of each per-lane maximum. */
         pos0 = WASM_LANESELECT(count, pos0, wasm_f32x4_gt(r0, best0));
         count = wasm_i32x4_add(count, wasm_i32x4_splat(4));
         pos1 = WASM_LANESELECT(count, pos1, wasm_f32x4_gt(r1, best1));
         count = wasm_i32x4_add(count, wasm_i32x4_splat(4));
         best0 = wasm_f32x4_pmax(best0, r0);
         best1 = wasm_f32x4_pmax(best1, r1);
      }
      /* Merge the two sets, keeping the lowest index on ties. */
      pos0 = WASM_LANESELECT(pos1, pos0, wasm_f32x4_gt(best1, best0));
      best0 = wasm_f32x4_pmax(best0, best1);
      /* Horizontal max */
      max = wasm_f32x4_pmax(best0, wasm_i64x2_shuffle(best0, best0, 1, 0));
      max = wasm_f32x4_pmax(max, wasm_i32x4_shuffle(max, max, 1, 0, 3, 2));
      /* Among the lanes that reach the global max, pick the lowest index,
         like the C code does. */
      pos0 = WASM_LANESELECT(pos0, wasm_i32x4_splat(0x7fffffff),
            wasm_f32x4_eq(best0, max));
      pos0 = wasm_i32x4_min(pos0, wasm_i64x2_shuffle(pos0, pos0, 1, 0));
      pos0 = wasm_i32x4_min(pos0, wasm_i32x4_shuffle(pos0, pos0, 1, 0, 3, 2));
      best_id = wasm_i32x4_extract_lane(pos0, 0);

      /* Updating the sums of the new pulse(s) */
      xy = ADD32(xy, EXTEND32(X[best_id]));
      /* We're multiplying y[j] by two so we don't have to do it here */
      yy = ADD16(yy, y[best_id]);

      /* Only now that we've made the final choice, update y/iy */
      /* Multiplying y[j] by 2 so we don't have to do it everywhere else */
      y[best_id] += 2;
      iy[best_id]++;
   }

   /* Put the original sign back */
   for (j=0;j<N;j+=4)
   {
      v128_t y4, s4;
      y4 = wasm_v128_load(&iy[j]);
      s4 = wasm_v128_load(&signy[j]);
      y4 = wasm_v128_xor(wasm_i32x4_add(y4, s4), s4);
      wasm_v128_store(&iy[j], y4);
   }
   RESTORE_STACK;
   return yy;
}

#endif /* !FIXED_POINT */
