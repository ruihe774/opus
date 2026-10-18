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
#include "macros.h"
#include "pitch.h"

#if defined(FIXED_POINT)

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

static OPUS_INLINE opus_int32 wasm_i32x4_hsum(v128_t a)
{
   a = wasm_i32x4_add(a, wasm_i64x2_shuffle(a, a, 1, 0));
   a = wasm_i32x4_add(a, wasm_i32x4_shuffle(a, a, 1, 0, 3, 2));
   return wasm_i32x4_extract_lane(a, 0);
}

/* All the fixed-point kernels below accumulate exactly the same 16x16-bit
   products as the C code into 32-bit lanes with wrap-around additions, so
   they are bit-exact with it (only the summation order differs, which does
   not matter for modular integer arithmetic). i32x4.dot_i16x8_s is exact
   too: the only pair sum that does not fit in 32 bits,
   (-32768)*(-32768)*2, wraps exactly like the C sum does. */

opus_val32 celt_inner_prod_wasm(const opus_val16 *x, const opus_val16 *y,
      int N)
{
   int i;
   opus_val32 xy;
   v128_t acc0, acc1;
   acc0 = wasm_i32x4_splat(0);
   acc1 = wasm_i32x4_splat(0);
   for (i=0;i<N-15;i+=16)
   {
      acc0 = wasm_i32x4_add(acc0, wasm_i32x4_dot_i16x8(
            wasm_v128_load(x+i), wasm_v128_load(y+i)));
      acc1 = wasm_i32x4_add(acc1, wasm_i32x4_dot_i16x8(
            wasm_v128_load(x+i+8), wasm_v128_load(y+i+8)));
   }
   if (i<N-7)
   {
      acc0 = wasm_i32x4_add(acc0, wasm_i32x4_dot_i16x8(
            wasm_v128_load(x+i), wasm_v128_load(y+i)));
      i += 8;
   }
   if (i<N-3)
   {
      acc1 = wasm_i32x4_add(acc1, wasm_i32x4_mul(
            wasm_i32x4_load16x4(x+i), wasm_i32x4_load16x4(y+i)));
      i += 4;
   }
   xy = wasm_i32x4_hsum(wasm_i32x4_add(acc0, acc1));
   for (;i<N;i++)
      xy = MAC16_16(xy, x[i], y[i]);
   return xy;
}

void dual_inner_prod_wasm(const opus_val16 *x, const opus_val16 *y01,
      const opus_val16 *y02, int N, opus_val32 *xy1, opus_val32 *xy2)
{
   int i;
   opus_val32 xy01, xy02;
   v128_t acc1, acc2;
   acc1 = wasm_i32x4_splat(0);
   acc2 = wasm_i32x4_splat(0);
   for (i=0;i<N-7;i+=8)
   {
      v128_t xi = wasm_v128_load(x+i);
      acc1 = wasm_i32x4_add(acc1, wasm_i32x4_dot_i16x8(xi,
            wasm_v128_load(y01+i)));
      acc2 = wasm_i32x4_add(acc2, wasm_i32x4_dot_i16x8(xi,
            wasm_v128_load(y02+i)));
   }
   if (i<N-3)
   {
      v128_t xi = wasm_i32x4_load16x4(x+i);
      acc1 = wasm_i32x4_add(acc1, wasm_i32x4_mul(xi,
            wasm_i32x4_load16x4(y01+i)));
      acc2 = wasm_i32x4_add(acc2, wasm_i32x4_mul(xi,
            wasm_i32x4_load16x4(y02+i)));
      i += 4;
   }
   xy01 = wasm_i32x4_hsum(acc1);
   xy02 = wasm_i32x4_hsum(acc2);
   for (;i<N;i++)
   {
      xy01 = MAC16_16(xy01, x[i], y01[i]);
      xy02 = MAC16_16(xy02, x[i], y02[i]);
   }
   *xy1 = xy01;
   *xy2 = xy02;
}

/* Returns {sum_j x[j]*y[j+k], k=0..3}. Reads y[0..len+2]. */
static OPUS_INLINE v128_t xcorr4_fixed(const opus_val16 *x,
      const opus_val16 *y, int len)
{
   int j;
   v128_t sum0, sum1, sum2, sum3, sum;
   sum0 = sum1 = sum2 = sum3 = wasm_i32x4_splat(0);
   for (j=0;j<len-7;j+=8)
   {
      v128_t xj = wasm_v128_load(x+j);
      sum0 = wasm_i32x4_add(sum0, wasm_i32x4_dot_i16x8(xj,
            wasm_v128_load(y+j)));
      sum1 = wasm_i32x4_add(sum1, wasm_i32x4_dot_i16x8(xj,
            wasm_v128_load(y+j+1)));
      sum2 = wasm_i32x4_add(sum2, wasm_i32x4_dot_i16x8(xj,
            wasm_v128_load(y+j+2)));
      sum3 = wasm_i32x4_add(sum3, wasm_i32x4_dot_i16x8(xj,
            wasm_v128_load(y+j+3)));
   }
   sum = wasm_i32x4_hadd4(sum0, sum1, sum2, sum3);
   for (;j<len;j++)
   {
      sum = wasm_i32x4_add(sum, wasm_i32x4_mul(wasm_i32x4_splat(x[j]),
            wasm_i32x4_load16x4(y+j)));
   }
   return sum;
}

void xcorr_kernel_wasm(const opus_val16 *x, const opus_val16 *y,
      opus_val32 sum[4], int len)
{
   celt_assert(len>=3);
   wasm_v128_store(sum, wasm_i32x4_add(wasm_v128_load(sum),
         xcorr4_fixed(x, y, len)));
}

opus_val32 celt_pitch_xcorr_wasm(const opus_val16 *_x, const opus_val16 *_y,
      opus_val32 *xcorr, int len, int max_pitch, int arch)
{
   int i;
   opus_val32 maxcorr;
   v128_t vmax;
   (void)arch;
   celt_assert(max_pitch>0);
   vmax = wasm_i32x4_splat(1);
   /* Eight lags at a time: each x vector is loaded once for eight dot
      products. */
   for (i=0;i<max_pitch-7;i+=8)
   {
      int j;
      const opus_val16 *y = _y+i;
      v128_t sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, lo, hi;
      sum0 = sum1 = sum2 = sum3 = wasm_i32x4_splat(0);
      sum4 = sum5 = sum6 = sum7 = wasm_i32x4_splat(0);
      for (j=0;j<len-7;j+=8)
      {
         v128_t xj = wasm_v128_load(_x+j);
         sum0 = wasm_i32x4_add(sum0, wasm_i32x4_dot_i16x8(xj,
               wasm_v128_load(y+j)));
         sum1 = wasm_i32x4_add(sum1, wasm_i32x4_dot_i16x8(xj,
               wasm_v128_load(y+j+1)));
         sum2 = wasm_i32x4_add(sum2, wasm_i32x4_dot_i16x8(xj,
               wasm_v128_load(y+j+2)));
         sum3 = wasm_i32x4_add(sum3, wasm_i32x4_dot_i16x8(xj,
               wasm_v128_load(y+j+3)));
         sum4 = wasm_i32x4_add(sum4, wasm_i32x4_dot_i16x8(xj,
               wasm_v128_load(y+j+4)));
         sum5 = wasm_i32x4_add(sum5, wasm_i32x4_dot_i16x8(xj,
               wasm_v128_load(y+j+5)));
         sum6 = wasm_i32x4_add(sum6, wasm_i32x4_dot_i16x8(xj,
               wasm_v128_load(y+j+6)));
         sum7 = wasm_i32x4_add(sum7, wasm_i32x4_dot_i16x8(xj,
               wasm_v128_load(y+j+7)));
      }
      lo = wasm_i32x4_hadd4(sum0, sum1, sum2, sum3);
      hi = wasm_i32x4_hadd4(sum4, sum5, sum6, sum7);
      for (;j<len;j++)
      {
         v128_t xj = wasm_i32x4_splat(_x[j]);
         lo = wasm_i32x4_add(lo, wasm_i32x4_mul(xj,
               wasm_i32x4_load16x4(y+j)));
         hi = wasm_i32x4_add(hi, wasm_i32x4_mul(xj,
               wasm_i32x4_load16x4(y+j+4)));
      }
      wasm_v128_store(xcorr+i, lo);
      wasm_v128_store(xcorr+i+4, hi);
      vmax = wasm_i32x4_max(vmax, wasm_i32x4_max(lo, hi));
   }
   if (i<max_pitch-3)
   {
      v128_t sum = xcorr4_fixed(_x, _y+i, len);
      wasm_v128_store(xcorr+i, sum);
      vmax = wasm_i32x4_max(vmax, sum);
      i += 4;
   }
   vmax = wasm_i32x4_max(vmax, wasm_i64x2_shuffle(vmax, vmax, 1, 0));
   vmax = wasm_i32x4_max(vmax, wasm_i32x4_shuffle(vmax, vmax, 1, 0, 3, 2));
   maxcorr = wasm_i32x4_extract_lane(vmax, 0);
   /* In case max_pitch isn't a multiple of 4, do non-unrolled version. */
   for (;i<max_pitch;i++)
   {
      opus_val32 sum = celt_inner_prod_wasm(_x, _y+i, len);
      xcorr[i] = sum;
      maxcorr = MAX32(maxcorr, sum);
   }
   return maxcorr;
}

#else /* !FIXED_POINT */

/* Multiply-add a*b+c. With relaxed SIMD this may be fused (single rounding),
   which is fine for these float kernels: like the SSE/NEON versions they do
   not match the C summation order anyway. */
#if defined(__wasm_relaxed_simd__)
# define WASM_F32X4_MADD(a, b, c) wasm_f32x4_relaxed_madd(a, b, c)
#else
# define WASM_F32X4_MADD(a, b, c) wasm_f32x4_add(wasm_f32x4_mul(a, b), c)
#endif

/* Sums the lanes of a, b, c and d: returns {sum(a), sum(b), sum(c), sum(d)}. */
static OPUS_INLINE v128_t wasm_f32x4_hadd4(v128_t a, v128_t b, v128_t c,
      v128_t d)
{
   v128_t ab, cd;
   ab = wasm_f32x4_add(wasm_i32x4_shuffle(a, b, 0, 4, 1, 5),
                       wasm_i32x4_shuffle(a, b, 2, 6, 3, 7));
   cd = wasm_f32x4_add(wasm_i32x4_shuffle(c, d, 0, 4, 1, 5),
                       wasm_i32x4_shuffle(c, d, 2, 6, 3, 7));
   return wasm_f32x4_add(wasm_i64x2_shuffle(ab, cd, 0, 2),
                         wasm_i64x2_shuffle(ab, cd, 1, 3));
}

static OPUS_INLINE float wasm_f32x4_hsum(v128_t a)
{
   a = wasm_f32x4_add(a, wasm_i64x2_shuffle(a, a, 1, 0));
   a = wasm_f32x4_add(a, wasm_i32x4_shuffle(a, a, 1, 0, 3, 2));
   return wasm_f32x4_extract_lane(a, 0);
}

opus_val32 celt_inner_prod_wasm(const opus_val16 *x, const opus_val16 *y,
      int N)
{
   int i;
   float xy;
   v128_t sum0, sum1;
   sum0 = wasm_f32x4_splat(0);
   sum1 = wasm_f32x4_splat(0);
   for (i=0;i<N-7;i+=8)
   {
      sum0 = WASM_F32X4_MADD(wasm_v128_load(x+i), wasm_v128_load(y+i), sum0);
      sum1 = WASM_F32X4_MADD(wasm_v128_load(x+i+4), wasm_v128_load(y+i+4),
            sum1);
   }
   if (i<N-3)
   {
      sum0 = WASM_F32X4_MADD(wasm_v128_load(x+i), wasm_v128_load(y+i), sum0);
      i += 4;
   }
   xy = wasm_f32x4_hsum(wasm_f32x4_add(sum0, sum1));
   for (;i<N;i++)
      xy = MAC16_16(xy, x[i], y[i]);
   return xy;
}

void dual_inner_prod_wasm(const opus_val16 *x, const opus_val16 *y01,
      const opus_val16 *y02, int N, opus_val32 *xy1, opus_val32 *xy2)
{
   int i;
   float xy01, xy02;
   v128_t sum1a, sum1b, sum2a, sum2b;
   sum1a = sum1b = sum2a = sum2b = wasm_f32x4_splat(0);
   for (i=0;i<N-7;i+=8)
   {
      v128_t xi = wasm_v128_load(x+i);
      v128_t xi4 = wasm_v128_load(x+i+4);
      sum1a = WASM_F32X4_MADD(xi, wasm_v128_load(y01+i), sum1a);
      sum2a = WASM_F32X4_MADD(xi, wasm_v128_load(y02+i), sum2a);
      sum1b = WASM_F32X4_MADD(xi4, wasm_v128_load(y01+i+4), sum1b);
      sum2b = WASM_F32X4_MADD(xi4, wasm_v128_load(y02+i+4), sum2b);
   }
   if (i<N-3)
   {
      v128_t xi = wasm_v128_load(x+i);
      sum1a = WASM_F32X4_MADD(xi, wasm_v128_load(y01+i), sum1a);
      sum2a = WASM_F32X4_MADD(xi, wasm_v128_load(y02+i), sum2a);
      i += 4;
   }
   xy01 = wasm_f32x4_hsum(wasm_f32x4_add(sum1a, sum1b));
   xy02 = wasm_f32x4_hsum(wasm_f32x4_add(sum2a, sum2b));
   for (;i<N;i++)
   {
      xy01 = MAC16_16(xy01, x[i], y01[i]);
      xy02 = MAC16_16(xy02, x[i], y02[i]);
   }
   *xy1 = xy01;
   *xy2 = xy02;
}

/* Returns {sum_j x[j]*y[j+k], k=0..3}. Reads y[0..len+2]. */
static OPUS_INLINE v128_t xcorr4_float(const float *x, const float *y,
      int len)
{
   int j;
   v128_t sum0, sum1, sum2, sum3, sum;
   sum0 = sum1 = sum2 = sum3 = wasm_f32x4_splat(0);
   /* Same lag-per-lane layout as the C kernel and SSE port: each x[j] is
      broadcast and multiplied with y[j..j+3]. */
   for (j=0;j<len-3;j+=4)
   {
      v128_t xj = wasm_v128_load(x+j);
      sum0 = WASM_F32X4_MADD(wasm_i32x4_shuffle(xj, xj, 0, 0, 0, 0),
            wasm_v128_load(y+j), sum0);
      sum1 = WASM_F32X4_MADD(wasm_i32x4_shuffle(xj, xj, 1, 1, 1, 1),
            wasm_v128_load(y+j+1), sum1);
      sum2 = WASM_F32X4_MADD(wasm_i32x4_shuffle(xj, xj, 2, 2, 2, 2),
            wasm_v128_load(y+j+2), sum2);
      sum3 = WASM_F32X4_MADD(wasm_i32x4_shuffle(xj, xj, 3, 3, 3, 3),
            wasm_v128_load(y+j+3), sum3);
   }
   sum = wasm_f32x4_add(wasm_f32x4_add(sum0, sum1),
                        wasm_f32x4_add(sum2, sum3));
   for (;j<len;j++)
   {
      sum = WASM_F32X4_MADD(wasm_f32x4_splat(x[j]), wasm_v128_load(y+j),
            sum);
   }
   return sum;
}

void xcorr_kernel_wasm(const opus_val16 *x, const opus_val16 *y,
      opus_val32 sum[4], int len)
{
   celt_assert(len>=3);
   wasm_v128_store(sum, wasm_f32x4_add(wasm_v128_load(sum),
         xcorr4_float(x, y, len)));
}

void celt_pitch_xcorr_wasm(const opus_val16 *_x, const opus_val16 *_y,
      opus_val32 *xcorr, int len, int max_pitch, int arch)
{
   int i;
   (void)arch;
   celt_assert(max_pitch>0);
   /* Eight lags at a time, like the AVX2 version: one partial inner product
      per lag (no broadcasts in the inner loop), then a transpose-add. */
   for (i=0;i<max_pitch-7;i+=8)
   {
      int j;
      const float *y = _y+i;
      v128_t sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, lo, hi;
      sum0 = sum1 = sum2 = sum3 = wasm_f32x4_splat(0);
      sum4 = sum5 = sum6 = sum7 = wasm_f32x4_splat(0);
      for (j=0;j<len-3;j+=4)
      {
         v128_t xj = wasm_v128_load(_x+j);
         sum0 = WASM_F32X4_MADD(xj, wasm_v128_load(y+j), sum0);
         sum1 = WASM_F32X4_MADD(xj, wasm_v128_load(y+j+1), sum1);
         sum2 = WASM_F32X4_MADD(xj, wasm_v128_load(y+j+2), sum2);
         sum3 = WASM_F32X4_MADD(xj, wasm_v128_load(y+j+3), sum3);
         sum4 = WASM_F32X4_MADD(xj, wasm_v128_load(y+j+4), sum4);
         sum5 = WASM_F32X4_MADD(xj, wasm_v128_load(y+j+5), sum5);
         sum6 = WASM_F32X4_MADD(xj, wasm_v128_load(y+j+6), sum6);
         sum7 = WASM_F32X4_MADD(xj, wasm_v128_load(y+j+7), sum7);
      }
      lo = wasm_f32x4_hadd4(sum0, sum1, sum2, sum3);
      hi = wasm_f32x4_hadd4(sum4, sum5, sum6, sum7);
      for (;j<len;j++)
      {
         v128_t xj = wasm_f32x4_splat(_x[j]);
         lo = WASM_F32X4_MADD(xj, wasm_v128_load(y+j), lo);
         hi = WASM_F32X4_MADD(xj, wasm_v128_load(y+j+4), hi);
      }
      wasm_v128_store(xcorr+i, lo);
      wasm_v128_store(xcorr+i+4, hi);
   }
   if (i<max_pitch-3)
   {
      wasm_v128_store(xcorr+i, xcorr4_float(_x, _y+i, len));
      i += 4;
   }
   /* In case max_pitch isn't a multiple of 4, do non-unrolled version. */
   for (;i<max_pitch;i++)
      xcorr[i] = celt_inner_prod_wasm(_x, _y+i, len);
}

void comb_filter_const_wasm(opus_val32 *y, opus_val32 *x, int T, int N,
      opus_val16 g10, opus_val16 g11, opus_val16 g12)
{
   int i;
   v128_t x0v, g10v, g11v, g12v;
   /* T >= COMBFILTER_MINPERIOD, so when filtering in place (x == y) every
      x value read below has either not been written yet or was written by
      an earlier iteration, exactly as in the C code. */
   g10v = wasm_f32x4_splat(g10);
   g11v = wasm_f32x4_splat(g11);
   g12v = wasm_f32x4_splat(g12);
   x0v = wasm_v128_load(&x[-T-2]);
   for (i=0;i<N-3;i+=4)
   {
      v128_t yi, x1v, x2v, x3v, x4v;
      yi = wasm_v128_load(x+i);
      x4v = wasm_v128_load(&x[i-T+2]);
      x1v = wasm_i32x4_shuffle(x0v, x4v, 1, 2, 3, 4);
      x2v = wasm_i32x4_shuffle(x0v, x4v, 2, 3, 4, 5);
      x3v = wasm_i32x4_shuffle(x0v, x4v, 3, 4, 5, 6);
#if defined(__wasm_relaxed_simd__)
      yi = wasm_f32x4_relaxed_madd(g10v, x2v, yi);
      yi = wasm_f32x4_relaxed_madd(g11v, wasm_f32x4_add(x3v, x1v), yi);
      yi = wasm_f32x4_relaxed_madd(g12v, wasm_f32x4_add(x4v, x0v), yi);
#else
      /* Use partial sums, like the SSE version. */
      yi = wasm_f32x4_add(yi, wasm_f32x4_mul(g10v, x2v));
      yi = wasm_f32x4_add(yi, wasm_f32x4_add(
            wasm_f32x4_mul(g11v, wasm_f32x4_add(x3v, x1v)),
            wasm_f32x4_mul(g12v, wasm_f32x4_add(x4v, x0v))));
#endif
      x0v = x4v;
      wasm_v128_store(y+i, yi);
   }
   for (;i<N;i++)
   {
      y[i] = x[i]
               + MULT16_32_Q15(g10,x[i-T])
               + MULT16_32_Q15(g11,ADD32(x[i-T+1],x[i-T-1]))
               + MULT16_32_Q15(g12,ADD32(x[i-T+2],x[i-T-2]));
   }
}

/* De-emphasis y[n] = x[n] + coef*y[n-1], four samples at a time:
     Y = X + {1,c,c^2,c^3}*m + x0*{0,c,c^2,c^3} + x1*{0,0,c,c^2}
           + x2*{0,0,0,c},
   with m = c*y[-1]. Only the m term depends on the previous block, which
   breaks the one-sample-long dependency chain of the C version. */
typedef struct {
   v128_t w0;  /* {1, c, c^2, c^3} */
   v128_t w1;  /* {0, c, c^2, c^3} */
   v128_t w2;  /* {0, 0, c, c^2} */
   v128_t w3;  /* {0, 0, 0, c} */
} deemph_weights;

static OPUS_INLINE void deemph_init(deemph_weights *w, float c)
{
   float c2 = c*c;
   float c3 = c2*c;
   w->w0 = wasm_f32x4_make(1.f, c, c2, c3);
   w->w1 = wasm_f32x4_make(0.f, c, c2, c3);
   w->w2 = wasm_f32x4_make(0.f, 0.f, c, c2);
   w->w3 = wasm_f32x4_make(0.f, 0.f, 0.f, c);
}

/* Filters 4 samples. On input *m is splat(coef*y[-1]); on output it is
   splat(coef*y[3]). Returns y[0..3] (not yet scaled). */
static OPUS_INLINE v128_t deemph4(const deemph_weights *w, v128_t xv,
      v128_t *m, v128_t cv)
{
   v128_t acc;
   xv = wasm_f32x4_add(xv, wasm_f32x4_splat(VERY_SMALL));
   acc = WASM_F32X4_MADD(wasm_i32x4_shuffle(xv, xv, 0, 0, 0, 0), w->w1, xv);
   acc = WASM_F32X4_MADD(wasm_i32x4_shuffle(xv, xv, 1, 1, 1, 1), w->w2, acc);
   acc = WASM_F32X4_MADD(wasm_i32x4_shuffle(xv, xv, 2, 2, 2, 2), w->w3, acc);
   acc = WASM_F32X4_MADD(*m, w->w0, acc);
   *m = wasm_f32x4_mul(wasm_i32x4_shuffle(acc, acc, 3, 3, 3, 3), cv);
   return acc;
}

opus_val32 celt_deemphasis_wasm(opus_res *y, const opus_val32 *x,
      opus_val16 coef0, opus_val32 m, int N)
{
   int j;
   deemph_weights w;
   v128_t mv, cv, scale;
   deemph_init(&w, coef0);
   mv = wasm_f32x4_splat(m);
   cv = wasm_f32x4_splat(coef0);
   scale = wasm_f32x4_splat(1/CELT_SIG_SCALE);
   for (j=0;j<N-3;j+=4)
   {
      v128_t yv = deemph4(&w, wasm_v128_load(x+j), &mv, cv);
      wasm_v128_store(y+j, wasm_f32x4_mul(yv, scale));
   }
   m = wasm_f32x4_extract_lane(mv, 0);
   for (;j<N;j++)
   {
      celt_sig tmp = x[j] + VERY_SMALL + m;
      m = MULT16_32_Q15(coef0, tmp);
      y[j] = SIG2RES(tmp);
   }
   return m;
}

void deemphasis_stereo_simple_wasm(celt_sig *in[], opus_res *pcm, int N,
      opus_val16 coef0, celt_sig *mem)
{
   const celt_sig *x0;
   const celt_sig *x1;
   celt_sig m0, m1;
   int j;
   deemph_weights w;
   v128_t m0v, m1v, cv, scale;
   x0 = in[0];
   x1 = in[1];
   deemph_init(&w, coef0);
   m0v = wasm_f32x4_splat(mem[0]);
   m1v = wasm_f32x4_splat(mem[1]);
   cv = wasm_f32x4_splat(coef0);
   scale = wasm_f32x4_splat(1/CELT_SIG_SCALE);
   /* The two channels are independent chains, which the CPU can overlap. */
   for (j=0;j<N-3;j+=4)
   {
      v128_t y0, y1;
      y0 = deemph4(&w, wasm_v128_load(x0+j), &m0v, cv);
      y1 = deemph4(&w, wasm_v128_load(x1+j), &m1v, cv);
      y0 = wasm_f32x4_mul(y0, scale);
      y1 = wasm_f32x4_mul(y1, scale);
      wasm_v128_store(pcm+2*j, wasm_i32x4_shuffle(y0, y1, 0, 4, 1, 5));
      wasm_v128_store(pcm+2*j+4, wasm_i32x4_shuffle(y0, y1, 2, 6, 3, 7));
   }
   m0 = wasm_f32x4_extract_lane(m0v, 0);
   m1 = wasm_f32x4_extract_lane(m1v, 0);
   for (;j<N;j++)
   {
      celt_sig tmp0, tmp1;
      tmp0 = x0[j] + VERY_SMALL + m0;
      tmp1 = x1[j] + VERY_SMALL + m1;
      m0 = MULT16_32_Q15(coef0, tmp0);
      m1 = MULT16_32_Q15(coef0, tmp1);
      pcm[2*j  ] = SIG2RES(tmp0);
      pcm[2*j+1] = SIG2RES(tmp1);
   }
   mem[0] = m0;
   mem[1] = m1;
}

#endif /* FIXED_POINT */
