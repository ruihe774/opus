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
/* WebAssembly SIMD128 implementation of the DNN vector operations.
   Compile with -msimd128, or with -mrelaxed-simd to also use relaxed SIMD
   (fused multiply-add for the float code, and the i8 x i7 dot product for
   the int8 matrices). */

#ifndef VEC_WASM_H
#define VEC_WASM_H

#include <wasm_simd128.h>
#include "os_support.h"

#define MAX_INPUTS (2048)

/* Like vec_neon.h, the int8 matrices use signed activations
   (x_q = round(127*x)) and signed weights, so USE_SU_BIAS must not be
   defined: compute_linear() then uses the regular bias. */

#if defined(__wasm_relaxed_simd__)
/* a*b + c, possibly fused. */
#define vec_wasm_madd(a, b, c) wasm_f32x4_relaxed_madd(a, b, c)
#else
#define vec_wasm_madd(a, b, c) wasm_f32x4_add(wasm_f32x4_mul(a, b), c)
#endif

#ifndef LPCNET_TEST
static OPUS_INLINE v128_t exp4_approx(v128_t X)
{
   const v128_t K0 = wasm_f32x4_splat(0.99992522f);
   const v128_t K1 = wasm_f32x4_splat(0.69583354f);
   const v128_t K2 = wasm_f32x4_splat(0.22606716f);
   const v128_t K3 = wasm_f32x4_splat(0.078024523f);
   const v128_t log2_E = wasm_f32x4_splat(1.44269504f);
   const v128_t max_in = wasm_f32x4_splat(50.f);
   const v128_t min_in = wasm_f32x4_splat(-50.f);
   /* 1.5*2^23: adding it to an integer-valued float puts the integer in the
      low mantissa bits. */
   const v128_t magic = wasm_f32x4_splat(12582912.f);
   const v128_t mask = wasm_i32x4_splat(0x7fffffff);
   v128_t XF, Y, I;
   X = wasm_f32x4_mul(X, log2_E);
   /* pmin()/pmax() map to single instructions. In this order, a NaN input
      ends up as min_in. */
   X = wasm_f32x4_pmax(min_in, wasm_f32x4_pmin(X, max_in));
   XF = wasm_f32x4_floor(X);
   X = wasm_f32x4_sub(X, XF);
   Y = vec_wasm_madd(vec_wasm_madd(vec_wasm_madd(K3, X, K2), X, K1), X, K0);
   /* The magic constant's own bits are all shifted out, leaving integer<<23. */
   I = wasm_i32x4_shl(wasm_f32x4_add(XF, magic), 23);
   Y = wasm_v128_and(mask, wasm_i32x4_add(I, Y));
   return Y;
}

/* Approximating tanh() using a Padé-like rational function:
   tanh(x) ~= x * (N0 + N1*x^2 + N2*x^4)/(D0 + D1*x^2 + D2*x^4)
   subject to the +/- 1 bounds. There is no reciprocal estimate in
   WebAssembly, so this uses a real division, which makes it slightly more
   accurate than the x86 and ARM versions. */
static OPUS_INLINE v128_t tanh4_approx(v128_t X)
{
   const v128_t N0 = wasm_f32x4_splat(952.52801514f);
   const v128_t N1 = wasm_f32x4_splat(96.39235687f);
   const v128_t N2 = wasm_f32x4_splat(0.60863042f);
   const v128_t D0 = wasm_f32x4_splat(952.72399902f);
   const v128_t D1 = wasm_f32x4_splat(413.36801147f);
   const v128_t D2 = wasm_f32x4_splat(11.88600922f);
   const v128_t max_out = wasm_f32x4_splat(1.f);
   const v128_t min_out = wasm_f32x4_splat(-1.f);
   v128_t X2, num, den;
   X2 = wasm_f32x4_mul(X, X);
   num = vec_wasm_madd(vec_wasm_madd(N2, X2, N1), X2, N0);
   den = vec_wasm_madd(vec_wasm_madd(D2, X2, D1), X2, D0);
   num = wasm_f32x4_div(wasm_f32x4_mul(num, X), den);
   return wasm_f32x4_pmax(min_out, wasm_f32x4_pmin(num, max_out));
}

/* Sigmoid approximation using the same rational function as tanh4_approx(),
   with the coefficients scaled to compute 0.5 + 0.5*tanh(0.5*x). */
static OPUS_INLINE v128_t sigmoid4_approx(v128_t X)
{
   const v128_t N0 = wasm_f32x4_splat(238.13200378f);
   const v128_t N1 = wasm_f32x4_splat(6.02452230f);
   const v128_t N2 = wasm_f32x4_splat(0.00950985f);
   const v128_t D0 = wasm_f32x4_splat(952.72399902f);
   const v128_t D1 = wasm_f32x4_splat(103.34200287f);
   const v128_t D2 = wasm_f32x4_splat(0.74287558f);
   const v128_t half = wasm_f32x4_splat(0.5f);
   const v128_t max_out = wasm_f32x4_splat(1.f);
   const v128_t min_out = wasm_f32x4_splat(0.f);
   v128_t X2, num, den;
   X2 = wasm_f32x4_mul(X, X);
   num = vec_wasm_madd(vec_wasm_madd(N2, X2, N1), X2, N0);
   den = vec_wasm_madd(vec_wasm_madd(D2, X2, D1), X2, D0);
   num = wasm_f32x4_add(wasm_f32x4_div(wasm_f32x4_mul(num, X), den), half);
   return wasm_f32x4_pmax(min_out, wasm_f32x4_pmin(num, max_out));
}

static OPUS_INLINE float lpcnet_exp(float x)
{
   return wasm_f32x4_extract_lane(exp4_approx(wasm_f32x4_splat(x)), 0);
}

static OPUS_INLINE float tanh_approx(float x)
{
   return wasm_f32x4_extract_lane(tanh4_approx(wasm_f32x4_splat(x)), 0);
}

static OPUS_INLINE float sigmoid_approx(float x)
{
   return wasm_f32x4_extract_lane(sigmoid4_approx(wasm_f32x4_splat(x)), 0);
}

static OPUS_INLINE void softmax(float *y, const float *x, int N)
{
   int i;
   for (i=0;i<N-3;i+=4)
      wasm_v128_store(&y[i], exp4_approx(wasm_v128_load(&x[i])));
   for (;i<N;i++)
      y[i] = lpcnet_exp(x[i]);
}

static OPUS_INLINE void vec_tanh(float *y, const float *x, int N)
{
   int i;
   for (i=0;i<N-3;i+=4)
      wasm_v128_store(&y[i], tanh4_approx(wasm_v128_load(&x[i])));
   for (;i<N;i++)
      y[i] = tanh_approx(x[i]);
}

static OPUS_INLINE void vec_sigmoid(float *y, const float *x, int N)
{
   int i;
   for (i=0;i<N-3;i+=4)
      wasm_v128_store(&y[i], sigmoid4_approx(wasm_v128_load(&x[i])));
   for (;i<N;i++)
      y[i] = sigmoid_approx(x[i]);
}
#endif

/* Each output keeps its own accumulator and the products are added in the
   same order as the C code. 32 rows (8 accumulators) are processed at a
   time when possible, so that enough additions are in flight to hide their
   latency. */
static OPUS_INLINE void sgemv_wasm_rows(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   i=0;
   for (;i<rows-31;i+=32)
   {
      v128_t vy0, vy4, vy8, vy12, vy16, vy20, vy24, vy28;
      vy0 = vy4 = vy8 = vy12 = vy16 = vy20 = vy24 = vy28 = wasm_f32x4_splat(0.f);
      for (j=0;j<cols;j++)
      {
         const float *w;
         v128_t vxj;
         w = &weights[j*col_stride + i];
         vxj = wasm_v128_load32_splat(&x[j]);
         vy0 = vec_wasm_madd(wasm_v128_load(&w[0]), vxj, vy0);
         vy4 = vec_wasm_madd(wasm_v128_load(&w[4]), vxj, vy4);
         vy8 = vec_wasm_madd(wasm_v128_load(&w[8]), vxj, vy8);
         vy12 = vec_wasm_madd(wasm_v128_load(&w[12]), vxj, vy12);
         vy16 = vec_wasm_madd(wasm_v128_load(&w[16]), vxj, vy16);
         vy20 = vec_wasm_madd(wasm_v128_load(&w[20]), vxj, vy20);
         vy24 = vec_wasm_madd(wasm_v128_load(&w[24]), vxj, vy24);
         vy28 = vec_wasm_madd(wasm_v128_load(&w[28]), vxj, vy28);
      }
      wasm_v128_store(&out[i], vy0);
      wasm_v128_store(&out[i+4], vy4);
      wasm_v128_store(&out[i+8], vy8);
      wasm_v128_store(&out[i+12], vy12);
      wasm_v128_store(&out[i+16], vy16);
      wasm_v128_store(&out[i+20], vy20);
      wasm_v128_store(&out[i+24], vy24);
      wasm_v128_store(&out[i+28], vy28);
   }
   for (;i<rows-15;i+=16)
   {
      v128_t vy0, vy4, vy8, vy12;
      vy0 = vy4 = vy8 = vy12 = wasm_f32x4_splat(0.f);
      for (j=0;j<cols;j++)
      {
         const float *w;
         v128_t vxj;
         w = &weights[j*col_stride + i];
         vxj = wasm_v128_load32_splat(&x[j]);
         vy0 = vec_wasm_madd(wasm_v128_load(&w[0]), vxj, vy0);
         vy4 = vec_wasm_madd(wasm_v128_load(&w[4]), vxj, vy4);
         vy8 = vec_wasm_madd(wasm_v128_load(&w[8]), vxj, vy8);
         vy12 = vec_wasm_madd(wasm_v128_load(&w[12]), vxj, vy12);
      }
      wasm_v128_store(&out[i], vy0);
      wasm_v128_store(&out[i+4], vy4);
      wasm_v128_store(&out[i+8], vy8);
      wasm_v128_store(&out[i+12], vy12);
   }
   for (;i<rows-7;i+=8)
   {
      v128_t vy0, vy4;
      vy0 = vy4 = wasm_f32x4_splat(0.f);
      for (j=0;j<cols;j++)
      {
         const float *w;
         v128_t vxj;
         w = &weights[j*col_stride + i];
         vxj = wasm_v128_load32_splat(&x[j]);
         vy0 = vec_wasm_madd(wasm_v128_load(&w[0]), vxj, vy0);
         vy4 = vec_wasm_madd(wasm_v128_load(&w[4]), vxj, vy4);
      }
      wasm_v128_store(&out[i], vy0);
      wasm_v128_store(&out[i+4], vy4);
   }
   for (;i<rows-3;i+=4)
   {
      v128_t vy0;
      vy0 = wasm_f32x4_splat(0.f);
      for (j=0;j<cols;j++)
         vy0 = vec_wasm_madd(wasm_v128_load(&weights[j*col_stride + i]), wasm_v128_load32_splat(&x[j]), vy0);
      wasm_v128_store(&out[i], vy0);
   }
   for (;i<rows;i++)
   {
      out[i] = 0;
      for (j=0;j<cols;j++) out[i] += weights[j*col_stride + i]*x[j];
   }
}

static OPUS_INLINE void sgemv16x1(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   sgemv_wasm_rows(out, weights, rows, cols, col_stride, x);
}

static OPUS_INLINE void sgemv8x1(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   sgemv_wasm_rows(out, weights, rows, cols, col_stride, x);
}

static OPUS_INLINE void sgemv(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   sgemv_wasm_rows(out, weights, rows, cols, col_stride, x);
}

/* Columns 0 and 2 of each block go to one set of accumulators, and columns
   1 and 3 to another, which halves the length of the dependency chains. */
static OPUS_INLINE void sparse_sgemv8x4(float *out, const float *w, const int *idx, int rows, const float *x)
{
   int i, j;
   for (i=0;i<rows;i+=8)
   {
      int cols;
      v128_t vy0, vy4, vz0, vz4;
      vy0 = vy4 = vz0 = vz4 = wasm_f32x4_splat(0.f);
      cols = *idx++;
      for (j=0;j<cols;j++)
      {
         int pos;
         v128_t vxj;
         pos = *idx++;
         vxj = wasm_v128_load32_splat(&x[pos]);
         vy0 = vec_wasm_madd(wasm_v128_load(&w[0]), vxj, vy0);
         vy4 = vec_wasm_madd(wasm_v128_load(&w[4]), vxj, vy4);
         vxj = wasm_v128_load32_splat(&x[pos+1]);
         vz0 = vec_wasm_madd(wasm_v128_load(&w[8]), vxj, vz0);
         vz4 = vec_wasm_madd(wasm_v128_load(&w[12]), vxj, vz4);
         vxj = wasm_v128_load32_splat(&x[pos+2]);
         vy0 = vec_wasm_madd(wasm_v128_load(&w[16]), vxj, vy0);
         vy4 = vec_wasm_madd(wasm_v128_load(&w[20]), vxj, vy4);
         vxj = wasm_v128_load32_splat(&x[pos+3]);
         vz0 = vec_wasm_madd(wasm_v128_load(&w[24]), vxj, vz0);
         vz4 = vec_wasm_madd(wasm_v128_load(&w[28]), vxj, vz4);
         w += 32;
      }
      wasm_v128_store(&out[i], wasm_f32x4_add(vy0, vz0));
      wasm_v128_store(&out[i+4], wasm_f32x4_add(vy4, vz4));
   }
}

/* Quantizes x[0..3] to round(127*x), saturated to [-127, 127], as i32x4.
   This rounds exactly like the floor(.5+127*x) of the C code (which rounds
   the product to float, then adds .5 in double precision): the product is
   rounded up when its fractional part, which is computed exactly, is at
   least .5. A NaN gives -127. */
static OPUS_INLINE v128_t quantize4_wasm(const float *x)
{
   const v128_t const127 = wasm_f32x4_splat(127.f);
   const v128_t constm127 = wasm_f32x4_splat(-127.f);
   const v128_t half = wasm_f32x4_splat(.5f);
   /* 1.5*2^23, to convert small integer-valued floats to i32. */
   const v128_t magic = wasm_f32x4_splat(12582912.f);
   v128_t X, XF, up;
   X = wasm_f32x4_mul(wasm_v128_load(x), const127);
   X = wasm_f32x4_pmax(constm127, wasm_f32x4_pmin(X, const127));
   XF = wasm_f32x4_floor(X);
   up = wasm_f32x4_ge(wasm_f32x4_sub(X, XF), half);
   return wasm_i32x4_sub(wasm_i32x4_sub(wasm_f32x4_add(XF, magic), magic), up);
}

#if defined(__wasm_relaxed_simd__)

/* With relaxed SIMD, the int8 products use i32x4.relaxed_dot_i8x16_i7x16_add,
   whose result is only deterministic when every byte of the second operand
   is in [0, 127]. The activations are therefore split into |x_q| (0..127)
   and a sign mask s (0 or -1), and the sign is moved to the weights:
   (w ^ s) - s = s ? -w : w. This is exact as long as no weight is -128,
   which the weight export code guarantees (quantize_weight() rejects it).
   Each pair of products is at most 2*127*127 in magnitude, so the i16
   intermediate sums cannot saturate either. */

static OPUS_INLINE void vector_ps_to_epi8_wasm(unsigned char *xa, unsigned char *xs, const float *_x, int len)
{
   int i;
   for (i=0;i<len;i+=4)
   {
      v128_t q, a, s;
      q = quantize4_wasm(&_x[i]);
      a = wasm_i32x4_abs(q);
      s = wasm_i32x4_shr(q, 31);
      a = wasm_i16x8_narrow_i32x4(a, a);
      s = wasm_i16x8_narrow_i32x4(s, s);
      wasm_v128_store32_lane(&xa[i], wasm_i8x16_narrow_i16x8(a, a), 0);
      wasm_v128_store32_lane(&xs[i], wasm_i8x16_narrow_i16x8(s, s), 0);
   }
}

/* Accumulates 4 rows x 4 columns of signed weights times the activations
   given as magnitude a and sign mask s (both broadcast to all 4 rows). */
static OPUS_INLINE v128_t dot4x4_wasm(v128_t acc, v128_t w, v128_t a, v128_t s)
{
   w = wasm_i8x16_sub(wasm_v128_xor(w, s), s);
   return wasm_i32x4_relaxed_dot_i8x16_i7x16_add(w, a, acc);
}

static OPUS_INLINE void cgemv8x4(float *_out, const opus_int8 *w, const float *scale, int rows, int cols, const float *_x)
{
   int i, j;
   unsigned char xa[MAX_INPUTS];
   unsigned char xs[MAX_INPUTS];
   vector_ps_to_epi8_wasm(xa, xs, _x, cols);
   for (i=0;i<rows;i+=8)
   {
      v128_t acc0, acc1, acc2, acc3;
      acc0 = acc1 = acc2 = acc3 = wasm_i32x4_splat(0);
      j=0;
      for (;j<cols-4;j+=8)
      {
         v128_t a, s;
         a = wasm_v128_load32_splat(&xa[j]);
         s = wasm_v128_load32_splat(&xs[j]);
         acc0 = dot4x4_wasm(acc0, wasm_v128_load(&w[0]), a, s);
         acc1 = dot4x4_wasm(acc1, wasm_v128_load(&w[16]), a, s);
         a = wasm_v128_load32_splat(&xa[j+4]);
         s = wasm_v128_load32_splat(&xs[j+4]);
         acc2 = dot4x4_wasm(acc2, wasm_v128_load(&w[32]), a, s);
         acc3 = dot4x4_wasm(acc3, wasm_v128_load(&w[48]), a, s);
         w += 64;
      }
      for (;j<cols;j+=4)
      {
         v128_t a, s;
         a = wasm_v128_load32_splat(&xa[j]);
         s = wasm_v128_load32_splat(&xs[j]);
         acc0 = dot4x4_wasm(acc0, wasm_v128_load(&w[0]), a, s);
         acc1 = dot4x4_wasm(acc1, wasm_v128_load(&w[16]), a, s);
         w += 32;
      }
      acc0 = wasm_i32x4_add(acc0, acc2);
      acc1 = wasm_i32x4_add(acc1, acc3);
      wasm_v128_store(&_out[i], wasm_f32x4_mul(wasm_v128_load(&scale[i]), wasm_f32x4_convert_i32x4(acc0)));
      wasm_v128_store(&_out[i+4], wasm_f32x4_mul(wasm_v128_load(&scale[i+4]), wasm_f32x4_convert_i32x4(acc1)));
   }
}

static OPUS_INLINE void sparse_cgemv8x4(float *_out, const opus_int8 *w, const int *idx, const float *scale, int rows, int cols, const float *_x)
{
   int i, j;
   unsigned char xa[MAX_INPUTS];
   unsigned char xs[MAX_INPUTS];
   vector_ps_to_epi8_wasm(xa, xs, _x, cols);
   for (i=0;i<rows;i+=8)
   {
      int colblocks;
      v128_t acc0, acc1;
      acc0 = acc1 = wasm_i32x4_splat(0);
      colblocks = *idx++;
      for (j=0;j<colblocks;j++)
      {
         int pos;
         v128_t a, s;
         pos = *idx++;
         a = wasm_v128_load32_splat(&xa[pos]);
         s = wasm_v128_load32_splat(&xs[pos]);
         acc0 = dot4x4_wasm(acc0, wasm_v128_load(&w[0]), a, s);
         acc1 = dot4x4_wasm(acc1, wasm_v128_load(&w[16]), a, s);
         w += 32;
      }
      wasm_v128_store(&_out[i], wasm_f32x4_mul(wasm_v128_load(&scale[i]), wasm_f32x4_convert_i32x4(acc0)));
      wasm_v128_store(&_out[i+4], wasm_f32x4_mul(wasm_v128_load(&scale[i+4]), wasm_f32x4_convert_i32x4(acc1)));
   }
}

#else

/* Without relaxed SIMD, the activations are stored as i16 and each block of
   4 rows x 4 columns uses two i32x4.dot_i16x8 on the sign-extended weights.
   Each lane then holds the sum of two products (one half of a row), and the
   halves are only added together at the end. */

static OPUS_INLINE void vector_ps_to_epi16_wasm(opus_int16 *x, const float *_x, int len)
{
   int i;
   for (i=0;i<len;i+=4)
   {
      v128_t q;
      q = quantize4_wasm(&_x[i]);
      wasm_v128_store64_lane(&x[i], wasm_i16x8_narrow_i32x4(q, q), 0);
   }
}

/* Adds the pairs of adjacent lanes: [a0+a1, a2+a3, b0+b1, b2+b3]. */
static OPUS_INLINE v128_t hadd_pairs_wasm(v128_t a, v128_t b)
{
   return wasm_i32x4_add(wasm_i32x4_shuffle(a, b, 0, 2, 4, 6), wasm_i32x4_shuffle(a, b, 1, 3, 5, 7));
}

static OPUS_INLINE void cgemv8x4(float *_out, const opus_int8 *w, const float *scale, int rows, int cols, const float *_x)
{
   int i, j;
   opus_int16 x[MAX_INPUTS];
   vector_ps_to_epi16_wasm(x, _x, cols);
   for (i=0;i<rows;i+=8)
   {
      v128_t acc0, acc1, acc2, acc3;
      acc0 = acc1 = acc2 = acc3 = wasm_i32x4_splat(0);
      for (j=0;j<cols;j+=4)
      {
         v128_t vx, vw0, vw1;
         vx = wasm_v128_load64_splat(&x[j]);
         vw0 = wasm_v128_load(&w[0]);
         vw1 = wasm_v128_load(&w[16]);
         acc0 = wasm_i32x4_add(acc0, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(vw0), vx));
         acc1 = wasm_i32x4_add(acc1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(vw0), vx));
         acc2 = wasm_i32x4_add(acc2, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(vw1), vx));
         acc3 = wasm_i32x4_add(acc3, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(vw1), vx));
         w += 32;
      }
      wasm_v128_store(&_out[i], wasm_f32x4_mul(wasm_v128_load(&scale[i]), wasm_f32x4_convert_i32x4(hadd_pairs_wasm(acc0, acc1))));
      wasm_v128_store(&_out[i+4], wasm_f32x4_mul(wasm_v128_load(&scale[i+4]), wasm_f32x4_convert_i32x4(hadd_pairs_wasm(acc2, acc3))));
   }
}

static OPUS_INLINE void sparse_cgemv8x4(float *_out, const opus_int8 *w, const int *idx, const float *scale, int rows, int cols, const float *_x)
{
   int i, j;
   opus_int16 x[MAX_INPUTS];
   vector_ps_to_epi16_wasm(x, _x, cols);
   for (i=0;i<rows;i+=8)
   {
      int colblocks;
      v128_t acc0, acc1, acc2, acc3;
      acc0 = acc1 = acc2 = acc3 = wasm_i32x4_splat(0);
      colblocks = *idx++;
      for (j=0;j<colblocks;j++)
      {
         v128_t vx, vw0, vw1;
         vx = wasm_v128_load64_splat(&x[*idx++]);
         vw0 = wasm_v128_load(&w[0]);
         vw1 = wasm_v128_load(&w[16]);
         acc0 = wasm_i32x4_add(acc0, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(vw0), vx));
         acc1 = wasm_i32x4_add(acc1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(vw0), vx));
         acc2 = wasm_i32x4_add(acc2, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(vw1), vx));
         acc3 = wasm_i32x4_add(acc3, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(vw1), vx));
         w += 32;
      }
      wasm_v128_store(&_out[i], wasm_f32x4_mul(wasm_v128_load(&scale[i]), wasm_f32x4_convert_i32x4(hadd_pairs_wasm(acc0, acc1))));
      wasm_v128_store(&_out[i+4], wasm_f32x4_mul(wasm_v128_load(&scale[i+4]), wasm_f32x4_convert_i32x4(hadd_pairs_wasm(acc2, acc3))));
   }
}

#endif

#define SCALE (128.f*127.f)
#define SCALE_1 (1.f/128.f/127.f)

#endif /*VEC_WASM_H*/
