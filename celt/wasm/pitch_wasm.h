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

#ifndef PITCH_WASM_H
#define PITCH_WASM_H

/* WebAssembly SIMD128 kernels. There is no run-time CPU detection on
   WebAssembly: they are selected at compile time and the arch argument is
   ignored. */

opus_val32 celt_inner_prod_wasm(const opus_val16 *x, const opus_val16 *y,
      int N);

void dual_inner_prod_wasm(const opus_val16 *x, const opus_val16 *y01,
      const opus_val16 *y02, int N, opus_val32 *xy1, opus_val32 *xy2);

void xcorr_kernel_wasm(const opus_val16 *x, const opus_val16 *y,
      opus_val32 sum[4], int len);

#ifdef FIXED_POINT
opus_val32
#else
void
#endif
celt_pitch_xcorr_wasm(const opus_val16 *_x, const opus_val16 *_y,
      opus_val32 *xcorr, int len, int max_pitch, int arch);

#define OVERRIDE_CELT_INNER_PROD
#define celt_inner_prod(x, y, N, arch) \
    ((void)(arch), celt_inner_prod_wasm(x, y, N))

#define OVERRIDE_DUAL_INNER_PROD
#define dual_inner_prod(x, y01, y02, N, xy1, xy2, arch) \
    ((void)(arch), dual_inner_prod_wasm(x, y01, y02, N, xy1, xy2))

#define OVERRIDE_XCORR_KERNEL
#define xcorr_kernel(x, y, sum, len, arch) \
    ((void)(arch), xcorr_kernel_wasm(x, y, sum, len))

#define OVERRIDE_PITCH_XCORR
#define celt_pitch_xcorr celt_pitch_xcorr_wasm

#ifndef FIXED_POINT

void comb_filter_const_wasm(opus_val32 *y, opus_val32 *x, int T, int N,
      opus_val16 g10, opus_val16 g11, opus_val16 g12);

#define OVERRIDE_COMB_FILTER_CONST
#define comb_filter_const(y, x, T, N, g10, g11, g12, arch) \
    ((void)(arch), comb_filter_const_wasm(y, x, T, N, g10, g11, g12))

opus_val32 celt_deemphasis_wasm(opus_res *y, const opus_val32 *x,
      opus_val16 coef0, opus_val32 m, int N);

void deemphasis_stereo_simple_wasm(celt_sig *in[], opus_res *pcm, int N,
      opus_val16 coef0, celt_sig *mem);

#define OVERRIDE_CELT_DEEMPHASIS
#define celt_deemphasis(y, x, coef, m, N, arch) \
    ((void)(arch), celt_deemphasis_wasm(y, x, coef, m, N))

#define OVERRIDE_DEEMPHASIS_STEREO_SIMPLE
#define deemphasis_stereo_simple(in, pcm, N, coef, mem, arch) \
    ((void)(arch), deemphasis_stereo_simple_wasm(in, pcm, N, coef, mem))

#endif /* !FIXED_POINT */

#endif /* PITCH_WASM_H */
