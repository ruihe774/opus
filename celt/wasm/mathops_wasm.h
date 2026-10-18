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

#ifndef MATHOPS_WASM_H
#define MATHOPS_WASM_H

#if !defined(DISABLE_FLOAT_API)

void celt_float2int16_wasm(const float * OPUS_RESTRICT in,
      short * OPUS_RESTRICT out, int cnt);

#define OVERRIDE_FLOAT2INT16
#define celt_float2int16(in, out, cnt, arch) \
    ((void)(arch), celt_float2int16_wasm(in, out, cnt))

int opus_limit2_checkwithin1_wasm(float *samples, int cnt);

#define OVERRIDE_LIMIT2_CHECKWITHIN1
#define opus_limit2_checkwithin1(samples, cnt, arch) \
    ((void)(arch), opus_limit2_checkwithin1_wasm(samples, cnt))

#endif /* !DISABLE_FLOAT_API */

#endif /* MATHOPS_WASM_H */
