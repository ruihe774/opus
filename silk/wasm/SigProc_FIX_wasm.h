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

#ifndef SIGPROC_FIX_WASM_H
#define SIGPROC_FIX_WASM_H

void silk_biquad_alt_stride2_wasm(
    const opus_int16            *in,                /* I     input signal                                               */
    const opus_int32            *B_Q28,             /* I     MA coefficients [3]                                        */
    const opus_int32            *A_Q28,             /* I     AR coefficients [2]                                        */
    opus_int32                  *S,                 /* I/O   State vector [4]                                           */
    opus_int16                  *out,               /* O     output signal                                              */
    const opus_int32            len                 /* I     signal length (must be even)                               */
);

#define OVERRIDE_silk_biquad_alt_stride2
#define silk_biquad_alt_stride2(in, B_Q28, A_Q28, S, out, len, arch) \
    ((void)(arch), silk_biquad_alt_stride2_wasm(in, B_Q28, A_Q28, S, out, len))

opus_int32 silk_LPC_inverse_pred_gain_wasm(         /* O   Returns inverse prediction gain in energy domain, Q30        */
    const opus_int16            *A_Q12,             /* I   Prediction coefficients, Q12 [order]                         */
    const opus_int              order               /* I   Prediction order                                             */
);

#define OVERRIDE_silk_LPC_inverse_pred_gain
#define silk_LPC_inverse_pred_gain(A_Q12, order, arch) \
    ((void)(arch), silk_LPC_inverse_pred_gain_wasm(A_Q12, order))

/* The kernels below are only used (and only compiled) in fixed-point builds. */
#ifdef FIXED_POINT
void silk_burg_modified_wasm(
    opus_int32                  *res_nrg,           /* O    Residual energy                                             */
    opus_int                    *res_nrg_Q,         /* O    Residual energy Q value                                     */
    opus_int32                  A_Q16[],            /* O    Prediction coefficients (length order)                      */
    const opus_int16            x[],                /* I    Input signal, length: nb_subfr * ( D + subfr_length )       */
    const opus_int32            minInvGain_Q30,     /* I    Inverse of max prediction gain                              */
    const opus_int              subfr_length,       /* I    Input signal subframe length (incl. D preceding samples)    */
    const opus_int              nb_subfr,           /* I    Number of subframes stacked in x                            */
    const opus_int              D,                  /* I    Order                                                       */
    int                         arch                /* I    Run-time architecture                                       */
);

#define OVERRIDE_silk_burg_modified
#define silk_burg_modified(res_nrg, res_nrg_Q, A_Q16, x, minInvGain_Q30, subfr_length, nb_subfr, D, arch) \
    silk_burg_modified_wasm(res_nrg, res_nrg_Q, A_Q16, x, minInvGain_Q30, subfr_length, nb_subfr, D, arch)

opus_int64 silk_inner_prod16_wasm(
    const opus_int16            *inVec1,            /*    I input vector 1                                              */
    const opus_int16            *inVec2,            /*    I input vector 2                                              */
    const opus_int              len                 /*    I vector lengths                                              */
);

#define OVERRIDE_silk_inner_prod16
#define silk_inner_prod16(inVec1, inVec2, len, arch) \
    ((void)(arch), silk_inner_prod16_wasm(inVec1, inVec2, len))
#endif

#endif /* SIGPROC_FIX_WASM_H */
