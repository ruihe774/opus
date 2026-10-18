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

#ifndef MAIN_WASM_H
#define MAIN_WASM_H

#include "NSQ_wasm.h"

void silk_VQ_WMat_EC_wasm(
    opus_int8                   *ind,                           /* O    index of best codebook vector               */
    opus_int32                  *res_nrg_Q15,                   /* O    best residual energy                        */
    opus_int32                  *rate_dist_Q8,                  /* O    best total bitrate                          */
    opus_int                    *gain_Q7,                       /* O    sum of absolute LTP coefficients            */
    const opus_int32            *XX_Q17,                        /* I    correlation matrix                          */
    const opus_int32            *xX_Q17,                        /* I    correlation vector                          */
    const opus_int8             *cb_Q7,                         /* I    codebook                                    */
    const opus_uint8            *cb_gain_Q7,                    /* I    codebook effective gain                     */
    const opus_uint8            *cl_Q5,                         /* I    code length for each codebook vector        */
    const opus_int              subfr_len,                      /* I    number of samples per subframe              */
    const opus_int32            max_gain_Q7,                    /* I    maximum sum of absolute LTP coefficients    */
    const opus_int              L                               /* I    number of vectors in codebook               */
);

#define OVERRIDE_silk_VQ_WMat_EC
#define silk_VQ_WMat_EC(ind, res_nrg_Q15, rate_dist_Q8, gain_Q7, XX_Q17, xX_Q17, cb_Q7, cb_gain_Q7, cl_Q5, \
                        subfr_len, max_gain_Q7, L, arch) \
    ((void)(arch),silk_VQ_WMat_EC_wasm(ind, res_nrg_Q15, rate_dist_Q8, gain_Q7, XX_Q17, xX_Q17, cb_Q7, cb_gain_Q7, cl_Q5, \
                       subfr_len, max_gain_Q7, L))

/**************************/
/* Noise level estimation */
/**************************/
void silk_VAD_GetNoiseLevels(
    const opus_int32            pX[ VAD_N_BANDS ],  /* I    subband energies                            */
    silk_VAD_state              *psSilk_VAD         /* I/O  Pointer to Silk VAD state                   */
);

opus_int silk_VAD_GetSA_Q8_wasm(
    silk_encoder_state *psEnC,
    const opus_int16   pIn[]
);

#define OVERRIDE_silk_VAD_GetSA_Q8
#define silk_VAD_GetSA_Q8(psEnC, pIn, arch) ((void)(arch),silk_VAD_GetSA_Q8_wasm(psEnC, pIn))

#ifndef FIXED_POINT
double silk_inner_product_FLP_wasm(
    const silk_float    *data1,
    const silk_float    *data2,
    opus_int            dataSize
);

#define OVERRIDE_inner_product_FLP
#define silk_inner_product_FLP(data1, data2, dataSize, arch) ((void)(arch),silk_inner_product_FLP_wasm(data1, data2, dataSize))
#endif

#endif /* MAIN_WASM_H */
