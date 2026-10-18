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

#ifndef NSQ_WASM_H
#define NSQ_WASM_H

void silk_NSQ_wasm(
    const silk_encoder_state    *psEncC,                                      /* I    Encoder State                   */
    silk_nsq_state              *NSQ,                                         /* I/O  NSQ state                       */
    SideInfoIndices             *psIndices,                                   /* I/O  Quantization Indices            */
    const opus_int16            x16[],                                        /* I    Input                           */
    opus_int8                   pulses[],                                     /* O    Quantized pulse signal          */
    const opus_int16            *PredCoef_Q12,                                /* I    Short term prediction coefs     */
    const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],      /* I    Long term prediction coefs      */
    const opus_int16            AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I    Noise shaping coefs             */
    const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],            /* I    Long term shaping coefs         */
    const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                     /* I    Spectral tilt                   */
    const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                   /* I    Low frequency shaping coefs     */
    const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                    /* I    Quantization step sizes         */
    const opus_int              pitchL[ MAX_NB_SUBFR ],                       /* I    Pitch lags                      */
    const opus_int              Lambda_Q10,                                   /* I    Rate/distortion tradeoff        */
    const opus_int              LTP_scale_Q14                                 /* I    LTP state scaling               */
);

#define OVERRIDE_silk_NSQ
#define silk_NSQ(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, \
                 HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((void)(arch),silk_NSQ_wasm(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, \
                 HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))

void silk_NSQ_del_dec_wasm(
    const silk_encoder_state    *psEncC,                                      /* I    Encoder State                   */
    silk_nsq_state              *NSQ,                                         /* I/O  NSQ state                       */
    SideInfoIndices             *psIndices,                                   /* I/O  Quantization Indices            */
    const opus_int16            x16[],                                        /* I    Input                           */
    opus_int8                   pulses[],                                     /* O    Quantized pulse signal          */
    const opus_int16            *PredCoef_Q12,                                /* I    Short term prediction coefs     */
    const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],      /* I    Long term prediction coefs      */
    const opus_int16            AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I    Noise shaping coefs             */
    const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],            /* I    Long term shaping coefs         */
    const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                     /* I    Spectral tilt                   */
    const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                   /* I    Low frequency shaping coefs     */
    const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                    /* I    Quantization step sizes         */
    const opus_int              pitchL[ MAX_NB_SUBFR ],                       /* I    Pitch lags                      */
    const opus_int              Lambda_Q10,                                   /* I    Rate/distortion tradeoff        */
    const opus_int              LTP_scale_Q14                                 /* I    LTP state scaling               */
);

#define OVERRIDE_silk_NSQ_del_dec
#define silk_NSQ_del_dec(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, \
                         HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14, arch) \
    ((void)(arch),silk_NSQ_del_dec_wasm(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13, \
                         HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10, LTP_scale_Q14))

#endif /* NSQ_WASM_H */
