#ifndef ZCU104_DMA_SHELL_MBCONV_K3_E3_PILOT_H
#define ZCU104_DMA_SHELL_MBCONV_K3_E3_PILOT_H

#include "zcu104_dma_shell_v1.h"

#define DMA_SHELL_MBCONV_K3_E3_OPERATOR_ID UINT32_C(0x00040002)

#define DMA_SHELL_WEIGHTS_BASE_ADDR UINTPTR_C(0xA0100000)
#define DMA_SHELL_BIASES_BASE_ADDR  UINTPTR_C(0xA0200000)

#define DMA_SHELL_MBCONV_K3_E3_WEIGHT_BYTES 7008u
#define DMA_SHELL_MBCONV_K3_E3_BIAS_COUNT   224u
#define DMA_SHELL_MBCONV_K3_E3_BIAS_BYTES   896u

/*
 * Parameter loading is deliberately outside the pilot build gate. Before a
 * future board transaction, software must write the frozen flattened weights
 * byte-for-byte and the 224 signed INT32 biases in native little-endian words,
 * then verify DMA_SHELL_OPERATOR_ID. This pilot does not run that transaction.
 */

#endif  /* ZCU104_DMA_SHELL_MBCONV_K3_E3_PILOT_H */
