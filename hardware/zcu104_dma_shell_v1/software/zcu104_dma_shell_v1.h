#ifndef ZCU104_DMA_SHELL_V1_H
#define ZCU104_DMA_SHELL_V1_H

#include <stdint.h>

/* Frozen hardware addresses for zcu104_dma_shell_v1. */
#define DMA_SHELL_BASE_ADDR          UINTPTR_C(0xA0000000)
#define AXI_DMA_BASE_ADDR            UINTPTR_C(0xA0010000)

#define DMA_SHELL_CONTROL            0x00u
#define DMA_SHELL_STATUS             0x04u
#define DMA_SHELL_EXEC_COUNT_LO      0x08u
#define DMA_SHELL_EXEC_COUNT_HI      0x0cu
#define DMA_SHELL_CORE_CYCLES_LO     0x10u
#define DMA_SHELL_CORE_CYCLES_HI     0x14u
#define DMA_SHELL_EXPECTED_IN_WORDS  0x18u
#define DMA_SHELL_EXPECTED_OUT_WORDS 0x1cu
#define DMA_SHELL_INPUT_SEEN         0x20u
#define DMA_SHELL_OUTPUT_SEEN        0x24u
#define DMA_SHELL_ERROR_FLAGS        0x28u
#define DMA_SHELL_ID                 0x2cu
#define DMA_SHELL_VERSION            0x30u
#define DMA_SHELL_OPERATOR_ID        0x34u
#define DMA_SHELL_AXIS_DATA_BITS     0x38u
#define DMA_SHELL_CLOCK_HZ           0x3cu

#define DMA_SHELL_CONTROL_START        (1u << 0)
#define DMA_SHELL_CONTROL_CLEAR_DONE   (1u << 1)
#define DMA_SHELL_CONTROL_CLEAR_ERRORS (1u << 2)

#define DMA_SHELL_STATUS_IDLE          (1u << 0)
#define DMA_SHELL_STATUS_BUSY          (1u << 1)
#define DMA_SHELL_STATUS_DONE          (1u << 2)
#define DMA_SHELL_STATUS_ERROR         (1u << 3)

#define DMA_SHELL_STAGE4_BYTES          25088u
#define DMA_SHELL_STAGE4_DMA_WORDS_256  784u

/*
 * Runtime sequencing contract (polling v1):
 *   1. Flush the 25,088-byte input buffer and invalidate the output buffer.
 *   2. Arm AXI DMA S2MM for 25,088 bytes.
 *   3. Clear shell DONE/errors and write START.
 *   4. Arm AXI DMA MM2S for 25,088 bytes.
 *   5. Poll both DMA channels and DMA_SHELL_STATUS_DONE; then invalidate output.
 *
 * Measure PS/DMA end-to-end time in software around steps 2--5. Never store it
 * as DMA_SHELL_CORE_CYCLES, which is the hardware handshake interval only.
 */

#endif  /* ZCU104_DMA_SHELL_V1_H */
