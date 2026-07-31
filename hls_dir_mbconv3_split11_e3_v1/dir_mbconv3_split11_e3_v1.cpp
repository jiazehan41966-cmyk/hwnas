#include "dir_mbconv3_split11_e3_v1.h"

#include <limits.h>

namespace {

int32_t round_shift_signed(int32_t value, int shift) {
    const int32_t half = 1 << (shift - 1);
    if (value >= 0) {
        return (value + half) >> shift;
    }
    return -(((-value) + half) >> shift);
}

int8_t saturate_int8(int32_t value) {
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return static_cast<int8_t>(value);
}

int8_t relu6_int8(int32_t value) {
    if (value < 0) {
        return 0;
    }
    if (value > 96) {
        return 96;
    }
    return static_cast<int8_t>(value);
}

int chw_index(int channel, int y, int x, int channels) {
    (void)channels;
    return (channel * DIR_H + y) * DIR_W + x;
}

}  // namespace

void dir_mbconv3_split11_e3_v1_int8(
    const int8_t input[DIR_C * DIR_H * DIR_W],
    const int8_t expand_w[DIR_E * DIR_C],
    const int8_t dw_1x3_w[DIR_B * 3],
    const int8_t dw_3x1_w[DIR_B * 3],
    const int8_t project_w[DIR_C * DIR_E],
    int8_t output[DIR_C * DIR_H * DIR_W]) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=expand_w offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=dw_1x3_w offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=dw_3x1_w offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=project_w offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem5
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int8_t expanded[DIR_E][DIR_H][DIR_W];
    int8_t directional[DIR_E][DIR_H][DIR_W];

expand_channels:
    for (int oc = 0; oc < DIR_E; ++oc) {
    expand_y:
        for (int y = 0; y < DIR_H; ++y) {
        expand_x:
            for (int x = 0; x < DIR_W; ++x) {
                int32_t accumulator = 0;
            expand_reduce:
                for (int ic = 0; ic < DIR_C; ++ic) {
                    accumulator +=
                        static_cast<int32_t>(
                            input[chw_index(ic, y, x, DIR_C)]) *
                        static_cast<int32_t>(expand_w[oc * DIR_C + ic]);
                }
                expanded[oc][y][x] = relu6_int8(
                    saturate_int8(round_shift_signed(accumulator, 10)));
            }
        }
    }

direction_first_1x3:
    for (int channel = 0; channel < DIR_B; ++channel) {
        for (int y = 0; y < DIR_H; ++y) {
            for (int x = 0; x < DIR_W; ++x) {
                int32_t accumulator = 0;
                for (int tap = 0; tap < 3; ++tap) {
                    const int source_x = x + tap - 1;
                    if (source_x >= 0 && source_x < DIR_W) {
                        accumulator +=
                            static_cast<int32_t>(
                                expanded[channel][y][source_x]) *
                            static_cast<int32_t>(
                                dw_1x3_w[channel * 3 + tap]);
                    }
                }
                directional[channel][y][x] = relu6_int8(
                    saturate_int8(round_shift_signed(accumulator, 8)));
            }
        }
    }

direction_second_3x1:
    for (int channel = 0; channel < DIR_B; ++channel) {
        for (int y = 0; y < DIR_H; ++y) {
            for (int x = 0; x < DIR_W; ++x) {
                int32_t accumulator = 0;
                for (int tap = 0; tap < 3; ++tap) {
                    const int source_y = y + tap - 1;
                    if (source_y >= 0 && source_y < DIR_H) {
                        accumulator +=
                            static_cast<int32_t>(
                                expanded[channel + DIR_B][source_y][x]) *
                            static_cast<int32_t>(
                                dw_3x1_w[channel * 3 + tap]);
                    }
                }
                directional[channel + DIR_B][y][x] = relu6_int8(
                    saturate_int8(round_shift_signed(accumulator, 8)));
            }
        }
    }

project_channels:
    for (int oc = 0; oc < DIR_C; ++oc) {
        for (int y = 0; y < DIR_H; ++y) {
            for (int x = 0; x < DIR_W; ++x) {
                int32_t accumulator = 0;
            project_reduce:
                for (int ic = 0; ic < DIR_E; ++ic) {
                    accumulator +=
                        static_cast<int32_t>(directional[ic][y][x]) *
                        static_cast<int32_t>(project_w[oc * DIR_E + ic]);
                }
                const int32_t projected =
                    round_shift_signed(accumulator, 12);
                const int index = chw_index(oc, y, x, DIR_C);
                output[index] = saturate_int8(
                    projected + static_cast<int32_t>(input[index]));
            }
        }
    }
}
