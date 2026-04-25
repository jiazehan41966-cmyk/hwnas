#ifndef HLS_LUT_COMMON_DEFS_H_
#define HLS_LUT_COMMON_DEFS_H_

#include "common_types.h"

typedef data_t weight_t;

#define MAKE_AXIS_TYPE(width_bits) ap_axiu<width_bits, 0, 0, 0>

#define DECLARE_AXIS_INTERFACE() \
    _Pragma("HLS INTERFACE axis port=input_stream") \
    _Pragma("HLS INTERFACE axis port=output_stream")

#define DECLARE_BRAM_WEIGHTS() \
    _Pragma("HLS INTERFACE bram port=weights") \
    _Pragma("HLS INTERFACE bram port=biases")

#define DECLARE_CONTROL_INTERFACE() \
    _Pragma("HLS INTERFACE s_axilite port=return bundle=control")

template <int CHANNELS>
static inline void unpack_packed_channels(
    const ap_uint<CHANNELS * 8> &packed,
    data_t values[CHANNELS]
) {
    for (int idx = 0; idx < CHANNELS; ++idx) {
        values[idx] = static_cast<data_t>(packed.range(((idx + 1) * 8) - 1, idx * 8));
    }
}

template <int CHANNELS, typename AxisPacket>
static inline void unpack_axis_channels(
    const AxisPacket &packet,
    data_t values[CHANNELS]
) {
    unpack_packed_channels<CHANNELS>(packet.data, values);
}

template <int CHANNELS>
static inline ap_uint<CHANNELS * 8> pack_channels(
    const data_t values[CHANNELS]
) {
    ap_uint<CHANNELS * 8> packed = 0;
    for (int idx = 0; idx < CHANNELS; ++idx) {
        packed.range(((idx + 1) * 8) - 1, idx * 8) = static_cast<ap_uint<8>>(values[idx]);
    }
    return packed;
}

template <typename AxisPacket, int CHANNELS>
static inline AxisPacket make_axis_packet(
    const data_t values[CHANNELS],
    bool last
) {
    AxisPacket packet;
    packet.data = pack_channels<CHANNELS>(values);
    packet.keep = -1;
    packet.strb = -1;
    packet.last = last ? 1 : 0;
    return packet;
}

#endif  // HLS_LUT_COMMON_DEFS_H_
