#ifndef HLS_LUT_BUILDER_COMMON_TYPES_H_
#define HLS_LUT_BUILDER_COMMON_TYPES_H_

// Deprecated as a direct include for operator kernels.
// New kernels should include common_defs.h so interface and packed-stream
// helpers stay aligned. common_defs.h still includes this file for the
// low-level scalar types and legacy helper primitives that remain shared.

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

typedef ap_int<8> data_t;
typedef ap_int<32> acc_t;
typedef ap_axis<8, 0, 0, 0> axis_t;

static inline data_t saturate_to_data(acc_t value) {
    if (value > 127) {
        return static_cast<data_t>(127);
    }
    if (value < -128) {
        return static_cast<data_t>(-128);
    }
    return static_cast<data_t>(value);
}

static inline data_t relu6_quant(acc_t value) {
    if (value < 0) {
        return static_cast<data_t>(0);
    }
    if (value > 6) {
        return static_cast<data_t>(6);
    }
    return static_cast<data_t>(value);
}

static inline data_t stream_read_data(hls::stream<axis_t> &input_stream) {
    axis_t packet = input_stream.read();
    return packet.data;
}

static inline void stream_write_data(
    hls::stream<axis_t> &output_stream,
    data_t value,
    bool last
) {
    axis_t packet;
    packet.data = value;
    packet.keep = -1;
    packet.strb = -1;
    packet.last = last ? 1 : 0;
    output_stream.write(packet);
}

template <int SIZE>
static inline void read_stream_tensor(
    hls::stream<axis_t> &input_stream,
    data_t buffer[SIZE]
) {
    for (int idx = 0; idx < SIZE; ++idx) {
#pragma HLS PIPELINE II=1
        buffer[idx] = stream_read_data(input_stream);
    }
}

template <int SIZE>
static inline void write_stream_tensor(
    hls::stream<axis_t> &output_stream,
    const data_t buffer[SIZE]
) {
    for (int idx = 0; idx < SIZE; ++idx) {
#pragma HLS PIPELINE II=1
        stream_write_data(output_stream, buffer[idx], idx == (SIZE - 1));
    }
}

#endif  // HLS_LUT_BUILDER_COMMON_TYPES_H_
