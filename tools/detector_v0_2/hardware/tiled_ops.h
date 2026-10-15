#ifndef HWNAS_V02_TILED_OPS_H
#define HWNAS_V02_TILED_OPS_H

#include <ap_int.h>
#include <stdint.h>
#include "numeric_v1.h"

// Standalone operator interface: NHWC, channels padded to a multiple of 16.
// Each ap_uint<128> word contains 16 signed INT8 lanes in little-endian order.
// This padded interface is a new V0.2 layout; it is not the old compact arena.
typedef ap_uint<128> v02_word;
static constexpr int V02_PE = 16;
static constexpr int V02_SIMD = 8;
static constexpr int V02_DW_LANES = 32;
static constexpr int V02_TILE = 16;
static constexpr int V02_PIXELS = V02_TILE * V02_TILE;

inline ap_int<8> v02_unpack(v02_word value, int lane) {
    return ap_int<8>(value.range(lane * 8 + 7, lane * 8));
}

// Eight explicitly concurrent signed multiplies and a balanced modulo-32 tree.
// Unsigned addition exactly preserves NUMERIC_CONTRACT_V1 overflow semantics.
inline ap_uint<32> v02_dot8(const ap_int<8> a[V02_SIMD],
                          const ap_int<8> b[V02_SIMD]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=a complete
#pragma HLS ARRAY_PARTITION variable=b complete
    ap_uint<32> products[V02_SIMD];
#pragma HLS ARRAY_PARTITION variable=products complete
    for (int s = 0; s < V02_SIMD; ++s) {
#pragma HLS UNROLL
        ap_int<16> product = a[s] * b[s];
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=2
        products[s] = ap_uint<32>(ap_int<32>(product));
    }
    ap_uint<32> p0 = products[0] + products[1];
    ap_uint<32> p1 = products[2] + products[3];
    ap_uint<32> p2 = products[4] + products[5];
    ap_uint<32> p3 = products[6] + products[7];
    return ap_uint<32>(ap_uint<32>(p0 + p1) + ap_uint<32>(p2 + p3));
}

// The input tile is loaded once and reused for every output-channel block.
// Partial sums remain on chip until the entire channel/kernel reduction ends.
// Source word counts below are diagnostic accesses, not measured AXI traffic.
template<int CIN, int COUT, int K, int STRIDE, int PAD>
int v02_dense(const v02_word* input, const v02_word* weights,
              const int32_t* biases, v02_word* output, uint64_t* stats,
              int height, int width, int multiplier, int shift,
              int activation, int q6) {
#pragma HLS INLINE off
    static constexpr int IB = (CIN + V02_SIMD - 1) / V02_SIMD;
    static constexpr int OB = (COUT + V02_PE - 1) / V02_PE;
    static constexpr int IW = (CIN + 15) / 16;
    static constexpr int OW = (COUT + 15) / 16;
    static constexpr int INPUT_SPAN = (V02_TILE - 1) * STRIDE + K;
    static constexpr int INPUT_PIXELS = INPUT_SPAN * INPUT_SPAN;
    if (height < 1 || width < 1 || height > 640 || width > 640 ||
        multiplier < 1 || shift < 0 || shift > 31 ||
        activation < 0 || activation > 2 || q6 < 0 || q6 > 127) return 1;
    const int oh = (height + 2 * PAD - K) / STRIDE + 1;
    const int ow = (width + 2 * PAD - K) / STRIDE + 1;
    if (oh < 1 || ow < 1) return 2;
    ap_int<8> local_w[K * K][OB][IB][V02_PE][V02_SIMD];
    ap_int<8> input_tile[INPUT_PIXELS][IB][V02_SIMD];
    ap_uint<32> psum[V02_PIXELS][V02_PE];
    int32_t local_b[OB][V02_PE];
#pragma HLS ARRAY_PARTITION variable=local_w dim=4 complete
#pragma HLS ARRAY_PARTITION variable=local_w dim=5 complete
#pragma HLS BIND_STORAGE variable=local_w type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=input_tile dim=3 complete
#pragma HLS BIND_STORAGE variable=input_tile type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=psum dim=2 complete
#pragma HLS BIND_STORAGE variable=psum type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=local_b dim=2 complete
    // Weight layout is [oc][ky][kx][ceil(CIN/16)] packed, OIHW values.
    uint64_t weight_words = 0;
    for (int ob = 0; ob < OB; ++ob) {
        for (int p = 0; p < V02_PE; ++p) {
            const int oc = ob * V02_PE + p;
            local_b[ob][p] = oc < COUT ? biases[oc] : 0;
            for (int tap = 0; tap < K * K; ++tap) {
                for (int iw = 0; iw < IW; ++iw) {
#pragma HLS PIPELINE II=1
                    v02_word packed = 0;
                    if (oc < COUT) {
                        packed = weights[(oc * K * K + tap) * IW + iw];
                        ++weight_words;
                    }
                    for (int lane = 0; lane < 16; ++lane) {
#pragma HLS UNROLL
                        const int ic = iw * 16 + lane;
                        if (ic < IB * V02_SIMD) {
                            local_w[tap][ob][ic / V02_SIMD][p][ic % V02_SIMD] =
                                (oc < COUT && ic < CIN) ? v02_unpack(packed, lane) : ap_int<8>(0);
                        }
                    }
                }
            }
        }
    }
    uint64_t input_words = 0, output_words = 0, tiles = 0;
    for (int ty = 0; ty < oh; ty += V02_TILE) {
        for (int tx = 0; tx < ow; tx += V02_TILE) {
            ++tiles;
            for (int iy = 0; iy < INPUT_SPAN; ++iy) {
                for (int ix = 0; ix < INPUT_SPAN; ++ix) {
                    const int sy = ty * STRIDE + iy - PAD;
                    const int sx = tx * STRIDE + ix - PAD;
                    for (int iw = 0; iw < IW; ++iw) {
#pragma HLS PIPELINE II=1
                        v02_word packed = 0;
                        if (sy >= 0 && sy < height && sx >= 0 && sx < width) {
                            packed = input[(sy * width + sx) * IW + iw];
                            ++input_words;
                        }
                        for (int lane = 0; lane < 16; ++lane) {
#pragma HLS UNROLL
                            const int ic = iw * 16 + lane;
                            if (ic < IB * V02_SIMD)
                                input_tile[iy * INPUT_SPAN + ix][ic / V02_SIMD][ic % V02_SIMD] =
                                    ic < CIN ? v02_unpack(packed, lane) : ap_int<8>(0);
                        }
                    }
                }
            }
            for (int ob = 0; ob < OB; ++ob) {
                for (int px = 0; px < V02_PIXELS; ++px) {
#pragma HLS PIPELINE II=1
                    for (int p = 0; p < V02_PE; ++p) {
#pragma HLS UNROLL
                        psum[px][p] = ap_uint<32>(local_b[ob][p]);
                    }
                }
                // Interleave all 256 pixels before the next reduction term.
                // Re-visiting the same BRAM partial sum is now 256 iterations
                // apart, rather than immediately after a pipelined multiplier.
                // Actual achieved II still has to be read from synthesis.
                DENSE_MAC_STEPS:
                for (int step = 0; step < V02_PIXELS * K * K * IB; ++step) {
#pragma HLS PIPELINE II=1
                    const int px = step % V02_PIXELS;
                    const int ib = (step / V02_PIXELS) % IB;
                    const int tap = step / (V02_PIXELS * IB);
                    const int row = (px / V02_TILE) * STRIDE + tap / K;
                    const int col = (px % V02_TILE) * STRIDE + tap % K;
                    ap_int<8> sample[V02_SIMD];
#pragma HLS ARRAY_PARTITION variable=sample complete
                    for (int s = 0; s < V02_SIMD; ++s) {
#pragma HLS UNROLL
                        sample[s] = input_tile[row * INPUT_SPAN + col][ib][s];
                    }
                    PE_LANES:
                    for (int p = 0; p < V02_PE; ++p) {
#pragma HLS UNROLL
                        ap_int<8> kernel[V02_SIMD];
#pragma HLS ARRAY_PARTITION variable=kernel complete
                        for (int s = 0; s < V02_SIMD; ++s) {
#pragma HLS UNROLL
                            kernel[s] = local_w[tap][ob][ib][p][s];
                        }
                        psum[px][p] += v02_dot8(sample, kernel);
                    }
                }
                for (int px = 0; px < V02_PIXELS; ++px) {
#pragma HLS PIPELINE II=1
                    const int y = ty + px / V02_TILE;
                    const int x = tx + px % V02_TILE;
                    v02_word packed = 0;
                    const int params[5] = {multiplier, shift, activation, q6, 0};
#pragma HLS ARRAY_PARTITION variable=params complete
                    for (int p = 0; p < V02_PE; ++p) {
#pragma HLS UNROLL
                        const int oc = ob * V02_PE + p;
                        ap_int<8> q = oc < COUT ? h2::finish(ap_int<32>(psum[px][p]), params) : ap_int<8>(0);
                        packed.range(p * 8 + 7, p * 8) = ap_uint<8>(q);
                    }
                    if (y < oh && x < ow) {
                        output[(y * ow + x) * OW + ob] = packed;
                        ++output_words;
                    }
                }
            }
        }
    }
    stats[0] = uint64_t(oh) * ow * CIN * COUT * K * K;
    stats[1] = tiles * V02_PIXELS * OB * IB * K * K;
    stats[2] = V02_PE * V02_SIMD;
    stats[3] = input_words;
    stats[4] = weight_words;
    stats[5] = output_words;
    stats[6] = V02_PE;
    stats[7] = V02_SIMD;
    stats[8] = tiles;
    stats[9] = input_words * 16;
    stats[10] = weight_words * 16;
    stats[11] = output_words * 16;
    return 0;
}

template<int CHANNELS, int STRIDE>
int v02_depthwise(const v02_word* input, const v02_word* weights,
                  const int32_t* biases, v02_word* output, uint64_t* stats,
                  int height, int width, int multiplier, int shift,
                  int activation, int q6) {
#pragma HLS INLINE off
    static constexpr int CB = (CHANNELS + V02_DW_LANES - 1) / V02_DW_LANES;
    static constexpr int WORDS = (CHANNELS + 15) / 16;
    static constexpr int SPAN = (V02_TILE - 1) * STRIDE + 3;
    if (height < 1 || width < 1 || height > 640 || width > 640 ||
        multiplier < 1 || shift < 0 || shift > 31 ||
        activation < 0 || activation > 2 || q6 < 0 || q6 > 127) return 1;
    const int oh = (height + STRIDE - 1) / STRIDE;
    const int ow = (width + STRIDE - 1) / STRIDE;
    ap_int<8> local_w[CB][9][V02_DW_LANES];
    int32_t local_b[CB][V02_DW_LANES];
    ap_int<8> input_tile[SPAN * SPAN][CB][V02_DW_LANES];
    ap_uint<32> psum[V02_PIXELS][V02_DW_LANES];
#pragma HLS ARRAY_PARTITION variable=local_w dim=3 complete
#pragma HLS ARRAY_PARTITION variable=local_b dim=2 complete
#pragma HLS ARRAY_PARTITION variable=input_tile dim=3 complete
#pragma HLS BIND_STORAGE variable=input_tile type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=psum dim=2 complete
#pragma HLS BIND_STORAGE variable=psum type=ram_2p impl=bram
    for (int cb = 0; cb < CB; ++cb) {
        for (int lane = 0; lane < V02_DW_LANES; ++lane) {
#pragma HLS PIPELINE II=1
            const int c = cb * V02_DW_LANES + lane;
            v02_word packed = c < CHANNELS ? weights[c] : v02_word(0);
            local_b[cb][lane] = c < CHANNELS ? biases[c] : 0;
            for (int tap = 0; tap < 9; ++tap) {
#pragma HLS UNROLL
                local_w[cb][tap][lane] = v02_unpack(packed, tap);
            }
        }
    }
    uint64_t input_words = 0, output_words = 0, tiles = 0;
    for (int ty = 0; ty < oh; ty += V02_TILE) {
        for (int tx = 0; tx < ow; tx += V02_TILE) {
            ++tiles;
            for (int iy = 0; iy < SPAN; ++iy) {
                for (int ix = 0; ix < SPAN; ++ix) {
                    const int sy = ty * STRIDE + iy - 1;
                    const int sx = tx * STRIDE + ix - 1;
                    for (int iw = 0; iw < WORDS; ++iw) {
#pragma HLS PIPELINE II=1
                        v02_word packed = 0;
                        if (sy >= 0 && sy < height && sx >= 0 && sx < width) {
                            packed = input[(sy * width + sx) * WORDS + iw];
                            ++input_words;
                        }
                        for (int lane = 0; lane < 16; ++lane) {
#pragma HLS UNROLL
                            const int c = iw * 16 + lane;
                            if (c < CHANNELS)
                                input_tile[iy * SPAN + ix][c / V02_DW_LANES][c % V02_DW_LANES] = v02_unpack(packed, lane);
                        }
                    }
                }
            }
            for (int cb = 0; cb < CB; ++cb) {
                for (int px = 0; px < V02_PIXELS; ++px) {
#pragma HLS PIPELINE II=1
                    for (int lane = 0; lane < V02_DW_LANES; ++lane) {
#pragma HLS UNROLL
                        psum[px][lane] = ap_uint<32>(local_b[cb][lane]);
                    }
                }
                DW_MAC_STEPS:
                for (int step = 0; step < V02_PIXELS * 9; ++step) {
#pragma HLS PIPELINE II=1
                    // Interleave pixels to avoid a loop-carried BRAM
                    // read/modify/write recurrence on every kernel tap.
                    const int px = step % V02_PIXELS;
                    const int tap = step / V02_PIXELS;
                    const int row = (px / V02_TILE) * STRIDE + tap / 3;
                    const int col = (px % V02_TILE) * STRIDE + tap % 3;
                    DW_CHANNEL_LANES:
                    for (int lane = 0; lane < V02_DW_LANES; ++lane) {
#pragma HLS UNROLL
                        const int c = cb * V02_DW_LANES + lane;
                        ap_int<8> sample = c < CHANNELS ? input_tile[row * SPAN + col][cb][lane] : ap_int<8>(0);
                        ap_int<16> product = sample * local_w[cb][tap][lane];
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=2
                        psum[px][lane] += ap_uint<32>(ap_int<32>(product));
                    }
                }
                for (int px = 0; px < V02_PIXELS; ++px) {
                    for (int packet = 0; packet < V02_DW_LANES / 16; ++packet) {
#pragma HLS PIPELINE II=1
                        const int y = ty + px / V02_TILE;
                        const int x = tx + px % V02_TILE;
                        const int output_packet = cb * (V02_DW_LANES / 16) + packet;
                        v02_word packed = 0;
                        const int params[5] = {multiplier, shift, activation, q6, 0};
#pragma HLS ARRAY_PARTITION variable=params complete
                        for (int lane = 0; lane < 16; ++lane) {
#pragma HLS UNROLL
                            const int local = packet * 16 + lane;
                            const int c = cb * V02_DW_LANES + local;
                            ap_int<8> q = c < CHANNELS ? h2::finish(ap_int<32>(psum[px][local]), params) : ap_int<8>(0);
                            packed.range(lane * 8 + 7, lane * 8) = ap_uint<8>(q);
                        }
                        if (y < oh && x < ow && output_packet < WORDS) {
                            output[(y * ow + x) * WORDS + output_packet] = packed;
                            ++output_words;
                        }
                    }
                }
            }
        }
    }
    stats[0] = uint64_t(oh) * ow * CHANNELS * 9;
    stats[1] = tiles * V02_PIXELS * CB * 9;
    stats[2] = V02_DW_LANES;
    stats[3] = input_words;
    stats[4] = CHANNELS;
    stats[5] = output_words;
    stats[6] = V02_DW_LANES;
    stats[7] = 1;
    stats[8] = tiles;
    stats[9] = input_words * 16;
    stats[10] = CHANNELS * 16;
    stats[11] = output_words * 16;
    return 0;
}

#endif
