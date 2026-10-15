// Experiment 3. Standard fixed-width C++ is shared by native and HLS C runs.
// All signed accumulator overflow is implemented via unsigned modulo arithmetic.
#include "config.h"
#include <cstdint>
#ifdef __SYNTHESIS__
#include <ap_utils.h>
#else
static inline void ap_wait() {}
#endif

using input_t = INPUT_TYPE;
using data_t = OUTPUT_TYPE;
using weight_t = WEIGHT_TYPE;

static int32_t wrap32(uint32_t v) {
    return v <= 0x7fffffffU ? int32_t(v) : int32_t(int64_t(v) - 4294967296LL);
}

static data_t requant(int32_t a, int m, int s) {
    int64_t p = int64_t(a) * int64_t(m);
    bool negative = p < 0;
    uint64_t mag = negative ? uint64_t(-p) : uint64_t(p);
    uint64_t q = mag >> s;
    if (s) {
        uint64_t r = mag & ((uint64_t(1) << s) - 1);
        uint64_t half = uint64_t(1) << (s - 1);
        if (r > half || (r == half && (q & 1))) ++q;
    }
    int64_t value = negative ? -int64_t(q) : int64_t(q);
    const int64_t low = -(int64_t(1) << (DATA_BITS-1));
    const int64_t high = (int64_t(1) << (DATA_BITS-1))-1;
    return data_t(value < low ? low : value > high ? high : value);
}

static data_t finish(int32_t acc, const int32_t p[8]) {
    data_t q = requant(acc,p[0],p[1]);
    if (p[6] && q < 0) q=0;
    if (p[6]==2 && q>p[7]) q=data_t(p[7]);
    return q;
}

extern "C" void pilot_kernel(const input_t* a, const input_t* b,
                              const weight_t* weight, const int32_t* bias,
                              data_t* output, const int32_t params[8], volatile uint8_t& phase) {
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=inputs depth=INPUT_ELEMENTS
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=inputs depth=INPUT_ELEMENTS
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=parameters depth=WEIGHT_ELEMENTS
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=parameters depth=CO
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=parameters depth=8
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=outputs depth=OUTPUT_ELEMENTS
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE ap_vld port=phase
    // Phases are observed by an external clocked counter in the integrated shell.
    // Native C timing is never accepted as an FPGA latency measurement.
    phase_load: {
#pragma HLS protocol fixed
        phase=1;
        ap_wait();
    }
    input_t input_local[CI][H*W];
    data_t output_local[CO][OH*OW];
#pragma HLS BIND_STORAGE variable=input_local type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=output_local type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=input_local cyclic factor=INPUT_BANKS dim=1
#pragma HLS ARRAY_PARTITION variable=output_local cyclic factor=PE dim=1
    for(int c=0;c<CI;++c) for(int i=0;i<H*W;++i) {
#pragma HLS PIPELINE II=1
        input_local[c][i]=a[c*H*W+i];
    }
#if OP == 0 || OP == 1
    weight_t weights_local[CO][WEIGHTS_PER_OUT];
    int32_t bias_local[CO];
#pragma HLS ARRAY_PARTITION variable=weights_local cyclic factor=PE dim=1
#pragma HLS ARRAY_PARTITION variable=bias_local cyclic factor=PE
    for(int c=0;c<CO;++c) {
        bias_local[c]=bias[c];
        for(int i=0;i<WEIGHTS_PER_OUT;++i) {
#pragma HLS PIPELINE II=1
            weights_local[c][i]=weight[c*WEIGHTS_PER_OUT+i];
        }
    }
#endif
#if OP == 3 || OP == 4
    input_t second_local[CI][H*W];
#pragma HLS BIND_STORAGE variable=second_local type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=second_local cyclic factor=INPUT_BANKS dim=1
    for(int c=0;c<CI;++c) for(int i=0;i<H*W;++i) {
#pragma HLS PIPELINE II=1
        second_local[c][i]=b[c*H*W+i];
    }
#endif
    int32_t p[8];
#pragma HLS ARRAY_PARTITION variable=p complete
    for (int i=0;i<8;++i) p[i]=params[i];
    phase_compute: {
#pragma HLS protocol fixed
        phase=2;
        ap_wait();
    }
#if OP == 0 || OP == 1
    for (int cb=0;cb<CO;cb+=PE) {
        for (int y=0;y<OH;++y) for (int x=0;x<OW;++x) {
            uint32_t accum[PE];
#pragma HLS ARRAY_PARTITION variable=accum complete
            for (int lane=0;lane<PE;++lane) {
#pragma HLS UNROLL
                accum[lane] = cb+lane<CO ? uint32_t(bias_local[cb+lane]) : 0;
            }
#if OP == 0
            for (int ib=0;ib<CI;ib+=SIMD) {
                for (int lane=0;lane<PE;++lane) {
#pragma HLS UNROLL
                    for (int si=0;si<SIMD;++si) {
#pragma HLS UNROLL
                        int oc=cb+lane, ic=ib+si;
                        if (oc<CO && ic<CI) {
                            int64_t product=int64_t(input_local[ic][y*W+x])*int64_t(weights_local[oc][ic]);
                            accum[lane] += uint32_t(product);
                        }
                    }
                }
            }
#else
            for (int ky=0;ky<K;++ky) for (int kx=0;kx<K;++kx) {
                for (int lane=0;lane<PE;++lane) {
#pragma HLS UNROLL
                    int oc=cb+lane, iy=y*STRIDE+ky-PAD, ix=x*STRIDE+kx-PAD;
                    if (oc<CO && iy>=0 && iy<H && ix>=0 && ix<W) {
                        int64_t product=int64_t(input_local[oc][iy*W+ix])*int64_t(weights_local[oc][ky*K+kx]);
                        accum[lane] += uint32_t(product);
                    }
                }
            }
#endif
            for (int lane=0;lane<PE;++lane) {
#pragma HLS UNROLL
                if (cb+lane<CO) output_local[cb+lane][y*OW+x]=finish(wrap32(accum[lane]),p);
            }
        }
    }
#else
    for (int base=0;base<OUTPUT_ELEMENTS;base+=PE) {
        for (int lane=0;lane<PE;++lane) {
#pragma HLS UNROLL
            int i=base+lane;
            if (i>=OUTPUT_ELEMENTS) continue;
#if OP == 2
            int c=i/(OH*OW), y=(i/OW)%OH, x=i%OW;
            output_local[c][y*OW+x]=input_local[c][(y/2)*W+x/2];
#elif OP == 3
            data_t qa=requant(int32_t(input_local[i/(H*W)][i%(H*W)]),p[0],p[1]);
            data_t qb=requant(int32_t(second_local[i/(H*W)][i%(H*W)]),p[2],p[3]);
            output_local[i/(OH*OW)][i%(OH*OW)]=requant(int32_t(qa)+int32_t(qb),p[4],p[5]);
#elif OP == 4
            int ci=(i%INPUT_ELEMENTS)/(H*W), spatial=i%(H*W);
            output_local[i/(OH*OW)][i%(OH*OW)]=i<INPUT_ELEMENTS ? requant(int32_t(input_local[ci][spatial]),p[0],p[1])
                : requant(int32_t(second_local[ci][spatial]),p[2],p[3]);
#elif OP == 5
            output_local[i/(OH*OW)][i%(OH*OW)]=requant(int32_t(input_local[i/(H*W)][i%(H*W)]),p[0],p[1]);
#endif
        }
    }
#endif
    phase_store: {
#pragma HLS protocol fixed
        phase=3;
        ap_wait();
    }
    for(int c=0;c<CO;++c) for(int i=0;i<OH*OW;++i) {
#pragma HLS PIPELINE II=1
        output[c*OH*OW+i]=output_local[c][i];
    }
    phase_idle: {
#pragma HLS protocol fixed
        phase=0;
        ap_wait();
    }
}
