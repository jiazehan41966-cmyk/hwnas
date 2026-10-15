// T6 v1: bounded spatial tiles; full activation frames reside in DDR.
// Modulo INT32 accumulation, INT64 requantization, ties-to-even, saturation.
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
    return v <= 0x7fffffffU ? int32_t(v) : int32_t(int64_t(v)-4294967296LL);
}
static data_t requant(int32_t a, int32_t m, int32_t s) {
    int64_t p=int64_t(a)*int64_t(m);
    bool neg=p<0;
    uint64_t mag=neg?uint64_t(-p):uint64_t(p), q=mag>>s;
    if(s) {
        uint64_t r=mag&((uint64_t(1)<<s)-1), half=uint64_t(1)<<(s-1);
        if(r>half || (r==half && (q&1))) ++q;
    }
    int64_t v=neg?-int64_t(q):int64_t(q);
    const int64_t lo=-(int64_t(1)<<(DATA_BITS-1)), hi=(int64_t(1)<<(DATA_BITS-1))-1;
    return data_t(v<lo?lo:v>hi?hi:v);
}
static data_t finish(int32_t acc,const int32_t p[8]) {
    data_t q=requant(acc,p[0],p[1]);
    if(p[6] && q<0) q=0;
    if(p[6]==2 && q>p[7]) q=data_t(p[7]);
    return q;
}

extern "C" void pilot_kernel(const input_t* a,const input_t* b,
                              const weight_t* weight,const int32_t* bias,
                              data_t* output,const int32_t params[8],volatile uint8_t& phase) {
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
    phase_parameters: {
#pragma HLS protocol fixed
        phase=1;
        ap_wait();
    }
    int32_t p[8];
#pragma HLS ARRAY_PARTITION variable=p complete
    for(int i=0;i<8;++i) p[i]=params[i];
#if OP == 0 || OP == 1 || OP == 6
    weight_t weights_local[CO][WEIGHTS_PER_OUT];
    int32_t bias_local[CO];
#pragma HLS BIND_STORAGE variable=weights_local type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=weights_local cyclic factor=PE dim=1
#pragma HLS ARRAY_PARTITION variable=weights_local cyclic factor=SIMD dim=2
#pragma HLS ARRAY_PARTITION variable=bias_local cyclic factor=PE
    for(int oc=0;oc<CO;++oc) {
        bias_local[oc]=bias[oc];
        for(int i=0;i<WEIGHTS_PER_OUT;++i) {
#pragma HLS PIPELINE II=1
            weights_local[oc][i]=weight[oc*WEIGHTS_PER_OUT+i];
        }
    }
#endif
#if OP == 0 || OP == 6
    // All input channels for ONE spatial patch; never CI*H*W.
    input_t input_tile[CI][PATCH_H*PATCH_W];
    data_t output_tile[CO][TILE_H*TILE_W];
#pragma HLS BIND_STORAGE variable=input_tile type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=output_tile type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=input_tile cyclic factor=SIMD dim=1
#pragma HLS ARRAY_PARTITION variable=output_tile cyclic factor=PE dim=1
    for(int yb=0;yb<OH;yb+=TILE_H) for(int xb=0;xb<OW;xb+=TILE_W) {
        phase_load: {
#pragma HLS protocol fixed
            phase=1;
            ap_wait();
        }
        for(int ic=0;ic<CI;++ic) for(int iy=0;iy<PATCH_H;++iy) for(int ix=0;ix<PATCH_W;++ix) {
#pragma HLS PIPELINE II=1
            int gy=yb*STRIDE+iy-PAD, gx=xb*STRIDE+ix-PAD;
            input_tile[ic][iy*PATCH_W+ix]=(gy>=0 && gy<H && gx>=0 && gx<W)?a[(ic*H+gy)*W+gx]:input_t(0);
        }
        phase_compute: {
#pragma HLS protocol fixed
            phase=2;
            ap_wait();
        }
        for(int cb=0;cb<CO;cb+=PE) for(int yy=0;yy<TILE_H;++yy) for(int xx=0;xx<TILE_W;++xx) {
            uint32_t accum[PE];
#pragma HLS ARRAY_PARTITION variable=accum complete
            for(int lane=0;lane<PE;++lane) {
#pragma HLS UNROLL
                accum[lane]=cb+lane<CO?uint32_t(bias_local[cb+lane]):0;
            }
            for(int ky=0;ky<K;++ky) for(int kx=0;kx<K;++kx) for(int ib=0;ib<CI;ib+=SIMD) {
#pragma HLS PIPELINE II=1
                for(int lane=0;lane<PE;++lane) {
#pragma HLS UNROLL
                    for(int si=0;si<SIMD;++si) {
#pragma HLS UNROLL
                        int oc=cb+lane, ic=ib+si;
                        if(oc<CO && ic<CI) accum[lane]+=uint32_t(int32_t(input_tile[ic][(yy*STRIDE+ky)*PATCH_W+xx*STRIDE+kx])*int32_t(weights_local[oc][(ic*K+ky)*K+kx]));
                    }
                }
            }
            for(int lane=0;lane<PE;++lane) {
#pragma HLS UNROLL
                if(cb+lane<CO) output_tile[cb+lane][yy*TILE_W+xx]=finish(wrap32(accum[lane]),p);
            }
        }
        phase_store: {
#pragma HLS protocol fixed
            phase=3;
            ap_wait();
        }
        for(int oc=0;oc<CO;++oc) for(int yy=0;yy<TILE_H;++yy) for(int xx=0;xx<TILE_W;++xx) {
#pragma HLS PIPELINE II=1
            if(yb+yy<OH && xb+xx<OW) output[(oc*OH+yb+yy)*OW+xb+xx]=output_tile[oc][yy*TILE_W+xx];
        }
    }
#else
    // Depthwise/pool/resize/elementwise process PE channels at a time.
    input_t input_tile[PE][PATCH_H*PATCH_W];
    data_t output_tile[PE][TILE_H*TILE_W];
#pragma HLS BIND_STORAGE variable=input_tile type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=output_tile type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=1
#if OP == 3
    input_t second_tile[PE][TILE_H*TILE_W];
#pragma HLS BIND_STORAGE variable=second_tile type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=second_tile complete dim=1
#endif
    for(int cb=0;cb<CO;cb+=PE) for(int yb=0;yb<OH;yb+=TILE_H) for(int xb=0;xb<OW;xb+=TILE_W) {
        phase_load: {
#pragma HLS protocol fixed
            phase=1;
            ap_wait();
        }
        for(int lane=0;lane<PE;++lane) for(int iy=0;iy<PATCH_H;++iy) for(int ix=0;ix<PATCH_W;++ix) {
#pragma HLS PIPELINE II=1
#if OP == 2
            int gy=yb/2+iy, gx=xb/2+ix;
#else
            int gy=yb*STRIDE+iy-PAD, gx=xb*STRIDE+ix-PAD;
#endif
            bool inside=cb+lane<CI && gy>=0 && gy<H && gx>=0 && gx<W;
#if OP == 7
            input_tile[lane][iy*PATCH_W+ix]=inside?a[((cb+lane)*H+gy)*W+gx]:input_t(-(1<<(INPUT_BITS-1)));
#else
            input_tile[lane][iy*PATCH_W+ix]=inside?a[((cb+lane)*H+gy)*W+gx]:input_t(0);
#endif
#if OP == 3
            second_tile[lane][iy*TILE_W+ix]=inside?b[((cb+lane)*H+gy)*W+gx]:input_t(0);
#endif
        }
        phase_compute: {
#pragma HLS protocol fixed
            phase=2;
            ap_wait();
        }
        for(int yy=0;yy<TILE_H;++yy) for(int xx=0;xx<TILE_W;++xx) {
            for(int lane=0;lane<PE;++lane) {
#pragma HLS UNROLL
#if OP == 1
                uint32_t acc=cb+lane<CO?uint32_t(bias_local[cb+lane]):0;
                for(int ky=0;ky<K;++ky) for(int kx=0;kx<K;++kx) {
                    if(cb+lane<CO) acc+=uint32_t(int32_t(input_tile[lane][(yy*STRIDE+ky)*PATCH_W+xx*STRIDE+kx])*int32_t(weights_local[cb+lane][ky*K+kx]));
                }
                output_tile[lane][yy*TILE_W+xx]=finish(wrap32(acc),p);
#elif OP == 7
                input_t best=input_t(-(1<<(INPUT_BITS-1)));
                for(int ky=0;ky<K;++ky) for(int kx=0;kx<K;++kx) {
                    input_t v=input_tile[lane][(yy*STRIDE+ky)*PATCH_W+xx*STRIDE+kx];
                    if(v>best) best=v;
                }
                output_tile[lane][yy*TILE_W+xx]=data_t(best);
#elif OP == 2
                output_tile[lane][yy*TILE_W+xx]=data_t(input_tile[lane][(yy/2)*PATCH_W+xx/2]);
#elif OP == 3
                data_t qa=requant(int32_t(input_tile[lane][yy*TILE_W+xx]),p[0],p[1]);
                data_t qb=requant(int32_t(second_tile[lane][yy*TILE_W+xx]),p[2],p[3]);
                output_tile[lane][yy*TILE_W+xx]=requant(int32_t(qa)+int32_t(qb),p[4],p[5]);
#elif OP == 5
                output_tile[lane][yy*TILE_W+xx]=requant(int32_t(input_tile[lane][yy*TILE_W+xx]),p[0],p[1]);
#else
#error Unsupported operator
#endif
            }
        }
        phase_store: {
#pragma HLS protocol fixed
            phase=3;
            ap_wait();
        }
        for(int lane=0;lane<PE;++lane) for(int yy=0;yy<TILE_H;++yy) for(int xx=0;xx<TILE_W;++xx) {
#pragma HLS PIPELINE II=1
            if(cb+lane<CO && yb+yy<OH && xb+xx<OW) output[((cb+lane)*OH+yb+yy)*OW+xb+xx]=output_tile[lane][yy*TILE_W+xx];
        }
    }
#endif
    phase_idle: {
#pragma HLS protocol fixed
        phase=0;
        ap_wait();
    }
}
