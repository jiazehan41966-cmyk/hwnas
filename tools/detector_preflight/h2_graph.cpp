// Functional serial-DDR graph executor. Not the searched H3 operator mapping.
#include <stdint.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include "numeric_v1.h"
typedef ap_axiu<256,0,0,0> axis256;

int h2_graph(hls::stream<axis256>& input_stream,hls::stream<axis256>& output_stream,
             int8_t* arena,const int8_t* weights,const int32_t* biases,const int* descriptor) {
#pragma HLS INTERFACE axis port=input_stream
#pragma HLS INTERFACE axis port=output_stream
#ifdef H2_SMALL_COSIM
// Small RTL witnesses touch bytes 0..4103 only. Depth sizes the co-sim memory
// model; all production interfaces remain 64-bit AXI addresses.
#pragma HLS INTERFACE m_axi port=arena offset=slave bundle=parameters depth=8192
#else
#pragma HLS INTERFACE m_axi port=arena offset=slave bundle=parameters depth=16777216
#endif
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=parameters depth=153600
#pragma HLS INTERFACE m_axi port=biases offset=slave bundle=parameters depth=960
#pragma HLS INTERFACE m_axi port=descriptor offset=slave bundle=parameters depth=32
#pragma HLS INTERFACE s_axilite port=arena bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=biases bundle=control
#pragma HLS INTERFACE s_axilite port=descriptor bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    int d[32];
#pragma HLS ARRAY_PARTITION variable=d complete
    for(int i=0;i<32;i++)d[i]=descriptor[i];
    if(d[26]!=0x48324731)return 1;
    int op=d[0],ci=d[1],co=d[2],h=d[3],w=d[4],oh=d[5],ow=d[6],k=d[7],s=d[8],pad=d[9],groups=d[10];
    if(op<0 || op>4 || ci<1 || ci>960 || co<1 || co>960 || h<1 || h>640 || w<1 || w>640 || oh<1 || oh>640 || ow<1 || ow>640)return 2;
    if(d[22]<=0 || d[22]>16777216)return 3;
    for(int i=0;i<3;i++)if(d[11+i]<0 || d[23+i]<0 || d[11+i]>d[22]-d[23+i])return 3;
    if(d[23]!=ci*h*w || d[25]!=co*oh*ow)return 3;
    int a=d[11],b=d[12],o=d[13];
    if(a<o+d[25] && o<a+d[23])return 4;
    if(op==3) {
        if(d[23]!=d[25] || d[23]%32)return 5;
        for(int i=0;i<d[23];i+=32) {
            axis256 t=input_stream.read();
            for(int j=0;j<32;j++)arena[a+i+j]=int8_t(t.data.range(j*8+7,j*8).to_uint());
            output_stream.write(t);
        }
        return 0;
    }
    // Op 4 is an explicit consumer-scale bridge. Do not fuse it into conv:
    // its rounding and saturation are part of the saved PTQ graph semantics.
    if(op==4) {
        if(ci!=co || h!=oh || w!=ow || d[24]!=0)return 9;
        if(d[16]<1 || d[17]<0 || d[17]>31 || d[18]!=0)return 8;
        for(int i=0;i<d[25];i++) {
// timing02: allow serial scheduling of canonical multiply/round/saturate,
// as for Add. The arithmetic and external clock are unchanged.
#pragma HLS PIPELINE off
            arena[o+i]=int8_t(h2::requant(ap_int<32>(arena[a+i]),d[16],d[17]).to_int());
        }
        return 0;
    }
    if(op==0) {
        if((k!=1&&k!=3&&k!=5) || (s!=1&&s!=2) || pad<0 || pad>2)return 6;
        // Only stride 1/2 is supported. Avoid a variable signed divider in
        // descriptor validation; the original generated RTL rejects valid shapes.
        int nh=h+2*pad-k,nw=w+2*pad-k;
        if(nh<0 || nw<0)return 6;
        int expected_h=(s==2?(nh>>1):nh)+1;
        int expected_w=(s==2?(nw>>1):nw)+1;
        if(oh!=expected_h || ow!=expected_w)return 6;
        if(groups!=1 && !(groups==ci && co==ci))return 6;
        int channels=(groups==1?ci:1);
        int count=co*channels*k*k;
        if(count!=d[14] || count>153600 || d[15]!=co)return 7;
        if(d[16]<1 || d[17]<0 || d[17]>31 || d[18]<0 || d[18]>2 || d[19]<0 || d[19]>127)return 8;
        int8_t local_w[153600];int32_t local_b[960];
        for(int i=0;i<count;i++)local_w[i]=weights[i];
        for(int i=0;i<co;i++)local_b[i]=biases[i];
        for(int y=0;y<oh;y++) {
#pragma HLS LOOP_FLATTEN off
          for(int x=0;x<ow;x++) {
#pragma HLS LOOP_FLATTEN off
           for(int oc=0;oc<co;oc++) {
#pragma HLS LOOP_FLATTEN off
            uint32_t acc=uint32_t(local_b[oc]);
            for(int ky=0;ky<k;ky++)for(int kx=0;kx<k;kx++) {
                int iy=y*s+ky-pad,ix=x*s+kx-pad;
                if(iy>=0 && iy<h && ix>=0 && ix<w)for(int ch=0;ch<channels;ch++) {
#pragma HLS PIPELINE II=1
                    int ic=(groups==1?ch:oc);
                    int32_t product=int32_t(arena[a+(iy*w+ix)*ci+ic])*int32_t(local_w[((oc*channels+ch)*k+ky)*k+kx]);
                    acc+=uint32_t(product); // exact modulo 2^32, no signed C++ overflow
                }
            }
            arena[o+(y*ow+x)*co+oc]=int8_t(h2::finish(ap_int<32>(acc),d+16).to_int());
           }
          }
        }
    } else {
        if(ci!=co || h!=oh || w!=ow || (op==2 && (h%2 || w%2)))return 9;
        int bh=op==2?h/2:h,bw=op==2?w/2:w;
        if(d[24]!=co*bh*bw || (b<o+d[25] && o<b+d[24]))return 9;
        for(int i=16;i<22;i+=2)if(d[i]<1 || d[i+1]<0 || d[i+1]>31)return 8;
        for(int y=0;y<h;y++) {
#pragma HLS LOOP_FLATTEN off
          for(int x=0;x<w;x++) {
#pragma HLS LOOP_FLATTEN off
           for(int c=0;c<co;c++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
            int by=op==2?y/2:y,bx=op==2?x/2:x;
            arena[o+(y*w+x)*co+c]=int8_t(h2::add(ap_int<8>(arena[a+(y*w+x)*co+c]),ap_int<8>(arena[b+(by*bw+bx)*co+c]),d+16).to_int());
           }
          }
        }
    }
    return 0;
}
