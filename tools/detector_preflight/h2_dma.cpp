// H2 V1 bring-up core: PW32->96 at 80x80 and FPN96 80->160.
// New execution mapping, not an H3 cost-table row or a complete detector.
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include "numeric_v1.h"
typedef ap_axiu<256,0,0,0> axis256;

int h2_dma(hls::stream<axis256>& input_stream,
           hls::stream<axis256>& output_stream,
           const ap_int<8>* weights, const ap_int<32>* biases,
           const int* numeric, const ap_int<8>* low, int mode) {
#pragma HLS INTERFACE axis port=input_stream
#pragma HLS INTERFACE axis port=output_stream
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=parameters depth=3072
#pragma HLS INTERFACE m_axi port=biases offset=slave bundle=parameters depth=96
#pragma HLS INTERFACE m_axi port=numeric offset=slave bundle=parameters depth=6
#pragma HLS INTERFACE m_axi port=low offset=slave bundle=parameters depth=614400
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=biases bundle=control
#pragma HLS INTERFACE s_axilite port=numeric bundle=control
#pragma HLS INTERFACE s_axilite port=low bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    if(mode!=0 && mode!=1) return 1;
    int p[6];
    for(int j=0;j<(mode==0?5:6);j++) p[j]=numeric[j];
    if(mode==0) {
        if(p[0]<1 || p[1]<0 || p[1]>31 || p[2]<0 || p[2]>2 || p[3]<0 || p[3]>127) return 2;
    } else {
        for(int j=0;j<6;j+=2) if(p[j]<1 || p[j+1]<0 || p[j+1]>31) return 2;
    }
    int errors=0;
    if(mode==0) {
        ap_int<8> w[96][32]; ap_int<32> b[96];
        for(int co=0;co<96;co++) {
            b[co]=biases[co];
            for(int ci=0;ci<32;ci++) w[co][ci]=weights[co*32+ci];
        }
        for(int pixel=0;pixel<6400;pixel++) {
            axis256 in=input_stream.read();
            if(in.keep!=ap_uint<32>(-1) || in.strb!=ap_uint<32>(-1) || bool(in.last)!=(pixel==6399)) errors|=4;
            for(int group=0;group<3;group++) {
                axis256 out; out.data=0; out.keep=-1; out.strb=-1;
                out.last=(pixel==6399 && group==2);
                for(int lane=0;lane<32;lane++) {
                    int co=group*32+lane;
                    ap_int<32> acc=b[co];
                    for(int ci=0;ci<32;ci++) {
#pragma HLS PIPELINE II=1
                        ap_int<8> x=in.data.range(ci*8+7,ci*8);
                        acc+=ap_int<32>(x)*ap_int<32>(w[co][ci]);
                    }
                    ap_int<8> y=h2::finish(acc,p);
                    out.data.range(lane*8+7,lane*8)=ap_uint<8>(y);
                }
                output_stream.write(out);
            }
        }
    } else {
        for(int row=0;row<160;row++) for(int col=0;col<160;col++) for(int group=0;group<3;group++) {
            axis256 in=input_stream.read(),out;
            bool tail=(row==159 && col==159 && group==2);
            if(in.keep!=ap_uint<32>(-1) || in.strb!=ap_uint<32>(-1) || bool(in.last)!=tail) errors|=4;
            out.data=0;out.keep=-1;out.strb=-1;out.last=tail;
            for(int lane=0;lane<32;lane++) {
#pragma HLS PIPELINE II=1
                int channel=group*32+lane;
                ap_int<8> a=in.data.range(lane*8+7,lane*8);
                ap_int<8> b=low[((row/2)*80+col/2)*96+channel];
                out.data.range(lane*8+7,lane*8)=ap_uint<8>(h2::add(a,b,p));
            }
            output_stream.write(out);
        }
    }
    return errors;
}
