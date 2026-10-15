#include <fstream>
#include <vector>
#include <iostream>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
typedef ap_axiu<256,0,0,0> axis256;
int h2_dma(hls::stream<axis256>&,hls::stream<axis256>&,const ap_int<8>*,const ap_int<32>*,const int*,const ap_int<8>*,int);
std::vector<char> read(const std::string& name) {
    std::ifstream f(name,std::ios::binary);
    if(!f)throw std::runtime_error(name);
    return std::vector<char>((std::istreambuf_iterator<char>(f)),{});
}
int main() {
    for(int mode=0;mode<2;mode++) {
        std::string dir=mode==0?"pw/":"fpn/";
        auto input=read(dir+"input.bin"),expected=read(dir+"reference.bin"),wr=read(dir+"weights.bin"),br=read(dir+"biases.bin"),nr=read(dir+"numeric.bin"),lr=read(dir+"low.bin");
        ap_int<8> weights[3072];ap_int<32> biases[96];int numeric[6];
        std::vector<ap_int<8>> low(614400);
        for(int i=0;i<3072;i++)weights[i]=(signed char)wr.at(i);
        for(int i=0;i<96;i++){unsigned v=0;for(int j=0;j<4;j++)v|=(unsigned char)br.at(i*4+j)<<(j*8);biases[i]=v;}
        for(int i=0;i<6;i++){unsigned v=0;for(int j=0;j<4;j++)v|=(unsigned char)nr.at(i*4+j)<<(j*8);numeric[i]=(int)v;}
        for(int i=0;i<614400;i++)low[i]=(signed char)lr.at(i);
        hls::stream<axis256> ins,outs;
        for(size_t off=0;off<input.size();off+=32){axis256 t;t.data=0;t.keep=-1;t.strb=-1;t.last=(off+32==input.size());for(int j=0;j<32;j++)t.data.range(j*8+7,j*8)=(unsigned char)input[off+j];ins.write(t);}
        int status=h2_dma(ins,outs,weights,biases,numeric,low.data(),mode);
        size_t mismatch=0;
        for(size_t off=0;off<expected.size();off+=32){if(outs.empty())return 3;auto t=outs.read();if(t.keep!=ap_uint<32>(-1)||t.strb!=ap_uint<32>(-1)||bool(t.last)!=(off+32==expected.size()))return 4;for(int j=0;j<32;j++)if((unsigned)t.data.range(j*8+7,j*8)!=(unsigned char)expected[off+j])mismatch++;}
        std::cout<<"mode="<<mode<<" bytes="<<expected.size()<<" mismatch="<<mismatch<<" status="<<status<<std::endl;
        if(mismatch||status||!ins.empty()||!outs.empty())return 5;
    }
    return 0;
}
