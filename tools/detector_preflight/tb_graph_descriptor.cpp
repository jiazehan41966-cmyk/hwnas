#include <stdint.h>
#include <iostream>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
typedef ap_axiu<256,0,0,0> axis256;
int h2_graph(hls::stream<axis256>&,hls::stream<axis256>&,int8_t*,const int8_t*,const int32_t*,const int*);
int main() {
    static int8_t arena[16777216]={0},weights[153600]={0};
    static int32_t biases[960]={0};
    hls::stream<axis256> in,out;
    for(int stride=1;stride<=2;stride++) {
        int d[32]={0};
        d[0]=0;d[1]=1;d[2]=32;d[3]=32;d[4]=32;
        d[5]=32/stride;d[6]=32/stride;d[7]=3;d[8]=stride;d[9]=1;d[10]=1;
        d[13]=409600;d[14]=0;d[15]=32;d[16]=1;d[17]=8;d[18]=1;d[19]=127;
        d[22]=11168832;d[23]=1024;d[25]=32*d[5]*d[6];d[26]=0x48324731;
        int result=h2_graph(in,out,arena,weights,biases,d);
        std::cout<<"stride="<<stride<<" expected=7 actual="<<result<<std::endl;
        if(result!=7)return 1;
    }
    int d[32]={0};
    d[1]=1;d[2]=1;d[3]=2;d[4]=2;d[5]=2;d[6]=2;
    d[7]=1;d[8]=1;d[10]=1;d[13]=4096;d[14]=1;d[15]=1;
    d[16]=1;d[19]=127;d[22]=16777216;d[23]=4;d[25]=4;d[26]=0x48324731;
    weights[0]=3;biases[0]=1;
    for(int i=0;i<4;i++)arena[i]=i+1;
    int result=h2_graph(in,out,arena,weights,biases,d);
    if(result!=0)return 2;
    for(int i=0;i<4;i++)if(arena[4096+i]!=(i+1)*3+1)return 3;
    std::cout<<"tiny_conv status=0 mismatch=0"<<std::endl;
    for(int op=0;op<3;op++) {
        int p[32]={0};
        p[0]=op;p[1]=2;p[2]=2;p[3]=2;p[4]=2;p[5]=2;p[6]=2;
        p[7]=1;p[8]=1;p[10]=2;p[12]=64;p[13]=4096;p[14]=2;p[15]=2;
        p[16]=1;p[19]=127;p[22]=16777216;p[23]=8;p[25]=8;p[26]=0x48324731;
        if(op) {p[18]=1;p[19]=0;p[20]=1;p[24]=(op==1?8:2);}
        weights[0]=2;weights[1]=-3;biases[0]=1;biases[1]=-1;
        for(int i=0;i<8;i++){arena[i]=i+1;arena[64+i]=10-i;arena[4096+i]=99;}
        int rc=h2_graph(in,out,arena,weights,biases,p);
        if(rc!=0)return 4;
        for(int i=0;i<8;i++) {
            int expected=(i%2==0?2*(i+1)+1:-3*(i+1)-1);
            if(op==1)expected=11;
            if(op==2)expected=i+1+10-(i%2);
            if(arena[4096+i]!=expected) {
                std::cout<<"op="<<op<<" i="<<i<<" actual="<<int(arena[4096+i])<<" expected="<<expected<<std::endl;
                return 5;
            }
        }
        std::cout<<"small_op="<<op<<" status=0 mismatch=0"<<std::endl;
    }
    return 0;
}
