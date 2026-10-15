#ifndef HWNAS_H2_NUMERIC_V1_H
#define HWNAS_H2_NUMERIC_V1_H
#include <ap_int.h>
namespace h2 {
// Validated loader domain: multiplier in [1,2^31-1], shift in [0,31].
inline ap_int<8> requant(ap_int<32> a,int multiplier,int shift) {
    ap_int<64> product=ap_int<64>(a)*ap_int<64>(multiplier);
    bool neg=product<0;
    ap_uint<64> absolute=neg?ap_uint<64>(-product):ap_uint<64>(product);
    ap_uint<64> q=absolute>>shift;
    if(shift>0) {
        ap_uint<64> mask=(ap_uint<64>(1)<<shift)-1;
        ap_uint<64> remainder=absolute&mask, half=ap_uint<64>(1)<<(shift-1);
        if(remainder>half || (remainder==half && q[0])) ++q;
    }
    ap_int<64> value=q;
    if(neg)value=-value;
    if(value>127)return 127;
    if(value<-128)return -128;
    return ap_int<8>(value);
}
// params: multiplier, shift, activation(0 linear,1 relu,2 relu6), q6, output exponent.
inline ap_int<8> finish(ap_int<32> a,const int* p) {
    ap_int<8> q=requant(a,p[0],p[1]);
    if(p[2] && q<0)q=0;
    if(p[2]==2 && q>p[3])q=p[3];
    return q;
}
// Same staged scale alignment for FPN and residual. args: A m/s,B m/s,out m/s.
inline ap_int<8> add(ap_int<8> a,ap_int<8> b,const int* p) {
    ap_int<8> qa=requant(ap_int<32>(a),p[0],p[1]);
    ap_int<8> qb=requant(ap_int<32>(b),p[2],p[3]);
    ap_int<32> sum=ap_int<32>(qa)+ap_int<32>(qb);
    return requant(sum,p[4],p[5]);
}
}
#endif
