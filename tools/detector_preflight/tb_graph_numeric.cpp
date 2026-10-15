// Descriptor regressions plus real-PTQ consumer bridges. Independent scalar
// oracle; never call h2::requant from the expected-value path.
#define main legacy_main
#include "tb_graph_descriptor.cpp"
#undef main
#include <limits.h>

static int8_t oracle(int32_t value,int multiplier,int shift) {
    int64_t product=int64_t(value)*int64_t(multiplier);
    bool negative=product<0;
    uint64_t absolute=negative?uint64_t(-product):uint64_t(product);
    uint64_t q=absolute>>shift;
    if(shift) {
        uint64_t remainder=absolute&((uint64_t(1)<<shift)-1);
        uint64_t half=uint64_t(1)<<(shift-1);
        if(remainder>half || (remainder==half && (q&1)))q++;
    }
    int64_t result=negative?-int64_t(q):int64_t(q);
    return int8_t(result>127?127:(result<-128?-128:result));
}

int main() {
    if(legacy_main())return 1;
    static int8_t arena[8192]={0},weights[153600]={0};
    static int32_t biases[960]={0};
    hls::stream<axis256> input,output;
    const int8_t values[16]={-128,-127,-65,-63,-7,-5,-3,-1,0,1,3,5,7,63,65,127};
    for(int mode=0;mode<3;mode++) {
        int d[32]={0};
        d[0]=4;d[1]=d[2]=1;d[3]=d[4]=d[5]=d[6]=4;
        d[13]=4096;d[16]=(mode==1?4:1);d[17]=(mode==0?1:(mode==2?31:0));
        d[22]=8192;d[23]=d[25]=16;d[26]=0x48324731;
        for(int i=0;i<16;i++)arena[i]=values[i];
        int rc=h2_graph(input,output,arena,weights,biases,d);
        if(rc)return 2;
        for(int i=0;i<16;i++)if(arena[4096+i]!=oracle(values[i],d[16],d[17]))return 3;
        std::cout<<"requant_case="<<mode<<" status=0 mismatch=0"<<std::endl;
    }
    for(int kernel=3;kernel<=5;kernel+=2)for(int stride=1;stride<=2;stride++) {
        int d[32]={0},height=4,width=4,channels=2,oh=4/stride;
        d[0]=0;d[1]=d[2]=channels;d[3]=d[4]=4;d[5]=d[6]=oh;
        d[7]=kernel;d[8]=stride;d[9]=kernel/2;d[10]=channels;d[13]=4096;
        d[14]=channels*kernel*kernel;d[15]=channels;d[16]=1;d[17]=1;d[18]=2;d[19]=6;
        d[22]=8192;d[23]=channels*height*width;d[25]=channels*oh*oh;d[26]=0x48324731;
        for(int i=0;i<32;i++)arena[i]=int8_t(i%9-4);
        for(int i=0;i<d[14];i++)weights[i]=int8_t(i%5-2);
        biases[0]=INT_MAX-3;biases[1]=INT_MIN+3;
        int rc=h2_graph(input,output,arena,weights,biases,d);
        if(rc)return 4;
        for(int y=0;y<oh;y++)for(int x=0;x<oh;x++)for(int c=0;c<channels;c++) {
            uint32_t acc=uint32_t(biases[c]);
            for(int ky=0;ky<kernel;ky++)for(int kx=0;kx<kernel;kx++) {
                int iy=y*stride+ky-kernel/2,ix=x*stride+kx-kernel/2;
                if(iy>=0 && iy<height && ix>=0 && ix<width)
                    acc+=uint32_t(int32_t(arena[(iy*width+ix)*channels+c])*int32_t(weights[(c*kernel+ky)*kernel+kx]));
            }
            int expected=oracle(int32_t(acc),1,1);
            if(expected<0)expected=0;if(expected>6)expected=6;
            if(arena[4096+(y*oh+x)*channels+c]!=expected)return 5;
        }
        std::cout<<"depthwise_kernel="<<kernel<<" stride="<<stride<<" wrap_relu6 status=0 mismatch=0"<<std::endl;
    }
    if(!input.empty() || !output.empty())return 6;
    std::cout<<"REAL_PTQ_NUMERIC_REGRESSION_PASS"<<std::endl;
    return 0;
}
