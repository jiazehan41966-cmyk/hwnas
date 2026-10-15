#include <fstream>
#include <vector>
#include <iostream>
#include <stdint.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
typedef ap_axiu<256,0,0,0> axis256;
int h2_graph(hls::stream<axis256>&,hls::stream<axis256>&,int8_t*,const int8_t*,const int32_t*,const int*);
std::vector<char> read(const char* name){std::ifstream f(name,std::ios::binary);if(!f)throw 1;return std::vector<char>((std::istreambuf_iterator<char>(f)),{});}
int main(){
 auto arena=read("arena.bin"),raw=read("descriptors.bin"),expected=read("expected_layers.bin");
 if(raw.size()%128)return 1;
 hls::stream<axis256> input,output;
 for(size_t index=0;index<raw.size()/128;index++){
  int d[32];for(int j=0;j<32;j++){uint32_t v=0;for(int b=0;b<4;b++)v|=uint32_t((unsigned char)raw[index*128+j*4+b])<<(b*8);d[j]=int32_t(v);}
  int rc=h2_graph(input,output,reinterpret_cast<int8_t*>(arena.data()),reinterpret_cast<const int8_t*>(arena.data()+d[27]),reinterpret_cast<const int32_t*>(arena.data()+d[28]),d);
  size_t mismatch=0;for(int j=0;j<d[25];j++)if(arena.at(d[13]+j)!=expected.at(d[29]+j))mismatch++;
  std::cout<<"node="<<index<<" bytes="<<d[25]<<" mismatch="<<mismatch<<" status="<<rc<<std::endl;
  if(rc||mismatch||!input.empty()||!output.empty())return 2;
 }
 return 0;
}
