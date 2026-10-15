#include "config.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using input_t=INPUT_TYPE;
using data_t=OUTPUT_TYPE;
using weight_t=WEIGHT_TYPE;
extern "C" void pilot_kernel(const input_t*,const input_t*,const weight_t*,const int32_t*,data_t*,const int32_t*,volatile uint8_t&);

template<class T> std::vector<T> read(const std::string& file, size_t count) {
    std::ifstream f(file,std::ios::binary);
    if (!f) throw std::runtime_error("missing "+file);
    std::vector<T> data(count);
    f.read(reinterpret_cast<char*>(data.data()),count*sizeof(T));
    if (!f || f.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("wrong length "+file);
    return data;
}
int main() {
    try {
        std::ifstream list("fixtures.txt");
        if (!list) throw std::runtime_error("missing fixture list");
        std::string id; uint64_t total=0, bad=0; int count=0;
        while (list>>id) {
            auto a=read<input_t>(id+"_a.bin",INPUT_ELEMENTS);
            auto b=read<input_t>(id+"_b.bin",INPUT_ELEMENTS);
            auto w=read<weight_t>(id+"_w.bin",WEIGHT_ELEMENTS);
            auto bias=read<int32_t>(id+"_bias.bin",CO);
            auto p=read<int32_t>(id+"_params.bin",8);
            auto reference=read<data_t>(id+"_expected.bin",OUTPUT_ELEMENTS);
            std::vector<data_t> out(OUTPUT_ELEMENTS,data_t(0x55));
            uint8_t phase=0;
            pilot_kernel(a.data(),b.data(),w.data(),bias.data(),out.data(),p.data(),phase);
            if(phase!=0) throw std::runtime_error("phase did not return idle");
            uint64_t mismatch=0;
            for (int i=0;i<OUTPUT_ELEMENTS;++i) if (out[i]!=reference[i]) ++mismatch;
            std::ofstream raw(id+"_actual.bin",std::ios::binary);
            raw.write(reinterpret_cast<char*>(out.data()),out.size()*sizeof(data_t));
            if(!raw) throw std::runtime_error("output write failed");
            std::cout<<id<<" elements="<<OUTPUT_ELEMENTS<<" mismatches="<<mismatch<<"\n";
            total+=OUTPUT_ELEMENTS;bad+=mismatch;++count;
        }
        std::ofstream result("c_result.json");
        result<<"{\"fixtures\":"<<count<<",\"elements\":"<<total<<",\"mismatches\":"<<bad<<"}\n";
        return count>0 && bad==0 ? 0 : 1;
    } catch (const std::exception& e) {std::cerr<<e.what()<<"\n";return 2;}
}
