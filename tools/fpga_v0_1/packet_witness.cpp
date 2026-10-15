// Actual shared HLS helpers, host C model; no RTL/DMA readiness claim.
#include "common_defs.h"
#include <iostream>
int main() {
    data_t source[21], restored[21];
    for (int c=0;c<21;++c) {source[c]=c-10;restored[c]=-100;}
    typedef MAKE_AXIS_TYPE(128) packet_t;
    packet_t a=make_axis_segment_packet<packet_t,21,16>(source,0,false);
    packet_t b=make_axis_segment_packet<packet_t,21,16>(source,16,true);
    unpack_axis_channel_segment<21,16>(a,0,restored);
    unpack_axis_channel_segment<21,16>(b,16,restored);
    for(int c=0;c<21;++c) if(source[c]!=restored[c]) return 1;
    for(int c=5;c<16;++c) if(b.data.range(c*8+7,c*8)!=0) return 2;
    if(a.last||!b.last||b.keep!=ap_uint<16>(65535)||b.strb!=ap_uint<16>(65535)) return 3;
    std::cout << "packet21_roundtrip=PASS tail_zero_lanes=11 tail_keep=65535 logical_valid_mask=31 MASK_GAP_CONFIRMED\n";
    std::cout << "acc16_raw=" << int(saturate_to_data(16)) << " expected_requant_ratio_half=8\n";
    std::cout << "relu6_raw16=" << int(relu6_quant(16)) << " expected_relu6_output_scale_half=12\n";
    std::cout << "saturate_negative=" << int(saturate_to_data(-200)) << " symmetric_contract_min=-127\n";
    return 0;
}
