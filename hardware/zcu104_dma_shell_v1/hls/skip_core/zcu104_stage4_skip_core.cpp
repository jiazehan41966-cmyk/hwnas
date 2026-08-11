#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

static constexpr int kChannels = 32;
static constexpr int kHeight = 28;
static constexpr int kWidth = 28;

typedef ap_axiu<kChannels * 8, 0, 0, 0> axis256_t;

// Interface-only variant of the frozen Stage4 Skip operator. The mathematical
// body is the same pixel-major identity used by hls_lut_builder/templates/skip.cpp.tmpl.
// ap_ctrl_hs is kept as direct pins so the benchmark wrapper owns the only
// software-visible AXI4-Lite register bank.
void zcu104_stage4_skip_core(
    hls::stream<axis256_t> &input_stream,
    hls::stream<axis256_t> &output_stream
) {
#pragma HLS INTERFACE axis port=input_stream
#pragma HLS INTERFACE axis port=output_stream
#pragma HLS INTERFACE ap_ctrl_hs port=return

    for (int h = 0; h < kHeight; ++h) {
        for (int w = 0; w < kWidth; ++w) {
#pragma HLS PIPELINE II=1
            axis256_t in_packet = input_stream.read();
            axis256_t out_packet;
            out_packet.data = in_packet.data;
            out_packet.keep = -1;
            out_packet.strb = -1;
            out_packet.last = ((h == (kHeight - 1)) && (w == (kWidth - 1))) ? 1 : 0;
            output_stream.write(out_packet);
        }
    }
}
