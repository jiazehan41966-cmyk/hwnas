#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#include <cstdint>
#include <iostream>

static constexpr int kChannels = 32;
static constexpr int kWords = 28 * 28;

typedef ap_axiu<kChannels * 8, 0, 0, 0> axis256_t;

void zcu104_stage4_skip_core(
    hls::stream<axis256_t> &input_stream,
    hls::stream<axis256_t> &output_stream
);

int main() {
    hls::stream<axis256_t> input_stream;
    hls::stream<axis256_t> output_stream;

    for (int word = 0; word < kWords; ++word) {
        axis256_t packet;
        packet.data = 0;
        for (int channel = 0; channel < kChannels; ++channel) {
            const std::int8_t value = static_cast<std::int8_t>((word * 17 + channel * 13) & 0xff);
            packet.data.range((channel + 1) * 8 - 1, channel * 8) =
                static_cast<ap_uint<8>>(value);
        }
        packet.keep = -1;
        packet.strb = -1;
        packet.last = (word == (kWords - 1)) ? 1 : 0;
        input_stream.write(packet);
    }

    zcu104_stage4_skip_core(input_stream, output_stream);

    int errors = 0;
    for (int word = 0; word < kWords; ++word) {
        if (output_stream.empty()) {
            std::cerr << "missing output word " << word << "\n";
            return 1;
        }
        const axis256_t packet = output_stream.read();
        for (int channel = 0; channel < kChannels; ++channel) {
            const std::int8_t expected = static_cast<std::int8_t>((word * 17 + channel * 13) & 0xff);
            const ap_uint<8> actual = packet.data.range((channel + 1) * 8 - 1, channel * 8);
            if (actual != static_cast<ap_uint<8>>(expected)) {
                ++errors;
            }
        }
        if (packet.keep != ap_uint<32>(-1) || packet.strb != ap_uint<32>(-1)) {
            ++errors;
        }
        const bool expected_last = word == (kWords - 1);
        if (static_cast<bool>(packet.last) != expected_last) {
            ++errors;
        }
    }

    if (!output_stream.empty()) {
        std::cerr << "unexpected extra output data\n";
        return 1;
    }
    if (errors != 0) {
        std::cerr << "Stage4 Skip C-sim mismatches: " << errors << "\n";
        return 1;
    }

    std::cout << "PASS: 784 pixel-major words, bit-exact identity, one final TLAST\n";
    return 0;
}
