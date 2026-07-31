#include "dir_mbconv3_split11_e3_v1.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool read_exact(
    const std::string& path,
    std::vector<int8_t>* values,
    std::size_t count) {
    values->assign(count, 0);
    std::ifstream stream(path.c_str(), std::ios::binary);
    if (!stream) {
        std::cerr << "cannot open " << path << std::endl;
        return false;
    }
    stream.read(
        reinterpret_cast<char*>(values->data()),
        static_cast<std::streamsize>(count));
    return stream.good() || stream.gcount() == static_cast<std::streamsize>(count);
}

int run_case(const std::string& root, const std::string& name) {
    const std::string prefix = root + "/" + name + "/";
    std::vector<int8_t> input;
    std::vector<int8_t> expand;
    std::vector<int8_t> dw_1x3;
    std::vector<int8_t> dw_3x1;
    std::vector<int8_t> project;
    std::vector<int8_t> expected;
    if (!read_exact(prefix + "input.bin", &input, DIR_C * DIR_H * DIR_W) ||
        !read_exact(prefix + "expand_w.bin", &expand, DIR_E * DIR_C) ||
        !read_exact(prefix + "dw_1x3_w.bin", &dw_1x3, DIR_B * 3) ||
        !read_exact(prefix + "dw_3x1_w.bin", &dw_3x1, DIR_B * 3) ||
        !read_exact(prefix + "project_w.bin", &project, DIR_C * DIR_E) ||
        !read_exact(prefix + "expected.bin", &expected, DIR_C * DIR_H * DIR_W)) {
        return 2;
    }
    std::vector<int8_t> observed(DIR_C * DIR_H * DIR_W, 0);
    dir_mbconv3_split11_e3_v1_int8(
        input.data(),
        expand.data(),
        dw_1x3.data(),
        dw_3x1.data(),
        project.data(),
        observed.data());
    int mismatches = 0;
    for (std::size_t index = 0; index < observed.size(); ++index) {
        if (observed[index] != expected[index]) {
            if (mismatches < 5) {
                std::cerr << name << " mismatch index=" << index
                          << " observed=" << static_cast<int>(observed[index])
                          << " expected=" << static_cast<int>(expected[index])
                          << std::endl;
            }
            ++mismatches;
        }
    }
    std::cout << name << " mismatches=" << mismatches << std::endl;
    return mismatches == 0 ? 0 : 1;
}

}  // namespace

int main() {
    const char* root_value = std::getenv("DIR_V1_VECTOR_DIR");
    if (root_value == NULL) {
        std::cerr << "DIR_V1_VECTOR_DIR is not set" << std::endl;
        return 3;
    }
    const std::string root(root_value);
    int status = 0;
    status |= run_case(root, "random");
    status |= run_case(root, "negative_full_scale");
    status |= run_case(root, "positive_full_scale");
    return status;
}
