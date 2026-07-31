#pragma once

#include <stdint.h>

static const int DIR_C = 32;
static const int DIR_E = 96;
static const int DIR_B = 48;
static const int DIR_H = 28;
static const int DIR_W = 28;

void dir_mbconv3_split11_e3_v1_int8(
    const int8_t input[DIR_C * DIR_H * DIR_W],
    const int8_t expand_w[DIR_E * DIR_C],
    const int8_t dw_1x3_w[DIR_B * 3],
    const int8_t dw_3x1_w[DIR_B * 3],
    const int8_t project_w[DIR_C * DIR_E],
    int8_t output[DIR_C * DIR_H * DIR_W]);
