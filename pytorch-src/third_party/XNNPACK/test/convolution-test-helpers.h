// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace xnnpack {
void compute_convolution_qs8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_channel_stride,
    int8_t input_zero_point,
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias);

void compute_convolution_qs8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    int8_t input_zero_point,
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias);

void compute_convolution_qu8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    uint8_t kernel_zero_point,
    const std::vector<uint8_t>& input,
    const std::vector<uint8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias);

void compute_convolution_qu8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_channel_stride,
    uint8_t input_zero_point,
    uint8_t kernel_zero_point,
    const std::vector<uint8_t>& input,
    const std::vector<uint8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias);

void compute_depthwise_convolution_qs8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t input_channels,
    size_t depth_multiplier,
    size_t input_channel_stride,
    int8_t input_zero_point,
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias);

void compute_depthwise_convolution_qs8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t input_channels,
    size_t depth_multiplier,
    int8_t input_zero_point,
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias);

void compute_depthwise_convolution_qu8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t input_channels,
    size_t depth_multiplier,
    size_t input_channel_stride,
    uint8_t input_zero_point,
    uint8_t kernel_zero_point,
    const std::vector<uint8_t>& input,
    const std::vector<uint8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias);

void compute_depthwise_convolution_qu8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t input_channels,
    size_t depth_multiplier,
    uint8_t input_zero_point,
    uint8_t kernel_zero_point,
    const std::vector<uint8_t>& input,
    const std::vector<uint8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias);
}  // namespace xnnpack
