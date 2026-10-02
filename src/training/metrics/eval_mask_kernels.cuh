/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include "core/cuda_types.hpp"

namespace lfs::training {

    [[nodiscard]] lfs::core::Tensor erode_metrics_mask(
        const lfs::core::Tensor& mask, int radius, cudaStream_t stream);

} // namespace lfs::training
