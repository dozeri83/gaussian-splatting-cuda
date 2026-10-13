/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <chrono>
#include <cstdint>
#include <glm/glm.hpp>
#include <optional>

namespace lfs::vis::input {

    // Cursor offset from the render-image center in render pixels. ortho_scale
    // counts render pixels per world unit about that center, which differs from
    // the panel's on-screen pixels when the render scale is not 1 or on HiDPI.
    [[nodiscard]] inline glm::vec2 orthoCursorOffset(const glm::vec2 panel_local,
                                                     const glm::vec2 panel_size,
                                                     const glm::ivec2 render_size) {
        const glm::vec2 render(render_size);
        return panel_local * (render / panel_size) - render * 0.5f;
    }

    // Camera translation that keeps the world point at `offset` (render pixels
    // from the center) fixed when ortho_scale changes from old_scale to
    // new_scale. Screen y grows downward while the camera up axis points up.
    [[nodiscard]] inline glm::vec3 orthoZoomShift(const glm::mat3& rotation, const glm::vec2 offset,
                                                  const float old_scale, const float new_scale) {
        const glm::vec2 shift = offset * (1.0f / old_scale - 1.0f / new_scale);
        return rotation[0] * shift.x - rotation[1] * shift.y;
    }

    // Zooming toward a world point keeps that point under the cursor, so one
    // depth read serves a whole zoom gesture. The anchor (or its absence over
    // background) is reused while the cursor stays put in the same view and
    // zoom events keep arriving; moving the cursor or pausing starts a new one.
    class ZoomAnchorCache {
    public:
        using Clock = std::chrono::steady_clock;
        static constexpr double kMaxCursorDrift = 2.0;
        static constexpr std::chrono::milliseconds kGestureIdle{300};

        [[nodiscard]] bool matches(const std::uint32_t view, const glm::dvec2 cursor,
                                   const Clock::time_point now) const {
            return valid_ && view == view_ && glm::length(cursor - cursor_) <= kMaxCursorDrift &&
                   now - last_event_ <= kGestureIdle;
        }

        // Returns the cached anchor and extends the gesture.
        [[nodiscard]] const std::optional<glm::vec3>& reuse(const Clock::time_point now) {
            last_event_ = now;
            return anchor_;
        }

        void store(const std::uint32_t view, const glm::dvec2 cursor,
                   const std::optional<glm::vec3>& anchor, const Clock::time_point now) {
            valid_ = true;
            view_ = view;
            cursor_ = cursor;
            anchor_ = anchor;
            last_event_ = now;
        }

        void clear() { valid_ = false; }

    private:
        bool valid_ = false;
        std::uint32_t view_ = 0;
        glm::dvec2 cursor_{0.0};
        std::optional<glm::vec3> anchor_;
        Clock::time_point last_event_{};
    };

} // namespace lfs::vis::input
