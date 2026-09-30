/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "splat_tile_streamer.hpp"
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#include "io/loader.hpp"
#include <algorithm>
#include <chrono>

namespace lfs::vis {

    namespace {
        constexpr float kSseFactorStep = 1.02f;
        constexpr float kMaxSseFactor = 64.0f;

        std::uint64_t tile_bytes(const io::SplatTile& tile) {
            const int rest = (tile.sh_degree + 1) * (tile.sh_degree + 1) - 1;
            return tile.splat_count * (14 + 3 * rest) * sizeof(float);
        }

        bool same_view(const io::SplatTileView& a, const io::SplatTileView& b) {
            return a.camera == b.camera && a.planes == b.planes && a.sse_per_error == b.sse_per_error &&
                   a.max_sse == b.max_sse;
        }
    } // namespace

    SplatTileStreamer::SplatTileStreamer(std::shared_ptr<const io::SplatTileSource> source,
                                         core::SplatTensorAllocator allocator)
        : source_(std::move(source)),
          allocator_(std::move(allocator)) {
        gpu_total_bytes_ = core::gpu_backend_memory_info(core::default_gpu_backend()).total_bytes;
        cache_limit_bytes_ = static_cast<std::uint64_t>(SplatTileStreamSettings{}.cache_fraction *
                                                        static_cast<double>(gpu_total_bytes_));
        const auto everything = [](std::uint32_t) { return true; };
        full_detail_splats_ = io::select_splat_tiles(*source_, {.sse_per_error = 1.0f, .max_sse = 0.0f}, everything)
                                  .render_splats;
        shown_set_ = io::select_splat_tiles(*source_, {.max_sse = std::numeric_limits<float>::infinity()}, everything)
                         .render;
        std::ranges::sort(shown_set_);
        requested_set_ = shown_set_;
        worker_ = std::jthread([this](const std::stop_token& stop) { work(stop); });
    }

    SplatTileStreamer::~SplatTileStreamer() {
        worker_.request_stop();
        cv_.notify_all();
    }

    std::unique_ptr<core::SplatData> SplatTileStreamer::update(const io::SplatTileView& view,
                                                               const SplatTileStreamSettings& settings,
                                                               std::function<void()> wake) {
        std::lock_guard lock(mutex_);
        if (!wake_)
            wake_ = std::move(wake);
        const auto cache_limit = static_cast<std::uint64_t>(
            std::clamp(settings.cache_fraction, 0.0f, 1.0f) * static_cast<double>(gpu_total_bytes_));
        if (cache_limit != cache_limit_bytes_) {
            // A new size starts from an empty cache; the drawn model is a separate copy
            // and stays until the reloaded tiles cover the view again.
            cache_limit_bytes_ = cache_limit;
            cache_.clear();
            cache_bytes_ = 0;
            over_budget_ = false;
            sse_factor_ = 1.0f;
            requested_set_.clear();
            build_request_.clear();
            cache_changed_ = true;
            cv_.notify_all();
        }
        // Memory-adjusted error: when the view's tiles do not fit the cache, accept a
        // slightly larger error each frame; relax it again once memory frees up.
        if (over_budget_ && sse_factor_ < kMaxSseFactor) {
            sse_factor_ = std::min(sse_factor_ * kSseFactorStep, kMaxSseFactor);
            cache_changed_ = true;
        } else if (!over_budget_ && sse_factor_ > 1.0f && cache_bytes_ < cache_limit_bytes_ * 4 / 5) {
            sse_factor_ = std::max(sse_factor_ / kSseFactorStep, 1.0f);
            cache_changed_ = true;
        }
        if (!settings.freeze && (cache_changed_ || !same_view(view, last_view_))) {
            cache_changed_ = false;
            last_view_ = view;
            ++frame_;
            auto adjusted = view;
            adjusted.max_sse *= sse_factor_;
            auto selection = io::select_splat_tiles(*source_, adjusted, [this](const std::uint32_t tile) {
                return cache_.contains(tile);
            });
            if (sse_factor_ > 1.0f && wake_)
                wake_(); // keep adjusting while the cache is over or under budget
            wanted_ = std::move(selection.wanted);
            for (const auto tile : wanted_)
                if (const auto it = cache_.find(tile); it != cache_.end())
                    it->second.last_wanted = frame_;
            std::ranges::sort(selection.render);
            // An incomplete cut would open holes; keep showing the current one until it fills in.
            if (selection.complete && !selection.render.empty() && selection.render != requested_set_) {
                requested_set_ = selection.render;
                build_request_ = std::move(selection.render);
            }
            cv_.notify_all();
        }
        if (!built_)
            return nullptr;
        shown_set_ = std::move(built_set_);
        return std::move(built_);
    }

    SplatTileStreamStats SplatTileStreamer::stats() {
        std::lock_guard lock(mutex_);
        const auto tiles = source_->tiles();
        SplatTileStreamStats out{.tiles = tiles.size(),
                                 .drawn_tiles = shown_set_.size(),
                                 .cached_tiles = cache_.size(),
                                 .failed_tiles = failed_.size(),
                                 .full_detail_splats = full_detail_splats_,
                                 .cache_bytes = cache_bytes_,
                                 .cache_limit_bytes = cache_limit_bytes_,
                                 .gpu_total_bytes = gpu_total_bytes_,
                                 .build_ms = build_ms_,
                                 .max_sse = last_view_.max_sse * sse_factor_};
        for (const auto tile : shown_set_)
            out.drawn_splats += tiles[tile].splat_count;
        for (const auto tile : wanted_)
            out.loading_tiles += !cache_.contains(tile) && !failed_.contains(tile);
        return out;
    }

    void SplatTileStreamer::evictLocked(const std::uint64_t incoming) {
        while (cache_bytes_ + incoming > cache_limit_bytes_) {
            auto victim = cache_.end();
            for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                if (it->second.last_wanted == frame_ || std::ranges::binary_search(requested_set_, it->first))
                    continue;
                if (victim == cache_.end() || it->second.last_wanted < victim->second.last_wanted)
                    victim = it;
            }
            if (victim == cache_.end())
                return;
            cache_bytes_ -= victim->second.bytes;
            cache_.erase(victim);
        }
    }

    void SplatTileStreamer::work(const std::stop_token& stop) {
        const auto tiles = source_->tiles();
        std::unique_lock lock(mutex_);
        while (!stop.stop_requested()) {
            // Next tile to load: the most urgent missing one, releasing tiles the view no
            // longer needs to make room. If it still does not fit, the view is over budget.
            std::uint32_t next = io::SplatTile::kNone;
            if (build_request_.empty()) {
                evictLocked();
                over_budget_ = false;
                for (const auto tile : wanted_) {
                    if (cache_.contains(tile) || failed_.contains(tile))
                        continue;
                    const auto bytes = tile_bytes(tiles[tile]);
                    evictLocked(bytes);
                    if (cache_bytes_ + bytes <= cache_limit_bytes_)
                        next = tile;
                    else if (!over_budget_) {
                        over_budget_ = true;
                        if (wake_)
                            wake_(); // the next update coarsens the selection
                    }
                    break;
                }
            }
            if (build_request_.empty() && next == io::SplatTile::kNone) {
                const auto selection = frame_;
                const auto limit = cache_limit_bytes_;
                cv_.wait(lock, stop, [&] {
                    return !build_request_.empty() || frame_ != selection || cache_limit_bytes_ != limit;
                });
                continue;
            }

            if (!build_request_.empty()) {
                auto set = std::move(build_request_);
                build_request_.clear();
                std::unordered_map<std::uint32_t, std::shared_ptr<const core::SplatData>> pieces;
                for (const auto tile : set)
                    if (const auto it = cache_.find(tile); it != cache_.end())
                        pieces.emplace(tile, it->second.data);
                if (pieces.size() != set.size()) {
                    requested_set_.clear(); // evicted meanwhile; ask the next selection again
                    cache_changed_ = true;
                    continue;
                }
                lock.unlock();
                const auto build_start = std::chrono::steady_clock::now();
                auto merged = io::merge_splat_tiles(*source_, set, [&](const std::uint32_t tile) {
                    return pieces.at(tile).get();
                });
                if (merged) {
                    if (auto migrated = io::migrateSplatTensorsToAllocator(*merged, allocator_); !migrated) {
                        LOG_ERROR("3D Tiles: cannot prepare streamed model: {}", migrated.error().format());
                        merged.reset();
                    }
                }
                const auto build_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_start).count();
                lock.lock();
                build_ms_ = build_ms;
                if (merged && set == requested_set_) {
                    built_ = std::move(merged);
                    built_set_ = std::move(set);
                    if (wake_)
                        wake_();
                }
                continue;
            }

            lock.unlock();
            auto loaded = io::load_splat_tile_gpu(*source_, next);
            lock.lock();
            if (!loaded) {
                LOG_ERROR("3D Tiles: tile {}: {}", next, loaded.error());
                failed_.insert(next);
                continue;
            }
            const auto bytes = tile_bytes(tiles[next]);
            cache_.emplace(next, CachedTile{std::make_shared<const core::SplatData>(std::move(*loaded)), bytes, frame_});
            cache_bytes_ += bytes;
            cache_changed_ = true;
            if (wake_)
                wake_();
        }
    }

} // namespace lfs::vis
