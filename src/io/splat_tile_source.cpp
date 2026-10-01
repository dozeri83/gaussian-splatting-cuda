/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/splat_tile_source.hpp"
#include "core/scene.hpp"
#include "core/splat_data_transform.hpp"
#include <algorithm>

namespace lfs::io {

    namespace {
        struct Traversal {
            const SplatTileSource& source;
            const SplatTileView& view;
            const std::function<bool(std::uint32_t)>& resident;
            std::vector<std::pair<float, std::uint32_t>> wanted;

            bool visible(const SplatTile& tile) const {
                const float r = tile.radius();
                return std::ranges::all_of(view.planes, [&](const glm::vec4& p) {
                    return glm::dot(glm::vec3(p), tile.center) + p.w >= -r;
                });
            }

            // Appends the tiles covering `index` to `render`; true when fully covered.
            bool visit(const std::uint32_t index, std::vector<std::uint32_t>& render) {
                const auto& tile = source.tiles()[index];
                if (!visible(tile))
                    return true;
                const float distance = std::max(tile.distance(view.camera), 1e-6f);
                const float sse = tile.geometric_error * view.sse_per_error / distance;
                const bool has = tile.splat_count > 0;
                const bool ready = has && resident(index);
                if (has)
                    wanted.emplace_back(sse, index);
                const bool refine = tile.child_count > 0 && (sse > view.max_sse || !has);
                if (!refine) {
                    if (ready)
                        render.push_back(index);
                    return !has || ready;
                }
                std::vector<std::uint32_t> children;
                bool covered = true;
                for (std::uint32_t c = 0; c < tile.child_count; ++c)
                    covered = visit(tile.first_child + c, children) && covered;
                if (tile.additive) {
                    if (ready)
                        render.push_back(index);
                    render.insert(render.end(), children.begin(), children.end());
                    return covered && (!has || ready);
                }
                if (covered || !ready) {
                    render.insert(render.end(), children.begin(), children.end());
                    return covered;
                }
                render.push_back(index); // children still loading: keep the parent
                return true;
            }
        };
    } // namespace

    SplatTileSelection select_splat_tiles(const SplatTileSource& source, const SplatTileView& view,
                                          const std::function<bool(std::uint32_t)>& resident) {
        SplatTileSelection out;
        if (source.tiles().empty())
            return out;
        Traversal traversal{source, view, resident, {}};
        out.complete = traversal.visit(0, out.render);
        for (const auto index : out.render)
            out.render_splats += source.tiles()[index].splat_count;
        std::ranges::sort(traversal.wanted, std::greater{});
        for (const auto& [_, index] : traversal.wanted)
            out.wanted.push_back(index);
        return out;
    }

    std::expected<core::SplatData, std::string> load_splat_tile_gpu(const SplatTileSource& source,
                                                                    const std::uint32_t tile) {
        auto data = source.load_tile(tile);
        if (!data)
            return data;
        using core::Device;
        data->means_raw() = data->means_raw().to(Device::GPU);
        data->sh0_raw() = data->sh0_raw().to(Device::GPU);
        if (data->shN_raw().is_valid() && data->shN_raw().numel() > 0)
            data->shN_raw() = data->shN_raw().to(Device::GPU);
        data->scaling_raw() = data->scaling_raw().to(Device::GPU);
        data->rotation_raw() = data->rotation_raw().to(Device::GPU);
        data->opacity_raw() = data->opacity_raw().to(Device::GPU);
        // Bake the tile's placement into the splats once, here, so repeated cut
        // rebuilds can concatenate the cached tiles on the fast identity path
        // instead of re-transforming and re-cloning every tile each time.
        if (const auto& placement = source.tiles()[tile].transform; placement != glm::mat4(1.0f))
            core::transform(*data, placement);
        return data;
    }

    std::unique_ptr<core::SplatData> merge_splat_tiles(
        const SplatTileSource& /*source*/, const std::span<const std::uint32_t> tiles,
        const std::function<const core::SplatData*(std::uint32_t)>& splats) {
        std::vector<std::pair<const core::SplatData*, glm::mat4>> pieces;
        for (const auto index : tiles)
            if (const auto* data = splats(index))
                // Placement is already baked in by load_splat_tile_gpu, so merge on
                // the identity fast path (preallocate once, slice-copy each tile).
                pieces.emplace_back(data, glm::mat4(1.0f));
        return core::Scene::mergeSplatsWithTransforms(pieces);
    }

} // namespace lfs::io
