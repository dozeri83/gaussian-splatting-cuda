/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// 3D Tiles 1.0/1.1 tilesets with KHR_gaussian_splatting (SPZ) GLB content.
// Implements the OGC 3D Tiles specification (tile tree, transforms, bounding
// volumes, refinement); tile content is decoded by load_spz.

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "formats/spz.hpp"
#include "io/splat_tile_source.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <deque>
#include <format>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <tbb/parallel_for.h>

namespace lfs::io {

    namespace {
        using nlohmann::json;
        namespace fs = std::filesystem;

        constexpr std::size_t kMaxTiles = 1'000'000;
        constexpr int kMaxExternalDepth = 8;
        // Parsing recurses per tile level; real tilesets stay far below this.
        constexpr int kMaxTreeDepth = 128;
        constexpr const char* kSpzPointer =
            "/extensions/KHR_gaussian_splatting/extensions/KHR_gaussian_splatting_compression_spz_2";
        constexpr std::uint32_t kGlbMagic = 0x46546C67;     // "glTF"
        constexpr std::uint32_t kGlbChunkJson = 0x4E4F534A; // "JSON"
        constexpr std::uint32_t kMaxGlbJsonBytes = 64u << 20;
        constexpr std::uintmax_t kMaxTilesetJsonBytes = 512u << 20;

        struct ParsedTile {
            SplatTile tile;
            std::vector<fs::path> contents;
            std::vector<ParsedTile> children;
        };

        json read_json(const fs::path& path) {
            std::ifstream in;
            if (!core::open_file_for_read(path, std::ios::binary, in))
                throw std::runtime_error(std::format("Cannot open '{}'", core::path_to_utf8(path)));
            return json::parse(in);
        }

        glm::dmat4 read_matrix(const json& tile) {
            if (!tile.contains("transform"))
                return glm::dmat4(1.0);
            const auto m = tile.at("transform").get<std::vector<double>>();
            if (m.size() != 16)
                throw std::runtime_error("tile transform must have 16 values");
            return glm::make_mat4(m.data()); // column-major, like glTF
        }

        double uniform_scale(const glm::dmat4& m) {
            return std::cbrt(std::abs(glm::determinant(glm::dmat3(m))));
        }

        glm::dvec3 wgs84_to_ecef(const double lon, const double lat, const double height) {
            constexpr double a = 6378137.0;
            constexpr double e2 = 6.69437999014e-3;
            const double n = a / std::sqrt(1.0 - e2 * std::sin(lat) * std::sin(lat));
            return {(n + height) * std::cos(lat) * std::cos(lon),
                    (n + height) * std::cos(lat) * std::sin(lon),
                    (n * (1.0 - e2) + height) * std::sin(lat)};
        }

        // Bounding volume -> oriented box in the source-local frame. `to_local` maps the
        // tile frame; regions are always EPSG:4979 and map through `ecef_to_local`.
        void read_bounds(const json& volume, const glm::dmat4& to_local,
                         const glm::dmat4& ecef_to_local, SplatTile& tile) {
            if (volume.contains("box")) {
                const auto b = volume.at("box").get<std::vector<double>>();
                if (b.size() != 12)
                    throw std::runtime_error("bounding box must have 12 values");
                tile.center = glm::vec3(to_local * glm::dvec4(b[0], b[1], b[2], 1.0));
                const glm::dmat3 axes(b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11]);
                tile.half_axes = glm::mat3(glm::dmat3(to_local) * axes);
            } else if (volume.contains("sphere")) {
                const auto s = volume.at("sphere").get<std::vector<double>>();
                if (s.size() != 4)
                    throw std::runtime_error("bounding sphere must have 4 values");
                tile.center = glm::vec3(to_local * glm::dvec4(s[0], s[1], s[2], 1.0));
                tile.half_axes = glm::mat3(static_cast<float>(s[3] * uniform_scale(to_local)));
            } else if (volume.contains("region")) {
                const auto r = volume.at("region").get<std::vector<double>>();
                if (r.size() != 6)
                    throw std::runtime_error("bounding region must have 6 values");
                glm::dvec3 lo(std::numeric_limits<double>::max()), hi(std::numeric_limits<double>::lowest());
                for (int i = 0; i <= 2; ++i)
                    for (int j = 0; j <= 2; ++j)
                        for (int k = 0; k <= 1; ++k) {
                            const auto p = glm::dvec3(ecef_to_local * glm::dvec4(wgs84_to_ecef(
                                                                                     r[0] + (r[2] - r[0]) * i / 2.0,
                                                                                     r[1] + (r[3] - r[1]) * j / 2.0,
                                                                                     k ? r[5] : r[4]),
                                                                                 1.0));
                            lo = glm::min(lo, p);
                            hi = glm::max(hi, p);
                        }
                tile.center = glm::vec3((lo + hi) * 0.5);
                const glm::vec3 half((hi - lo) * 0.5);
                tile.half_axes = glm::mat3(half.x, 0, 0, 0, half.y, 0, 0, 0, half.z);
            } else {
                throw std::runtime_error("tile has no supported bounding volume");
            }
        }

        // A tile has one `content` or (3D Tiles 1.1) several `contents`, all shown together.
        std::vector<std::string> content_uris(const json& tile) {
            std::vector<std::string> uris;
            if (tile.contains("content"))
                uris.push_back(tile.at("content").at("uri").get<std::string>());
            if (tile.contains("contents"))
                for (const auto& content : tile.at("contents"))
                    uris.push_back(content.at("uri").get<std::string>());
            return uris;
        }

        // Content URIs are relative URI references (RFC 3986): percent-encoded, possibly
        // with a query or fragment. Only local files are supported.
        fs::path resolve_uri(const fs::path& base, const std::string& uri) {
            if (const auto colon = uri.find(':'); colon != std::string::npos) {
                const auto scheme = uri.substr(0, colon);
                if (!scheme.empty() && std::isalpha(static_cast<unsigned char>(scheme[0])) &&
                    std::ranges::all_of(scheme, [](const char c) {
                        return std::isalnum(static_cast<unsigned char>(c)) || c == '+' || c == '-' || c == '.';
                    }))
                    throw std::runtime_error(std::format(
                        "content URI '{}' uses the '{}' scheme; only relative file URIs are supported", uri, scheme));
            }
            const auto hex = [](const char c) -> int {
                if (c >= '0' && c <= '9')
                    return c - '0';
                if (c >= 'a' && c <= 'f')
                    return c - 'a' + 10;
                if (c >= 'A' && c <= 'F')
                    return c - 'A' + 10;
                return -1;
            };
            std::string decoded;
            for (std::size_t i = 0; i < uri.size() && uri[i] != '?' && uri[i] != '#'; ++i) {
                if (uri[i] == '%' && i + 2 < uri.size() && hex(uri[i + 1]) >= 0 && hex(uri[i + 2]) >= 0) {
                    decoded.push_back(static_cast<char>(hex(uri[i + 1]) * 16 + hex(uri[i + 2])));
                    i += 2;
                } else {
                    decoded.push_back(uri[i]);
                }
            }
            return (base / core::utf8_to_path(decoded)).lexically_normal();
        }

        struct ParseContext {
            glm::dmat4 ecef_to_local{1.0};
            std::size_t tiles = 0;
        };

        ParsedTile parse_tile(const json& node, const fs::path& base, const glm::dmat4& parent_world,
                              const bool parent_additive, ParseContext& context, const int depth,
                              const int tree_depth) {
            if (++context.tiles > kMaxTiles)
                throw std::runtime_error(std::format("tileset exceeds {} tiles", kMaxTiles));
            if (tree_depth > kMaxTreeDepth)
                throw std::runtime_error(std::format("tile tree nests deeper than {} levels", kMaxTreeDepth));
            ParsedTile parsed;
            const glm::dmat4 world = parent_world * read_matrix(node);
            const glm::dmat4 to_local = context.ecef_to_local * world;
            auto& tile = parsed.tile;
            tile.transform = glm::mat4(to_local);
            tile.geometric_error = static_cast<float>(node.at("geometricError").get<double>() *
                                                      uniform_scale(context.ecef_to_local));
            tile.additive = node.contains("refine") ? node.at("refine").get<std::string>() == "ADD" : parent_additive;
            read_bounds(node.at("boundingVolume"), to_local, context.ecef_to_local, tile);

            for (const auto& uri : content_uris(node)) {
                const auto path = resolve_uri(base, uri);
                if (path.extension() == ".json") {
                    // External tileset: its root becomes a child of this tile.
                    if (depth >= kMaxExternalDepth)
                        throw std::runtime_error("external tilesets nest too deeply");
                    const auto external = read_json(path);
                    parsed.children.push_back(parse_tile(external.at("root"), path.parent_path(), world,
                                                         tile.additive, context, depth + 1, tree_depth + 1));
                } else {
                    parsed.contents.push_back(path);
                }
            }
            if (node.contains("children"))
                for (const auto& child : node.at("children"))
                    parsed.children.push_back(
                        parse_tile(child, base, world, tile.additive, context, depth, tree_depth + 1));
            return parsed;
        }

        struct ContentProbe {
            std::uint64_t splats = 0;
            int sh_degree = 0;
        };

        // Splat count and SH degree from the GLB JSON chunk, without reading the payload.
        // Uses the primitive the decoder reads (the first carrying SPZ-compressed Gaussian
        // splats); empty when the content is not such a GLB (another glTF, b3dm, pnts...).
        std::optional<ContentProbe> probe_content(const fs::path& path) {
            std::ifstream in;
            if (!core::open_file_for_read(path, std::ios::binary, in))
                throw std::runtime_error(std::format("Cannot open '{}'", core::path_to_utf8(path)));
            std::uint32_t header[5] = {};
            if (!in.read(reinterpret_cast<char*>(header), sizeof(header)) || header[0] != kGlbMagic ||
                header[4] != kGlbChunkJson || header[3] > kMaxGlbJsonBytes)
                return std::nullopt;
            std::string text(header[3], '\0');
            if (!in.read(text.data(), static_cast<std::streamsize>(text.size())))
                throw std::runtime_error(std::format("'{}' has a truncated JSON chunk", core::path_to_utf8(path)));
            const auto doc = json::parse(text);
            const json::json_pointer spz(kSpzPointer);
            for (const auto& mesh : doc.value("meshes", json::array()))
                for (const auto& primitive : mesh.value("primitives", json::array())) {
                    if (!primitive.contains(spz))
                        continue;
                    const auto& attributes = primitive.value("attributes", json::object());
                    ContentProbe probe;
                    probe.splats = doc.at("accessors")
                                       .at(attributes.at("POSITION").get<std::size_t>())
                                       .at("count")
                                       .get<std::uint64_t>();
                    for (int degree = 3; degree > 0; --degree)
                        if (attributes.contains(std::format("KHR_gaussian_splatting:SH_DEGREE_{}_COEF_0", degree))) {
                            probe.sh_degree = degree;
                            break;
                        }
                    return probe;
                }
            return std::nullopt;
        }

        // Content kind for the skipped-content summary: "GLB" or the file extension.
        std::string content_kind(const fs::path& path) {
            auto extension = core::path_to_utf8(path.extension());
            std::ranges::transform(extension, extension.begin(),
                                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return extension == ".glb" ? std::string("GLB") : extension.empty() ? std::string("unknown")
                                                                                : extension;
        }

        class Tiles3dSource final : public SplatTileSource {
        public:
            std::vector<SplatTile> tiles_;
            std::vector<std::vector<fs::path>> contents_; // splat GLBs of each tile

            std::span<const SplatTile> tiles() const override { return tiles_; }

            std::expected<core::SplatData, std::string> load_tile(const std::uint32_t tile) const override {
                if (tile >= tiles_.size() || contents_[tile].empty())
                    return std::unexpected(std::format("tile {} has no content", tile));
                const auto& paths = contents_[tile];
                if (paths.size() == 1)
                    return load_spz(paths.front());
                // Several contents make up the tile together: concatenate them (on the GPU,
                // where merge_splat_tiles pads lower SH degrees) into the tile's model.
                std::vector<core::SplatData> parts;
                parts.reserve(paths.size());
                for (const auto& path : paths) {
                    auto part = load_spz(path);
                    if (!part)
                        return std::unexpected(part.error());
                    using core::Device;
                    part->means_raw() = part->means_raw().to(Device::GPU);
                    part->sh0_raw() = part->sh0_raw().to(Device::GPU);
                    if (part->shN_raw().is_valid() && part->shN_raw().numel() > 0)
                        part->shN_raw() = part->shN_raw().to(Device::GPU);
                    part->scaling_raw() = part->scaling_raw().to(Device::GPU);
                    part->rotation_raw() = part->rotation_raw().to(Device::GPU);
                    part->opacity_raw() = part->opacity_raw().to(Device::GPU);
                    parts.push_back(std::move(*part));
                }
                std::vector<std::uint32_t> indices(parts.size());
                std::iota(indices.begin(), indices.end(), 0u);
                auto merged = merge_splat_tiles(*this, indices, [&](const std::uint32_t i) { return &parts[i]; });
                if (!merged)
                    return std::unexpected(std::format("tile {} has no splats", tile));
                return std::move(*merged);
            }
        };

        // Breadth-first so every tile's children are contiguous.
        void flatten(ParsedTile& root, Tiles3dSource& source) {
            std::deque<std::pair<ParsedTile*, std::uint32_t>> queue{{&root, 0u}};
            source.tiles_.push_back(root.tile);
            source.contents_.push_back(std::move(root.contents));
            while (!queue.empty()) {
                auto [parsed, self] = queue.front();
                queue.pop_front();
                source.tiles_[self].first_child = static_cast<std::uint32_t>(source.tiles_.size());
                source.tiles_[self].child_count = static_cast<std::uint32_t>(parsed->children.size());
                for (auto& child : parsed->children) {
                    child.tile.parent = self;
                    const auto child_index = static_cast<std::uint32_t>(source.tiles_.size());
                    source.tiles_.push_back(child.tile);
                    source.contents_.push_back(std::move(child.contents));
                    queue.emplace_back(&child, child_index);
                }
            }
        }
    } // namespace

    bool is_tiles3d_path(const std::filesystem::path& path) {
        // Recognized by the tileset's required properties, not its name: asset.version,
        // geometricError and a root tile with boundingVolume and geometricError.
        std::error_code ec;
        if (path.extension() != ".json" || !fs::is_regular_file(path, ec) ||
            fs::file_size(path, ec) > kMaxTilesetJsonBytes)
            return false;
        try {
            const auto doc = read_json(path);
            const auto has_volume = [](const json& volume) {
                return volume.is_object() && (volume.contains("box") || volume.contains("region") ||
                                              volume.contains("sphere"));
            };
            return doc.is_object() && doc.contains("asset") && doc["asset"].is_object() &&
                   doc["asset"].contains("version") && doc["asset"]["version"].is_string() &&
                   doc.contains("geometricError") && doc["geometricError"].is_number() &&
                   doc.contains("root") && doc["root"].is_object() &&
                   doc["root"].contains("geometricError") && doc["root"]["geometricError"].is_number() &&
                   doc["root"].contains("boundingVolume") && has_volume(doc["root"]["boundingVolume"]);
        } catch (const std::exception&) {
            return false;
        }
    }

    std::expected<std::shared_ptr<SplatTileSource>, std::string> open_tiles3d(const std::filesystem::path& path) {
        try {
            const auto doc = read_json(path);
            const auto& root = doc.at("root");
            // Keep float geometry near the origin: the root frame (often ECEF) becomes local.
            auto source = std::make_shared<Tiles3dSource>();
            source->local_to_world = read_matrix(root);
            ParseContext context{.ecef_to_local = glm::inverse(source->local_to_world)};
            auto parsed = parse_tile(root, path.parent_path(), glm::dmat4(1.0), false, context, 0, 0);
            flatten(parsed, *source);

            // Probe every content; keep only the splat GLBs. Other content (meshes, point
            // clouds, b3dm, ...) is skipped, so a mixed tileset still shows its splats.
            std::vector<std::string> errors(source->tiles_.size());
            std::vector<std::vector<std::string>> skipped(source->tiles_.size());
            tbb::parallel_for(std::size_t{0}, source->tiles_.size(), [&](const std::size_t i) {
                auto& tile = source->tiles_[i];
                std::vector<fs::path> splat_contents;
                try {
                    for (auto& content : source->contents_[i]) {
                        if (const auto probe = probe_content(content)) {
                            tile.splat_count += probe->splats;
                            tile.sh_degree = std::max(tile.sh_degree, probe->sh_degree);
                            splat_contents.push_back(std::move(content));
                        } else {
                            skipped[i].push_back(content_kind(content));
                        }
                    }
                } catch (const std::exception& e) {
                    errors[i] = e.what();
                }
                source->contents_[i] = std::move(splat_contents);
            });
            for (const auto& error : errors)
                if (!error.empty())
                    return std::unexpected(error);

            std::map<std::string, std::size_t> skipped_by_kind;
            for (const auto& kinds : skipped)
                for (const auto& kind : kinds)
                    ++skipped_by_kind[kind];
            std::string skipped_summary;
            for (const auto& [kind, count] : skipped_by_kind) {
                source->skipped_contents += count;
                skipped_summary += std::format("{}{} {}", skipped_summary.empty() ? "" : ", ", count, kind);
            }
            std::uint64_t splats = 0;
            for (const auto& tile : source->tiles_)
                splats += tile.splat_count;
            if (splats == 0)
                return std::unexpected(std::format(
                    "3D Tiles tileset '{}' has no Gaussian splat content. Only glTF tiles with KHR_gaussian_splatting "
                    "(SPZ compression) are supported{}",
                    core::path_to_utf8(path),
                    skipped_summary.empty() ? std::string(".") : std::format("; found {} contents.", skipped_summary)));
            if (source->skipped_contents > 0)
                LOG_WARN("3D Tiles '{}': skipped {} contents without SPZ Gaussian splats ({})", core::path_to_utf8(path),
                         source->skipped_contents, skipped_summary);
            LOG_INFO("3D Tiles '{}': {} tiles, {} splats", core::path_to_utf8(path), source->tiles_.size(), splats);
            return source;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Invalid 3D Tiles tileset '{}': {}", core::path_to_utf8(path), e.what()));
        }
    }

} // namespace lfs::io
