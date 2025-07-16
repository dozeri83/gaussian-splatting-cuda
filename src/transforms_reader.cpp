#include <torch/torch.h>

#include "core/transforms_reader.hpp"

namespace F = torch::nn::functional;


std::tuple<std::vector<CameraData>, torch::Tensor> read_transforms_cameras_and_images(
    const std::filesystem::path& base,
    const std::string& images_folder) {

    std::vector<CameraData> cameras;

    std::ifstream trans_file{base.string()};
    nlohmann::json transforms = nlohmann::json::parse(trans_file, nullptr, true, true);



    return std::tuple<std::vector<CameraData>, torch::Tensor>();
}
