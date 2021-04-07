//
// Created by Vansh Sikka on 4/1/21.
//

#include "core/probability_model.h"

#include <core/probability_computation_engine.h>

#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using std::vector;
using std::map;
using std::string;

namespace naivebayes {

ProbabilityModel::ProbabilityModel(
    map<size_t, vector<naivebayes::Image>>& images_map) {

  std::map<size_t , std::vector<std::vector<float>>> shaded_probability_model;
  std::map<size_t , std::vector<std::vector<float>>> unshaded_probability_model;

  for (int i = 0; i < images_map.size(); i++) {

    class_probability_model_[i] = ProbabilityComputationEngine::CalculateClassProbability(i, images_map);

    vector<string> image_string_vector = images_map[0][0].image_string_vector();

    vector<vector<float>> shaded_vec(
        image_string_vector.size(),vector<float>(image_string_vector.size()));

    std::vector<std::vector<float>> unshaded_vec(
        image_string_vector.size(),vector<float>(image_string_vector.size()));

    for (int j = 0; j < images_map[0][0].image_string_vector().size(); j++) {

      for (int k = 0; k < images_map[0][0].image_string_vector().size(); k++) {

        std::pair<float, float> shaded_unshaded =
            (ProbabilityComputationEngine::CalculatePixelProbability(
                i, images_map, std::pair<size_t, size_t>(j, k)));


        shaded_vec[j][k] = (shaded_unshaded.first);
        unshaded_vec[j][k] = (shaded_unshaded.second);
      }
    }

    shaded_probability_model[i] = shaded_vec;
    unshaded_probability_model[i] = unshaded_vec;
    shaded_vec.clear();
    unshaded_vec.clear();
  }
  pixel_probability_model_.push_back(shaded_probability_model);
  pixel_probability_model_.push_back(unshaded_probability_model);

}

void ProbabilityModel::WriteJsonOutputFile(std::string file_path) {

  json j;
  j["pixel_probability_model"] = pixel_probability_model_;
  j["class_probability_model"] = class_probability_model_;

  std::ofstream output_file(file_path);

  if (output_file.is_open()) {
    output_file << j << std::endl;
  }

  output_file.close();
}

std::istream& operator>>(std::istream& is,
                         naivebayes::ProbabilityModel& probability_model) {
  json j = json::parse(is);

  probability_model.pixel_probability_model_ =
      j["pixel_probability_model"].get<vector<map<size_t, vector<vector<float>>>>>();
  probability_model.class_probability_model_ =
      j["class_probability_model"].get<map<size_t, float>>();

  return is;
}

ProbabilityModel::ProbabilityModel() {}

std::vector<std::map<size_t, std::vector<std::vector<float>>>> & ProbabilityModel::pixel_probability_model() {
  return pixel_probability_model_;
}

std::map<size_t, float> ProbabilityModel::class_probability_model() {
  return class_probability_model_;
}

}  // namespace naivebayes
