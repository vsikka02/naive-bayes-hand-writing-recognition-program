//
// Created by Vansh Sikka on 4/1/21.
//

#include "core/probability_model.h"

#include <core/data_processing_engine.h>
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

std::map<size_t, float> ProbabilityModel::class_probability_model() {
  return class_probability_model_;
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

ProbabilityModel::ProbabilityModel(std::string input_file) {
    std::fstream input (input_file);
    json j = json::parse(input);

    pixel_probability_model_ =
        j["pixel_probability_model"].get<vector<map<size_t, vector<vector<float>>>>>();
    class_probability_model_ =
        j["class_probability_model"].get<map<size_t, float>>();
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

std::pair<int, float> ProbabilityModel::Classifier(const naivebayes::Image& image) {
  float max_likelihood_score = -(std::numeric_limits<float>::max());
  int class_label = -1;
  vector<string> image_vector = image.image_string_vector();
  std::pair<int, float> to_return (0,0);

  for (size_t i = 0; i < class_probability_model_.size(); i++) {
    float current_likelihood = class_probability_model_[i];
    std::vector<std::map<size_t, std::vector<std::vector<float>>>> pixel_probability_model = pixel_probability_model_;
    for (int j = 0; j < pixel_probability_model_[0][0][0].size(); j++) {
      for (int k = 0; k < pixel_probability_model_[0][0][0].size(); k++) {
        if(image_vector[j][k] == ' ') {
          current_likelihood += pixel_probability_model_[1][i][j][k];
        } else {
          current_likelihood += pixel_probability_model_[0][i][j][k];
        }
      }
    }
    if (max_likelihood_score < current_likelihood) {
      max_likelihood_score = current_likelihood;
      class_label = i;
      to_return.first = i;
      to_return.second = max_likelihood_score;
    }
  }

  return to_return;
}

float ProbabilityModel::AccuracyOfClassifier(const std::string& file_name) {
  std::ifstream input_file (file_name);
  naivebayes::DataProcessingEngine data_engine;
  input_file >> data_engine;

  float accuracy = 0;
  for (int i = 0; i < data_engine.image_map().size(); i++) {
    for (int j = 0; j < data_engine.image_map()[i].size(); j++) {
      naivebayes::Image image = data_engine.image_map()[i][j];
      size_t label = Classifier(image).first;

      if (image.image_class_label() == label) {
        accuracy++;
      }
    }
  }

  return accuracy/ProbabilityComputationEngine::CalculateTotalNumberOfImages(data_engine.image_map());
}

}  // namespace naivebayes
