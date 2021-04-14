//
// Created by Vansh Sikka on 4/1/21.
//

#include "core/probability_computation_engine.h"

#include <core/image.h>

namespace ProbabilityComputationEngine {
float CalculateClassProbability(
    size_t class_label,
    std::map<size_t, std::vector<naivebayes::Image>>& images_map) {
  if (images_map.empty()) {
    throw std::invalid_argument("Empty Map Inputted!");
  }

  if (images_map.find(class_label) == images_map.end()) {
    throw std::invalid_argument("Invalid Class Label");
  }

  float images_in_class = images_map[class_label].size();
  float class_probability =
      log((kK + images_in_class) / (images_map.size() * kK +
                                    CalculateTotalNumberOfImages(images_map)));

  return (class_probability);
}

std::pair<float, float> CalculatePixelProbability(
    size_t class_label,
    std::map<size_t, std::vector<naivebayes::Image>>& images_map,
    const std::pair<size_t, size_t>& coordinates) {
  if (images_map.empty()) {
    throw std::invalid_argument("Empty Map Inputted!");
  }

  if (images_map.find(class_label) == images_map.end()) {
    throw std::invalid_argument("Class Label Not Found");
  }

  size_t image_size = images_map[class_label][0].image_string_vector()[0].size() - 1;

  if (image_size < coordinates.first || image_size < coordinates.second) {
    throw std::out_of_range("Coordinates do not exist");
  }

  float count_shaded = 0;
  float count_unshaded = 0;
  std::vector<naivebayes::Image> images = images_map[class_label];

  for (size_t i = 0; i < images_map[class_label].size(); i++) {
    std::vector<string> training_image = images[i].image_string_vector();
    if (std::find(kShadedCharacterSet.begin(),
                  kShadedCharacterSet.end(),
                  training_image[coordinates.first][coordinates.second])
        != kShadedCharacterSet.end()) {
      count_shaded++;
    } else if (std::find(kUnshadedCharacterSet.begin(),
                         kUnshadedCharacterSet.end(),
                         training_image[coordinates.first][coordinates.second])
               != kUnshadedCharacterSet.end()) {
      count_unshaded++;
    }
  }

  float images_in_class = images_map[class_label].size();

  float shaded_pixel_probability =
      log((kK + count_shaded) / (kShadedOrUnshadedV * kK + images_in_class));
  float unshaded_pixel_probability =
      log((kK + count_unshaded) / (kShadedOrUnshadedV * kK + images_in_class));

  return std::pair<float, float>(shaded_pixel_probability,
                                 unshaded_pixel_probability);
}

float CalculateTotalNumberOfImages(
    std::map<size_t, std::vector<naivebayes::Image>>& images_map) {
  float total_count = 0;
  for (size_t i = 0; i < images_map.size(); i++) {
    total_count += images_map[i].size();
  }

  return total_count;
}
}  // namespace ProbabilityComputationEngine