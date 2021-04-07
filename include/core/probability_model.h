//
// Created by Vansh Sikka on 4/1/21.
//
#pragma once

#include <map>
#include <vector>
#include <fstream>

#include "image.h"

using std::vector;
namespace naivebayes {
class ProbabilityModel {
 public:
  //Empty Constructor.
  ProbabilityModel();

  // Constructs a probability model that goes through all the labels and
  // will generate a vector of a map from size_t to a 2D vector of floats
  // that will store the shaded and unshaded probabilities by class_label.
  // This will also generate the map from class_label to class probability.
  ProbabilityModel(std::map<size_t , std::vector<naivebayes::Image>>&);

  // This utilizes the nlohmann class in order to write a json file that
  // serializes the pixel_probability_model_ and the class_probability_model_.
  std::string WriteJsonOutputFile(std::string file_name);

  // Input operator overrided. This will take an input file stream and then
  // load an empty probability_model with the data from a JSON file.
  friend std::istream& operator >> (std::istream& is,
                                    ProbabilityModel& probability_model);

  //Getter for the pixel_probability_model_.
  std::vector<std::map<size_t , std::vector<std::vector<float>>>>&
    pixel_probability_model();

  //Getter for the class_probability_model_.
  std::map<size_t, float> class_probability_model();

 private:
  std::vector<std::map<size_t , std::vector<std::vector<float>>>>
      pixel_probability_model_;

  std::map<size_t, float> class_probability_model_;
};
}
