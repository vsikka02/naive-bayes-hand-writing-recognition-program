#include <core/data_processing_engine.h>
#include <core/probability_model.h>

#include <fstream>
#include <iostream>

int main() {
  bool train_new_probability_model = false;
  bool train_test_probability_model = true;

  if (train_new_probability_model) {
    std::ifstream input_file("../data/trainingimagesandlabels.txt");
    naivebayes::DataProcessingEngine data_engine = naivebayes::DataProcessingEngine();
    input_file >> data_engine;

    naivebayes::ProbabilityModel probability_model = naivebayes::ProbabilityModel(data_engine.image_map());
    probability_model.WriteJsonOutputFile("output_probability_model");
  } else {
    std::ifstream json_file("../data/output_probability_model.json");
    naivebayes::ProbabilityModel probability_model = naivebayes::ProbabilityModel();
    json_file >> probability_model;
  }
  if (train_test_probability_model) {
    std::ifstream input_file("../data/testtrainingimagesandlabels.txt");
    naivebayes::DataProcessingEngine data_engine = naivebayes::DataProcessingEngine();
    input_file >> data_engine;

    naivebayes::ProbabilityModel probability_model = naivebayes::ProbabilityModel(data_engine.image_map());

    probability_model.WriteJsonOutputFile("test_output_probability_model");
  }

  return 0;
}
