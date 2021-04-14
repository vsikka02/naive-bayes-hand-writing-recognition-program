#include <core/data_processing_engine.h>
#include <core/probability_model.h>

#include <fstream>
#include <iostream>

int main() {
  bool train_new_probability_model = false;
  bool test_classifier = true;

  if (train_new_probability_model) {
    std::ifstream input_file("../data/trainingimagesandlabels.txt");
    naivebayes::DataProcessingEngine data_engine = naivebayes::DataProcessingEngine();
    input_file >> data_engine;

    naivebayes::ProbabilityModel probability_model = naivebayes::ProbabilityModel(data_engine.image_map());
    probability_model.WriteJsonOutputFile("../data/output_probability_model.json");
  } else {
    std::ifstream json_file("../data/output_probability_model.json");
    naivebayes::ProbabilityModel probability_model = naivebayes::ProbabilityModel();
    json_file >> probability_model;
  }
  if (test_classifier) {
    std::ifstream input_file("../data/testimagesandlabels.txt");
    naivebayes::DataProcessingEngine data_engine = naivebayes::DataProcessingEngine();
    input_file >> data_engine;

    std::ifstream json_file("../data/output_probability_model.json");
    naivebayes::ProbabilityModel probability_model = naivebayes::ProbabilityModel();
    json_file >> probability_model;

    float acc = probability_model.AccuracyOfClassifier("../data/testimagesandlabels.txt");
    std::cout<<acc<<std::endl;
  }

  return 0;
}
