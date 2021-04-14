//
// Created by Vansh Sikka on 4/6/21.
//

#include <core/data_processing_engine.h>
#include <core/probability_model.h>

#include <catch2/catch.hpp>
#include <sstream>

TEST_CASE("Test Probability Model Constructor") {
  std::ifstream input_file(
      "/Users/vanshsikka/Documents/CS126/Cinder/my_projects/"
      "naive-bayes-vsikka2/tests/data/testtrainingimagesandlabels.txt");

  naivebayes::DataProcessingEngine test_data_engine = naivebayes::DataProcessingEngine();
  input_file >> test_data_engine;

  naivebayes::ProbabilityModel probability_model = naivebayes::ProbabilityModel(test_data_engine.image_map());

  SECTION("Generated Probability Model Size Check") {
    REQUIRE(probability_model.pixel_probability_model()[0].size() == 2);
    REQUIRE(probability_model.pixel_probability_model()[1].size() == 2);
    REQUIRE(probability_model.pixel_probability_model()[0][0].size() == 5);
    REQUIRE(probability_model.pixel_probability_model()[0][0][0].size() == 5);
  }

  SECTION("Check Calculations in Class 1 Shaded (2 Images)") {
    std::vector<std::vector<float>> shaded_probability_model_1{
        {1.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f},
        {1.0f/4.0f, 2.0f/4.0f, 3.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f},
        {1.0f/4.0f, 1.0f/4.0f, 3.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f},
        {1.0f/4.0f, 2.0f/4.0f, 3.0f/4.0f, 2.0f/4.0f, 1.0f/4.0f},
        {1.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f}};

    for (size_t i = 0; i < shaded_probability_model_1.size(); i++) {
      for (size_t j = 0; j < shaded_probability_model_1.size(); j++) {
        REQUIRE(
            log(shaded_probability_model_1[i][j]) ==
            Approx(probability_model.pixel_probability_model()[0][1][i][j]));
      }
    }
  }

  SECTION("Check Calculations in Class 0 Unshaded (1 Image)") {
    std::vector<std::vector<float>> unshaded_probability_model_0{
        {2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f},
        {2.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 2.0f/3.0f},
        {2.0f/3.0f, 1.0f/3.0f, 2.0f/3.0f, 1.0f/3.0f, 2.0f/3.0f},
        {2.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 2.0f/3.0f},
        {2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f}};

    for (size_t i = 0; i < unshaded_probability_model_0.size(); i++) {
      for (size_t j = 0; j < unshaded_probability_model_0.size(); j++) {
        REQUIRE(
            log(unshaded_probability_model_0[i][j]) ==
            Approx(probability_model.pixel_probability_model()[1][0][i][j]));
      }
    }
  }
  SECTION("Check Class Probabilities") {
    std::map<size_t, float> class_probabilities;
    class_probabilities[0] = log((1.0f + 1.0f) / (2.0f + 3.0f));
    class_probabilities[1] = log ((1.0f + 2.0f) / (2.0f + 3.0f));

    REQUIRE(probability_model.class_probability_model()[0] ==
            Approx(class_probabilities[0]));
    REQUIRE(probability_model.class_probability_model()[1] ==
            Approx(class_probabilities[1]));
  }
}

TEST_CASE("Read and Write JSON File", "Overridden << Operator") {
  std::ifstream input_file(
      "/Users/vanshsikka/Documents/CS126/Cinder/my_projects/"
      "naive-bayes-vsikka2/tests/data/testtrainingimagesandlabels.txt");

  naivebayes::DataProcessingEngine test_data_engine = naivebayes::DataProcessingEngine();
  input_file >> test_data_engine;

  naivebayes::ProbabilityModel probability_model = naivebayes::ProbabilityModel(test_data_engine.image_map());

  SECTION("Write JSON file") {
    probability_model.WriteJsonOutputFile("/Users/vanshsikka/Documents/"
        "CS126/Cinder/my_projects/naive-bayes-vsikka2/tests/data/test_output_probability_model.json");
  }

  naivebayes::ProbabilityModel new_model = naivebayes::ProbabilityModel();

  std::ifstream json_file(
      "/Users/vanshsikka/Documents/CS126/Cinder/my_projects/naive-bayes-vsikka2/"
      "tests/data/test_output_probability_model.json");

  json_file >> new_model;

  SECTION("Check Correct Class Probability Model") {
    std::map<size_t, float> class_probabilities;
    class_probabilities[0] = log((1.0f + 1.0f) / (2.0f + 3.0f));
    class_probabilities[1] = log ((1.0f + 2.0f) / (2.0f + 3.0f));

    REQUIRE(new_model.class_probability_model()[0] ==
            Approx(class_probabilities[0]));
    REQUIRE(new_model.class_probability_model()[1] ==
            Approx(class_probabilities[1]));

  }
  SECTION("Check if Probability Model 0 Unshaded is Correct") {

    std::vector<std::vector<float>> unshaded_probability_model_0{
        {2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f},
        {2.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 2.0f/3.0f},
        {2.0f/3.0f, 1.0f/3.0f, 2.0f/3.0f, 1.0f/3.0f, 2.0f/3.0f},
        {2.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 2.0f/3.0f},
        {2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f}};

    for (size_t i = 0; i < unshaded_probability_model_0.size(); i++) {
      for (size_t j = 0; j < unshaded_probability_model_0.size(); j++) {
        REQUIRE(log(unshaded_probability_model_0[i][j]) ==
                Approx(new_model.pixel_probability_model()[1][0][i][j]));
      }
    }

  }

  SECTION("Check if Probability Model 1 Shaded is Correct") {
    std::vector<std::vector<float>> shaded_probability_model_1{
        {1.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f},
        {1.0f/4.0f, 2.0f/4.0f, 3.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f},
        {1.0f/4.0f, 1.0f/4.0f, 3.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f},
        {1.0f/4.0f, 2.0f/4.0f, 3.0f/4.0f, 2.0f/4.0f, 1.0f/4.0f},
        {1.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f, 1.0f/4.0f}};

    for (size_t i = 0; i < shaded_probability_model_1.size(); i++) {
      for (size_t j = 0; j < shaded_probability_model_1.size(); j++) {
        REQUIRE(log(shaded_probability_model_1[i][j]) ==
                Approx(new_model.pixel_probability_model()[0][1][i][j]));
      }
    }

  }
}

TEST_CASE("Classifier") {
  std::ifstream test_file ("/Users/vanshsikka/Documents/CS126/Cinder/my_projects/"
      "naive-bayes-vsikka2/tests/data/testtestimagesandlabels.txt");

  naivebayes::DataProcessingEngine data_processing = naivebayes::DataProcessingEngine();
  test_file >> data_processing;

  naivebayes::ProbabilityModel probability_model ("/Users/vanshsikka/Documents/CS126/Cinder/my_projects/"
      "naive-bayes-vsikka2/tests/data/test_output_probability_model.json");

  REQUIRE(probability_model.Classifier(data_processing.image_map()[0][0]) == 0);
  REQUIRE(probability_model.Classifier(data_processing.image_map()[1][0]) == 1);
}

TEST_CASE("Accuracy Test") {
  naivebayes::ProbabilityModel probability_model ("/Users/vanshsikka/Documents/"
      "CS126/Cinder/my_projects/naive-bayes-vsikka2/data/output_probability_model.json");

  REQUIRE(
      probability_model.AccuracyOfClassifier("/Users/vanshsikka/Documents/CS126/Cinder/"
                                             "my_projects/naive-bayes-vsikka2/data/testimagesandlabels.txt") >= 0.7);
}