//
// Created by Vansh Sikka on 4/6/21.
//

#include <core/data_processing_engine.h>
#include <core/probability_computation_engine.h>

#include <catch2/catch.hpp>
#include <fstream>

using std::pair;
using std::map;

TEST_CASE("Test Calculate Class Probability") {
  std::ifstream input_file(
      "/Users/vanshsikka/Documents/CS126/Cinder/my_projects/"
      "naive-bayes-vsikka2/tests/data/testtrainingimagesandlabels.txt");

  naivebayes::DataProcessingEngine test_data_engine = naivebayes::DataProcessingEngine();
  input_file >> test_data_engine;

  map<size_t, vector<naivebayes::Image>> image_map = test_data_engine.image_map();

  SECTION("Test 0 Class Probability") {
    float class_0_probability =
        ProbabilityComputationEngine::CalculateClassProbability(0, image_map);

    REQUIRE(class_0_probability == Approx(log((1.0f + 1.0f)/ (2.0f + 3.0f))));
  }

  SECTION("Test 1 Class Probability") {
    float class_1_probability =
        ProbabilityComputationEngine::CalculateClassProbability(1, image_map);

    REQUIRE(class_1_probability == Approx(log((1.0f + 2.0f)/ (2.0f + 3.0f))));
  }

  SECTION("Test Invalid Class") {
    REQUIRE_THROWS_AS(
        ProbabilityComputationEngine::CalculateClassProbability(3, image_map),std::invalid_argument);
  }
  SECTION("Test Empty Map") {
    std::map<size_t, std::vector<naivebayes::Image>> empty_map;

    REQUIRE_THROWS_AS(
        ProbabilityComputationEngine::CalculateClassProbability(0, empty_map),std::invalid_argument);
  }
}

TEST_CASE("Test Calculate Total Number of Images") {
  std::ifstream input_file(
      "/Users/vanshsikka/Documents/CS126/Cinder/my_projects/"
      "naive-bayes-vsikka2/tests/data/testtrainingimagesandlabels.txt");

  naivebayes::DataProcessingEngine test_data_engine = naivebayes::DataProcessingEngine();
  input_file >> test_data_engine;

  map<size_t, vector<naivebayes::Image>> image_map = test_data_engine.image_map();

  SECTION("Test Valid Calculation") {
    size_t total = ProbabilityComputationEngine::CalculateTotalNumberOfImages(image_map);

    REQUIRE(total == 3);
  }
  SECTION("Empty Image Map") {
    map<size_t, std::vector<naivebayes::Image>> empty_map;

    size_t total = ProbabilityComputationEngine::CalculateTotalNumberOfImages(empty_map);

    REQUIRE(total == 0);
  }
}

TEST_CASE("Test Pixel Probability") {
  std::ifstream input_file(
      "/Users/vanshsikka/Documents/CS126/Cinder/my_projects/"
      "naive-bayes-vsikka2/tests/data/testtrainingimagesandlabels.txt");

  naivebayes::DataProcessingEngine test_data_engine = naivebayes::DataProcessingEngine();
  input_file >> test_data_engine;

  map<size_t, vector<naivebayes::Image>> image_map = test_data_engine.image_map();

  SECTION("Invalid Class Label") {
    REQUIRE_THROWS_AS(
        ProbabilityComputationEngine::CalculatePixelProbability(3, image_map, pair<size_t, size_t>(0, 0)),
        std::invalid_argument);
  }

  SECTION("Invalid Coordinate Point") {
    REQUIRE_THROWS_AS(ProbabilityComputationEngine::CalculatePixelProbability(
                          0, image_map, pair<size_t, size_t>(10, 10)),
                      std::out_of_range);
  }

  SECTION("Empty Image Map") {
    map<size_t, std::vector<naivebayes::Image>> empty_map;
    REQUIRE_THROWS_AS(ProbabilityComputationEngine::CalculatePixelProbability(
                          0, empty_map, pair<size_t, size_t>(0, 0)),
                      std::invalid_argument);
  }

  SECTION("Calculate Pixel with class 1 (2 Images)") {
    SECTION("Probability at (0,0)", "Both Unshaded") {
      pair<float, float> probability_1_00 =
          ProbabilityComputationEngine::CalculatePixelProbability(
              1, image_map, pair<size_t, size_t>(0, 0));

      REQUIRE(probability_1_00.first == Approx(log((1.0f) / (2.0f + 2.0f))));
      REQUIRE(probability_1_00.second == Approx(log((1.0f + 2.0f) / (2.0f + 2.0f))));
    }

    SECTION("Probability at (1,1)", "One Image Shaded Other Unshaded") {
      pair<float, float> probability_1_11 =
          ProbabilityComputationEngine::CalculatePixelProbability(
              1, image_map, pair<size_t, size_t>(1, 1));

      REQUIRE(probability_1_11.first == Approx(log((1.0f + 1.0f) / (2.0f + 2.0f))));
      REQUIRE(probability_1_11.second == Approx(log((1.0f + 1.0f) / (2.0f + 2.0f))));
    }

    SECTION("Probability at (1,2)", "Both Shaded") {
      pair<float, float> probability_1_12 =
          ProbabilityComputationEngine::CalculatePixelProbability(
              1, image_map, pair<size_t, size_t>(1, 2));

      REQUIRE(probability_1_12.first == Approx(log((1.0f + 2.0f) / (2.0f + 2.0f))));
      REQUIRE(probability_1_12.second == Approx(log((1.0f) / (2.0f + 2.0f))));
    }
  }

  SECTION("Calculate Pixel with class 0 (1 image)") {
    SECTION("Probability at (0,0)", "Unshaded Pixel") {
      pair<float, float> probability_0_00 =
          ProbabilityComputationEngine::CalculatePixelProbability(
              0, image_map, pair<size_t, size_t>(0, 0));

      REQUIRE(probability_0_00.first == Approx(log((1.0f) / (2.0f + 1.0f))));
      REQUIRE(probability_0_00.second == Approx(log((1.0f + 1.0f) / (2.0f + 1.0f))));
    }

    SECTION("Probability at (2,3)", "Shaded Pixel") {
      pair<float, float> probability_0_23 =
          ProbabilityComputationEngine::CalculatePixelProbability(
              0, image_map, pair<size_t, size_t>(2, 3));

      REQUIRE(probability_0_23.first == Approx(log((1.0f + 1.0f) / (2.0f + 1.0f))));
      REQUIRE(probability_0_23.second == Approx(log((1.0f) / (2.0f + 1.0f))));
    }
  }
}
