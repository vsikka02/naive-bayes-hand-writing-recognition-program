#include <core/data_processing_engine.h>

#include <catch2/catch.hpp>
#include <fstream>

TEST_CASE("Test Training Data Parsing with Overloaded Operator") {
  std::ifstream input_file(
      "/Users/vanshsikka/Documents/CS126/Cinder/my_projects/"
      "naive-bayes-vsikka2/tests/data/testtrainingimagesandlabels.txt");

  naivebayes::DataProcessingEngine test_data_engine = naivebayes::DataProcessingEngine();

  input_file >> test_data_engine;

  std::map<size_t, vector<naivebayes::Image>> image_map =
      test_data_engine.image_map();

  SECTION("Check Size of Data Structure") {
    REQUIRE(image_map.size() == 2);
    REQUIRE(image_map[0].size() == 1);
    REQUIRE(image_map[1].size() == 2);
  }
  SECTION("Check Images in Image Map") {
    REQUIRE(image_map[0][0].image_string_vector()[0] == ("     "));
    REQUIRE(image_map[0][0].image_string_vector()[3] == (" ### "));
  }
  SECTION("Check Image Map Labels") {
    REQUIRE(image_map[0][0].image_class_label() == 0);
    REQUIRE(image_map[1][0].image_class_label() == 1);
    REQUIRE(image_map[1][1].image_class_label() == 1);
  }
}

TEST_CASE("String to Integer Converter") {
  SECTION("No trailing spaces") {
    size_t converted =
        naivebayes::DataProcessingEngine::ConvertStringToSizeT("5");

    REQUIRE(converted == 5);
  }
  SECTION("With trailing spaces") {
    size_t converted =
        naivebayes::DataProcessingEngine::ConvertStringToSizeT("  10   ");

    REQUIRE(converted == 10);
  }
}
