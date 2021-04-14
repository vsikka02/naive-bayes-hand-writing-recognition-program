//
// Created by Vansh Sikka on 4/6/21.
//

#include <core/data_processing_engine.h>
#include <core/image.h>

#include <catch2/catch.hpp>
#include <fstream>
#include <sstream>

TEST_CASE("Test Constructor") {
  std::vector<std::string> string_image;
  string_image.push_back(" * * ");
  string_image.push_back("* * *");
  string_image.push_back(" * * ");

  naivebayes::Image image = naivebayes::Image(string_image, 0);

  SECTION("Test Vector and Class") {
    REQUIRE(image.image_string_vector().size() == 3);
    REQUIRE(image.image_class_label() == 0);
  }
}

TEST_CASE("Print Override Operator") {
  std::vector<std::string> string_image;
  string_image.push_back(" * * ");
  string_image.push_back("* * *");
  string_image.push_back(" * * ");

  naivebayes::Image image = naivebayes::Image(string_image, 0);

  std::stringstream ss;
  ss << image;

  REQUIRE(ss.str() == " * * \n* * *\n * * \n");
}

TEST_CASE("Test Loading Different Sized Images") {
  naivebayes::DataProcessingEngine data_engine_1;
  std::ifstream three_by_three_images ("/Users/vanshsikka/Documents/CS126/Cinder/my_projects/"
      "naive-bayes-vsikka2/tests/data/testothersizedimages.txt");

  three_by_three_images >> data_engine_1;

  SECTION("3 by 3 Image") {
    std::stringstream ss;
    ss << data_engine_1.image_map()[0][0];
    REQUIRE(ss.str() == ("###\n# #\n###\n"));
  }

  naivebayes::DataProcessingEngine data_engine_2;
  std::ifstream five_by_five_images ("/Users/vanshsikka/Documents/CS126/Cinder/my_projects/"
      "naive-bayes-vsikka2/tests/data/testtrainingimagesandlabels.txt");

  five_by_five_images >> data_engine_2;

  SECTION("5 by 5 Image") {
    std::stringstream ss;
    ss << data_engine_2.image_map()[0][0];
    REQUIRE(ss.str() == ("     \n ### \n # # \n ### \n     \n"));
  }
}
