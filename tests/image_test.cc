//
// Created by Vansh Sikka on 4/6/21.
//

#include <core/image.h>

#include <catch2/catch.hpp>
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