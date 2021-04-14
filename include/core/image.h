//
// Created by Vansh Sikka on 4/1/21.
//

#pragma once

#include <iostream>
#include <vector>
#include <utility>

using std::string;
using std::vector;
namespace naivebayes {

class Image {
 public:
  // Constructs a single image object which holds a vector of strings and contains
  // the corresponding class label.
  Image(const vector<string>& image_vector, size_t class_label);
  Image();

  // Overrode the output operator to print the string vector as it would
  // appear in the training data file.
  friend std::ostream& operator << (std::ostream& os, const Image& image);

  //Getter for the image_string_vector.
  vector<string>& image_string_vector();

  const vector<string>& image_string_vector() const;

  //Getter for the image_class_label.
  size_t image_class_label();

 private:
  size_t image_class_label_;
  vector<string> image_string_vector_;
};
}
