//
// Created by Vansh Sikka on 4/1/21.
//

#include "core/image.h"
namespace naivebayes {

Image::Image(const vector<string>& image_vector, size_t class_label) {
  image_string_vector_ = image_vector;
  image_class_label_ = class_label;
}

std::ostream& operator<<(std::ostream& os, const Image& image) {

  for (std::string value : image.image_string_vector_) {
    os << value << "\n";
  }

  return os;
}

const vector<string>& Image::image_string_vector() const {
  return image_string_vector_;
}

vector<string>& Image::image_string_vector() {
  return image_string_vector_;
}

size_t Image::image_class_label() {
  return image_class_label_;
}
Image::Image() {}
}  // namespace naivebayes