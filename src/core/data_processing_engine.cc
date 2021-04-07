#include <core/data_processing_engine.h>

#include <istream>
#include <sstream>

namespace naivebayes {

using std::map;
using std::vector;
using std::string;

std::istream& operator>>(std::istream& is,
                         naivebayes::DataProcessingEngine& data_engine) {

  data_engine.GenerateImageMap(is);
  return is;

}

void DataProcessingEngine::GenerateImageMap(std::istream& is) {
  string string_line;
  vector<string> line_image;
  size_t prev_label;

  while (getline(is, string_line)) {

    if (std::isdigit(string_line[0])) {

      if (line_image.empty()) {
        prev_label = ConvertStringToSizeT(string_line);
        continue;
      }

      naivebayes::Image image(line_image, prev_label);
      image_map_[prev_label].push_back(image);
      prev_label = ConvertStringToSizeT(string_line);
      line_image.clear();
      continue;
    }

    line_image.push_back(string_line);

  }

  //Ensures that the map is not off by one and gathers the last image.
  if (!(line_image.empty())) {
    naivebayes::Image image(line_image, prev_label);
    image_map_[prev_label].push_back(image);
  }
}

size_t DataProcessingEngine::ConvertStringToSizeT(const string& string) {

  size_t label;
  std::stringstream ss(string);
  ss >> label;

  return label;
}

map<size_t, vector<naivebayes::Image>>& DataProcessingEngine::image_map() {
  return image_map_;
}

}  // namespace naivebayes