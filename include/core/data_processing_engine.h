#include <istream>
#include <map>
#include <string>
#include <vector>

#include "image.h"

namespace naivebayes {

using std::vector;

class DataProcessingEngine {

 public:

  // Over-rode the input operator, to use input file stream to retrieve the
  // training data set, and then call the GenerateImageMap function which
  // processes the passed input stream.
  friend std::istream& operator>>(std::istream& is,
                                  DataProcessingEngine& data_engine);

  // Fill image_map by going through the whole training data file and sorting
  // data based off of labels and images that are in the txt file.
  void GenerateImageMap(std::istream& is);

  //Converts string to a size_t value.
  static size_t ConvertStringToSizeT(const std::string& string);

  //Getter for image_map_ data structure.
  std::map<size_t, std::vector<naivebayes::Image>>& image_map();

 private:
  std::map<size_t, vector<naivebayes::Image>> image_map_;
};

}  // namespace naivebayes
