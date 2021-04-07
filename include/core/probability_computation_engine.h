//
// Created by Vansh Sikka on 4/1/21.
//
#pragma once
#include <map>

#include "image.h"

namespace ProbabilityComputationEngine {

//All constants that are needed to compute the probability.
static const size_t kK = 1;
static const size_t kShadedOrUnshadedV = 2;

static const std::vector<char> kShadedCharacterSet({'+', '#'});
static const std::vector<char> kUnshadedCharacterSet({' '});

// Function utilized to Calculate the Class Probability given a certain label
// and a map from class_label to vector of Image objects.
float CalculateClassProbability(
    size_t class_label,
    std::map<size_t, std::vector<naivebayes::Image>>& images_map);

// Calculates the Probability at a certain Pixel for a given class using the
// map of class labels to a vector of Image objects.
std::pair<float, float> CalculatePixelProbability(
    size_t class_label,
    std::map<size_t, std::vector<naivebayes::Image>>& images_map,
    const std::pair<size_t, size_t>& coordinates);

//Counts the total number of Image objects that are inside the image_map.
float CalculateTotalNumberOfImages(
    std::map<size_t,std::vector<naivebayes::Image>>& images_map);

}  // namespace ProbabilityComputationEngine
