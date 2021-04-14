#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "sketchpad.h"
#include "core/probability_model.h"

namespace naivebayes {

namespace visualizer {

/**
 * Allows a user to draw a digit on a sketchpad and uses Naive Bayes to
 * classify it.
 */
class NaiveBayesApp : public ci::app::App {
 public:
  // Constructor for the Naive Bayes Application.
  NaiveBayesApp();

  // The Draw Function is constantly updated and draws the actual sketchpad and
  // shades in the necessary pixels.
  void draw() override;

  // This mouseDown and mouseDrag method are used to edit the image that is
  // linked to the sketchpad and fills the image up with #'s if you are pushing
  // down or dragging your mouse.
  void mouseDown(ci::app::MouseEvent event) override;
  void mouseDrag(ci::app::MouseEvent event) override;

  // This method allows the user to hit delete in order to clear the board or hit
  // enter to create a prediction of what is on the sketchpad.
  void keyDown(ci::app::KeyEvent event) override;

  // Display Constants for Cinder UI.
  const double kWindowSize = 800;
  const double kMargin = 100;
  const size_t kImageDimension = 28;

 private:
  Sketchpad sketchpad_;
  naivebayes::ProbabilityModel probability_model;
  int current_prediction_ = -1;
};

}  // namespace visualizer

}  // namespace naivebayes
