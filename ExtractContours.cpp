/**
 * @function ExtractContours.cpp
 * @brief Extract single object contour from image
 * @author Tien Ha Chu
 */
 
#include "ExtractContours.h"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

template <class T> bool compare_length(const vector<T> &contour1, const vector<T> &contour2) {
    return arcLength(contour1, true) > arcLength(contour2, true) && contourArea(contour1) > contourArea(contour2);
}

std::vector<cv::Point> extractTemplateContours(cv::Mat &template_canny_output, std::string filename) {
  std::vector<std::vector<cv::Point> > template_contours;
  std::vector<Vec4i> hierarchy;
  findContours( template_canny_output, template_contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0) );

  std::sort(template_contours.begin(), template_contours.end(), compare_length<cv::Point>);
  /// Draw contours
  Mat drawing = Mat::zeros(template_canny_output.size(), CV_8U);
  size_t NUMBER_CONTOURS = 3;  
  for (size_t i = 0; i < NUMBER_CONTOURS; i++)
     {
       if (i>0 && std::abs(contourArea(template_contours[i]) - contourArea(template_contours[i-1])) < .5)
         continue;
       // in case of whole gear
       if (i == 2)
         drawContours(drawing, template_contours, (int)i, Scalar(0), CV_FILLED, 8, hierarchy, 0, Point());
       else
         drawContours(drawing, template_contours, (int)i, Scalar(255), CV_FILLED, 8, hierarchy, 0, Point());
     }
     
  imwrite(filename, drawing);
  return template_contours[0];
    
}

bool compare_x(cv::Point &point1, cv::Point &point2) {
  return point1.x < point2.x;
}

bool compare_y(cv::Point &point1, cv::Point &point2) {
  return point1.y < point2.y;
}

void findMinMaxCoords(std::vector<cv::Point> &contour, cv::Point &minX, cv::Point &maxX, cv::Point &minY, cv::Point &maxY) {
  minX = *std::min_element(contour.begin(), contour.end(), compare_x);
  maxX = *std::max_element(contour.begin(), contour.end(), compare_x);
  minY = *std::min_element(contour.begin(), contour.end(), compare_y);
  maxY = *std::max_element(contour.begin(), contour.end(), compare_y);  
  
}

