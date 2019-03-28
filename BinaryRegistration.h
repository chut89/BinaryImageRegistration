#include "ExtractContours.h"

#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat computeMahalanobis(const Mat &img, std::vector<cv::Point> contour, Mat &cov);

void calculateIntegral(const Mat &img, std::vector<cv::Point> &contour, 
  double (*func)(double), const Mat &gaussImg, double *integrals);
  
cv::Mat binaryRegistration(Mat &refImg, std::vector<cv::Point> &templateContour, 
      Mat &sensedImg, std::vector<cv::Point> &contour);
