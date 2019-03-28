#include "ExtractContours.h"

#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat computeMahalanobis(const Mat &img, std::vector<cv::Point> contour, Mat &cov) {
  cv::Point minX, maxX, minY, maxY;

  uint width = img.size().width;
  uint height = img.size().height;

  minX.x = 0;
  maxX.x = width - 1;
  minY.y = 0;
  maxY.y = height - 1;
  
  Vec2d mup = Vec2d(0., 0.);
  int count = 0;
  for (uint i = 0; i < img.rows; ++i)
    for (uint j = 0; j < img.cols; ++j) {
      if (img.at<uchar>(i, j)) {
        mup += Vec2d(static_cast<double>(j), static_cast<double>(i));
        ++count;
      }
    }
  mup /= count;
    
  Mat samples(0, 2, CV_64F);
  for (uint i = minY.y; i < maxY.y; ++i)
    for (uint j = minX.x; j < maxX.x; ++j)
      if (img.at<uchar>(i, j)) {
        Mat row(1, 2, CV_64F);
        row.at<double>(0, 0) = (static_cast<double>(j) - mup[0]) / 1.;
        row.at<double>(0, 1) = (static_cast<double>(i) - mup[1]) / 1.;
        samples.push_back(row);
        ++count;
      } 
  Mat mu;
  Matx22d covp;
  cv::calcCovarMatrix(samples, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
  cov = cov / samples.rows;
  Mat gaussImg = Mat::zeros(img.size(), CV_64F);
  for (uint i = 0; i < img.rows; ++i)
    for (uint j = 0; j < img.cols; ++j)
      if (img.at<uchar>(i, j))
        // consider this or exponential function
        // consider scaling x, y to [-1, 1], in order to reduce elements of cov matrix
        gaussImg.at<double>(i, j) = (cv::Mahalanobis(Vec2d(static_cast<double>(j) / 1. - mup[0], static_cast<double>(i) / 1. - mup[1]), Vec2d(mu), cov));
        
  return gaussImg;
  
}

double identity(double val) {
  return val;
}

double x2(double val) {
  return val * val;
}

double x3(double val) {
  return pow(val, 3.);
}

double x5(double val) {
  return pow(val, 5.);
}

double x1_3(double val) {
  return pow(val, 1./3);
}

double x1_5(double val) {
  return pow(val, 1./5);
}

double xSinx(double val) {
  return val * sin(val);
}

double xCosx(double val) {
  return val * cos(val);
}

double x2Sinx(double val) {
  return val * val * sin(val);
}

double x2Cosx(double val) {
  return val * val * cos(val);
}

double xSin2(double val) {
  return val * sin(val) * sin(val);
}

double xCos2(double val) {
  return val * cos(val) * cos(val);
}

double cos2(double val) {
  return cos(val) * cos(val);
}

double sin2(double val) {
  return sin(val) * sin(val);
}

double cos2x(double val) {
  return cos(2 * val);
}

double sin2x(double val) {
  return sin(2 * val);
}

double cosSin(double val) {
  return cos(val) * sin(val);
}

double cosx2(double val) {
  return cos(val * val);
}

double sinx2(double val) {
  return sin(val * val);
}

int f_test_area(unsigned dim, const double *x, void *data, unsigned fdim, double *retval) {
      // firstly load image into gaussImg, should be passed via *data
      cv::Mat img = *reinterpret_cast<cv::Mat*>(data);
      retval[0] = img.at<double>(x[1], x[0]) / 255.;
      return 0;

}

int f_test(unsigned dim, const double *x, void *data, unsigned fdim, double *retval) {
  cv::Mat img = *reinterpret_cast<cv::Mat *>(data);
  retval[0] = img.at<double>(x[1], x[0]);
  retval[1] = img.at<double>(x[1], x[0]) * (x[0] - 1224.);
  retval[2] = img.at<double>(x[1], x[0]) * (x[1] - 1025.);
  return 0;
}

void calculateIntegral(const Mat &img, std::vector<cv::Point> &contour, double (*func)(double), const Mat &gaussImg, 
    double *integrals) {
  cv::Point minX, maxX, minY, maxY;
  uint width = img.size().width;
  uint height = img.size().height;

  minX.x = 0;
  maxX.x = width - 1;
  minY.y = 0;
  maxY.y = height - 1;

  double dxy = 1. / (width * height);
  integrals[0] = 0.;
  integrals[1] = 0.;
  integrals[2] = 0.;
  // this is to compare with contourArea, remember to remove afterwards
  // 30.08.2018: it's OK to keep it here, the overload is not high
  integrals[3] = 0.;

  Mat cpdGaussImg;
  gaussImg.copyTo(cpdGaussImg);

  for (uint y = minY.y; y <= maxY.y; ++y)
    for (uint x = minX.x; x <= maxX.x; ++x)
      if (img.at<uchar>(y, x)) {
        // these integrals can be better approximated by trapezoidal rule or simpson rule
        integrals[0] += (*func)(gaussImg.at<double>(y, x)) * dxy;
        integrals[1] += (static_cast<double>(x) / 1.- .5 * width) * (*func)(gaussImg.at<double>(y, x)) * dxy;
        integrals[2] += (static_cast<double>(y) / 1.- .5 * height) * (*func)(gaussImg.at<double>(y, x)) * dxy;
        integrals[3] += 1.;
      }

}

cv::Mat binaryRegistration(Mat &refImg, std::vector<cv::Point> &templateContour, 
      Mat &sensedImg, std::vector<cv::Point> &contour) {

  Mat cpdRefImg;
  refImg.copyTo(cpdRefImg);
  
  cv::threshold(refImg, refImg, 129, 255, 0);
  cv::threshold(sensedImg, sensedImg, 129, 255, 0);
  
  Mat refCov;
  Mat gaussRefImg = computeMahalanobis(refImg, templateContour, refCov);
  // imwrite doesn't support floating point write
  cv::normalize(gaussRefImg, gaussRefImg, 20., 0., NORM_INF, CV_64F);
  Mat sensedCov;
  Mat gaussSensedImg = computeMahalanobis(sensedImg, contour, sensedCov);
  cv::normalize(gaussSensedImg, gaussSensedImg, 20., 0., NORM_INF, CV_64F);
    
  double jacobian_by_sigma_ratio = sqrt(cv::determinant(refCov) / cv::determinant(sensedCov));

  const int NUMBER_VARIANT_FUNCTIONS = 10;
  double refInts[NUMBER_VARIANT_FUNCTIONS][4];
  double sensedInts[NUMBER_VARIANT_FUNCTIONS][4];
  
  int i = 0;
  calculateIntegral(refImg, templateContour, cosSin, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, cosSin, gaussSensedImg, sensedInts[i++]);

  calculateIntegral(refImg, templateContour, cos2, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, cos2, gaussSensedImg, sensedInts[i++]);
  
  calculateIntegral(refImg, templateContour, sin2, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, sin2, gaussSensedImg, sensedInts[i++]);

  calculateIntegral(refImg, templateContour, cos, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, cos, gaussSensedImg, sensedInts[i++]);  
  
  calculateIntegral(refImg, templateContour, sin, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, sin, gaussSensedImg, sensedInts[i++]);

  calculateIntegral(refImg, templateContour, cos2x, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, cos2x, gaussSensedImg, sensedInts[i++]);
  
  //calculateIntegral(refImg, templateContour, sin2x, gaussRefImg, refInts[i]);
  //calculateIntegral(sensedImg, contour, sin2x, gaussSensedImg, sensedInts[i++]);
  
  calculateIntegral(refImg, templateContour, cosx2, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, cosx2, gaussSensedImg, sensedInts[i++]);
  
  calculateIntegral(refImg, templateContour, sinx2, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, sinx2, gaussSensedImg, sensedInts[i++]);

  calculateIntegral(refImg, templateContour, log, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, log, gaussSensedImg, sensedInts[i++]);
  
  calculateIntegral(refImg, templateContour, identity, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, identity, gaussSensedImg, sensedInts[i++]);
  /*/
  calculateIntegral(refImg, templateContour, x3, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, x3, gaussSensedImg, sensedInts[i++]);
  
  calculateIntegral(refImg, templateContour, x5, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, x5, gaussSensedImg, sensedInts[i++]);
  
  calculateIntegral(refImg, templateContour, x1_3, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, x1_3, gaussSensedImg, sensedInts[i++]);
  
  calculateIntegral(refImg, templateContour, x1_5, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, x1_5, gaussSensedImg, sensedInts[i++]);
  //*/
  // SVD computation gives better result than weighted least square and they take nearly the same amount of time  
  Mat points(0, 2, CV_32F);
  Mat pointsRow(1, 2, CV_32F);
  for (int i = 0; i < NUMBER_VARIANT_FUNCTIONS; ++i) {
    pointsRow.at<float>(0, 0) = refInts[i][0];
    pointsRow.at<float>(0, 1) = sensedInts[i][0];
    points.push_back(pointsRow);
  }
  Mat line;
  cv::fitLine(points, line, CV_DIST_HUBER, 1., .00, 0.);
  // We can either use jacobian, jacobian_by_svd, jacobian_by_sigma_ratio
  double jacobian = line.at<float>(0, 0) / line.at<float>(1, 0);
  Mat w, u, vt;
  cv::SVD::compute(points.t() * points, w, u, vt);
  double jacobian_by_svd = std::abs(u.at<double>(0, 0) / u.at<double>(0, 1));
    
  Mat inputMat(0, 6, CV_64F);
  Mat row(1, 6, CV_64F);

  for (int i = 0; i < NUMBER_VARIANT_FUNCTIONS; ++i) {
    row.at<double>(0, 0) = sensedInts[i][1];
    row.at<double>(0, 1) = sensedInts[i][2];
    row.at<double>(0, 2) = sensedInts[i][0];
    row.at<double>(0, 3) = 0.;
    row.at<double>(0, 4) = 0.;
    row.at<double>(0, 5) = 0.;
    inputMat.push_back(row);

    row.at<double>(0, 0) = 0.;
    row.at<double>(0, 1) = 0.;
    row.at<double>(0, 2) = 0.;
    row.at<double>(0, 3) = sensedInts[i][1];
    row.at<double>(0, 4) = sensedInts[i][2];
    row.at<double>(0, 5) = sensedInts[i][0];
    inputMat.push_back(row);
  }
      
  Mat sol;
  // could have been solved by over-determined system
  Mat argVec(2 * NUMBER_VARIANT_FUNCTIONS, 1, CV_64F);
  for (int i = 0; i < NUMBER_VARIANT_FUNCTIONS; ++i) {
    argVec.at<double>(2 * i, 0) = refInts[i][1];
    argVec.at<double>(2 * i + 1, 0) = refInts[i][2];
  }
  cv::solve(inputMat, jacobian_by_sigma_ratio * argVec, sol, DECOMP_SVD);
  
//[x' - x_center  = A * [(x - x_center)  + [Tx
// y' - y_center]        (y - y_center)]    Ty]
  Mat AMat(Matx22d(sol.at<double>(0, 0), sol.at<double>(1, 0), sol.at<double>(3, 0), sol.at<double>(4, 0)).inv());
  Mat affineMat(2, 3, CV_64F);
  AMat.col(0).copyTo(affineMat.col(0));
  AMat.col(1).copyTo(affineMat.col(1));
  Mat bMat(Matx21d(-sol.at<double>(2, 0) - 0.5 * refImg.cols, -sol.at<double>(5, 0) - 0.5 * refImg.rows));
  Mat TMat(AMat * bMat + (Mat_<double>(2, 1) << 0.5 * refImg.cols, 0.5 * refImg.rows));
  TMat.col(0).copyTo(affineMat.col(2));
  cv::warpAffine(cpdRefImg, sensedImg, affineMat, refImg.size(), cv::INTER_LINEAR);
  
  return affineMat;
}

double evaluate(cv::Mat &refImg, std::vector<cv::Point> &contour, cv::Mat &fullAffineMat, cv::Mat &fullAffineHatMat) {
  cv::Point minX, maxX, minY, maxY;
  findMinMaxCoords(contour, minX, maxX, minY, maxY);
  
  double sum = 0.;
  uint count = 0;
  for (int i = minY.y; i <= maxY.y; ++i)
    for (int j = minX.x; j <= maxX.x; ++j)
      if (refImg.at<double>(i, j)) {
        Mat pix_vec(Matx31d(j, i, 1.));
        sum += cv::norm((fullAffineMat - fullAffineHatMat) * pix_vec) / cv::norm(fullAffineMat * pix_vec);
        ++count;
      }

  return sum / count;
}

int main(int argc, char** argv)
{

  /// Load source image and convert it to gray
  cv::Mat thresholded_img = imread(argv[2], IMREAD_GRAYSCALE);
  cv::Mat canny_output = imread(argv[3], IMREAD_GRAYSCALE);
  
  std::vector<cv::Point> templateContour = extractTemplateContours(canny_output, "template_contours_img_whole_gear.jpg");
  // Here I create test image by doing transformation in openCV, be aware that Toolip conversion from RGB to grayscale gives different result than openCV
  double alpha;
  std::istringstream iss(argv[1]);
  iss>>alpha;
  cv::Mat fullAffineMat(cv::Matx33d(1., 0., .5*canny_output.cols, 0., 1., .5*canny_output.rows, 0., 0., 1.) * cv::Matx33d(cos(alpha), -sin(alpha), 0., sin(alpha), cos(alpha), 0., 0., 0., 1.) * cv::Matx33d(1., 0., -.5*canny_output.cols, 0., 1., -.5*canny_output.rows, 0., 0., 1.));
  cv::Mat affineMat(2, 3, CV_64F);
  fullAffineMat.row(0).copyTo(affineMat.row(0));
  fullAffineMat.row(1).copyTo(affineMat.row(1));

  cv::Mat thresholded_img_rotated;
  cv::Mat canny_output_rotated;
  cv::warpAffine(thresholded_img, thresholded_img_rotated, affineMat, thresholded_img.size());
  imwrite("thresholded_img_rotated.jpg", thresholded_img_rotated);
  cv::warpAffine(canny_output, canny_output_rotated, affineMat, canny_output.size());
  imwrite("canny_output_rotated.jpg", canny_output_rotated);
  std::vector<cv::Point> contour = extractTemplateContours(canny_output_rotated, "contours_img.jpg");
  
  /// Load source image and convert it to gray
  Mat sensedImg = imread("contours_img.jpg", IMREAD_GRAYSCALE);
  Mat refImg = imread("template_contours_img_whole_gear.jpg", IMREAD_GRAYSCALE);

  cv::Mat affineHatMat = binaryRegistration(refImg, templateContour, sensedImg, contour);

  cv::imshow("warpedImg", sensedImg);
  cv::imwrite("warpedImg.jpg", sensedImg);

  cv::Mat fullAffineHatMat(3, 3, CV_64F);
  affineHatMat.row(0).copyTo(fullAffineHatMat.row(0));
  affineHatMat.row(1).copyTo(fullAffineHatMat.row(1));
  cv::Mat last_row(cv::Matx13d(0., 0., 1.));
  last_row.row(0).copyTo(fullAffineHatMat.row(2));
  std::cout<<"fullAffineMat: "<<fullAffineMat<<" fullAffineHatMat: "<<fullAffineHatMat<<std::endl;
  std::cout<<"epsilon: "<<evaluate(refImg, templateContour, fullAffineMat, fullAffineHatMat);

  waitKey(0);
  return(0);
}

