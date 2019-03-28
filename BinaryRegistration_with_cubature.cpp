#include "ExtractContours.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include <iostream>
#include <cstdio>
#include <cmath>

//#include "cubature.h"

using namespace cv;
using namespace std;

//*/
void gaussianEstimate(const Mat &img, Vec2d &mu, Matx22d &cov) {
  uint width = img.size().width;
  uint height = img.size().height;

  mu = Vec2d(0., 0.);
  int count = 0;
  for (uint i = 0; i < img.rows; ++i)
    for (uint j = 0; j < img.cols; ++j) {
      if (img.at<uchar>(i, j)) {
        mu += Vec2d(static_cast<double>(j) / width - .5, static_cast<double>(i) / height - .5);
        ++count;
      }
    }
  mu /= count;

  
  cov = Matx22d(0., 0., 0., 0.);
  for (uint i = 0; i < img.rows; ++i)
    for (uint j = 0; j < img.cols; ++j) {
      if (img.at<uchar>(i, j)) {
        Vec2d x(static_cast<double>(j) / width - .5 - mu[0], static_cast<double>(i) / height - .5 - mu[1]);
        cov += x * x.t();
        //if (i<500&&j<500) std::cout<<"x:"<<x<<" cov:"<<cov<<"\n";
      }
    }
    
  cov *= 1./count;
}//*/

Mat computeMahalanobis(const Mat &img, std::vector<cv::Point> contour, Mat &cov) {
  cv::Point minX, maxX, minY, maxY;
  findMinMaxCoords(contour, minX, maxX, minY, maxY);

  // we do gain more foreground pixels
  /*/
  minX.x -= 5;
  maxX.x += 5;
  minY.y -= 5;
  maxY.y += 5;
  //*/
  uint width = img.size().width;
  uint height = img.size().height;
  
  Mat samples(0, 2, CV_64F);
  for (uint i = minY.y; i < maxY.y; ++i)
    for (uint j = minX.x; j < maxX.x; ++j)
      if (img.at<uchar>(i, j)) {
        Mat row(1, 2, CV_64F);
        row.at<double>(0, 0) = static_cast<double>(j) / width - .5;
        row.at<double>(0, 1) = static_cast<double>(i) / height - .5;
        samples.push_back(row);
      } 
  Mat mu;
  Vec2d mup;
  Matx22d covp;
  //gaussianEstimate(img, mup, covp);
  //std::cout<<"My GaussImg: mu="<<mup<<std::endl<<" cov="<<covp<<std::endl;
  cv::calcCovarMatrix(samples, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
  cov = cov / samples.rows;
  std::cout<<"GaussImg: mu="<<mu<<std::endl<<" cov="<<cov<<std::endl;
  Mat gaussImg = Mat::zeros(img.size(), CV_64F);
  for (uint i = 0; i < img.rows; ++i)
    for (uint j = 0; j < img.cols; ++j)
      if (img.at<uchar>(i, j))
        // consider this or exponential function
        // consider scaling x, y to [-1, 1], in order to reduce elements of cov matrix
        // verify Sigma' = AT * Sigma * A
        // accuracy mostly depends at this point and choices of variant functions
        gaussImg.at<double>(i, j) = //1. / (2 * M_PI * sqrt(cv::determinant(cov))) * 
            (cv::Mahalanobis(Vec2d(static_cast<double>(j) / width - .5, static_cast<double>(i) / height - .5), Vec2d(mu), cov));
        
  //circle( gaussImg, Point2f(mu.at<double>(0, 0), mu.at<double>(0, 1)), 4, Scalar(255, 0, 0), -1, 8, 0 );
  //imshow("bivariate normal distr in ref image", gaussImg);
  //waitKey(0);
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
      cv::Mat img = *(cv::Mat *) data;
      retval[0] = img.at<uchar>(x[1], x[0]) / 255.;
      return 0;

}

int f_test(unsigned dim, const double *x, void *data, unsigned fdim, double *retval) {
  cv::Mat img = *(cv::Mat *) data;
  retval[0] = img.at<double>(x[1], x[0]);
  retval[1] = img.at<double>(x[1], x[0]) * (x[0] - 128.);
  retval[2] = img.at<double>(x[1], x[0]) * (x[1] - 128.);
  return 0;
}

void calculateIntegral(const Mat &img, std::vector<cv::Point> &contour, double (*func)(double), const Mat &gaussImg, 
    double *integrals) {
  cv::Point minX, maxX, minY, maxY;
  //findMinMaxCoords(contour, minX, maxX, minY, maxY);
  // we do gain more foreground pixels
  /*/
  minX.x -= 5;
  maxX.x += 5;
  minY.y -= 5;
  maxY.y += 5;
  //*/
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
  integrals[3] = 0.;

  Mat cpdGaussImg(gaussImg.size(), CV_64F);
  for (uint y = minY.y; y <= maxY.y; ++y)
    for (uint x = minX.x; x <= maxX.x; ++x)
      if (img.at<uchar>(y, x)) {
      #if USE_CUBATURE == 0
        // these integrals can be better approximated by trapezoidal rule or simpson rule
        integrals[0] += (*func)(gaussImg.at<double>(y, x)) * dxy;
        integrals[1] += (static_cast<double>(x) / width - .5) * (*func)(gaussImg.at<double>(y, x)) * dxy;
        integrals[2] += (static_cast<double>(y) / height - .5) * (*func)(gaussImg.at<double>(y, x)) * dxy;
        integrals[3] += 1.;
      #else
        cpdGaussImg.at<double>(y, x) = (*func)(gaussImg.at<double>(y, x));
      #endif
      }
      #if USE_CUBATURE == 1
      else 
        cpdGaussImg.at<double>(y, x) = 0.;
      #endif
        
  #if USE_CUBATURE == 1
  double xmin[2];
  xmin[0] = minX.x;
  xmin[1] = minY.y;
  double xmax[2];
  xmax[0] = maxX.x;
  xmax[1] = maxY.y;
  double val[3];
  double err[3];
  pcubature(3, f_test, &cpdGaussImg, 2, xmin, xmax, 0, 0, 5e-5, ERROR_INDIVIDUAL, val, err);
  integrals[0] = val[0];
  integrals[1] = val[1];
  integrals[2] = val[2];
  Mat cpdImg;
  img.copyTo(cpdImg);
  pcubature(1, f_test_area, &cpdImg, 2, xmin, xmax, 0, 0, 5e-5, ERROR_INDIVIDUAL, val, err);
  integrals[3] = val[0];
  #endif
  
}

void binaryRegistration(Mat &refImg, std::vector<cv::Point> &templateContour, 
      Mat &sensedImg, std::vector<cv::Point> &contour) {

  cv::threshold(refImg, refImg, 200, 255, 0);
  cv::threshold(sensedImg, sensedImg, 200, 255, 0);

  Mat refCov;
  Mat gaussRefImg = computeMahalanobis(refImg, templateContour, refCov);
  //Mat gaussRefImg;
  //refImg.convertTo(gaussRefImg, CV_64F);
  // imwrite doesn't support floating point write
  cv::normalize(gaussRefImg, gaussRefImg, 0, 255, NORM_MINMAX, CV_8U);
  imwrite("gaussRefImg.jpg", gaussRefImg);
  Mat sensedCov;
  Mat gaussSensedImg = computeMahalanobis(sensedImg, contour, sensedCov);
  //Mat gaussSensedImg;
  //sensedImg.convertTo(gaussSensedImg, CV_64F);
  cv::normalize(gaussSensedImg, gaussSensedImg, 0, 255, NORM_MINMAX, CV_8U);  
  imwrite("gaussSensedImg.jpg", gaussSensedImg);
    
  //*/
  //double jacobian = sqrt(cv::determinant(refCov) / cv::determinant(sensedCov));
  //std::cout<<"jacobian: "<<jacobian<<std::endl;

  const int NUMBER_VARIANT_FUNCTIONS = 11;
  double refInts[NUMBER_VARIANT_FUNCTIONS][4];
  double sensedInts[NUMBER_VARIANT_FUNCTIONS][4];
  
  Mat cpdRefImg;
  refImg.copyTo(cpdRefImg);
  for (uint y = 0; y < refImg.rows; ++y)
    for (uint x = 0; x < refImg.cols; ++x)
      if (refImg.at<uchar>(y, x)) {
        refImg.at<uchar>(y, x) = 255;
      } else {
        refImg.at<uchar>(y, x) = 0;
      }
      cv::imshow("ref img before being passed", refImg);
  for (uint y = 0; y < sensedImg.rows; ++y)
    for (uint x = 0; x < sensedImg.cols; ++x)
      if (sensedImg.at<uchar>(y, x))
        sensedImg.at<uchar>(y, x) = 255;
      else
        sensedImg.at<uchar>(y, x) = 0;
      cv::imshow("sensed img before being passed", sensedImg);
  // currently (sin, cos2, sin2) gives near to decent accuracy
  int i = 0;
  //*/
  calculateIntegral(refImg, templateContour, cosSin, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, cosSin, gaussSensedImg, sensedInts[i]);

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
  
  calculateIntegral(refImg, templateContour, sin2x, gaussRefImg, refInts[i]);
  calculateIntegral(sensedImg, contour, sin2x, gaussSensedImg, sensedInts[i++]);
  
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
  // sigmaA / sigmaB yields |A| = 1.16 which is better than result from linear regression
  
  Mat points(0, 2, CV_64F);
  Mat pointsRow(1, 2, CV_64F);
  for (int i = 0; i < NUMBER_VARIANT_FUNCTIONS; ++i) {
    pointsRow.at<double>(0, 0) = refInts[i][0];
    pointsRow.at<double>(0, 1) = sensedInts[i][0];
    points.push_back(pointsRow);
  }
  //Mat line;
  //cv::fitLine(points, line, CV_DIST_HUBER, 1., .00, 0.);
  //double jacobian = line.at<float>(0, 0) / line.at<float>(1, 0);
  //std::cout<<"jacobian by weighted least square:"<<jacobian<<std::endl;
  Mat w, u, vt;
  cv::SVD::compute(points.t() * points, w, u, vt);
  double jacobian = std::abs(u.at<double>(0, 0) / u.at<double>(0, 1));
  std::cout<<"jacobian by svd:"<<jacobian<<std::endl;
    
  //std::cout<<"contourArea in refImg: "<<contourArea(templateContour)<<std::endl<<"contourArea in sensedImg: "<<contourArea(contour)<<std::endl
    std::cout<<"roughArea in refImg: "<<refInts[0][3]<<std::endl<<"roughArea in sensedImg: "<<sensedInts[0][3]<<std::endl;
  std::cout<<"integral of variant functions applied on refImg: "<<refInts[10][0]<<" "<<refInts[3][0]<<" "<<refInts[4][0]<<" "<<refInts[5][0]<<" "<<refInts[6][0]<<std::endl
    <<"integral of variant functions applied on sensedImg: "<<sensedInts[10][0]<<" "<<sensedInts[3][0]<<" "<<sensedInts[4][0]<<" "<<sensedInts[5][0]<<" "<<sensedInts[6][0]<<std::endl;
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
  }std::cout<<"inputMat: "<<inputMat<<std::endl;
      
  Mat sol;
  // could have been solved by over-determined system
  Mat argVec(2 * NUMBER_VARIANT_FUNCTIONS, 1, CV_64F);
  for (int i = 0; i < NUMBER_VARIANT_FUNCTIONS; ++i) {
    argVec.at<double>(2 * i, 0) = refInts[i][1];
    argVec.at<double>(2 * i + 1, 0) = refInts[i][2];
  }
  cv::solve(inputMat, jacobian * argVec, sol, DECOMP_SVD);std::cout<<"argVec: "<<jacobian * argVec<<std::endl;
  std::cout<<"sol:"<<sol<<std::endl;
  
//[x' - x_center  = A * [(x - x_center)  + [Tx
// y' - y_center]        (y - y_center)]    Ty]
  Mat AMat(Matx22d(sol.at<double>(0, 0), sol.at<double>(1, 0), sol.at<double>(3, 0), sol.at<double>(4, 0)).inv());
  Mat affineMat(2, 3, CV_64F);
  AMat.col(0).copyTo(affineMat.col(0));
  AMat.col(1).copyTo(affineMat.col(1));
  Mat bMat(-Matx21d(sol.at<double>(2, 0) - 0.5 * refImg.cols, sol.at<double>(5, 0) - 0.5 * refImg.rows));
  Mat TMat(AMat * bMat + (Mat_<double>(2, 1) << 0.5 * refImg.cols, 0.5 * refImg.rows));
  TMat.col(0).copyTo(affineMat.col(2));
  std::cout<<"affineMat: "<<affineMat<<std::endl;
  cv::warpAffine(cpdRefImg, sensedImg, affineMat, refImg.size(), cv::INTER_LINEAR);
  cv::imshow("warpedImg", sensedImg);
}

int main(int, char** argv)
{

  return(0);
}

