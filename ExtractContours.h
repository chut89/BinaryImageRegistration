#include "opencv2/imgproc/imgproc.hpp"

std::vector<cv::Point> extractTemplateContours(cv::Mat &template_canny_output, std::string filename);
bool compare_length(std::vector<cv::Point> &contour1, std::vector<cv::Point> &contour2);
bool compare_x(cv::Point &point1, cv::Point &point2);
bool compare_y(cv::Point &point1, cv::Point &point2);
void findMinMaxCoords(std::vector<cv::Point> &contour, cv::Point &minX, cv::Point &maxX, cv::Point &minY, cv::Point &maxY);
