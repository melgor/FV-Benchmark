#ifndef DISTANCES_HPP
#define DISTANCES_HPP
#include "Utils.hpp"

void
chiSquaredDistance(cv::Mat feat1, cv::Mat feat2, cv::Mat& result);

void
cosineDistance(cv::Mat feat1, cv::Mat feat2, cv::Mat& result);

#endif //DISTANCES_HPP