#pragma once

#include <opencv2/core/core.hpp>

cv::Mat find_card(cv::Mat input);
cv::Mat find_target(cv::Mat input, std::string target, cv::Point fov_top, cv::Point fov_down);
bool comp(cv::Point2f a, cv::Point2f b);