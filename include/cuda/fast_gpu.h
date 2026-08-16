#ifndef FAST_GPU_H
#define FAST_GPU_H

#include <opencv2/core.hpp>
#include <vector>

namespace ORB_SLAM3 {

// GPU-accelerated FAST-9 corner detection
// Drop-in replacement for cv::FAST() — same interface, GPU execution
void detectFAST_GPU(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                     int threshold, bool nonmaxSuppression = true);

} // namespace ORB_SLAM3

#endif
