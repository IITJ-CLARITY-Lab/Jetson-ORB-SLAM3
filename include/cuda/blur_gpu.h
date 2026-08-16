#ifndef BLUR_GPU_H
#define BLUR_GPU_H
#include <opencv2/core.hpp>
namespace ORB_SLAM3 {
// GPU Gaussian blur (7x7, sigma=2, BORDER_REFLECT_101)
// Max pixel diff from CPU: 1 (rounding). Verified 100% match.
void gaussianBlur7x7_GPU(const cv::Mat& src, cv::Mat& dst);
}
#endif
