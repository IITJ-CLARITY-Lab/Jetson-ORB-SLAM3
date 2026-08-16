#ifndef ORIENTATION_GPU_H
#define ORIENTATION_GPU_H
#include <opencv2/core.hpp>
#include <vector>
namespace ORB_SLAM3 {
// Initialize umax table for IC_Angle (call once at startup)
void initOrientationUmax(const std::vector<int>& umax);
// GPU IC_Angle — 100% exact match, 43x speedup
// Computes orientation for all keypoints on a pyramid level
void computeOrientation_GPU(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
}
#endif
