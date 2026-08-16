#ifndef DESCRIPTORS_GPU_H
#define DESCRIPTORS_GPU_H
#include <opencv2/core.hpp>
#include <vector>
namespace ORB_SLAM3 {
// Initialize ORB bit pattern on GPU (call once with bit_pattern_31_ from ORBextractor.cc)
void initDescriptorPattern_GPU(const int* pattern_31);
// GPU ORB descriptor computation — uses same pattern + rotation as CPU
// Input: blurred image, keypoints with angle set
// Output: descriptors (32 bytes per keypoint)
void computeDescriptors_GPU(const cv::Mat& blurred_img,
                             const std::vector<cv::KeyPoint>& keypoints,
                             cv::Mat& descriptors);
}
#endif
