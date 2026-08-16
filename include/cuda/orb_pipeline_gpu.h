#ifndef ORB_PIPELINE_GPU_H
#define ORB_PIPELINE_GPU_H

#include <opencv2/core.hpp>
#include <vector>

namespace ORB_SLAM3 {

void initUmaxTable(const std::vector<int>& umax);
void initBitPattern(const int* pattern);

// Per-level GPU FAST (optimized: async upload, packed download)
void detectFAST_GPU_level(const cv::Mat& level_img,
                           std::vector<cv::KeyPoint>& keypoints,
                           int threshold);

// Batch all levels: handles cell-based threshold fallback
void detectFAST_GPU_allLevels(const std::vector<cv::Mat>& pyramid,
                               std::vector<std::vector<cv::KeyPoint>>& allKeypoints,
                               int iniThreshold, int minThreshold,
                               const std::vector<int>& nFeaturesPerLevel,
                               float cellW = 35.0f);

} // namespace ORB_SLAM3

#endif
