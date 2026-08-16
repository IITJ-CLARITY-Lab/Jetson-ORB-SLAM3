#pragma once

#include <vector>
#include <string>
#include <memory>
#include <opencv2/core/core.hpp>

namespace ORB_SLAM3 {

class KeyFrame;

/**
 * CNN-based global place recognition for loop closure.
 *
 * Replaces BoW retrieval with a ResNet50 GAP descriptor (2048-d, L2-normalised).
 * torch/script.h is included only in CNNLoopDetector.cpp to avoid name collisions
 * between LibTorch and g2o/Eigen allocator types in other translation units.
 */
class CNNLoopDetector {
public:
    explicit CNNLoopDetector(const std::string& model_path);
    ~CNNLoopDetector();

    // True only if a TensorRT engine was actually loaded. The constructor does
    // not throw when the engine is missing — it returns leaving the detector
    // inert — so callers must ask rather than assume construction succeeded.
    bool IsReady() const;

    // Extract descriptor for pKF and add it to the database.
    void AddKeyFrame(KeyFrame* pKF);

    // Return up to top_k loop candidates for pKF, excluding the last
    // exclude_recent frames. Returns empty vector if database is too small.
    std::vector<KeyFrame*> DetectLoopCandidates(KeyFrame* pKF,
                                                 int top_k = 5,
                                                 int exclude_recent = 20,
                                                 float min_sim = 0.75f);

    void Reset();

private:
    struct Impl;                   // defined in .cpp — keeps torch headers out of this header
    std::unique_ptr<Impl> mImpl;

    std::vector<std::vector<float>> mDescriptors;
    std::vector<KeyFrame*>          mKeyFrames;
};

} // namespace ORB_SLAM3
