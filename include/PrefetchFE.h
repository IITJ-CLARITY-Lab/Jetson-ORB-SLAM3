// PrefetchFE — pipelined front-end state (env PIPELINE_FE).
// Overlaps ORB extraction of frame t with tracking of frame t-1 (one frame of
// output latency; per-pose timestamps unchanged, so trajectories/ATE are
// unaffected). The async worker uses its OWN CPU extractor pair: the detector
// family is identical, the pyramids stay valid for the constructor's stereo
// matching, and there is no contention with the GPU extractor.
// Mutually exclusive with ADAPTIVE_FE (the worker's extractors are fixed).
#ifndef PREFETCH_FE_H
#define PREFETCH_FE_H
#include <opencv2/opencv.hpp>
#include <future>
#include <vector>
namespace ORB_SLAM3 {
class ORBextractor;
class PrefetchFE {
public:
    static PrefetchFE& I(){ static PrefetchFE s; return s; }

    // worker extractor pair (CPU; constructed by Tracking when PIPELINE_FE set)
    ORBextractor* eL = nullptr;
    ORBextractor* eR = nullptr;

    // fill bank — written by the async worker for the pending frame
    std::vector<cv::KeyPoint> kL, kR; cv::Mat dL, dR; int monoL = 0, monoR = 0;

    // consume bank — read once by Frame::ExtractORB for the frame being tracked
    bool hasL = false, hasR = false;
    std::vector<cv::KeyPoint> cKL, cKR; cv::Mat cDL, cDR; int cMonoL = 0, cMonoR = 0;

    // pending images (arrived this call; tracked next call)
    bool pendingValid = false;
    cv::Mat pimL, pimR; double pts = 0;

    std::future<void> fut;
    bool armed = false;   // launch worker after the current Frame is constructed

    void promote(){
        cKL = std::move(kL); cKR = std::move(kR);
        cDL = dL; cDR = dR; cMonoL = monoL; cMonoR = monoR;
        hasL = true; hasR = true;
    }
    void takeL(std::vector<cv::KeyPoint>& k, cv::Mat& d, int& m){ k = std::move(cKL); d = cDL; m = cMonoL; hasL = false; }
    void takeR(std::vector<cv::KeyPoint>& k, cv::Mat& d, int& m){ k = std::move(cKR); d = cDR; m = cMonoR; hasR = false; }
};
}
#endif
