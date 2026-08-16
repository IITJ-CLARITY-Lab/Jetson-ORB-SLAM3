/**
* This file is part of Jetson-ORB-SLAM3, a GPU-accelerated fork of ORB-SLAM3.
*
* Copyright (C) 2026 CLARITY Lab, Indian Institute of Technology Jodhpur
*
* Jetson-ORB-SLAM3 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Jetson-ORB-SLAM3 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

// Native TensorRT inference for CNN loop closure on Jetson Orin Nano.
// ORT CUDA/TRT EPs hang during graph partitioning on Orin ARM (see
// docs/ORIN_LEARNINGS.md); libnvinfer directly gives ~2 ms/query (FP16)
// vs ~396 ms on ORT CPU EP, ~8 ms on a desktop GTX 1060.
//
// Engine is built once with trtexec and loaded as a .trt file:
//     trtexec --onnx=cosplace_r50_512.onnx \
//             --saveEngine=cosplace_r50_512.fp16.trt --fp16
// The .trt file is device- and TRT-version-specific — regenerate on
// each target machine.

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "CNNLoopDetector.h"
#include "KeyFrame.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

namespace ORB_SLAM3 {

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Show warnings and errors only; suppress INFO to keep startup log clean.
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
    }
};

size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
    return v;
}

} // namespace

struct CNNLoopDetector::Impl {
    TrtLogger logger;
    nvinfer1::IRuntime*          runtime{nullptr};
    nvinfer1::ICudaEngine*       engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};
    cudaStream_t stream{nullptr};

    // Device buffers and CPU staging.
    void* d_input{nullptr};
    void* d_output{nullptr};
    size_t input_elts{0};
    size_t output_elts{0};
    std::string input_name;
    std::string output_name;

    std::vector<float> input_h;     // CHW, ImageNet-normalised
    std::vector<float> output_h;    // descriptor

    bool loaded{false};

    static constexpr int kSize = 224;

    std::vector<float> extract(const cv::Mat& imGray) {
        if (!loaded || imGray.empty()) return {};

        const float kMean[3] = {0.485f, 0.456f, 0.406f};
        const float kStd[3]  = {0.229f, 0.224f, 0.225f};

        cv::Mat resized;
        cv::resize(imGray, resized, cv::Size(kSize, kSize));
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
        rgb.convertTo(rgb, CV_32FC3, 1.0f / 255.0f);

        // CHW + ImageNet normalisation into the persistent host buffer.
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < kSize; ++h) {
                for (int w = 0; w < kSize; ++w) {
                    float v = rgb.at<cv::Vec3f>(h, w)[c];
                    input_h[c * kSize * kSize + h * kSize + w] =
                        (v - kMean[c]) / kStd[c];
                }
            }
        }

        // H2D → inference → D2H, all on one stream.
        cudaMemcpyAsync(d_input, input_h.data(),
                        input_elts * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        if (!context->enqueueV3(stream)) {
            std::cerr << "[CNN] enqueueV3 failed" << std::endl;
            return {};
        }

        cudaMemcpyAsync(output_h.data(), d_output,
                        output_elts * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // L2 normalise the descriptor.
        float norm = 0.f;
        for (size_t i = 0; i < output_elts; ++i)
            norm += output_h[i] * output_h[i];
        norm = std::sqrt(norm);
        if (norm > 1e-8f)
            for (size_t i = 0; i < output_elts; ++i)
                output_h[i] /= norm;

        return output_h;  // copy out (small — 512 floats)
    }
};

CNNLoopDetector::CNNLoopDetector(const std::string& model_path)
    : mImpl(std::make_unique<Impl>())
{
    // model_path is the ONNX path in the existing call site
    // (LoopClosing.cc passes "cosplace_r50_512.onnx"). We expect the
    // matching engine alongside it as "<basename>.fp16.trt".
    std::string engine_path = model_path;
    size_t dot = engine_path.find_last_of('.');
    if (dot != std::string::npos) engine_path = engine_path.substr(0, dot);
    engine_path += ".fp16.trt";

    std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cerr << "[CNN] TRT engine not found at " << engine_path
                  << "; run trtexec --onnx=" << model_path
                  << " --saveEngine=" << engine_path << " --fp16" << std::endl;
        return;
    }
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    if (!f.read(buf.data(), size)) {
        std::cerr << "[CNN] Failed to read engine " << engine_path << std::endl;
        return;
    }

    mImpl->runtime = nvinfer1::createInferRuntime(mImpl->logger);
    if (!mImpl->runtime) {
        std::cerr << "[CNN] createInferRuntime failed" << std::endl;
        return;
    }
    mImpl->engine = mImpl->runtime->deserializeCudaEngine(buf.data(), size);
    if (!mImpl->engine) {
        std::cerr << "[CNN] deserializeCudaEngine failed" << std::endl;
        return;
    }
    mImpl->context = mImpl->engine->createExecutionContext();
    if (!mImpl->context) {
        std::cerr << "[CNN] createExecutionContext failed" << std::endl;
        return;
    }

    // Discover input/output tensor names and sizes (TRT 10 API).
    int n_io = mImpl->engine->getNbIOTensors();
    for (int i = 0; i < n_io; ++i) {
        const char* name = mImpl->engine->getIOTensorName(i);
        auto mode = mImpl->engine->getTensorIOMode(name);
        auto shape = mImpl->engine->getTensorShape(name);
        size_t v = volume(shape);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            mImpl->input_name = name;
            mImpl->input_elts = v;
        } else {
            mImpl->output_name = name;
            mImpl->output_elts = v;
        }
    }

    mImpl->input_h.resize(mImpl->input_elts);
    mImpl->output_h.resize(mImpl->output_elts);

    cudaMalloc(&mImpl->d_input,  mImpl->input_elts  * sizeof(float));
    cudaMalloc(&mImpl->d_output, mImpl->output_elts * sizeof(float));
    cudaStreamCreate(&mImpl->stream);

    mImpl->context->setTensorAddress(mImpl->input_name.c_str(),  mImpl->d_input);
    mImpl->context->setTensorAddress(mImpl->output_name.c_str(), mImpl->d_output);

    mImpl->loaded = true;
    std::cout << "[CNN] EP: TensorRT FP16" << std::endl;
    std::cout << "[CNN] Loaded engine: " << engine_path
              << " (in=" << mImpl->input_name << "[" << mImpl->input_elts << "]"
              << " out=" << mImpl->output_name << "[" << mImpl->output_elts << "])"
              << std::endl;
}

bool CNNLoopDetector::IsReady() const { return mImpl && mImpl->loaded; }

CNNLoopDetector::~CNNLoopDetector() {
    if (mImpl->stream)   cudaStreamDestroy(mImpl->stream);
    if (mImpl->d_input)  cudaFree(mImpl->d_input);
    if (mImpl->d_output) cudaFree(mImpl->d_output);
    delete mImpl->context;
    delete mImpl->engine;
    delete mImpl->runtime;
}

void CNNLoopDetector::AddKeyFrame(KeyFrame* pKF)
{
    mDescriptors.push_back(mImpl->extract(pKF->mImGray));
    mKeyFrames.push_back(pKF);
}

std::vector<KeyFrame*> CNNLoopDetector::DetectLoopCandidates(
    KeyFrame* pKF, int top_k, int exclude_recent, float min_sim)
{
    int query_idx = -1;
    for (int i = (int)mKeyFrames.size() - 1; i >= 0; --i) {
        if (mKeyFrames[i] == pKF) { query_idx = i; break; }
    }
    if (query_idx < 0) return {};

    int past_limit = query_idx - exclude_recent;
    if (past_limit <= 0) return {};

    const auto& q = mDescriptors[query_idx];
    if (q.empty()) return {};

    // Compute all similarity scores to build the distribution
    std::vector<std::pair<float, int>> all_scores;
    float sum_sim = 0.0f, sum_sim2 = 0.0f;
    int n_scores = 0;
    for (int j = 0; j < past_limit; ++j) {
        const auto& d = mDescriptors[j];
        if (d.empty()) continue;
        float sim = std::inner_product(q.begin(), q.end(), d.begin(), 0.0f);
        all_scores.emplace_back(sim, j);
        sum_sim += sim;
        sum_sim2 += sim * sim;
        n_scores++;
    }

    // Adaptive threshold: mean + 3*std, clamped to [min_sim, 0.95]
    // This rejects perceptual aliasing in repetitive environments (e.g. machine hall)
    float adaptive_th = min_sim;
    if (n_scores > 10) {
        float mean = sum_sim / n_scores;
        float var = sum_sim2 / n_scores - mean * mean;
        float std_dev = (var > 0.f) ? std::sqrt(var) : 0.0f;
        adaptive_th = mean + 3.0f * std_dev;
        if (adaptive_th < min_sim) adaptive_th = min_sim;
        if (adaptive_th > 0.95f)  adaptive_th = 0.95f;
    }

    // Filter by adaptive threshold
    std::vector<std::pair<float, int>> scores;
    for (auto& p : all_scores) {
        if (p.first >= adaptive_th)
            scores.push_back(p);
    }

    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    std::vector<KeyFrame*> candidates;
    for (int k = 0; k < std::min(top_k, (int)scores.size()); ++k) {
        KeyFrame* pCand = mKeyFrames[scores[k].second];
        if (!pCand->isBad())
            candidates.push_back(pCand);
    }
    return candidates;
}

void CNNLoopDetector::Reset()
{
    mDescriptors.clear();
    mKeyFrames.clear();
}

} // namespace ORB_SLAM3
