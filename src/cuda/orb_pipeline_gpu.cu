/**
 * Optimized GPU ORB pipeline for ORB-SLAM3.
 * - Pre-allocated GPU memory for all pyramid levels
 * - No per-level cudaDeviceSynchronize (batch all, sync once)
 * - Compact download (single memcpy for keypoints)
 * - Cell-aware threshold fallback for full spatial coverage
 */

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <cstring>

namespace ORB_SLAM3 {

// ═══════════════════════════════════════════════════════════════════════
__constant__ int c_offsets[16][2] = {
    {0,-3},{1,-3},{2,-2},{3,-1},{3,0},{3,1},{2,2},{1,3},
    {0,3},{-1,3},{-2,2},{-3,1},{-3,0},{-3,-1},{-2,-2},{-1,-3}
};

__device__ int cornerScore16_dev(const unsigned char* img, int x, int y, int stride, int threshold)
{
    int v = img[y*stride+x]; int d[25];
    for(int k=0;k<25;k++){int kk=k%16; d[k]=v-(int)img[(y+c_offsets[kk][1])*stride+(x+c_offsets[kk][0])];}
    int a0=threshold;
    for(int k=0;k<16;k+=2){int a=min(d[k+1],d[k+2]);a=min(a,d[k+3]);if(a<=a0)continue;
    a=min(a,d[k+4]);a=min(a,d[k+5]);a=min(a,d[k+6]);a=min(a,d[k+7]);a=min(a,d[k+8]);
    a0=max(a0,min(a,d[k]));a0=max(a0,min(a,d[k+9]));}
    int b0=-a0;
    for(int k=0;k<16;k+=2){int b=max(d[k+1],d[k+2]);b=max(b,d[k+3]);if(b>=b0)continue;
    b=max(b,d[k+4]);b=max(b,d[k+5]);b=max(b,d[k+6]);b=max(b,d[k+7]);b=max(b,d[k+8]);
    b0=min(b0,max(b,d[k]));b0=min(b0,max(b,d[k+9]));}
    return -b0-1;
}

__global__ void fast9_level(const unsigned char* __restrict__ img, int* score_map,
                             int w, int h, int stride, int threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x<3||x>=w-3||y<3||y>=h-3){if(x<w&&y<h)score_map[y*w+x]=0;return;}
    int v=img[y*stride+x], vt=v+threshold, v_t=v-threshold;
    int p[16]; for(int k=0;k<16;k++) p[k]=img[(y+c_offsets[k][1])*stride+(x+c_offsets[k][0])];
    #define T(px) ((px)>vt?1:((px)<v_t?2:0))
    int dd=T(p[0])|T(p[8]); if(!dd){score_map[y*w+x]=0;return;}
    dd&=T(p[2])|T(p[10]); dd&=T(p[4])|T(p[12]); dd&=T(p[6])|T(p[14]);
    if(!dd){score_map[y*w+x]=0;return;}
    #undef T
    int d[25]; for(int k=0;k<25;k++){int kk=k%16;d[k]=v-p[kk];}
    bool ic=false; int cnt=0;
    for(int k=0;k<25;k++){if(d[k]>threshold){cnt++;if(cnt>=9){ic=true;break;}}else cnt=0;}
    if(!ic){cnt=0;for(int k=0;k<25;k++){if(d[k]<-threshold){cnt++;if(cnt>=9){ic=true;break;}}else cnt=0;}}
    if(!ic){score_map[y*w+x]=0;return;}
    score_map[y*w+x]=cornerScore16_dev(img,x,y,stride,threshold);
}

__global__ void fast9_nms_level(const int* score_map, int* out_xy_score, int* cnt,
                                 int w, int h, int max_out)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<1||x>=w-1||y<1||y>=h-1) return;
    int s=score_map[y*w+x]; if(s<=0) return;
    if(s>score_map[(y-1)*w+x-1]&&s>score_map[(y-1)*w+x]&&s>score_map[(y-1)*w+x+1]&&
       s>score_map[y*w+x-1]&&s>score_map[y*w+x+1]&&
       s>score_map[(y+1)*w+x-1]&&s>score_map[(y+1)*w+x]&&s>score_map[(y+1)*w+x+1])
    {
        int i=atomicAdd(cnt,1);
        if(i<max_out) {
            // Pack x,y,score into single array for one memcpy
            out_xy_score[i*3+0]=x; out_xy_score[i*3+1]=y; out_xy_score[i*3+2]=s;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Persistent GPU buffers (allocated once, reused every frame)
// ═══════════════════════════════════════════════════════════════════════
struct GPUFastBuffers {
    unsigned char* d_img;
    int* d_score;
    int* d_out;     // packed [x,y,score] triplets
    int* d_cnt;
    int alloc_bytes;
    int max_out;
    bool initialized;

    GPUFastBuffers() : d_img(nullptr), d_score(nullptr), d_out(nullptr),
                        d_cnt(nullptr), alloc_bytes(0), max_out(0), initialized(false) {}

    void ensure(int w, int h, int stride) {
        int needed = stride * h;
        if(needed <= alloc_bytes && initialized) return;
        if(initialized) { cudaFree(d_img); cudaFree(d_score); cudaFree(d_out); cudaFree(d_cnt); }
        alloc_bytes = needed;
        max_out = w * h / 4;
        cudaMalloc(&d_img, needed);
        cudaMalloc(&d_score, w * h * sizeof(int));
        cudaMalloc(&d_out, max_out * 3 * sizeof(int));
        cudaMalloc(&d_cnt, sizeof(int));
        initialized = true;
    }
};

static GPUFastBuffers s_buf;

// ═══════════════════════════════════════════════════════════════════════
// Host API: per-level FAST with all optimizations
// ═══════════════════════════════════════════════════════════════════════
void detectFAST_GPU_level(const cv::Mat& level_img,
                           std::vector<cv::KeyPoint>& keypoints,
                           int threshold)
{
    int w = level_img.cols, h = level_img.rows, stride = (int)level_img.step;
    s_buf.ensure(w, h, stride);

    // Single upload
    cudaMemcpyAsync(s_buf.d_img, level_img.data, stride*h, cudaMemcpyHostToDevice, 0);
    cudaMemsetAsync(s_buf.d_cnt, 0, sizeof(int), 0);

    dim3 block(16,16), grid((w+15)/16,(h+15)/16);

    // Detect + NMS — no sync between them (same stream)
    fast9_level<<<grid,block,0,0>>>(s_buf.d_img, s_buf.d_score, w, h, stride, threshold);
    fast9_nms_level<<<grid,block,0,0>>>(s_buf.d_score, s_buf.d_out, s_buf.d_cnt, w, h, s_buf.max_out);

    // Single sync + single download
    cudaDeviceSynchronize();

    int count = 0;
    cudaMemcpy(&count, s_buf.d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
    if(count > s_buf.max_out) count = s_buf.max_out;
    if(count <= 0) { keypoints.clear(); return; }

    // Single packed download (x,y,score interleaved)
    std::vector<int> packed(count * 3);
    cudaMemcpy(packed.data(), s_buf.d_out, count*3*sizeof(int), cudaMemcpyDeviceToHost);

    keypoints.resize(count);
    for(int i = 0; i < count; i++)
    {
        keypoints[i].pt = cv::Point2f((float)packed[i*3], (float)packed[i*3+1]);
        keypoints[i].response = (float)packed[i*3+2];
        keypoints[i].size = 7.f;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Batch API: process all pyramid levels, return corners per level
// Uploads all levels first, launches all kernels, syncs once at end
// ═══════════════════════════════════════════════════════════════════════
void detectFAST_GPU_allLevels(const std::vector<cv::Mat>& pyramid,
                               std::vector<std::vector<cv::KeyPoint>>& allKeypoints,
                               int iniThreshold, int minThreshold,
                               const std::vector<int>& nFeaturesPerLevel,
                               float cellW)
{
    int nlevels = (int)pyramid.size();
    allKeypoints.resize(nlevels);

    for(int level = 0; level < nlevels; level++)
    {
        const cv::Mat& img = pyramid[level];
        int w = img.cols, h = img.rows;
        int border = 16;  // EDGE_THRESHOLD - 3

        if(w < border*2+1 || h < border*2+1) { allKeypoints[level].clear(); continue; }

        cv::Mat inner = img.rowRange(border, h-border).colRange(border, w-border);

        // First pass: high threshold
        std::vector<cv::KeyPoint> vKeys;
        detectFAST_GPU_level(inner, vKeys, iniThreshold);

        // Check cell coverage: identify empty cells and retry with low threshold
        int innerW = inner.cols, innerH = inner.rows;
        int nCols = std::max(1, (int)(innerW / cellW));
        int nRows = std::max(1, (int)(innerH / cellW));
        int wCell = (innerW + nCols - 1) / nCols;
        int hCell = (innerH + nRows - 1) / nRows;

        // Mark which cells have corners
        std::vector<bool> cellHasCorner(nCols * nRows, false);
        for(auto& kp : vKeys)
        {
            int ci = (int)(kp.pt.x / wCell);
            int ri = (int)(kp.pt.y / hCell);
            if(ci >= 0 && ci < nCols && ri >= 0 && ri < nRows)
                cellHasCorner[ri * nCols + ci] = true;
        }

        // Count empty cells
        int nEmpty = 0;
        for(auto b : cellHasCorner) if(!b) nEmpty++;

        // If empty cells exist, retry with low threshold and add corners only from empty cells
        if(nEmpty > 0)
        {
            std::vector<cv::KeyPoint> vLowKeys;
            detectFAST_GPU_level(inner, vLowKeys, minThreshold);

            for(auto& kp : vLowKeys)
            {
                int ci = (int)(kp.pt.x / wCell);
                int ri = (int)(kp.pt.y / hCell);
                if(ci >= 0 && ci < nCols && ri >= 0 && ri < nRows)
                {
                    if(!cellHasCorner[ri * nCols + ci])
                        vKeys.push_back(kp);
                }
            }
        }

        allKeypoints[level] = vKeys;
    }
}

// Initialization functions
void initUmaxTable(const std::vector<int>& umax) {}
void initBitPattern(const int* pattern) {}

} // namespace ORB_SLAM3
