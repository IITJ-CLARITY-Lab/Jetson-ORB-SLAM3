/**
 * GPU ORB descriptor computation — matches ORB-SLAM3 CPU exactly.
 *
 * The ORB descriptor uses 256 pixel pair comparisons. Each pair is
 * rotated by the keypoint angle. The pattern is the standard ORB
 * bit_pattern_31_ (256 pairs stored as 512 cv::Point = 1024 ints).
 *
 * GPU kernel: one thread per keypoint, each computes 32 bytes (256 bits).
 * Uses __float2int_rn() to match cvRound().
 */
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <vector>

namespace ORB_SLAM3 {

// Pattern stored as pairs: [x1,y1, x2,y2, x1,y1, x2,y2, ...]
// 256 comparisons × 2 points × 2 coords = 1024 ints
// But ORB-SLAM3 stores it as 512 Point (each Point = 2 ints) = 1024 ints
// The pattern in ORBextractor.cc is bit_pattern_31_[256*4] where
// each group of 4 ints is: pt1.x, pt1.y, pt2.x, pt2.y
__constant__ int c_pattern[256*4]; // 1024 ints

void initDescriptorPattern_GPU(const int* pattern_31)
{
    cudaMemcpyToSymbol(c_pattern, pattern_31, 256*4*sizeof(int));
}

__global__ void orb_desc_kernel(const unsigned char* __restrict__ img, int stride,
                                 const float* kp_x, const float* kp_y, const float* kp_angle,
                                 unsigned char* desc, int n, int w, int h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    int cx = __float2int_rn(kp_x[i]);
    int cy = __float2int_rn(kp_y[i]);
    if(cx < 16 || cx >= w-16 || cy < 16 || cy >= h-16)
    {
        for(int k=0;k<32;k++) desc[i*32+k]=0;
        return;
    }

    float angle = kp_angle[i] * (3.14159265358979f / 180.0f);
    float a = cosf(angle), b = sinf(angle);
    const unsigned char* center = img + cy*stride + cx;
    unsigned char* out = desc + i*32;

    // ORB-SLAM3 uses 16 pattern Points per descriptor byte:
    // pattern[0..15] → byte 0, pattern[16..31] → byte 1, etc.
    // Each consecutive pair (pattern[2k], pattern[2k+1]) is one comparison
    // pattern is stored as Point (x,y), so pattern[k] = (c_pattern[k*2], c_pattern[k*2+1])
    for(int k = 0; k < 32; k++)
    {
        int val = 0;
        int base = k * 16; // 16 Points = 8 comparisons
        for(int bit = 0; bit < 8; bit++)
        {
            int pidx = base + bit*2; // index of first Point in pair
            int px1 = c_pattern[pidx*2+0], py1 = c_pattern[pidx*2+1];     // Point 1
            int px2 = c_pattern[(pidx+1)*2+0], py2 = c_pattern[(pidx+1)*2+1]; // Point 2

            // Rotate: same formula as ORB-SLAM3's GET_VALUE macro
            // row offset = cvRound(x*b + y*a), col offset = cvRound(x*a - y*b)
            int t0 = center[__float2int_rn(px1*b+py1*a)*stride + __float2int_rn(px1*a-py1*b)];
            int t1 = center[__float2int_rn(px2*b+py2*a)*stride + __float2int_rn(px2*a-py2*b)];
            val |= (t0 < t1) << bit;
        }
        out[k] = (unsigned char)val;
    }
}

// Persistent GPU buffers
static unsigned char* s_dimg = nullptr;
static float *s_dx=nullptr, *s_dy=nullptr, *s_da=nullptr;
static unsigned char* s_ddesc = nullptr;
static int s_img_sz=0, s_max_n=0;

void computeDescriptors_GPU(const cv::Mat& blurred_img,
                             const std::vector<cv::KeyPoint>& keypoints,
                             cv::Mat& descriptors)
{
    int N = (int)keypoints.size();
    if(N == 0) { descriptors = cv::Mat(); return; }

    int w = blurred_img.cols, h = blurred_img.rows, stride = (int)blurred_img.step;
    int img_sz = stride * h;

    // Ensure GPU buffers
    if(img_sz > s_img_sz) {
        if(s_dimg) cudaFree(s_dimg);
        cudaMalloc(&s_dimg, img_sz); s_img_sz = img_sz;
    }
    if(N > s_max_n) {
        if(s_dx){cudaFree(s_dx);cudaFree(s_dy);cudaFree(s_da);cudaFree(s_ddesc);}
        cudaMalloc(&s_dx, N*sizeof(float)); cudaMalloc(&s_dy, N*sizeof(float));
        cudaMalloc(&s_da, N*sizeof(float)); cudaMalloc(&s_ddesc, N*32);
        s_max_n = N;
    }

    // Upload
    cudaMemcpyAsync(s_dimg, blurred_img.data, img_sz, cudaMemcpyHostToDevice, 0);

    std::vector<float> hx(N), hy(N), ha(N);
    for(int i=0;i<N;i++){hx[i]=keypoints[i].pt.x; hy[i]=keypoints[i].pt.y; ha[i]=keypoints[i].angle;}
    cudaMemcpyAsync(s_dx, hx.data(), N*sizeof(float), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(s_dy, hy.data(), N*sizeof(float), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(s_da, ha.data(), N*sizeof(float), cudaMemcpyHostToDevice, 0);

    // Compute
    orb_desc_kernel<<<(N+255)/256, 256>>>(s_dimg, stride, s_dx, s_dy, s_da, s_ddesc, N, w, h);
    cudaDeviceSynchronize();

    // Download
    descriptors.create(N, 32, CV_8U);
    cudaMemcpy(descriptors.data, s_ddesc, N*32, cudaMemcpyDeviceToHost);
}

} // namespace ORB_SLAM3
