/**
 * GPU Gaussian blur — verified match with OpenCV GaussianBlur(7x7, sigma=2)
 */
#include <cuda_runtime.h>
#include <opencv2/core.hpp>

namespace ORB_SLAM3 {

__constant__ float c_g7[7] = {0.070159f,0.131075f,0.190713f,0.216106f,0.190713f,0.131075f,0.070159f};

__global__ void blur_row_k(const unsigned char* src, float* tmp, int w, int h, int stride) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return;
    float s=0;
    for(int k=-3;k<=3;k++){int xx=x+k;if(xx<0)xx=-xx;if(xx>=w)xx=2*w-xx-2;s+=c_g7[k+3]*src[y*stride+xx];}
    tmp[y*w+x]=s;
}
__global__ void blur_col_k(const float* tmp, unsigned char* dst, int w, int h, int stride) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return;
    float s=0;
    for(int k=-3;k<=3;k++){int yy=y+k;if(yy<0)yy=-yy;if(yy>=h)yy=2*h-yy-2;s+=c_g7[k+3]*tmp[yy*w+x];}
    dst[y*stride+x]=(unsigned char)(s+0.5f);
}

static unsigned char *s_src=nullptr,*s_dst=nullptr; static float *s_tmp=nullptr;
static int s_w=0,s_h=0;

void gaussianBlur7x7_GPU(const cv::Mat& src, cv::Mat& dst) {
    int w=src.cols,h=src.rows,stride=(int)src.step;
    dst.create(h,w,CV_8UC1);
    if(w!=s_w||h!=s_h){
        if(s_src){cudaFree(s_src);cudaFree(s_dst);cudaFree(s_tmp);}
        s_w=w;s_h=h;
        cudaMalloc(&s_src,stride*h);cudaMalloc(&s_dst,w*h);cudaMalloc(&s_tmp,w*h*sizeof(float));
    }
    cudaMemcpyAsync(s_src,src.data,stride*h,cudaMemcpyHostToDevice,0);
    dim3 b(16,16),g((w+15)/16,(h+15)/16);
    blur_row_k<<<g,b>>>(s_src,s_tmp,w,h,stride);
    blur_col_k<<<g,b>>>(s_tmp,s_dst,w,h,w);
    cudaDeviceSynchronize();
    cudaMemcpy(dst.data,s_dst,w*h,cudaMemcpyDeviceToHost);
}

} // namespace ORB_SLAM3
