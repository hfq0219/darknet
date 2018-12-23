#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
#ifdef OPENCL
    fprintf(stderr,"\nwork with opencl...\n");
    extern int BLOCK;
    extern cl_context *clContext;
    extern cl_command_queue *clCommandQueue;
    extern cl_program *clProgram;
    extern cl_kernel *clKernel;
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    cl_int height_col = (height + 2 * pad - ksize) / stride + 1;
    cl_int width_col = (width + 2 * pad - ksize) / stride + 1;
    cl_int num_kernels = channels * height_col * width_col;
    int size_im=channels*width*height;
    int size_col=num_kernels*ksize*ksize;
    size_t globalWorkSize[1]={(num_kernels+BLOCK-1)/BLOCK};
    size_t localWorkSize[1]={BLOCK};
    cl_int err;
    *clKernel=clCreateKernel(*clProgram, "im2col_opencl", &err);
    if(err!=CL_SUCCESS) {fprintf(stderr,"kernel error\n");exit(-1);}
    cl_mem data_im_opencl=clCreateBuffer(*clContext, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(float)*size_im, data_im, NULL);
    cl_mem data_col_opencl=clCreateBuffer(*clContext, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(float)*size_col, data_col, NULL);
    err=clSetKernelArg(*clKernel, 0, sizeof(cl_int), &num_kernels);
    err|=clSetKernelArg(*clKernel, 1, sizeof(cl_mem), &data_im_opencl);
    err|=clSetKernelArg(*clKernel, 2, sizeof(cl_int), &height);
    err|=clSetKernelArg(*clKernel, 3, sizeof(cl_int), &width);
    err|=clSetKernelArg(*clKernel, 4, sizeof(cl_int), &ksize);
    err|=clSetKernelArg(*clKernel, 5, sizeof(cl_int), &pad);
    err|=clSetKernelArg(*clKernel, 6, sizeof(cl_int), &stride);
    err|=clSetKernelArg(*clKernel, 7, sizeof(cl_int), &height_col);
    err|=clSetKernelArg(*clKernel, 8, sizeof(cl_int), &width_col);
    err|=clSetKernelArg(*clKernel, 9, sizeof(cl_mem), &data_col_opencl);
    if(err!=CL_SUCCESS||data_col_opencl==NULL||data_im_opencl==NULL){
        fprintf(stderr,"kernel arg set failed.\n");
        clean(clContext,clCommandQueue,clProgram,clKernel);
        exit(-1);
    }
    err=clEnqueueNDRangeKernel(*clCommandQueue,*clKernel,1,NULL,globalWorkSize,localWorkSize,0,NULL,NULL);
    if(err!=CL_SUCCESS){
        fprintf(stderr,"\ncompute error:%d\n",err);
        exit(-1);
    }
    clEnqueueReadBuffer(*clCommandQueue,data_col_opencl,CL_TRUE,0,sizeof(float)*size_col,data_col,0,NULL,NULL);
#else
    fprintf(stderr,"\nwork with cpu...\n");
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
#endif
}

