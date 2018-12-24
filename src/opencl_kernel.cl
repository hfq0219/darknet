
//im2col.c
__kernel void im2col_opencl(int n,__global float *data_im,int height,int width,
        int ksize,int pad,int stride,int height_col,int width_col,__global float *data_col)
{
    int x_=get_global_id(0),y_=get_global_id(1),z_=get_global_id(2);
    int m_=get_global_size(0),n_=get_global_size(1),k_=get_global_size(2);

    int index = x_+y_*m_+z_*m_*n_;
    for(; index < n; index += m_*n_*k_){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        __global float *data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        __global float *data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}
//col2im.c
__kernel void col2im_opencl(int n,__global float *data_col,int height,int width,
        int ksize,int pad,int stride,int height_col,int width_col,__global float *data_im)
{
    int x_=get_global_id(0),y_=get_global_id(1),z_=get_global_id(2);
    int m_=get_global_size(0),n_=get_global_size(1),k_=get_global_size(2);

    int index = x_+y_*m_+z_*m_*n_;
    for(; index < n; index += m_*n_*k_){
        float val = 0;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);
        // compute the start and end of the output
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        // equivalent implementation
        int offset =(c * ksize * ksize + h * ksize + w) * height_col * width_col;
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        int coeff_w_col = (1 - stride * height_col * width_col);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col){
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col){
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] += val;
    }
}
//gemm.c
__kernel void gemm_nn_opencl(int M,int N,int K,float ALPHA,__global float *weight,
        int lda,__global float *input,int ldb,__global float *output,int ldc)
{
    int x_=get_global_id(0),y_=get_global_id(1),z_=get_global_id(2);
    int m_=get_global_size(0),n_=get_global_size(1),k_=get_global_size(2);

    int index = x_+y_*m_+z_*m_*n_;

    if(index<M*ldc){
        int row = index / ldc;
        int col = index % ldc;
        for(int i = 0; i < K; i++){
            output[index] += ALPHA * weight[row*lda+i] * input[i*ldb+col];
        }
    }
}
//gemm.c
__kernel void gemm_nt_opencl(int M,int N,int K,float ALPHA,__global float *weight,
        int lda,__global float *input,int ldb,__global float *output,int ldc)
{
    int x_=get_global_id(0),y_=get_global_id(1),z_=get_global_id(2);
    int m_=get_global_size(0),n_=get_global_size(1),k_=get_global_size(2);

    int index = x_+y_*m_+z_*m_*n_;

    if(index<M*ldc){
        int row = index / ldc;
        int col = index % ldc;
        float sum=0;
        for(int i = 0; i < K; i++){
            sum += ALPHA*weight[row*lda+i]*input[col*ldb+i];
        }
        output[index] += sum;
    }
}
//gemm.c
__kernel void gemm_tn_opencl(int M,int N,int K,float ALPHA,__global float *weight,
        int lda,__global float *input,int ldb,__global float *output,int ldc)
{
    int x_=get_global_id(0),y_=get_global_id(1),z_=get_global_id(2);
    int m_=get_global_size(0),n_=get_global_size(1),k_=get_global_size(2);

    int index = x_+y_*m_+z_*m_*n_;

    if(index<M*ldc){
        int row = index / ldc;
        int col = index % ldc;
        for(int i = 0; i < K; i++){
            float A_PART = ALPHA*weight[i*lda+row];
            output[row*ldc+col] += A_PART*input[i*ldb+col];
        }
    }
}
//gemm.c
__kernel void gemm_opencl(int M,int ldc,__global float *output,float BETA)
{
    int x_=get_global_id(0),y_=get_global_id(1),z_=get_global_id(2);
    int m_=get_global_size(0),n_=get_global_size(1),k_=get_global_size(2);

    int index = x_+y_*m_+z_*m_*n_;

    if(index<M*ldc){
        output[index] *= BETA;
    }
}










