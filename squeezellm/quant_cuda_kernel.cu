#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// half-tensor
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDATensorMethods.cuh>

// atomicAdd for double-precision floating-point numbers on hardware with
// compute capability < 6.0 from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(
    double* address,
    double val
) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(
      address_as_ull,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed))
    );

  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

const int BLOCKWIDTH  = 128;
const int BLOCKHEIGHT3 =  12;
const int BLOCKHEIGHT4 =  16;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

__global__ void VecQuant3MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
);

__global__ void VecQuant4MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
);

__global__ void VecQuant3MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
);

__global__ void VecQuant4MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC(
    const       int* __restrict__ rows,
    const       int* __restrict__ cols,
    const  scalar_t* __restrict__ mat,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    const  int num_rows
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC_BATCHED(
    const       int* __restrict__ rows,
    const       int* __restrict__ cols,
    const  scalar_t* __restrict__ mat,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    const  int num_rows,
    int batch,
    int vec_height
);

__global__ void DenseMatVecKernel(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int topX,
    int full_width
);

__global__ void DenseMatVecKernelBatched(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int topX,
    int full_width,
    int batch,
    int vec_height,
    int matwidth
);

void vecquant3matmul_nuq_perchannel_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannel<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width
  );
}

// 4-bit matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannel<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width
  );
}

// 3-bit batched matvec kernel (LUT-based)
void vecquant3matmul_nuq_perchannel_batched_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, batch, vec_height
  );
}

// 4-bit batched matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_batched_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, batch, vec_height
  );
}

//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  VecQuant3MatMulKernelNUQPerChannel<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3
  );
}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  VecQuant4MatMulKernelNUQPerChannel<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4
  );
}


//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_batched_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_batched_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );
}


//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannel<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  //dense matvec kernel here!
  DenseMatVecKernel<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width
  );

}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannel<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  //dense matvec kernel here!
  DenseMatVecKernel<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width
  );

}

//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  DenseMatVecKernelBatched<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width,
    batch,
    vec_height,
    matwidth
  );

}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  DenseMatVecKernelBatched<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width,
    batch,
    vec_height,
    matwidth
  );

}


__global__ void VecQuant3MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
) {

  int row = BLOCKHEIGHT3 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT3) * BLOCKWIDTH + threadIdx.x];

  //Modified dequant block
  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;
  int column_offset = col * 8;
  for (int val = 0; val < 8; val += 1) {
    int lut_index = column_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  int i = width * row + col;
  int k = 0;

  float res = 0;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  __syncthreads();

  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);

    res += deq2[(tmp1 >>  0) & 0x7][off] * blockvec[k + 0];
    res += deq2[(tmp1 >>  3) & 0x7][off] * blockvec[k + 1];
    res += deq2[(tmp1 >>  6) & 0x7][off] * blockvec[k + 2];
    res += deq2[(tmp1 >>  9) & 0x7][off] * blockvec[k + 3];
    res += deq2[(tmp1 >>  12) & 0x7][off] * blockvec[k + 4];
    res += deq2[(tmp1 >>  15) & 0x7][off] * blockvec[k + 5];
    res += deq2[(tmp1 >>  18) & 0x7][off] * blockvec[k + 6];
    res += deq2[(tmp1 >>  21) & 0x7][off] * blockvec[k + 7];
    res += deq2[(tmp1 >>  24) & 0x7][off] * blockvec[k + 8];
    res += deq2[(tmp1 >>  27) & 0x7][off] * blockvec[k + 9];

    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += deq2[(tmp >>  0) & 0x7][off] * blockvec[k + 10];
    k += 11;
    res += deq2[(tmp2 >>  0) & 0x7][off] * blockvec[k + 0];
    res += deq2[(tmp2 >>  3) & 0x7][off] * blockvec[k + 1];
    res += deq2[(tmp2 >>  6) & 0x7][off] * blockvec[k + 2];
    res += deq2[(tmp2 >>  9) & 0x7][off] * blockvec[k + 3];
    res += deq2[(tmp2 >>  12) & 0x7][off] * blockvec[k + 4];
    res += deq2[(tmp2 >>  15) & 0x7][off] * blockvec[k + 5];
    res += deq2[(tmp2 >>  18) & 0x7][off] * blockvec[k + 6];
    res += deq2[(tmp2 >>  21) & 0x7][off] * blockvec[k + 7];
    res += deq2[(tmp2 >>  24) & 0x7][off] * blockvec[k + 8];
    res += deq2[(tmp2 >>  27) & 0x7][off] * blockvec[k + 9];

    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += deq2[(tmp >>  0) & 0x7][off] * blockvec[k + 10];
    k += 11;
    res += deq2[(tmp1 >>  0) & 0x7][off] * blockvec[k + 0];
    res += deq2[(tmp1 >>  3) & 0x7][off] * blockvec[k + 1];
    res += deq2[(tmp1 >>  6) & 0x7][off] * blockvec[k + 2];
    res += deq2[(tmp1 >>  9) & 0x7][off] * blockvec[k + 3];
    res += deq2[(tmp1 >>  12) & 0x7][off] * blockvec[k + 4];
    res += deq2[(tmp1 >>  15) & 0x7][off] * blockvec[k + 5];
    res += deq2[(tmp1 >>  18) & 0x7][off] * blockvec[k + 6];
    res += deq2[(tmp1 >>  21) & 0x7][off] * blockvec[k + 7];
    res += deq2[(tmp1 >>  24) & 0x7][off] * blockvec[k + 8];
    res += deq2[(tmp1 >>  27) & 0x7][off] * blockvec[k + 9];
    i += width;
    k += 10;
  }

  atomicAdd(&mul[col], res);
}

//4-bit per-channel
__global__ void VecQuant4MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
) {

  int row = BLOCKHEIGHT4 * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT4) * BLOCKWIDTH + threadIdx.x];

  //Modified dequant block
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;
  int column_offset = col * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = column_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  __syncthreads();

  float res = 0;
  int i = width * row + col;
  int k = 0;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    res += deq2[(tmp >>  0) & 0xf][off] * blockvec[k + 0];
    res += deq2[(tmp >>  4) & 0xf][off] * blockvec[k + 1];
    res += deq2[(tmp >>  8) & 0xf][off] * blockvec[k + 2];
    res += deq2[(tmp >>  12) & 0xf][off] * blockvec[k + 3];
    res += deq2[(tmp >>  16) & 0xf][off] * blockvec[k + 4];
    res += deq2[(tmp >>  20) & 0xf][off] * blockvec[k + 5];
    res += deq2[(tmp >>  24) & 0xf][off] * blockvec[k + 6];
    res += deq2[(tmp >>  28) & 0xf][off] * blockvec[k + 7];

    i += width;
    k += 8;
  }

  atomicAdd(&mul[col], res);
}


//batched version (3-bit)
__global__ void VecQuant3MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
) {

  int row = BLOCKHEIGHT3 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;
  int column_offset = col * 8;
  for (int val = 0; val < 8; val += 1) {
    int lut_index = column_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  int i;
  float res;
  int k;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  for (int b = 0; b < batch; ++b){
    //initialize vars
    i = width * row + col;
    res = 0;
    k = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + (row / BLOCKHEIGHT3) * BLOCKWIDTH + threadIdx.x];
    __syncthreads();

    while (k < BLOCKWIDTH) {
      tmp1 = as_unsigned(mat[i]);

      res += deq2[(tmp1 >>  0) & 0x7][off] * blockvec[k + 0];
      res += deq2[(tmp1 >>  3) & 0x7][off] * blockvec[k + 1];
      res += deq2[(tmp1 >>  6) & 0x7][off] * blockvec[k + 2];
      res += deq2[(tmp1 >>  9) & 0x7][off] * blockvec[k + 3];
      res += deq2[(tmp1 >>  12) & 0x7][off] * blockvec[k + 4];
      res += deq2[(tmp1 >>  15) & 0x7][off] * blockvec[k + 5];
      res += deq2[(tmp1 >>  18) & 0x7][off] * blockvec[k + 6];
      res += deq2[(tmp1 >>  21) & 0x7][off] * blockvec[k + 7];
      res += deq2[(tmp1 >>  24) & 0x7][off] * blockvec[k + 8];
      res += deq2[(tmp1 >>  27) & 0x7][off] * blockvec[k + 9];

      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
      tmp2 >>= 1;
      res += deq2[(tmp >>  0) & 0x7][off] * blockvec[k + 10];
      k += 11;
      res += deq2[(tmp2 >>  0) & 0x7][off] * blockvec[k + 0];
      res += deq2[(tmp2 >>  3) & 0x7][off] * blockvec[k + 1];
      res += deq2[(tmp2 >>  6) & 0x7][off] * blockvec[k + 2];
      res += deq2[(tmp2 >>  9) & 0x7][off] * blockvec[k + 3];
      res += deq2[(tmp2 >>  12) & 0x7][off] * blockvec[k + 4];
      res += deq2[(tmp2 >>  15) & 0x7][off] * blockvec[k + 5];
      res += deq2[(tmp2 >>  18) & 0x7][off] * blockvec[k + 6];
      res += deq2[(tmp2 >>  21) & 0x7][off] * blockvec[k + 7];
      res += deq2[(tmp2 >>  24) & 0x7][off] * blockvec[k + 8];
      res += deq2[(tmp2 >>  27) & 0x7][off] * blockvec[k + 9];

      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
      tmp1 >>= 2;
      res += deq2[(tmp >>  0) & 0x7][off] * blockvec[k + 10];
      k += 11;
      res += deq2[(tmp1 >>  0) & 0x7][off] * blockvec[k + 0];
      res += deq2[(tmp1 >>  3) & 0x7][off] * blockvec[k + 1];
      res += deq2[(tmp1 >>  6) & 0x7][off] * blockvec[k + 2];
      res += deq2[(tmp1 >>  9) & 0x7][off] * blockvec[k + 3];
      res += deq2[(tmp1 >>  12) & 0x7][off] * blockvec[k + 4];
      res += deq2[(tmp1 >>  15) & 0x7][off] * blockvec[k + 5];
      res += deq2[(tmp1 >>  18) & 0x7][off] * blockvec[k + 6];
      res += deq2[(tmp1 >>  21) & 0x7][off] * blockvec[k + 7];
      res += deq2[(tmp1 >>  24) & 0x7][off] * blockvec[k + 8];
      res += deq2[(tmp1 >>  27) & 0x7][off] * blockvec[k + 9];
      i += width;
      k += 10;
    }

    atomicAdd(&mul[b * width + col], res);
  }
}

//batched version (4-bit)
__global__ void VecQuant4MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
) {

  int row = BLOCKHEIGHT4 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  __shared__ float blockvec[BLOCKWIDTH];

  //Modified dequant block
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;
  int column_offset = col * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = column_offset + (val & 0xf);
    deq2[val][off] = lookup_table[lut_index];
  }

  int i;
  float res;
  int k;
  unsigned int tmp;

  for (int b = 0; b < batch; ++b){
    i = width * row + col;
    res = 0;
    k = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + (row / BLOCKHEIGHT4) * BLOCKWIDTH + threadIdx.x];
    __syncthreads();

    while (k < BLOCKWIDTH) {
      tmp = as_unsigned(mat[i]);

      res += deq2[(tmp >>  0) & 0xf][off] * blockvec[k + 0];
      res += deq2[(tmp >>  4) & 0xf][off] * blockvec[k + 1];
      res += deq2[(tmp >>  8) & 0xf][off] * blockvec[k + 2];
      res += deq2[(tmp >>  12) & 0xf][off] * blockvec[k + 3];
      res += deq2[(tmp >>  16) & 0xf][off] * blockvec[k + 4];
      res += deq2[(tmp >>  20) & 0xf][off] * blockvec[k + 5];
      res += deq2[(tmp >>  24) & 0xf][off] * blockvec[k + 6];
      res += deq2[(tmp >>  28) & 0xf][off] * blockvec[k + 7];

      i += width;
      k += 8;
    }

    atomicAdd(&mul[b * width + col], res);
  }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int i = start_elem; i < end_elem; i++) {
            dot += mat[i] * vec[cols[i]];
        }
        atomicAdd(&mul[row], dot);
    }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC_BATCHED(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows,
  int batch,
  int vec_height
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int b = 0; b < batch; ++b){
            float dot = 0;
            for (int i = start_elem; i < end_elem; i++) {
                dot += mat[i] * vec[b * vec_height + cols[i]];
                // dot += mat[i] * vec[cols[i] * batch + b];
                // dot += mat[i] * vec[cols[i]];
            }
            atomicAdd(&mul[b * num_rows + row], dot);
            // atomicAdd(&mul[row * batch + b], dot);
            // atomicAdd(&mul[row], dot);
        }
    }
}

// Dense kernel for only a subset of rows
__global__ void DenseMatVecKernel(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width
) {

  int row = BLOCKWIDTH * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[row + threadIdx.x];

  __syncthreads();

  int i = width * row + col;
  int k = 0;
  float res = 0;

  if (threadIdx.x < width) {
    while (k < BLOCKWIDTH) {
      res += full_rows[i] * blockvec[k];
      k += 1;
      i += width;
    }

    int col_idx = full_row_indices[col];
    atomicAdd(&mul[col_idx], res);
  }
}


// Dense kernel for only a subset of rows
__global__ void DenseMatVecKernelBatched(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width,
    int batch,
    int vec_height,
    int matwidth
) {

  int row = BLOCKWIDTH * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  for (int b = 0; b < batch; ++b){
    int i = width * row + col;
    int k = 0;
    float res = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + row + threadIdx.x];
    __syncthreads();

    if (threadIdx.x < width) {
      while (k < BLOCKWIDTH) {
        res += full_rows[i] * blockvec[k];
        k += 1;
        i += width;
      }

      int col_idx = full_row_indices[col];
      atomicAdd(&mul[b * matwidth + col_idx], res);
    }
  }
}
