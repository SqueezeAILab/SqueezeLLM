#include<sycl/sycl.hpp>
#include<dpct/dpct.hpp>
#include <torch/all.h>
#include <torch/python.h>

// half-tensor

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 600
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

inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

void VecQuant3MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2);

void VecQuant4MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2);

void VecQuant3MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2);

void VecQuant4MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2);

template <typename scalar_t>
void SPMV_ATOMIC(
    const       int* __restrict__ rows,
    const       int* __restrict__ cols,
    const  scalar_t* __restrict__ mat,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    const  int num_rows
,
    const sycl::nd_item<3> &item_ct1);

template <typename scalar_t>
void SPMV_ATOMIC_BATCHED(
    const       int* __restrict__ rows,
    const       int* __restrict__ cols,
    const  scalar_t* __restrict__ mat,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    const  int num_rows,
    int batch,
    int vec_height
,
    const sycl::nd_item<3> &item_ct1);

void DenseMatVecKernel(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int topX,
    int full_width
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec);

void DenseMatVecKernelBatched(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int topX,
    int full_width,
    int batch,
    int vec_height,
    int matwidth
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec);

void vecquant3matmul_nuq_perchannel_sycl(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                        (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3);
  sycl::range<3> threads(1, 1, BLOCKWIDTH);

  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
   
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);
    
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(8, 128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const int *mat_data_ptr_int_ct1 = mat.data_ptr<int>();
    float *mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    const float *lookup_table_data_ptr_float_ct3 =
        lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant3MatMulKernelNUQPerChannel(
              vec_data_ptr_float_ct0, mat_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height,
              width, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              deq2_acc_ct1);
        });
  });
}

// 4-bit matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_sycl(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                        (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads(1, 1, BLOCKWIDTH);

  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);
    
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(16, 128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const int *mat_data_ptr_int_ct1 = mat.data_ptr<int>();
    float *mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    const float *lookup_table_data_ptr_float_ct3 =
        lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant4MatMulKernelNUQPerChannel(
              vec_data_ptr_float_ct0, mat_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height,
              width, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              deq2_acc_ct1);
        });
  });
}

// 3-bit batched matvec kernel (LUT-based)
void vecquant3matmul_nuq_perchannel_batched_sycl(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                        (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3);
  sycl::range<3> threads(1, 1, BLOCKWIDTH);

  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);
    
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(8, 128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const int *mat_data_ptr_int_ct1 = mat.data_ptr<int>();
    float *mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    const float *lookup_table_data_ptr_float_ct3 =
        lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant3MatMulKernelNUQPerChannelBatched(
              vec_data_ptr_float_ct0, mat_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height,
              width, batch, vec_height, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              deq2_acc_ct1);
        });
  });
}

// 4-bit batched matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_batched_sycl(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                        (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads(1, 1, BLOCKWIDTH);

  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);
    
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(16, 128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const int *mat_data_ptr_int_ct1 = mat.data_ptr<int>();
    float *mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    const float *lookup_table_data_ptr_float_ct3 =
        lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant4MatMulKernelNUQPerChannelBatched(
              vec_data_ptr_float_ct0, mat_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height,
              width, batch, vec_height, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              deq2_acc_ct1);
        });
  });
}

//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_sycl(
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

  sycl::range<3> blocks3(1, (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3);
  sycl::range<3> threads3(1, 1, BLOCKWIDTH);

  
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic", ([&] {
      
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      const int *rows_data_int_ct0 = rows.data<int>();
      const int *cols_data_int_ct1 = cols.data<int>();
      const scalar_t *mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      const scalar_t *vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      scalar_t *mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC(rows_data_int_ct0, cols_data_int_ct1,
                                     mat_data_scalar_t_ct2,
                                     vec_data_scalar_t_ct3,
                                     mul_data_scalar_t_ct4, num_rows, item_ct1);
                       });
    });
                             }));

  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);
    
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(8, 128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const int *mat3_data_ptr_int_ct1 = mat3.data_ptr<int>();
    float *mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    const float *lookup_table_data_ptr_float_ct3 =
        lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks3 * threads3, threads3),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant3MatMulKernelNUQPerChannel(
              vec_data_ptr_float_ct0, mat3_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height3,
              width3, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              deq2_acc_ct1);
        });
  });
}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_sycl(
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

  sycl::range<3> blocks4(1, (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads4(1, 1, BLOCKWIDTH);

  
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic", ([&] {
      
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      const int *rows_data_int_ct0 = rows.data<int>();
      const int *cols_data_int_ct1 = cols.data<int>();
      const scalar_t *mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      const scalar_t *vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      scalar_t *mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC(rows_data_int_ct0, cols_data_int_ct1,
                                     mat_data_scalar_t_ct2,
                                     vec_data_scalar_t_ct3,
                                     mul_data_scalar_t_ct4, num_rows, item_ct1);
                       });
    });
                             }));

  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 ), cgh);
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(16, 128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const int *mat4_data_ptr_int_ct1 = mat4.data_ptr<int>();
    float *mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    const float *lookup_table_data_ptr_float_ct3 =
        lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks4 * threads4, threads4),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant4MatMulKernelNUQPerChannel(
              vec_data_ptr_float_ct0, mat4_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height4,
              width4, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              deq2_acc_ct1);
        });
  });
}


//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_batched_sycl(
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

  sycl::range<3> blocks3(1, (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3);
  sycl::range<3> threads3(1, 1, BLOCKWIDTH);

  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);
    
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(8, 128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const int *mat3_data_ptr_int_ct1 = mat3.data_ptr<int>();
    float *mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    const float *lookup_table_data_ptr_float_ct3 =
        lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks3 * threads3, threads3),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant3MatMulKernelNUQPerChannelBatched(
              vec_data_ptr_float_ct0, mat3_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height3,
              width3, batch, vec_height, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              deq2_acc_ct1);
        });
  });

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic_batched", ([&] {
      
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      const int *rows_data_int_ct0 = rows.data<int>();
      const int *cols_data_int_ct1 = cols.data<int>();
      const scalar_t *mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      const scalar_t *vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      scalar_t *mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC_BATCHED(
                             rows_data_int_ct0, cols_data_int_ct1,
                             mat_data_scalar_t_ct2, vec_data_scalar_t_ct3,
                             mul_data_scalar_t_ct4, num_rows, batch, vec_height,
                             item_ct1);
                       });
    });
                             }));
}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_batched_sycl(
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

  sycl::range<3> blocks4(1, (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads4(1, 1, BLOCKWIDTH);

  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);
    
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(16, 128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const int *mat4_data_ptr_int_ct1 = mat4.data_ptr<int>();
    float *mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    const float *lookup_table_data_ptr_float_ct3 =
        lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks4 * threads4, threads4),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant4MatMulKernelNUQPerChannelBatched(
              vec_data_ptr_float_ct0, mat4_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height4,
              width4, batch, vec_height, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              deq2_acc_ct1);
        });
  });

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic_batched", ([&] {
      
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      const int *rows_data_int_ct0 = rows.data<int>();
      const int *cols_data_int_ct1 = cols.data<int>();
      const scalar_t *mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      const scalar_t *vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      scalar_t *mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC_BATCHED(
                             rows_data_int_ct0, cols_data_int_ct1,
                             mat_data_scalar_t_ct2, vec_data_scalar_t_ct3,
                             mul_data_scalar_t_ct4, num_rows, batch, vec_height,
                             item_ct1);
                       });
    });
                             }));
}


//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_sycl(
    torch::Tensor rows, torch::Tensor cols, torch::Tensor mat,
    torch::Tensor vec, torch::Tensor full_rows, torch::Tensor full_row_indices,
    torch::Tensor mul, int num_rows, torch::Tensor mat3,
    torch::Tensor lookup_table) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  sycl::range<3> blocks3(1, (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3);
  sycl::range<3> threads3(1, 1, BLOCKWIDTH);

  q_ct1.submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);
    
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(8, 128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const int *mat3_data_ptr_int_ct1 = mat3.data_ptr<int>();
    float *mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    const float *lookup_table_data_ptr_float_ct3 =
        lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks3 * threads3, threads3),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant3MatMulKernelNUQPerChannel(
              vec_data_ptr_float_ct0, mat3_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height3,
              width3, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              deq2_acc_ct1);
        });
  });

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic", ([&] {
      
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      const int *rows_data_int_ct0 = rows.data<int>();
      const int *cols_data_int_ct1 = cols.data<int>();
      const scalar_t *mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      const scalar_t *vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      scalar_t *mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC(rows_data_int_ct0, cols_data_int_ct1,
                                     mat_data_scalar_t_ct2,
                                     vec_data_scalar_t_ct3,
                                     mul_data_scalar_t_ct4, num_rows, item_ct1);
                       });
    });
                             }));

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  sycl::range<3> blocks_topX(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                             (height + BLOCKWIDTH - 1) / BLOCKWIDTH);
  sycl::range<3> threads_topX(1, 1, BLOCKWIDTH);

  //dense matvec kernel here!
  q_ct1.submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const float *full_rows_data_ptr_float_ct1 = full_rows.data_ptr<float>();
    const int *full_row_indices_data_ptr_int_ct2 =
        full_row_indices.data_ptr<int>();
    float *mul_data_ptr_float_ct3 = mul.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks_topX * threads_topX, threads_topX),
        [=](sycl::nd_item<3> item_ct1) {
          DenseMatVecKernel(
              vec_data_ptr_float_ct0, full_rows_data_ptr_float_ct1,
              full_row_indices_data_ptr_int_ct2, mul_data_ptr_float_ct3, height,
              width, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_sycl(
    torch::Tensor rows, torch::Tensor cols, torch::Tensor mat,
    torch::Tensor vec, torch::Tensor full_rows, torch::Tensor full_row_indices,
    torch::Tensor mul, int num_rows, torch::Tensor mat4,
    torch::Tensor lookup_table) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  sycl::range<3> blocks4(1, (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads4(1, 1, BLOCKWIDTH);

  q_ct1.submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);
    
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(16, 128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const int *mat4_data_ptr_int_ct1 = mat4.data_ptr<int>();
    float *mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    const float *lookup_table_data_ptr_float_ct3 =
        lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks4 * threads4, threads4),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant4MatMulKernelNUQPerChannel(
              vec_data_ptr_float_ct0, mat4_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height4,
              width4, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              deq2_acc_ct1);
        });
  });

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic", ([&] {
     
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      const int *rows_data_int_ct0 = rows.data<int>();
      const int *cols_data_int_ct1 = cols.data<int>();
      const scalar_t *mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      const scalar_t *vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      scalar_t *mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC(rows_data_int_ct0, cols_data_int_ct1,
                                     mat_data_scalar_t_ct2,
                                     vec_data_scalar_t_ct3,
                                     mul_data_scalar_t_ct4, num_rows, item_ct1);
                       });
    });
                             }));

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  sycl::range<3> blocks_topX(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                             (height + BLOCKWIDTH - 1) / BLOCKWIDTH);
  sycl::range<3> threads_topX(1, 1, BLOCKWIDTH);

  //dense matvec kernel here!
  q_ct1.submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const float *full_rows_data_ptr_float_ct1 = full_rows.data_ptr<float>();
    const int *full_row_indices_data_ptr_int_ct2 =
        full_row_indices.data_ptr<int>();
    float *mul_data_ptr_float_ct3 = mul.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks_topX * threads_topX, threads_topX),
        [=](sycl::nd_item<3> item_ct1) {
          DenseMatVecKernel(
              vec_data_ptr_float_ct0, full_rows_data_ptr_float_ct1,
              full_row_indices_data_ptr_int_ct2, mul_data_ptr_float_ct3, height,
              width, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
}

//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_sycl(
    torch::Tensor rows, torch::Tensor cols, torch::Tensor mat,
    torch::Tensor vec, torch::Tensor full_rows, torch::Tensor full_row_indices,
    torch::Tensor mul, int num_rows, torch::Tensor mat3,
    torch::Tensor lookup_table) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  sycl::range<3> blocks3(1, (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3);
  sycl::range<3> threads3(1, 1, BLOCKWIDTH);

  q_ct1.submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);
    
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(8, 128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const int *mat3_data_ptr_int_ct1 = mat3.data_ptr<int>();
    float *mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    const float *lookup_table_data_ptr_float_ct3 =
        lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks3 * threads3, threads3),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant3MatMulKernelNUQPerChannelBatched(
              vec_data_ptr_float_ct0, mat3_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height3,
              width3, batch, vec_height, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              deq2_acc_ct1);
        });
  });

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic_batched", ([&] {
      
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      const int *rows_data_int_ct0 = rows.data<int>();
      const int *cols_data_int_ct1 = cols.data<int>();
      const scalar_t *mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      const scalar_t *vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      scalar_t *mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC_BATCHED(
                             rows_data_int_ct0, cols_data_int_ct1,
                             mat_data_scalar_t_ct2, vec_data_scalar_t_ct3,
                             mul_data_scalar_t_ct4, num_rows, batch, vec_height,
                             item_ct1);
                       });
    });
                             }));

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  sycl::range<3> blocks_topX(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                             (height + BLOCKWIDTH - 1) / BLOCKWIDTH);
  sycl::range<3> threads_topX(1, 1, BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  q_ct1.submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const float *full_rows_data_ptr_float_ct1 = full_rows.data_ptr<float>();
    const int *full_row_indices_data_ptr_int_ct2 =
        full_row_indices.data_ptr<int>();
    float *mul_data_ptr_float_ct3 = mul.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks_topX * threads_topX, threads_topX),
        [=](sycl::nd_item<3> item_ct1) {
          DenseMatVecKernelBatched(
              vec_data_ptr_float_ct0, full_rows_data_ptr_float_ct1,
              full_row_indices_data_ptr_int_ct2, mul_data_ptr_float_ct3, height,
              width, batch, vec_height, matwidth, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_sycl(
    torch::Tensor rows, torch::Tensor cols, torch::Tensor mat,
    torch::Tensor vec, torch::Tensor full_rows, torch::Tensor full_row_indices,
    torch::Tensor mul, int num_rows, torch::Tensor mat4,
    torch::Tensor lookup_table) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  sycl::range<3> blocks4(1, (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads4(1, 1, BLOCKWIDTH);

  q_ct1.submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(16, 128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const int *mat4_data_ptr_int_ct1 = mat4.data_ptr<int>();
    float *mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    const float *lookup_table_data_ptr_float_ct3 =
        lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks4 * threads4, threads4),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant4MatMulKernelNUQPerChannelBatched(
              vec_data_ptr_float_ct0, mat4_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height4,
              width4, batch, vec_height, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              deq2_acc_ct1);
        });
  });

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic_batched", ([&] {
      
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      const int *rows_data_int_ct0 = rows.data<int>();
      const int *cols_data_int_ct1 = cols.data<int>();
      const scalar_t *mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      const scalar_t *vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      scalar_t *mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC_BATCHED(
                             rows_data_int_ct0, cols_data_int_ct1,
                             mat_data_scalar_t_ct2, vec_data_scalar_t_ct3,
                             mul_data_scalar_t_ct4, num_rows, batch, vec_height,
                             item_ct1);
                       });
    });
                             }));

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  sycl::range<3> blocks_topX(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                             (height + BLOCKWIDTH - 1) / BLOCKWIDTH);
  sycl::range<3> threads_topX(1, 1, BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  q_ct1.submit([&](sycl::handler &cgh) {
    
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128), cgh);

    const float *vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    const float *full_rows_data_ptr_float_ct1 = full_rows.data_ptr<float>();
    const int *full_row_indices_data_ptr_int_ct2 =
        full_row_indices.data_ptr<int>();
    float *mul_data_ptr_float_ct3 = mul.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks_topX * threads_topX, threads_topX),
        [=](sycl::nd_item<3> item_ct1) {
          DenseMatVecKernelBatched(
              vec_data_ptr_float_ct0, full_rows_data_ptr_float_ct1,
              full_row_indices_data_ptr_int_ct2, mul_data_ptr_float_ct3, height,
              width, batch, vec_height, matwidth, item_ct1,
              blockvec_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
}


void VecQuant3MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2) {

  int row = BLOCKHEIGHT3 * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  blockvec[item_ct1.get_local_id(2)] =
      vec[(row / BLOCKHEIGHT3) * BLOCKWIDTH + item_ct1.get_local_id(2)];

  //Modified dequant block

  int off = item_ct1.get_local_id(2);
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

  
  item_ct1.barrier(sycl::access::fence_space::local_space);

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

  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[col],
                                                                     res);
}

//4-bit per-channel
void VecQuant4MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2) {

  int row = BLOCKHEIGHT4 * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  blockvec[item_ct1.get_local_id(2)] =
      vec[(row / BLOCKHEIGHT4) * BLOCKWIDTH + item_ct1.get_local_id(2)];

  //Modified dequant block

  int off = item_ct1.get_local_id(2);
  int column_offset = col * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = column_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  
  item_ct1.barrier(sycl::access::fence_space::local_space);

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

  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[col],
                                                                     res);
}


//batched version (3-bit)
void VecQuant3MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2) {

  int row = BLOCKHEIGHT3 * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  int off = item_ct1.get_local_id(2);
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

    
    item_ct1.barrier(sycl::access::fence_space::local_space);
    blockvec[item_ct1.get_local_id(2)] =
        vec[b * vec_height + (row / BLOCKHEIGHT3) * BLOCKWIDTH +
            item_ct1.get_local_id(2)];
    
    item_ct1.barrier(sycl::access::fence_space::local_space);

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

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &mul[b * width + col], res);
  }
}

//batched version (4-bit)
void VecQuant4MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2) {

  int row = BLOCKHEIGHT4 * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  //Modified dequant block

  int off = item_ct1.get_local_id(2);
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

    
    item_ct1.barrier(sycl::access::fence_space::local_space);
    blockvec[item_ct1.get_local_id(2)] =
        vec[b * vec_height + (row / BLOCKHEIGHT4) * BLOCKWIDTH +
            item_ct1.get_local_id(2)];
    
    item_ct1.barrier(sycl::access::fence_space::local_space);

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

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &mul[b * width + col], res);
  }
}

template <typename scalar_t>
void SPMV_ATOMIC(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows,
  const sycl::nd_item<3> &item_ct1) {
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (row < num_rows) {
        float dot = 0;
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int i = start_elem; i < end_elem; i++) {
            dot += mat[i] * vec[cols[i]];
        }
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &mul[row], dot);
    }
}

template <typename scalar_t>
void SPMV_ATOMIC_BATCHED(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows,
  int batch,
  int vec_height,
  const sycl::nd_item<3> &item_ct1) {
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);

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
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &mul[b * num_rows + row], dot);
            // atomicAdd(&mul[row * batch + b], dot);
            // atomicAdd(&mul[row], dot);
        }
    }
}

// Dense kernel for only a subset of rows
void DenseMatVecKernel(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec) {

  int row = BLOCKWIDTH * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  blockvec[item_ct1.get_local_id(2)] = vec[row + item_ct1.get_local_id(2)];

  item_ct1.barrier(sycl::access::fence_space::local_space);

  int i = width * row + col;
  int k = 0;
  float res = 0;

  if (item_ct1.get_local_id(2) < width) {
    while (k < BLOCKWIDTH) {
      res += full_rows[i] * blockvec[k];
      k += 1;
      i += width;
    }

    int col_idx = full_row_indices[col];
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &mul[col_idx], res);
  }
}


// Dense kernel for only a subset of rows
void DenseMatVecKernelBatched(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width,
    int batch,
    int vec_height,
    int matwidth,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec) {

  int row = BLOCKWIDTH * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  for (int b = 0; b < batch; ++b){
    int i = width * row + col;
    int k = 0;
    float res = 0;

    
    item_ct1.barrier(sycl::access::fence_space::local_space);
    blockvec[item_ct1.get_local_id(2)] =
        vec[b * vec_height + row + item_ct1.get_local_id(2)];
    
    
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (item_ct1.get_local_id(2) < width) {
      while (k < BLOCKWIDTH) {
        res += full_rows[i] * blockvec[k];
        k += 1;
        i += width;
      }

      int col_idx = full_row_indices[col];
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          &mul[b * matwidth + col_idx], res);
    }
  }
}