#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void vecquant3matmul_nuq_perchannel_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);
void vecquant4matmul_nuq_perchannel_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);
void vecquant3matmul_nuq_perchannel_batched_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);
void vecquant4matmul_nuq_perchannel_batched_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);

void vecquant3matmul_nuq_perchannel(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_cuda(vec, mat, mul, lookup_table);
}
void vecquant4matmul_nuq_perchannel(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_cuda(vec, mat, mul, lookup_table);
}
void vecquant3matmul_nuq_perchannel_batched(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_batched_cuda(vec, mat, mul, lookup_table);
}
void vecquant4matmul_nuq_perchannel_batched(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_batched_cuda(vec, mat, mul, lookup_table);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant3matmul_nuq_perchannel", &vecquant3matmul_nuq_perchannel, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant4matmul_nuq_perchannel", &vecquant4matmul_nuq_perchannel, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant3matmul_nuq_perchannel_batched", &vecquant3matmul_nuq_perchannel_batched, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant4matmul_nuq_perchannel_batched", &vecquant4matmul_nuq_perchannel_batched, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
}
