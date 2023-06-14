# SqueezeLLM: Dense-and-Sparse Quantization [[Paper](https://arxiv.org/abs/2306.07629)]


![image](https://github.com/chooper1/squeezegpt-tmp/assets/50283958/07d99c71-0342-42b2-9e02-364179bf0e72)

Post-training LLM Quantization for efficient and accurate **3bit** and **4bit** inference, allowing **4-5x memory footprint reduction** and **2x speedup**.

Our framework incorporates **2 simple ideas** to achieve near lossless 3-4 bit quantization:
* Sensitivity-based Non-uniform Quantization
* Dense-and-Sparse Decomposition for Outlier Handling

Check out our [paper](https://arxiv.org/abs/2306.07629) for more details. This code is based on [GPTQ](https://github.com/IST-DASLab/gptq) as well as [GPTQ-For-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa/).

---
## Installation

1. Create a conda environment
```
conda create --name sqllm python=3.9 -y
conda activate sqllm
```

2. Clone and install the dependencies
```
git clone https://github.com/SqueezeAILab/SqueezeLLM
cd SqueezeLLM
pip install -e .
cd squeezellm
python setup_cuda.py install
```

---

## Supported Models

Currently, we support [LLaMA](https://arxiv.org/abs/2302.13971) 7B, 13B, and 30B, as well as the instruction-tuned [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) 7B and 13B.
For each model, we support 3-bit and 4-bit quantized models, with sparse levels of 0% (dense-only), 0.05%, and 0.45%.
See our [Paper](https://arxiv.org/abs/2306.07629) for more detailed information on these configurations.
Below are the links to download the models.

### LLaMA

| Model |  Bitwidth | Dense-only (0%) |
| -------- | -------- | -------- |
| LLaMA-7B    | 3   |  [sq-llama-7b-w3-o0](https://huggingface.co/squeeze-ai-lab/sq-llama-7b-w3-o0/blob/main/sq-llama-7b-w3-o0.pt) |
| LLaMA-7B    | 4   | [sq-llama-7b-w4-o0](https://huggingface.co/squeeze-ai-lab/sq-llama-7b-w4-o0/blob/main/sq-llama-7b-w4-o0.pt) |
| LLaMA-13B    | 3   |  [sq-llama-13b-w3-o0](https://huggingface.co/squeeze-ai-lab/sq-llama-13b-w3-o0/blob/main/sq-llama-13b-w3-o0.pt) |
| LLaMA-13B    | 4   | [sq-llama-13b-w4-o0](https://huggingface.co/squeeze-ai-lab/sq-llama-13b-w4-o0/blob/main/sq-llama-13b-w4-o0.pt) |
| LLaMA-30B    | 3   |  [sq-llama-30b-w3-o0]() |
| LLaMA-30B    | 4   | [sq-llama-30b-w4-o0]() |

### Vicuna

| Model |  Bitwidth | Dense-only (0%) |
| -------- | -------- | -------- |
| Vicuna-7B    | 3   | [Link]() |
| Vicuna-7B    | 4     | [Link]() |
| Vicuna-13B    | 3     | [Link]() |
| Vicuna-13B    | 4    | [Link]() |

**NOTE:** Sparsity levels with 0.05% and 0.45% are coming soon!

---

## Run

The packed checkpoints of our quantized models are all available in [Huggingface Model Hub](https://huggingface.co/squeeze-ai-lab) and can be downloaded from the links above.
However, we require the users to have their own LLaMA or Vicuna checkpoints locally, as our checkpoints only include the quantized linear layers and do not contain other layers like Embedding layers to avoid uploading the full model.

### Benchmarking

The following code will run and benchmark the 3-bit quantized LLaMA-7B model on the C4 dataset. The "torch_profile" argument can be passed when running benchmarking to replicate the runtime results from the paper.
Follow the same procedure for other quantized models.
```
# After downloading the quantized model to sq-llama-7b-w3-o0.pt locally
CUDA_VISIBLE_DEVICES=0 python llama.py <path-to-llama-7b-hf> c4 --wbits 4 --load sq-llama-7b-w3-o0.pt --benchmark 128 --check
```

### Perplexity Evaluation

The following code will evaluate perplexity using the 3-bit quantized LLaMA-7B model on the C4 dataset, following the same evaluation methodology of [GPTQ](https://github.com/IST-DASLab/gptq) and [GPTQ-For-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa/).
Follow the same procedure for other quantized models.
```
CUDA_VISIBLE_DEVICES=0 python llama.py <path-to-llama-7b-hf> c4 --wbits 4 --load sq-llama-7b-w3-o0.pt --eval
```

The code was tested on A5000 and A6000 GPUs with Cuda 11.3 and CUDNN 8.2.

---

## Details

SqueezeLLM is a framework for efficient ultra-low precision inference with LLMs. SqueezeLLM achieves ultra-low precision of up to **3-bit** **without retraining** through two key methods:

1) **Sensitivity-Based Non-Uniform Quantization**: we employ LUT-based non-uniform quantization to minimize the performance degradation from quantizing to low precision.

   *  Generative LLM inference is *memory bandwidth-bound*
   *  That is, we can improve performance by reducing memory traffic, and the overhead from additional compute operations is unimportant

2) **Dense and Sparse Quantization**: we decompose the model into both dense and sparse components in order to optimally handle outliers without skewing the bits in the non-uniform quantization scheme.

   * We observed that 99% of the values in the weight matrices are clustered within under 10% of the dynamic range
   * By removing the outlier values, we can reduce the strain on our low-precision dense matrix
   * This is particularly useful at low-bit precisions


---

## Reference
```
@article{kim2023squeezellm,
  title={SqueezeLLM: Dense-and-Sparse Quantization},
  author={Kim, Sehoon and Hooper, Coleman and Gholami, Amir and Dong, Zhen and Li, Xiuyu and Shen, Sheng and Mahoney, Michael and Keutzer, Kurt},
  journal={arXiv},
  year={2023}
}
```
