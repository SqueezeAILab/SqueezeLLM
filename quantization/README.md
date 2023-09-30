## From-scratch Quantization

Here we provide the codes for quantizing your custom models from scratch. Follow the steps outlined below.

**NOTE:** Only dense-only quantization is supported now. We are working to release the code for the Dense-and-Sparse quantization soon.

### 0. Prerequisite

In addition to installing the dependencies required for the inference code, you will need to install additional dependencies by running the following command:
```
conda activate sqllm
pip install scikit-learn==1.3.1
```
Additionally, make sure you have your own LLaMA Huggingface checkpoint saved at `[MODEL_PATH]`.


### 1. Compute gradients (Fisher-based sensitivity score)
SqueezeLLM employs the Fisher Information matrix as a sensitivity metric.
To compute this, we offer a separate [separate framework](https://github.com/kssteven418/SqueezeLLM-gradients) where you can compute the gradient square for your target model. 
This framework will produce the gradient square in the same format as the original Huggingface model checkpoint for your target model, with the only difference being that the weight values are replaced by the gradient square.

### 2. Chunk model weights and gradients
You should now have the model checkpoint at `[MODEL_PATH]` and the gradient checkpoint computed in the previous step at `[GRADIENT_PATH]`. 
Our framework requires that both checkpoints are chunked at the layer granularity to reduce the model loading overhead. 
Run the following code to chunk both your model and gradient checkpoints:
```
python chunk_models.py --model [MODEL_PATH] --output [MODEL_CHUNKS_PATH] --model_type llama
python chunk_models.py --model [GRADIENT_PATH] --output [GRADIENT_CHUNKS_PATH] --model_type llama
```

This will save model weights and gradients in the layer granularity as `[MODEL_CHUNKS_PATH]` and `[GRADIENT_CHUNKS_PATH]`.

### 3. K-means clustering
Run the following code to perform K-means clustering, which will yield the non-uniform quantization look-up table (LUT):
```
python nuq.py --bit 4 --model_type llama --model [MODEL_CHUNKS_PATH] --gradient [GRADIENT_CHUNKS_PATH] --output [LUT_PATH]
```
The `--bit` argument is the bit-precision, and can be set to either 3 or 4. 
The `--model` and `--gradient` arguments should point to the chunked model weights and gradients obtained in the previous step. 
The resulting LUT entries will be stored in `[LUT_PATH]/lut`.

To only quantize a specific range of layers, you can use the `--range` option. For instance, assigning `--range 0,10` will only compute LUT entries for layers 0 to 9.

Please note that this process is highly CPU-intensive, so it is recommended to run the code in environments with multiple and stronger CPU cores for faster computation.

### 4. Packing
Finally, use the obtained LUT from the previous step to save your model into a packed format. Run the following command:
```
python pack.py --model [MODEL_PATH] --wbits 4 --folder [LUT_PATH] --save [PACKED_CKPT_PATH]
```
`[MODEL_PATH]` is the original model checkpoint, and `[LUT_PATH]` is the location where the LUT is stored from the previous step. 
The packed checkpoint will be saved at `[PACKED_CKPT_PATH]`, which can now be immediately used in your inference code.
