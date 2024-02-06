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

### 3. (Optional for D+S quantization) Outlier configuration generation
This is an optional step to generate an outlier configuration for the Dense-and-Sparse quantization.
This step creates a configuration file that defines thresholds for identifying outlier values in the weights.
Note that while SqueezeLLM extracts both outliers and sensitive values to sparse matrices, 
there's no need for a separate configuration step for sensitive values -- this is covered in step 4.
This step only generates a configuration of outlier values.

Run the following command to generate an outlier configuration:
```
python generate_outlier_config.py --model [MODEL_CHUNKS_PATH] --range [RANGE] --output [OUTLIERS_CONFIG_PATH]
```

* `--model`: Points to the chunked model weights obtained in the previous step
* `--range`: Determines the thresholds T_min and T_max as described in Section 4.2 of the paper. Adjusting this value changes the absolute values of T_min and T_max, consequently affecting the total number of outliers. A larger range value decreases the number of outliers.
* `--output`: The path where the outlier configuration will be saved, formatted as `[OUTLIERS_CONFIG_PATH]/outlier_config_o{outlier_percentage}.json`, where outlier_percentage represents the percentage of values classified as outliers. You will need to fine-tune the `range` argument to achieve a desired outlier percentage, with 1.5-2.0 being a recommended starting range.



### 4. K-means clustering
Run the following code to perform K-means clustering, which will yield the non-uniform quantization look-up table (LUT):
```
python nuq.py --bit 4 --model_type llama --model [MODEL_CHUNKS_PATH] --gradient [GRADIENT_CHUNKS_PATH] --output [LUT_PATH]
```
The `--bit` argument is the bit-precision, and can be set to either 3 or 4. 
The `--model` and `--gradient` arguments should point to the chunked model weights and gradients obtained in the previous step. 
The resulting LUT entries will be stored in `[LUT_PATH]/lut`.

To only quantize a specific range of layers, you can use the `--range` option. For instance, assigning `--range 0,10` will only compute LUT entries for layers 0 to 9.

Please note that this process is highly CPU-intensive, so it is recommended to run the code in environments with multiple and stronger CPU cores for faster computation.

**Additional arguments for D+S quantization:**
To perform Dense-and-Sparse quantization, e.g. with 0.45% outliers and 0.05% sensitive values as in the paper, you will first need to generate an outlier configuration file for 0.45% outliers by completing step 3. This file will be stored as `[OUTLIERS_CONFIG_PATH]/outlier_config_o0.45.json`.
Then, run the following command:
```
python nuq.py --bit 4 --model_type llama --model [MODEL_CHUNKS_PATH] --gradient [GRADIENT_CHUNKS_PATH] --output [LUT_PATH] --outlier_config [OUTLIERS_CONFIG_PATH]/outlier_config_o0.45.json --sensitivity 0.05
```
* `--outlier_config`: reads the outlier configuration file generated in the previous step to extract 0.45% of the outliers to the sparse matrices accordingly.
* `--sensitivity`: takes out an additional 0.05% of sensitive values to the sparse matrices. 



### 4. Packing
Finally, use the obtained LUT from the previous step to save your model into a packed format. Run the following command:
```
python pack.py --model [MODEL_PATH] --wbits 4 --folder [LUT_PATH] --save [PACKED_CKPT_PATH]
```
`[MODEL_PATH]` is the original model checkpoint, and `[LUT_PATH]` is the location where the LUT is stored from the previous step. 
The packed checkpoint will be saved at `[PACKED_CKPT_PATH]`, which can now be immediately used in your inference code.

**Additional arguments for D+S packing:**
When you performed D+S quantization in the previous step, you will obtain the outliers file as well as LUT in `[LUT_PATH]`.
In this case, you will need to  proceed with D+S packing using the following command:
```
python pack.py --model [MODEL_PATH] --wbits 4 --folder [LUT_PATH] --save [PACKED_CKPT_PATH] --include_sparse --balance
```
This command incorporates two additional arguments, `--include_sparse` and `--balance`.
