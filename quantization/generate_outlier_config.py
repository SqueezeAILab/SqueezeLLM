import argparse
import json
import os

import numpy as np
import torch
from squeezellm.model_parse import get_module_names, parse_model
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, help="llama model to load")
parser.add_argument(
    "--model_type", type=str, default=None, help="model type", choices=["llama", "opt"]
)
parser.add_argument("--cache_dir", type=str, default=None, help="cache directory")
parser.add_argument(
    "--range",
    type=float,
    required=True,
    help="threshold for outlier range, e.g. 1.8",
)
parser.add_argument(
    "--output_folder", type=str, required=None, help="path to dump the output"
)

args = parser.parse_args()

model_type = args.model_type or parse_model(args.model)
threshold_range = args.range

nlayers = len([f for f in os.listdir(args.model)])

total_params = 0
total_outliers = 0
json_data = []
for l in tqdm(range(nlayers)):
    try:
        model_layer = torch.load(f"{args.model}/layer_{l}.pt")
    except:
        raise Exception(f"Needs chunked model weight file at {args.model}/layer_{l}.pt")
    layer_json = {}

    for name in get_module_names(model_type):
        module_weight = model_layer[name]
        weights_np = module_weight.detach().numpy()
        q1 = np.quantile(weights_np, 0.25)
        q3 = np.quantile(weights_np, 0.75)
        minimum = q1 - threshold_range * (q3 - q1)
        maximum = q3 + threshold_range * (q3 - q1)
        minimum, maximum = -max(abs(minimum), abs(maximum)), max(
            abs(minimum), abs(maximum)
        )
        num_params = weights_np.shape[0] * weights_np.shape[1]
        num_outliers = (weights_np < minimum).sum() + (weights_np > maximum).sum()
        total_params += num_params
        total_outliers += num_outliers

        print(l, name, f"% outlier: {num_outliers / num_params * 100:.3f}%")
        layer_json[name] = maximum

    json_data.append(layer_json)

# Save the json data
outlier_percentage = total_outliers / total_params * 100

outlier_folder = f"{args.output_folder}"
if not os.path.exists(outlier_folder):
    os.makedirs(outlier_folder)

o = round(outlier_percentage, 2)
json_data = {
    "outlier_threshold": o,
    "outlier_config": json_data,
}

with open(f"{outlier_folder}/outlier_config_o{o}.json", "w") as f:
    json.dump(json_data, f, indent=4)
