import argparse
import os

import torch
from squeezellm.model_parse import (
    get_layers,
    get_module_names,
    get_modules,
    load_model,
    parse_model,
)
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path", type=str, default=None, help="chunk the model and store"
)
parser.add_argument("--model", type=str, help="model to load")
parser.add_argument(
    "--model_type",
    type=str,
    default=None,
    help="model type",
    choices=["llama", "opt", "mistral"],
)

args = parser.parse_args()
# if model type is not explicitly given, infer from the model name
model_type = args.model_type or parse_model(args.model)

# This path is only taken when we want to chunk the model and store it,
# which is used when '--output_path' is passed as an argument.
print(f"chunking the model: {args.model} and storing in {args.output_path}")
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
model = load_model(args.model, model_type)
layers = get_layers(model, model_type)
for i, layer in tqdm(enumerate(layers)):
    data = {}
    modules = get_modules(layer, model_type)
    module_names = get_module_names(model_type)

    for lin, name in zip(modules, module_names):
        data[name] = lin.weight.data
    torch.save(data, os.path.join(args.output_path, f"layer_{i}.pt"))
