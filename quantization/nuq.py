import os
import torch
import pickle
import argparse
import numpy as np
import json
from sklearn.cluster import KMeans

from tqdm import tqdm
from transformers import LlamaForCausalLM

from squeezellm.model_parse import parse_model, get_module_names

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model', type=str,
    help='model weights to load', required=True
)
parser.add_argument(
    '--model_type', type=str, default=None,
    help='model type', choices=['llama', 'opt']
)
parser.add_argument(
    '--gradient', type=str,
    help='model gradients to load', required=True
)
parser.add_argument(
    '--bit', type=int, default=3,
    help='bitwidth', choices=[3, 4],
)
parser.add_argument(
    '--range', type=str, default=None,
    help='range of layers to quantize'
)
parser.add_argument(
    '--output_folder', type=str, required=None,
    help='path to dump the output'
)
parser.add_argument(
    '--outlier_config', type=str, default=None,
    help='path to outlier config, obtained from generate_outlier_config.py'
)
parser.add_argument(
    '--sensitivity', type=float, default=0,
    help='sensitivity for outlier extraction'
)



def remove_outliers_by_sensitivity(
    model, 
    gradients, 
    sensitivity,
):
    module_names = list(model.keys())
    outlier_weights = []
    total_outliers = 0
    total_weights = 0

    def _body(gweight, weight):
        num_outliers = int(gweight.numel() * sensitivity / 100)
        thres = gweight.reshape(-1).topk(k=num_outliers).values[-1]
        
        t = gweight > thres

        outlier_weight = weight * t
        weight = weight * ~t
        #print((weight == 0).sum().item() / weight.numel())
        return  weight.to(weight.dtype), outlier_weight, t.sum().item(), t.numel()

    for _name in module_names:
        weight = model[_name].to(torch.float)
        gweight = gradients[_name].to(torch.float)
        new_weight, outlier_weight, _total_outliers, _total_weights = _body(
            gweight, weight
        )
        model[_name] = new_weight
        total_outliers += _total_outliers
        total_weights += _total_weights
        outlier_weights.append(outlier_weight)

    print("p outlier:", total_outliers / total_weights * 100)
    return [outlier_weights]


def remove_outliers_by_threshold(
    model, 
    outlier_config,
    outlier_weights=None, # to avoid memory leak
):
    module_names = list(model.keys())
    if outlier_weights is None:
        outlier_weights = [[0 for _ in range(len(module_names))]]

    total_outliers = 0
    total_weights = 0

    def _body(weight, thres):
        t = torch.logical_or(
            weight >= thres,  
            weight <= -thres,
        )

        outlier_weight = weight * t
        weight = weight * ~t
        return weight.to(weight.dtype), outlier_weight, t.sum().item(), t.numel()

    for i, name in enumerate(module_names):
        thres = outlier_config[name]
        weight = model[name].to(torch.float)
        new_weight, outlier_weight, _total_outliers, _total_weights = _body(
            weight, thres
        )
        model[name] = new_weight
        total_outliers += _total_outliers
        total_weights += _total_weights
        #import pdb; pdb.set_trace()
        outlier_weights[0][i] += outlier_weight

    print("p outlier:", total_outliers / total_weights * 100)
    return outlier_weights



def remove_outliers(
    model, 
    sensitivity, 
    outlier_config,
    gradients=None,
):
    # model and gradients are dictionary of a layer component
    # where the key is the layer name (e.g. q, k, v) and the value is the weight
    assert isinstance(model, dict)
    assert isinstance(gradients, dict) or gradients is None

    assert outlier_config is not None or sensitivity != 0
    if sensitivity != 0:
        assert gradients is not None

    if sensitivity != 0:
        print("removing outliers by sensitivity")
        outlier_weights = remove_outliers_by_sensitivity(
            model=model,
            gradients=gradients,
            sensitivity=sensitivity,
        )
    else:
        outlier_weights = None

    if outlier_config is not None:
        print("removing outliers by threshold")
        outlier_weights = remove_outliers_by_threshold(
            model=model, 
            outlier_config=outlier_config,
            outlier_weights=outlier_weights,
        )

    return outlier_weights










if __name__ == "__main__":
    args = parser.parse_args()

    # if model type is not explicitly given, infer from the model name
    model_type = args.model_type or parse_model(args.model)

    # Run as a outlier extraction mode if outlier config is given or sensitivity is non-zero
    is_outlier_mode = args.outlier_config is not None or args.sensitivity > 0

    if args.outlier_config is not None:
        # load json file
        with open(args.outlier_config, "r") as f:
            outlier_config = json.load(f)
        outlier_threshold = outlier_config["outlier_threshold"]
        outlier_config = outlier_config["outlier_config"]
    else:
        outlier_threshold = 0
        outlier_config = None

    lut_folder = f"{args.output_folder}/lut"
    if not os.path.exists(lut_folder):
        os.makedirs(lut_folder)

    if is_outlier_mode:
        outlier_folder = f"{args.output_folder}/outliers"
        print(
            f"Running outlier mode with outlier folder threshold {outlier_threshold} "
            f"and sensitivity {args.sensitivity}"
        )
        if not os.path.exists(outlier_folder):
            os.makedirs(outlier_folder)

        # store outlier config in the output folder
        with open(f"{args.output_folder}/outlier_config.json", "w") as f:
            json.dump(
                {
                    "outlier_threshold": outlier_threshold,
                    "sensitivity": args.sensitivity,
                    "outlier_config": outlier_config,
                },
                f,
                indent=4,
            )

    if args.range:
        ranges = args.range.split(",")
        ranges = [int(r) for r in ranges]
        ran = list(range(ranges[0], ranges[1]))
    else:
        # Count number of layers based on the chunk item count in the model folder
        # You should not add/delete anything in the folder to make this work
        nlayers = len([f for f in os.listdir(args.model)])
        ran = list(range(nlayers))

    print(f"Quantizing layers {ran}")

    for l in ran:
        if ran is not None and l not in ran:
            print(f"Skipping layer {l}")
            continue

        lut_file_name = f"{lut_folder}/l{l}.pkl"

        if is_outlier_mode:
            outlier_file_name = f"{outlier_folder}/l{l}.pkl"
        else:
            outlier_file_name = None
        
        if os.path.exists(lut_file_name):
            print(f"Skipping layer {l}, file already exists at {lut_file_name}")
            continue

        print(f"Quantizing layer {l}")

        try:
            gradient_layer = torch.load(f"{args.gradient}/layer_{l}.pt")
        except:
            raise Exception(f"Needs chunked gradient file at {gradient_layer}")
            
        try:
            model_layer = torch.load(f"{args.model}/layer_{l}.pt")
        except:
            raise Exception(f"Needs chunked model weight file at {args.model}")

        if is_outlier_mode:
            print(f"Removing outliers outlier={outlier_threshold}, sensitivity={args.sensitivity}")
            outlier_config_layer = outlier_config[l]
            outliers = remove_outliers(
                model=model_layer,
                sensitivity=args.sensitivity, 
                outlier_config=outlier_config_layer,
                gradients=gradient_layer,
            )

        config_per_layer = {}

        for name in tqdm(get_module_names(model_type)):
            g = gradient_layer[name].float().numpy()

            config_per_row = []
            module_weight = model_layer[name]
            _weights_np = module_weight.float().numpy()

            n_cluster = 2 ** args.bit

            # iterate over row
            for i in (range(module_weight.shape[0])):
                config_per_group = []
                weights_np_temp = _weights_np[i, :]
                weights_np = weights_np_temp.reshape(-1, 1)

                weight_mask = weights_np_temp != 0
                sample_weight = g[i, :]
                sample_weight = sample_weight * weight_mask

                if np.sum(sample_weight) == 0:
                    sample_weight = np.ones_like(sample_weight)

                kmeans = KMeans(
                    n_clusters=n_cluster, 
                    random_state=0, 
                    n_init="auto", 
                    max_iter=50,
                ).fit(
                    weights_np, 
                    sample_weight=sample_weight,
                )
                config_per_group.append(
                    (kmeans.cluster_centers_.reshape(-1), np.cast['byte'](kmeans.labels_))
                )
                config_per_row.append(config_per_group)

            config_per_layer[name] = config_per_row

        # save parts
        with open(lut_file_name, "wb") as f:
            print(f"Saving layer lut to {lut_folder}/l{l}.pkl")
            pickle.dump(config_per_layer, f)

        if is_outlier_mode:
            with open(outlier_file_name, "wb") as f:
                print(f"Saving layer outliers to {outlier_folder}/l{l}.pkl")
                # take the first item (since it is a list of length 1)
                pickle.dump([x.to_sparse() for x in outliers[0]], f)
