import os
os.environ["OMP_NUM_THREADS"] = "1"  # this is necessary to parallelize the kmeans

import argparse
import json
import pickle

import numpy as np
import torch
from sklearn.cluster import KMeans
from squeezellm.model_parse import get_module_names, parse_model
from squeezellm.outliers import remove_outliers
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, help="model weights to load", required=True)
parser.add_argument(
    "--model_type", type=str, default=None, help="model type", choices=["llama", "opt"]
)
parser.add_argument(
    "--gradient", type=str, help="model gradients to load", required=True
)
parser.add_argument(
    "--bit",
    type=int,
    default=3,
    help="bitwidth",
    choices=[2, 3, 4],
)
parser.add_argument(
    "--range", type=str, default=None, help="range of layers to quantize"
)
parser.add_argument(
    "--output_folder", type=str, required=None, help="path to dump the output"
)
parser.add_argument(
    "--outlier_config",
    type=str,
    default=None,
    help="path to outlier config, obtained from generate_outlier_config.py",
)
parser.add_argument(
    "--sensitivity", type=float, default=0, help="sensitivity for outlier extraction"
)


# Define the helper function for parallel k-means
def kmeans_fit(row_data):
    weights_np, sample_weight, n_cluster = row_data
    kmeans = KMeans(
        n_clusters=n_cluster,
        random_state=0,
        n_init="auto",
        max_iter=50,
    ).fit(weights_np, sample_weight=sample_weight)
    return kmeans.cluster_centers_.reshape(-1), np.cast["byte"](kmeans.labels_)


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

    pool = Pool(os.cpu_count())

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
            print(
                f"Removing outliers outlier={outlier_threshold}, sensitivity={args.sensitivity}"
            )
            outlier_config_layer = outlier_config[l]
            outliers = remove_outliers(
                model=model_layer,
                sensitivity=args.sensitivity,
                outlier_config=outlier_config_layer,
                gradients=gradient_layer,
            )

        config_per_layer = {}

        # Prepare data for parallel execution
        kmeans_tasks = []
        for name in get_module_names(model_type):
            g = gradient_layer[name].float().numpy()
            module_weight = model_layer[name]
            _weights_np = module_weight.float().numpy()
            n_cluster = 2**args.bit

            for i in range(module_weight.shape[0]):
                weights_np_temp = _weights_np[i, :]
                weights_np = weights_np_temp.reshape(-1, 1)
                weight_mask = weights_np_temp != 0
                sample_weight = g[i, :] * weight_mask
                if np.sum(sample_weight) == 0:
                    sample_weight = np.ones_like(sample_weight)
                kmeans_tasks.append((weights_np, sample_weight, n_cluster))

        # Parallel execution using Pool
        kmeans_results = list(tqdm(pool.imap(kmeans_fit, kmeans_tasks), total=len(kmeans_tasks)))

        # Organize results
        task_idx = 0
        for name in get_module_names(model_type):
            config_per_row = []
            module_weight = model_layer[name]
            for i in range(module_weight.shape[0]):
                centers, labels = kmeans_results[task_idx]
                task_idx += 1
                config_per_row.append([(centers, labels)])
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
    
    pool.close()
    pool.join()

