import json
import os
import pickle
import time

import torch
import torch.nn as nn
import transformers
from squeezellm.modelutils import *
from squeezellm.quant import *


@torch.no_grad()
def llama_sequential(model, folder, include_sparse):
    print("Starting ...")

    layers = model.model.layers

    quantizers = {}
    for i in range(len(layers)):
        with open(f"{folder}/lut/l{i}.pkl", "rb") as f:
            # dictionary: key ["q", "k", "v", "o", "gate", "up", "down"] -> list of length channel,
            # each of which is a list of #group lists, (here, it's always 1)
            # and each of them are a tuple (centroids, indices)
            lut_layer = pickle.load(f)

        if include_sparse:
            with open(f"{folder}/outliers/l{i}.pkl", "rb") as f:
                # dictionary: key ["q", "k", "v", "o", "gate", "up", "down"] -> list of length channel,
                # each of which is a list of #group lists, (here, it's always 1)
                # and each of them are a tuple (centroids, indices)
                outlier_list_layer = pickle.load(f)

        sequential_lut = ["q", "k", "v", "o", "gate", "up", "down"]
        sequential_lut_real_name = {
            "q": "self_attn.q_proj",
            "k": "self_attn.k_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.o_proj",
            "gate": "mlp.gate_proj",
            "up": "mlp.up_proj",
            "down": "mlp.down_proj",
        }

        outlier_index = {"q": 0, "k": 1, "v": 2, "o": 3, "gate": 4, "up": 5, "down": 6}

        for s in sequential_lut:
            lut = lut_layer[s]
            if include_sparse:
                idx = outlier_index[s]
                outliers = outlier_list_layer[idx]
            else:
                outliers = None
            name = sequential_lut_real_name[s]
            quantizers["model.layers.%d.%s" % (i, name)] = [lut, outliers]

    return quantizers


def llama_pack(
    model,
    quantizers,
    wbits,
    include_sparse,
    balanced,
    num_nonzero_per_thread,
):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant_lut(
        model,
        quantizers,
        wbits,
        include_sparse=include_sparse,
        balanced=balanced,
    )

    qlayers = find_layers(model, [QuantLinearLUT])
    print("Packing ...")
    sparsedict = {}

    for name in qlayers:
        print(name)
        lookup_table = quantizers[name]
        layers[name].cpu()
        qlayers[name].pack2(
            layers[name],
            lookup_table,
            include_sparse,
            num_nonzero_per_thread=num_nonzero_per_thread,
        )
        if include_sparse:
            sparsedict[name] = qlayers[name].vals.shape[-1]

    print("Done.")
    return model, sparsedict


if __name__ == "__main__":
    import argparse

    from squeezellm.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="llama model to load")
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--save",
        type=str,
        required=True,
        help="Save quantized checkpoint under this name.",
    )

    # sparse args
    parser.add_argument(
        "--folder",
        type=str,
        default="",
        help="Path to folder containing luts and outliers.",
    )
    parser.add_argument(
        "--include_sparse",
        action="store_true",
        help="Whether loaded checkpoint has sparse matrix.",
    )

    #balanced kernel arguments
    parser.add_argument(
        '--balanced', action='store_true',
        help='Whether to use balanced sparse kernel.'
    )
    parser.add_argument(
        '--num_nonzero_per_thread', type=int, default=10,
        help='Num nonzeros assigned to each thread.'
    )


    args = parser.parse_args()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype="auto"
    )
    model.eval()

    print("Running llama_sequential")
    tick = time.time()
    quantizers = llama_sequential(
        model=model,
        folder=args.folder,
        include_sparse=args.include_sparse,
    )
    print("llama_sequential Done:", time.time() - tick)

    print("Running llama_pack")
    tick = time.time()
    model, numvals = llama_pack(
        model=model,
        quantizers=quantizers,
        wbits=args.wbits,
        include_sparse=args.include_sparse,
        balanced=args.balanced,
        num_nonzero_per_thread=args.num_nonzero_per_thread,
    )
    print("llama_pack Done:", time.time() - tick)

    model_dict = model.state_dict()

    if args.include_sparse:
        # need to merge in sparse dict
        for k, v in numvals.items():
            model_dict["sparse_threshold." + k] = v

    # save model
    torch.save(model_dict, args.save)

    # get directory to save quant_config
    directory = os.path.dirname(args.save)
    data = {"wbits": args.wbits}
    output_fn = os.path.join(directory, "quant_config.json")

    # save quant_config
    with open(output_fn, "w") as f:
        json.dump(data, f, indent=4)
