import time

import torch
import torch.nn as nn

from squeezellm.modelutils import *
from squeezellm.quant import *

from squeezellm.model_parse import (
    parse_model,
    get_layers,
    get_embedding,
    get_norm,
)


def get_model(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model)
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_eval(model, testenc, dev):
    print("Evaluating ...")
    model_type = parse_model(model)

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_layers(model, model_type)
    embeddings = get_embedding(model, model_type)
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i].to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if "position_ids" in kwargs:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i].cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache.get("position_ids")

    for i in range(len(layers)):
        print("Layer", i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            if model_type == "opt":
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                )[0]
            else:
                assert model_type in ("llama", "mistral")
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    norm = get_norm(model, model_type)
    if norm is not None:
        norm = norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if norm is not None:
            hidden_states = norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


# loading quantized checkpoint
def load_quant(model, checkpoint, wbits, include_sparse, topX):
    if (
        "xgen" in checkpoint
        or "opt" in checkpoint
        or ("vicuna" in checkpoint and "v1.3" in checkpoint)
        or "llama-2" in checkpoint
        or "mistral" in checkpoint
    ):
        # TODO: this is a hacky solution, will be preperly implemented after all the model checkpoints are updated with
        # the new packing scheme that includes the non-linear weights
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model)
        model = AutoModelForCausalLM.from_config(config)
    else:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
    model = model.eval()
    layers = find_layers(model)

    state_dict = torch.load(checkpoint)

    # load sparse thresholds from checkpoint
    if include_sparse:
        num_vals = {}
        for k, v in state_dict.items():
            if "sparse_threshold." in k:
                key = k.replace("sparse_threshold.", "")
                num_vals[key] = v
        for k, v in num_vals.items():
            del state_dict["sparse_threshold." + k]
    else:
        num_vals = None

    # replace layers
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]
    make_quant_lut(
        model, layers, wbits, include_sparse=include_sparse, numvals=num_vals, topX=topX
    )
    del layers

    print("Loading model ...")
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.seqlen = 2048
    print("Done.")

    return model


# function for benchmarking runtime
def benchmark(model, input_ids, check=False):
    model_type = parse_model(model)
    layers = get_layers(model, model_type)

    input_ids = input_ids.to(model.gpus[0] if hasattr(model, "gpus") else DEV)
    torch.cuda.synchronize()

    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    for i, layer in enumerate(layers):
        layer.register_forward_hook(clear_past(i))

    print("Benchmarking ...")

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.0

    def sync():
        if hasattr(model, "gpus"):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i : i + 1],
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, : (i + 1)].reshape((1, -1)),
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            max_memory = max(max_memory, torch.cuda.memory_allocated() / 1024 / 1024)
            if check and i != input_ids.numel() - 1:
                tot += loss(
                    out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)
                ).float()
            cache["past"] = list(out.past_key_values)
            del out
        sync()
        import numpy as np

        print("Median:", np.median(times))
        if check:
            print("PPL:", torch.exp(tot / (input_ids.numel() - 1)).item())
            print("max memory(MiB):", max_memory)


if __name__ == "__main__":
    import argparse
    from squeezellm.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="llama model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Which dataset to use for benchmarking.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument("--eval", action="store_true", help="evaluate quantized model.")
    parser.add_argument("--load", type=str, default="", help="Load quantized model.")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Number of tokens to use for benchmarking.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Whether to compute perplexity during benchmarking for verification.",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--torch_profile",
        action="store_true",
        help="Use CUDA profiling tool for timing runs.",
    )
    parser.add_argument(
        "--include_sparse",
        action="store_true",
        help="Whether loaded checkpoint has sparse matrix.",
    )
    parser.add_argument(
        "--num_dense_channels",
        type=int,
        default=10,
        help="Number of dense channel used for hybrid kernel.",
    )

    DEV = torch.device("cuda:0")

    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        print(args.model)
        model = load_quant(
            args.model,
            args.load,
            args.wbits,
            args.include_sparse,
            args.num_dense_channels,
        )
    else:
        model = get_model(args.model)
        model.eval()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    if args.benchmark:
        model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, : args.benchmark]

            if args.torch_profile:
                from torch.profiler import profile, record_function, ProfilerActivity

                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as p:
                    benchmark(model, input_ids, check=args.check)
                print(
                    p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
                )
            else:
                benchmark(model, input_ids, check=args.check)

    if args.eval:
        datasets = ["wikitext2", "c4"]
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            llama_eval(model, testloader, DEV)
