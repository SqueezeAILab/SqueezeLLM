---
license: apache-2.0
---

# XGen-7B-8K-Base

Official research release for the family of **XGen** models (`7B`) by Salesforce AI Research:

*Title*: [Long Sequence Modeling with XGen: A 7B LLM Trained on 8K Input Sequence Length](https://blog.salesforceairesearch.com/xgen/)

*Authors*: [Erik Nijkamp](https://eriknijkamp.com)\*, Tian Xie\*, [Hiroaki Hayashi](https://hiroakih.me/)\*, [Bo Pang](https://scholar.google.com/citations?user=s9fNEVEAAAAJ&hl=en)\*, Congying Xia\*, Chen Xing, Jesse Vig, Semih Yavuz, Philippe Laban, Ben Krause, Senthil Purushwalkam, Tong Niu, Wojciech Kryscinski, Lidiya Murakhovs'ka, Prafulla Kumar Choubey, Alex Fabbri, Ye Liu, Rui Meng, Lifu Tu, Meghana Bhat, [Chien-Sheng Wu](https://jasonwu0731.github.io/), Silvio Savarese, [Yingbo Zhou](https://scholar.google.com/citations?user=H_6RQ7oAAAAJ&hl=en), [Shafiq Rayhan Joty](https://raihanjoty.github.io/), [Caiming Xiong](http://cmxiong.com/).

(* indicates equal contribution)

Correspondence to: [Shafiq Rayhan Joty](mailto:sjoty@salesforce.com), [Caiming Xiong](mailto:cxiong@salesforce.com)

## Models

### Base models
* [XGen-7B-4K-Base](https://huggingface.co/Salesforce/xgen-7b-4k-base): XGen-7B model pre-trained under 4K sequence length.
  * License: Apache-2.0
* [XGen-7B-8K-Base](https://huggingface.co/Salesforce/xgen-7b-8k-base): XGen-7B model pre-trained under 8K sequence length.
  * License: Apache-2.0

### Instruction-finetuned models

Supervised finetuned model on public domain instructional data. Released for ***research purpose*** only.

* [XGen-7B-8K-Inst](https://huggingface.co/Salesforce/xgen-7b-8k-inst)

## How to run

The training data for the models are tokenized with OpenAI Tiktoken library.
To use this model, install the package via `pip`:

```sh
pip install tiktoken
```

The models can be used as auto-regressive samplers as follows:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen-7b-8k-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Salesforce/xgen-7b-8k-base", torch_dtype=torch.bfloat16)
inputs = tokenizer("The world is", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0]))
```

## Citation

```bibtex
@misc{XGen,
  title={Long Sequence Modeling with XGen: A 7B LLM Trained on 8K Input Sequence Length},
  author={Erik Nijkamp, Tian Xie, Hiroaki Hayashi, Bo Pang, Congying Xia, Chen Xing, Jesse Vig, Semih Yavuz, Philippe Laban, Ben Krause, Senthil Purushwalkam, Tong Niu, Wojciech Kryscinski, Lidiya Murakhovs'ka, Prafulla Kumar Choubey, Alex Fabbri, Ye Liu, Rui Meng, Lifu Tu, Meghana Bhat, Chien-Sheng Wu, Silvio Savarese, Yingbo Zhou, Shafiq Rayhan Joty, Caiming Xiong},
  howpublished={Salesforce AI Research Blog},
  year={2023},
  url={https://blog.salesforceairesearch.com/xgen}
}
```
