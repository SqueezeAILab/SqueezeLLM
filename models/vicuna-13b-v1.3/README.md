---
inference: false
---

# Vicuna Model Card

## Model Details

Vicuna is a chat assistant trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT.

- **Developed by:** [LMSYS](https://lmsys.org/)
- **Model type:** An auto-regressive language model based on the transformer architecture.
- **License:** Non-commercial license
- **Finetuned from model:** [LLaMA](https://arxiv.org/abs/2302.13971).

### Model Sources

- **Repository:** https://github.com/lm-sys/FastChat
- **Blog:** https://lmsys.org/blog/2023-03-30-vicuna/
- **Paper:** https://arxiv.org/abs/2306.05685
- **Demo:** https://chat.lmsys.org/

## Uses

The primary use of Vicuna is research on large language models and chatbots.
The primary intended users of the model are researchers and hobbyists in natural language processing, machine learning, and artificial intelligence.

## How to Get Started with the Model

- Command line interface: https://github.com/lm-sys/FastChat#vicuna-weights.
- APIs (OpenAI API, Huggingface API): https://github.com/lm-sys/FastChat/tree/main#api.  

## Training Details

Vicuna v1.3 is fine-tuned from LLaMA with supervised instruction fine-tuning.
The training data is around 140K conversations collected from ShareGPT.com.
See more details in the "Training Details of Vicuna Models" section in the appendix of this [paper](https://arxiv.org/pdf/2306.05685.pdf).

## Evaluation

Vicuna is evaluated with standard benchmarks, human preference, and LLM-as-a-judge. See more details in this [paper](https://arxiv.org/pdf/2306.05685.pdf) and [leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard).

## Difference between different versions of Vicuna
See [vicuna_weights_version.md](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md)