# Third-party content

The folder `output/` contains the raw responses of third-party language models to the 65 test
instructions. No model weights are included. The table records, for every model, the checkpoint or
model ID that was used and the licence or terms that apply to it.

| Model | Checkpoint / model ID | Licence or terms | Folder |
|---|---|---|---|
| Qwen3-4B | `unsloth/Qwen3-4B` | Apache-2.0 | `output/4B/Qwen3-4B/` |
| Gemma-3-4B | `google/gemma-3-4b-it` | Gemma Terms of Use | `output/4B/Gemma-3-4B/` |
| Llama-3.2-3B | `unsloth/Llama-3.2-3B-Instruct` | Llama 3.2 Community License | `output/4B/Llama-3.2-3B/` |
| Qwen3-8B | `unsloth/Qwen3-8B` | Apache-2.0 | `output/8B/Qwen3-8B/` |
| Qwen2.5-7B | `unsloth/Qwen2.5-7B-Instruct` | Apache-2.0 | `output/8B/Qwen2.5-7B/` |
| Llama-3.1-8B | `unsloth/Meta-Llama-3.1-8B-Instruct` | Llama 3.1 Community License | `output/8B/Llama-3.1-8B/` |
| GLM-4-9B | `THUDM/glm-4-9b-chat-hf` | GLM-4 Model License | `output/8B/GLM-4-9B/` |
| Gemma-2-9B | `google/gemma-2-9b-it` | Gemma Terms of Use | `output/8B/Gemma-2-9B/` |
| Qwen3.7-Max | `qwen3.7-max` (Alibaba Cloud DashScope) | provider terms of use | `output/Cloud/Qwen3.7-Max/` |
| DeepSeek-V4-Pro | `deepseek-v4-pro` (DeepSeek API) | provider terms of use | `output/Cloud/DeepSeek-V4-Pro/` |
| Kimi-K2.7-Code | `kimi-k2.7-code` (Alibaba Cloud DashScope) | provider terms of use | `output/Cloud/Kimi-K2.7-Code/` |
| ChatGPT-5.5 | `gpt-5.5`, snapshot `gpt-5.5-2026-04-23` (OpenAI Batch API) | provider terms of use | `output/Cloud/ChatGPT-5.5/` |

The four fine-tuned models in `output/FT/` are LoRA adaptations of Qwen3-8B, Llama-3.1-8B,
GLM-4-9B and Gemma-2-9B; their outputs follow the licence of the respective base model. The
adapter weights are not included.

## Use of the outputs

The outputs are included only so that the results reported in the paper can be verified. They are
not covered by the MIT or CC BY 4.0 licences of this repository (see `LICENSE`): any reuse is subject
to the licence or terms of use of the respective model or provider, which may, for example, prohibit
using the outputs to develop competing models. If a provider's terms do not permit redistribution,
the authors will remove the affected files on request.
