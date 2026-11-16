# Practical tooling for a teacher→student agent distillation pipeline on one RTX 4090 (24 GB), Windows 10 / WSL2, zero budget — as of Nov 2026

Scope: (1) serve a quantized Qwen3.6-35B-A3B teacher (fallback Qwen3-30B-A3B) with tool calling and concurrent generation; (2) LoRA SFT of a Qwen3.5-2B student (fallback Qwen3-1.7B) on multi-turn tool-calling data, loss on assistant tokens only; (3) serve the student with parallel sub-agent calls; (4) a lightweight harness; (5) time estimates.

Research date: 2026-11-13. Versions seen in sources: vLLM recipes cite vLLM >= 0.17.0 (Qwen3.5/3.6 baseline), community AWQ checkpoint tested on vLLM 0.26.0, vLLM API docs exist up to v0.29.0; TRL docs v1.18.0 (built against transformers v5.23.0); Unsloth docs require transformers v5 for Qwen3.5.

---

## 1. Teacher serving: vLLM support, quantization on Ada, memory/context, parsers, Windows, alternatives

### Takeaway
Qwen3.5/3.6 are hybrid Gated-DeltaNet + gated-attention models (MoE for 35B-A3B, natively multimodal) supported by vLLM >= 0.17.0, served with `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`. The official FP8 checkpoint (~35 GB) does not fit in 24 GB, so on one 4090 you need a community 4-bit build (AWQ/W4A16, about 19–20.5 GB). One such build reports 32K context, `--max-num-seqs 16`, fp8 KV cache and about 156 tok/s decode on a 3090/4090. Run vLLM under WSL2 (or Docker with the WSL2 backend). Native Windows works only through a community fork. llama.cpp is the most Windows-friendly fallback, but parallel slots on hybrid Qwen3.5 models have open bugs.

### Cited Findings

**Architecture / version requirements**
- Qwen3.6-35B-A3B "shares the gated-delta-networks MoE architecture with Qwen3.5". It has 35B total and 3B active parameters, 256 experts (8 routed + 1 shared), is multimodal, and has 262,144 native context. The vLLM recipe requires vLLM >= 0.17.0, and NVFP4 needs >= 0.28.0. — [vLLM recipe Qwen3.6-35B-A3B](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B.md)
- Qwen3.5-2B has 24 layers laid out as 6 × (3 × Gated DeltaNet→FFN + 1 × Gated Attention→FFN), i.e. 18 DeltaNet layers and 6 attention layers. It has 262,144 native context and takes image/video input (Image-Text-to-Text). — [Qwen/Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- Unsloth docs: "vLLM 0.16.0 does not support Qwen3.5", wait for 0.17.0 or use nightly. — [Unsloth Qwen3.5 fine-tune docs](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- vLLM integrates Flash-Linear-Attention Triton kernels and a hybrid KV-cache manager for linear plus full attention layers (introduced for Qwen3-Next). — [vLLM blog: Qwen3-Next](https://vllm.ai/blog/2025-09-11-qwen3-next)

**Variants / quantization on Ada (sm_89)**
- The official recipe lists BF16, FP8 (Qwen official) and NVIDIA ModelOpt NVFP4, plus Red Hat FP8/NVFP4 variants. It lists no AWQ variant. — [vLLM recipe Qwen3.6-35B-A3B](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B.md)
- FP8 on Ada: a mixed 4090+3090 setup serving `Qwen/Qwen3.6-35B-A3B-FP8` reports that "the 4090 prefers W8A8 and the 3090 falls back to W8A16". `VLLM_TEST_FORCE_FP8_MARLIN=1` was needed to avoid drift in long conversations on the mixed setup. This confirms that vLLM runs FP8 natively on sm_89 (community report, TP=2 across two cards, vLLM 0.19.1). — [allanchan339 blog, Qwen3.6-35B-A3B tool calling](https://allanchan339.github.io/bug-fixes/2026/04/20/Qwen36-35B-A3B-tool-calling.html)
- NVFP4 on Ada: one community article says vLLM falls back to the Marlin kernel on the 4090. You get the memory savings but no FP4 activation speedup. — search summary of [besthub.dev Qwen3.6 article](https://besthub.dev/articles/qwen3-6-35b-quantized-model-on-vllm-local-deployment-and-performance-benchmark-30cf82451eb0) (community, not verified)
- Community AWQ build `hyperhuzaifa/Qwen3.6-35B-A3B-W4A16-AWQ` (community-only claims):
  - Built with llm-compressor 0.12.0: AWQ + W4A16, symmetric int4, group size 128, compressed-tensors format.
  - About 20.5 GB on disk and 19.06 GiB resident in VRAM. The vision tower stays in bf16 and lm_head is not quantized.
  - Requires vLLM >= 0.26.0.
  - Suggested single-24 GB command: `vllm serve hyperhuzaifa/Qwen3.6-35B-A3B-W4A16-AWQ --max-model-len 32768 --max-num-seqs 16 --kv-cache-dtype fp8 --gpu-memory-utilization 0.94`.
  - `--max-num-seqs` must be <= 46 on 24 GB, or CUDA graph capture aborts.
  - Measured: 32K context, "2.86×" concurrency, 156 tok/s decode with CUDA graphs. With `--enforce-eager`, 171,872 context at 1× and 19.5 tok/s.
  - "Don't enable MTP on a single 24 GB card": the draft head leaves under 8,192 usable context.
  - Quality was not benchmarked (coherence checks only).
  — [HF model card hyperhuzaifa/Qwen3.6-35B-A3B-W4A16-AWQ](https://huggingface.co/hyperhuzaifa/Qwen3.6-35B-A3B-W4A16-AWQ)
- Other community 4-bit builds: QuantTrio/Qwen3.6-35B-A3B-AWQ, cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, and an int4 AutoRound build. One listing claims about 19.5 GB and that a single 24 GB GPU is enough. — [QuantTrio AWQ discussion](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ/discussions/7); [cyburn AutoRound](https://huggingface.co/cyburn/Qwen3.6-35B-A3B-int4-AutoRound) (community)
- Quantization caveat (community model card): the Gated-DeltaNet linear-attention projections (`in_proj_qkvz`, `in_proj_ba`) should stay unquantized for correct vLLM inference, citing vLLM issue #40252. — search summary of [88plug/Qwen3.6-35B-A3B-W4A16](https://friendli.ai/models/88plug/Qwen3.6-35B-A3B-W4A16) (not independently verified)

**Useful vLLM flags (official recipe, Qwen3.5/3.6)**
- Tool calling: `--enable-auto-tool-choice --tool-call-parser qwen3_coder`. Reasoning: `--reasoning-parser qwen3`. Disable thinking server-wide: `--default-chat-template-kwargs '{"enable_thinking": false}'`. — [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html); [Qwen3.6 recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B.md)
- `--language-model-only` skips the vision encoder and frees memory for KV cache. — [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- Prefix caching for the Mamba/DeltaNet state works only in "align" mode and is "currently experimental". — [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- Known issue: `assert num_cache_lines >= batch` in `causal_conv1d_update` happens when the CUDA-graph capture size exceeds the Mamba cache size. Fix it by lowering `--max-cudagraph-capture-size` (default 512). — [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- MTP speculative decoding: `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` for 3.6. The recipe says MTP lowers per-token latency but reduces throughput under high concurrency. — [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- The recipe gives an RTX 5090 (32 GB) example that "caps context at 64K" with NVFP4. — [vLLM recipe Qwen3.6](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B.md)

**Tool-call parser reliability (community)**
- Community testing on 4090+3090, vLLM 0.19.1, FP8, Qwen3.6-35B-A3B compared three combinations in long OpenCode agent sessions:
  - `qwen3_xml` + a community "qwen3.5-enhanced.jinja" template was best, but still failed after about 13 min / 111k tokens.
  - `qwen3_coder` + the official template failed after 6.5 min.
  - `qwen3_xml` + the official template failed after 1 min (tool calls emitted inside the thinking block).
  - Qwen3.6-35B-A3B "shows more frequent reasoning loops and malformed tool calls than Qwen3.5-27B".
  — [allanchan339 blog](https://allanchan339.github.io/bug-fixes/2026/04/20/Qwen36-35B-A3B-tool-calling.html)
- An NVIDIA forum thread reports that `qwen3_xml` works better than `qwen3_coder` for Qwen3.5 (community). — [NVIDIA forum: Qwen3.5 tool calling finally fixed](https://forums.developer.nvidia.com/t/qwen3-5-tool-calling-finally-fixed-possibly/366451)
- Unsloth says Qwen3.5 tool calling "improved following our chat template fixes", "universal across Qwen3.5 formats and uploaders". — [Unsloth Qwen3.5 run guide](https://unsloth.ai/docs/models/qwen3.5)

**Fallback teacher Qwen3-30B-A3B**
- Qwen docs recommend Hermes-style tool use for Qwen3, so in vLLM use `--tool-call-parser hermes`. Qwen3-Instruct and Qwen3-Think map to the hermes parser, while Qwen3.5 and Qwen3-Coder use `qwen3_coder`. — [Qwen docs function calling](https://qwen.readthedocs.io/en/v3.0/framework/function_call.html); [NVIDIA Dynamo parser mapping](https://docs.dynamo.nvidia.com/dynamo/parsing/tool-call-parsing.md)
- Old vLLM (0.8.5) hermes parser had JSONDecodeError bugs with Qwen3-32B. — [vLLM issue #17790](https://github.com/vllm-project/vllm/issues/17790)

**vLLM on Windows**
- Unsloth docs: "currently vLLM does not support Windows directly but only via WSL or Linux." — [Unsloth Windows install](https://unsloth.ai/docs/get-started/install/windows-installation)
- Community fork SystemPanic/vllm-windows (v0.20.0, Apr 30 2026, native Python 3.12 + CUDA 13 + PyTorch 2.11 wheels) has no FlashInfer, FA3/4 or DeepGEMM on Windows. Docker Desktop >= 4.54 can run vLLM through the WSL2 backend. These claims come from blogs. — [fazm.ai vLLM Windows 2026](https://fazm.ai/t/vllm-windows-support-2026); [HackerNoon native vLLM on Windows](https://hackernoon.com/how-to-run-vllm-natively-on-windows); [Docker blog](https://www.docker.com/?p=83736)
- One write-up reports about 85 tok/s under WSL versus 160 tok/s on native Ubuntu on the same hardware. This is a single anecdote. — [dev.to: native vLLM vs WSL vs llama.cpp](https://dev.to/alanwest/running-llms-on-windows-native-vllm-vs-wsl-vs-llamacpp-compared-37a9)
- Sources conflict on official native Windows support: one claims there was never official support (as of May 2026), another claims native support arrived but needs CUDA 13 and RTX 6000 Ada or newer. Neither is confirmed in vLLM docs. — [fazm.ai](https://fazm.ai/t/vllm-windows-support-2026); [startupfortune](https://startupfortune.com/?p=9131)

**llama.cpp / Ollama / LM Studio**
- llama.cpp tool calling needs `llama-server --jinja`, with `--chat-template-file` for overrides. Parallel tool calls are "disabled by default" and enabled with `"parallel_tool_calls": true` in the request. Templates it does not recognize fall back to a "Generic" handler that "may consume more tokens". The doc warns that extreme KV quantization (`-ctk q4_0`) "can substantially degrade the model's tool calling performance". The fetched doc did not list Qwen3/3.5 among the native formats. — [llama.cpp function-calling.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
- Single-stream Qwen3.6-35B-A3B Q4_K_M (~21 GB) on a 4090 with llama.cpp: reports range from 84 to 120 to 165 tok/s depending on setup. One 3090 report gives 135.7 tok/s for llama-server versus 107 tok/s for Ollama 0.20.7. All are community numbers. — [markaicode 4090 benchmark](https://markaicode.com/benchmarks/llamacpp-llama-33-rtx-4090-throughput-benchmark/); [insiderllm guide](https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/); [benchmarklist](https://benchmarklist.com/benchmarks/localmaxxing/)
- Open hybrid-model issues in llama.cpp:
  - #21383: Qwen3.5-27B crashes in the prompt-cache checkpoint and tool-call turns.
  - #24714: full prompt re-processing on hybrid/recurrent models even with cached prompts.
  - In the ik_llama.cpp fork (PR #1933), Qwen3.5-MoE slots with np>=3 "bleed recurrent state into each other".
  — [llama.cpp #21383](https://github.com/ggml-org/llama.cpp/issues/21383); [llama.cpp #24714](https://github.com/ggml-org/llama.cpp/issues/24714); [ik_llama.cpp PR #1933](https://github.xiaohuzi123231346.workers.dev/ikawrakow/ik_llama.cpp/pull/1933)

### Inferences
- The teacher cannot run in FP8 or BF16 on 24 GB (FP8 weights alone are about 35 GB). The realistic choice is a 4-bit W4A16 AWQ/compressed-tensors checkpoint on vLLM >= 0.26 under WSL2, run with:
  - `--language-model-only --max-model-len 32768 --kv-cache-dtype fp8 --max-num-seqs 16..32 --gpu-memory-utilization 0.92-0.94 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`
  - Fall back to `qwen3_xml` if tool-call formatting breaks.
  - `--language-model-only` should free a few GB from the bf16 vision tower for KV cache (memory gain not measured).
- Hybrid models keep only 1 in 4 layers as full attention with KV cache; the DeltaNet layers hold a fixed-size state. KV per token is therefore much smaller than in a dense model of the same size. This is why 16+ concurrent 32K sequences fit next to about 19 GB of weights.
- Ada FP8 matters mostly for the KV cache (`--kv-cache-dtype fp8`) and for FP8 serving of the small student, not for the 35B teacher.
- For thousands of trajectories, vLLM's continuous batching is the right engine. llama.cpp `-np > 1` on Qwen3.5/3.6 hybrids carries correctness risk (state bleed, cache crashes). Use llama.cpp only if WSL2/vLLM is impossible, and then with `-np 1..2`. LM Studio and Ollama wrap llama.cpp and inherit the same limits, at lower throughput.
- Fallback teacher Qwen3-30B-A3B is a standard-attention MoE, so it avoids the hybrid-state caveats and uses `--tool-call-parser hermes`. Its 4-bit AWQ is also about 17–19 GB (same parameter class).

### Gaps
- No official or controlled benchmark of aggregate throughput (tok/s summed over 16–32 concurrent requests) for Qwen3.6-35B-A3B 4-bit on a single 4090 under vLLM. The 156 tok/s figure from the AWQ card is ambiguous (likely single-stream decode).
- Exact current vLLM stable version in Nov 2026 not confirmed from release notes. Docs exist through v0.29.0.
- SGLang Qwen3.5 docs page returned 404, so SGLang version, parser names and Windows status could not be verified. The model card says SGLang uses `--tool-call-parser qwen3_coder` and needed the main branch at launch ([Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)).
- Whether current mainline llama.cpp has a native Qwen3.5/3.6 (Qwen3-Coder XML) tool-call parser was not confirmed. The fetched function-calling doc may be outdated.
- vLLM issue #40252 (DeltaNet projections must stay unquantized) was not read directly.

---

## 2. Student training: Unsloth and TRL for Qwen3.5-2B (fallback Qwen3-1.7B)

### Takeaway
Use bf16 LoRA, not QLoRA. Unsloth explicitly advises against 4-bit training on Qwen3.5 and lists about 5 GB VRAM for a 2B bf16 LoRA at short context, so 8k–16k sequences should fit on 24 GB with gradient checkpointing. TRL 1.18 ships bundled Qwen3.5 training templates (`qwen3_5_nothink_training.jinja` for 2B and smaller) that add `{% generation %}` markers and keep historical thinking blocks. `assistant_only_loss=True` therefore works on tool-calling conversations. Because Qwen3.5 is a VLM, you must pass `processing_class=tokenizer` (or use a prompt-completion format) to avoid the "not supported for VLMs" error.

### Cited Findings

**Unsloth (official docs)**
- Supported sizes are 0.8B, 2B, 4B, 9B, 27B, 35B-A3B and 122B-A10B. bf16 LoRA VRAM: 0.8B 3 GB, 2B 5 GB, 4B 10 GB, 9B 22 GB. Unsloth claims "1.5× faster and uses 50% less VRAM than FA2 setups". — [Unsloth Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense, due to higher than normal quantization differences." The example uses `load_in_4bit=False, load_in_16bit=True`. — [Unsloth Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Example LoRA settings:
  - `r=16, lora_alpha=16, lora_dropout=0, bias="none"`
  - target modules `q,k,v,o,gate,up,down_proj`
  - `use_gradient_checkpointing="unsloth"`
  - `max_seq_length=2048` ("start small; scale up after it works")
  - "Recommended alpha == r at least."
  — [Unsloth Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- "Please use transformers v5 for Qwen3.5. Older versions will not work." Training can seem slow at first because of custom Mamba Triton kernel compilation. — [Unsloth Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Thinking: "If you want to preserve reasoning ability, you can mix reasoning-style examples with direct answers (keep a minimum of 75% reasoning)." — [Unsloth Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Notebooks: free Colab notebooks for Qwen3.5 0.8B/2B/4B (vision) and Qwen3.5-4B GRPO. A100 notebooks for 27B and 35B-A3B. No text-only tool-calling SFT notebook is listed. — [Unsloth Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Windows: Unsloth Core runs natively on Windows via Conda (Python 3.12, PyTorch cu130, Triton 3.6.0 in the sample output). Docker and WSL (Ubuntu-24.04) are alternatives. The sample output shows Unsloth 2026.1.4. Unsloth Studio installs with `irm https://unsloth.ai/install.ps1 | iex`. — [Unsloth Windows installation](https://unsloth.ai/docs/get-started/install/windows-installation); [Unsloth Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- A community Qwen3.5-2B tool-calling LoRA trained with Unsloth kept `enable_thinking` disabled throughout training and inference. — [cowWhySo/qwen3_5_2B_voice_assistant_tools](https://huggingface.co/cowWhySo/qwen3_5_2B_voice_assistant_tools) (community)

**TRL SFTTrainer (docs v1.18.0)**
- `assistant_only_loss=True` with a conversational dataset computes loss only on assistant responses. It "requires the chat template to include `{% generation %}` and `{% endgeneration %}`… For known model families (e.g. Qwen3), TRL automatically patches the template when `assistant_only_loss=True`." For vision datasets, assistant-only loss "requires transformers>=5.18.0". — [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- Tool calling: "SFTTrainer fully supports fine-tuning models with tool calling". Each example contains messages with `tool_calls` and `tool`-role messages, plus a `tools` column of JSON schemas. — [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- Pre-tokenized datasets: an optional `labels` column is used as-is, otherwise labels are built from an `assistant_masks` / `completion_mask` column. This gives an escape hatch for custom masking. — [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- SFTConfig defaults to watch:
  - `max_length=1024`: raise it to 8192–16384 or long trajectories get truncated.
  - `truncation_mode` supports only "keep_start".
  - `gradient_checkpointing=True` and `bf16=True` by default.
  - `packing` with "bfd" enables padding-free mode, which needs FlashAttention 2/3.
  - Model dtype defaults to float32 when the model is given as a string unless `model_init_kwargs={"dtype": torch.bfloat16}` is set.
  - The docs suggest LR around 1e-4 for adapters.
  — [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- Bundled training templates: Qwen3.5 and Qwen3.6 are recognized families.
  - `qwen3_5_think_training.jinja` (Qwen3.5-4B and larger, thinking on by default) and `qwen3_5_nothink_training.jinja` (Qwen3.5-2B and smaller, thinking off by default).
  - Both drop the `loop.index0 > ns.last_query_index` conditional "so the thinking block is always emitted" and wrap assistant output in generation markers.
  - The pre-filled `<think>\n\n</think>\n\n` (when `enable_thinking=False`) stays outside the markers.
  - Qwen3 (`qwen3_training.jinja`) and Qwen3.6 have equivalent templates, which covers the Qwen3-1.7B fallback.
  — [TRL Chat Templates](https://huggingface.co/docs/trl/chat_templates)
- Known issue (May 25–26 2026): on `Qwen3.5-4B-Base`, SFTTrainer raised "Assistant-only loss is not yet supported for vision-language models" because the config has vision capabilities. Fix: pass `processing_class=tokenizer` to the trainer, then verify with `trainer.data_collator.torch_call(...)` that only assistant tokens have `labels != -100`. Alternative: a prompt-completion dataset with `completion_only_loss=True`. Tracking issue: huggingface/trl#5471. — [HF forum thread](https://discuss.huggingface.co/t/sfttrainerflags-blocks-assistant-only-loss-true/176210); [TRL #5471](https://github.com/huggingface/trl/issues/5471)

**Qwen3.5 chat/thinking conventions relevant to data formatting**
- Qwen3.5-2B is non-thinking by default. Enable thinking with `"enable_thinking": True` via `chat_template_kwargs`. The `/think` and `/nothink` soft switches are not officially supported. In thinking mode the 2B is "more prone to entering thinking loops". — [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)
- Recommended sampling, non-thinking text: temp 1.0, top_p 1.0, top_k 20, presence_penalty 2.0. Thinking text: temp 1.0, top_p 0.95, top_k 20, presence_penalty 1.5. — [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)

### Inferences
- Recommended student recipe:
  - Qwen3.5-2B loaded text-only in bf16, LoRA r=16–32 with alpha = r on all linear layers in the language model, `max_length` 8192–16384, batch 1 with gradient accumulation, gradient checkpointing.
  - Either Unsloth (FastLanguageModel, 16-bit) or plain TRL+PEFT. With 2B at 16k tokens, bf16 LoRA should fit comfortably in 24 GB. Unsloth's 5 GB at 2k leaves headroom; activation memory scales roughly linearly with gradient checkpointing.
  - The 16k VRAM figure is not measured. Verify it with a dry run.
- Masking: the most robust path is TRL `assistant_only_loss=True` with the bundled `qwen3_5_nothink_training.jinja`, because it is maintained for exactly this model and masks tool responses (role `tool`) out of the loss. If you use Unsloth's `train_on_responses_only`, verify the boundaries by decoding labels yourself. No official Unsloth marker strings for Qwen3.5 tool turns were found.
- Thinking mismatch: the stock Qwen3.x template strips `<think>` from assistant turns before the last user message, while TRL's training template always keeps them. Decide deliberately:
  - (a) Distil non-thinking: generate the teacher with `enable_thinking=false` and train the 2B nothink. This is cheapest, matches the 2B default, and avoids its thinking loops.
  - (b) Distil reasoning: keep the teacher's `reasoning_content` and train with thinking on, following Unsloth's at-least-75%-reasoning mix rule if you want to retain it.
- Store teacher trajectories in OpenAI/HF message format: `{"role":"assistant","content":..., "reasoning_content":..., "tool_calls":[{"type":"function","function":{"name":..., "arguments":{...dict}}}]}` plus `{"role":"tool","content":...}` and a `tools` column. The Qwen3.5 template then renders Qwen3-Coder-style XML tool calls itself. Keep `arguments` as a dict; TRL notes dict arguments are the documented transformers format.
- Fallback student Qwen3-1.7B is a plain transformer with no VLM detection issue. TRL's `qwen3_training.jinja` covers it and QLoRA is less risky there, though at 1.7B bf16 LoRA fits easily anyway.

### Gaps
- No measured VRAM for Qwen3.5-2B LoRA at 8k/16k found. Only the 2k-context 5 GB figure is available.
- Whether Unsloth's FastLanguageModel loads Qwen3.5-2B text-only (dropping the vision tower) and whether its `train_on_responses_only` handles Qwen3.5 tool turns correctly: not documented.
- Whether native-Windows Unsloth runs the custom Mamba/FLA Triton kernels for Qwen3.5 reliably is unconfirmed. Triton on Windows is supported per the docs, but there is no Qwen3.5-specific Windows report. WSL2 is the safer default.

---

## 3. Serving the student for evaluation with parallel sub-agent calls

### Takeaway
Serve the 2B base model plus the LoRA (or a merged checkpoint) on vLLM with `qwen3_coder` and `qwen3` parsers. Small hybrid models leave lots of KV and state headroom on 24 GB, so dozens of concurrent sub-agent requests are feasible. LoRA-adapter serving on Qwen3.5 in upstream vLLM is not documented, so merging the adapter before serving is the safe route.

### Cited Findings
- Qwen3.5-2B serve example: `vllm serve Qwen/Qwen3.5-2B --max-model-len 262144`, optional `--language-model-only`, tool calling with `--enable-auto-tool-choice --tool-call-parser qwen3_coder`. The card asked for vLLM nightly at release (Feb–Mar 2026). — [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)
- The vLLM Qwen3.5 recipe does not cover LoRA. The vLLM Gaudi plugin has a "LoRA path, labeled as Qwen3.5 only" with separate `in_proj_qkv` / `in_proj_z` projections. That is indirect evidence that LoRA on DeltaNet projections needs special handling. — [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html); [vllm-gaudi qwen3_5](https://docs.vllm.ai/projects/gaudi/en/0.21.0/api/vllm_gaudi/models/qwen3_5.html)
- Parallel tool calls in one assistant turn: llama.cpp needs `"parallel_tool_calls": true` per request. — [llama.cpp function-calling.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)

### Inferences
- Simplest path: merge the LoRA into bf16 weights (`model.merge_and_unload()` or Unsloth `save_pretrained_merged`) and serve the merged model. This avoids any `--enable-lora` incompatibility with DeltaNet layers.
  - Suggested flags: `--max-model-len 32768 --max-num-seqs 64 --enable-prefix-caching` (prefix caching is experimental for hybrids, so test with and without), `--default-chat-template-kwargs '{"enable_thinking": false}'` if the student was trained without thinking.
  - If LoRA serving is needed (several adapters), test `--enable-lora --max-lora-rank 32` and fall back to merging if it fails.
- The teacher (~20 GB) and the student cannot run on the GPU at the same time with good batch sizes. Run phases sequentially: teacher generation → stop → train → serve student.

### Gaps
- Upstream vLLM `--enable-lora` support for Qwen3.5 (Qwen3_5ForConditionalGeneration) is not confirmed. Search vLLM issues for "enable-lora Qwen3.5".
- No throughput numbers for Qwen3.5-2B on a 4090.

---

## 4. Lightweight harness: OpenAI-compatible client + asyncio, recursive spawn_subagent, trajectory logging

### Takeaway
No heavy framework is needed. An `AsyncOpenAI` client pointed at vLLM's `/v1`, an `asyncio.Semaphore` capping in-flight requests at about `--max-num-seqs`, and a tool loop are enough. `spawn_subagent` is a tool whose handler recursively runs the same loop with a fresh message list and a depth limit. Several sub-agent tool calls are run with `asyncio.gather`. Qwen officially recommends Qwen-Agent as a ready-made framework.

### Cited Findings
- The Qwen3.5 model card recommends Qwen-Agent for building agents on Qwen3.5. — [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)
- Without a tool-call parser, vLLM returns the raw tool-call text instead of structured `tool_calls` objects. — search summary of [dev.to Qwen3.6 vLLM coding agent](https://dev.to/lavellehatcherjr/serving-qwen36-35b-a3b-with-vllm-and-building-a-coding-agent-with-tool-calling-2kob) (community)
- vLLM `--reasoning-parser qwen3` separates thinking into `reasoning_content` (Qwen3-family thinking format). — [vLLM recipe Qwen3.6](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B.md); search summary of [Baseten Qwen3.6 doc](https://docs.baseten.co/examples/models/llm/qwen3.6.md)
- Community reports of reasoning loops and malformed tool calls on Qwen3.6-35B-A3B in long sessions. — [allanchan339 blog](https://allanchan339.github.io/bug-fixes/2026/04/20/Qwen36-35B-A3B-tool-calling.html)

### Inferences
Design sketch (not sourced; standard engineering):
- `async def run_agent(messages, tools, depth)`:
  1. Loop: `resp = await client.chat.completions.create(model=..., messages=..., tools=..., extra_body={"chat_template_kwargs":{"enable_thinking":False}, "top_k":20})`.
  2. Append `resp.choices[0].message` and keep `reasoning_content` if present.
  3. For each `tool_call`, dispatch: local calculators/tools run directly, and `spawn_subagent(task, context)` calls `run_agent(new_messages, subagent_tools, depth+1)` and returns the sub-agent's final answer as the tool result string.
  4. Run all tool calls of one turn concurrently with `asyncio.gather`.
  5. Append `{"role":"tool","tool_call_id":..., "content":...}`.
  6. Stop on no tool calls, `max_turns`, or a token budget.
- Guard recursion with `max_depth` (e.g. 2), and do not offer `spawn_subagent` at the max depth. Use a global semaphore so nested agents cannot deadlock the server. Prefer a semaphore around each HTTP call, not around a whole agent; otherwise parents holding slots wait on children that cannot get one.
- Record trajectories as JSONL, one record per agent invocation: `{id, parent_id, depth, tools, messages (with reasoning_content and tool_calls with dict args), final_answer, gold, correct, usage tokens, latency}`. Sub-agent runs become separate training examples linked by `parent_id`. That lets you train the student both as orchestrator and as sub-agent.
- Add validation and repair: reject or regenerate turns whose `tool_calls` arguments fail JSON-schema validation, and filter on final-answer correctness before SFT (rejection sampling). This matters given the documented malformed-call and looping behaviour.
- Use streaming or a `max_tokens` cap per turn to kill runaway thinking loops, which the Qwen3.5-2B card flags.

### Gaps
- No benchmarked lightweight harness specific to recursive sub-agents on vLLM was researched. Qwen-Agent's support for parallel/recursive sub-agents was not verified.
- Whether vLLM's `qwen3_coder` parser reliably returns multiple tool calls in one turn (parallel calls) for Qwen3.5/3.6 was not verified.

---

## 5. Time estimates for 2k–5k trajectories (generation + training)

### Takeaway
No source gives end-to-end numbers for this exact setup. Back-of-envelope:
- Teacher generation of 2k–5k multi-turn trajectories takes roughly 3–15 hours on one 4090, depending heavily on thinking on/off and trajectory length.
- LoRA SFT of a 2B model on 20–40M tokens takes roughly 1–4 hours per epoch.

Treat these as planning estimates to be calibrated with a 50-trajectory pilot.

### Cited Findings
- Single-24 GB AWQ teacher: 156 tok/s decode (CUDA graphs, 32K, fp8 KV), `--max-num-seqs 16` recommended, <= 46 allowed. — [hyperhuzaifa AWQ card](https://huggingface.co/hyperhuzaifa/Qwen3.6-35B-A3B-W4A16-AWQ)
- llama.cpp single-stream on a 4090 Q4_K_M: 84–165 tok/s (community, varying setups). — [markaicode](https://markaicode.com/benchmarks/llamacpp-llama-33-rtx-4090-throughput-benchmark/); [benchmarklist](https://benchmarklist.com/benchmarks/localmaxxing/); [insiderllm](https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/)
- WSL2 versus native Linux throughput penalty: 85 vs 160 tok/s in one anecdote. — [dev.to comparison](https://dev.to/alanwest/running-llms-on-windows-native-vllm-vs-wsl-vs-llamacpp-compared-37a9)
- Unsloth claims 1.5× faster training and 50% less VRAM than FA2 for Qwen3.5, with an initial slow-down from Triton kernel compilation. — [Unsloth Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

### Inferences (assumptions explicit; not sourced)
- **Generation volume.** Assume a trajectory has about 6 model turns. Without thinking that is ~250 output tokens per turn, about 1.5k output tokens per trajectory. With thinking, budget about 4–6k output tokens per trajectory. Sub-agents multiply this.
  - 5k trajectories ≈ 7.5M output tokens (no thinking) or 20–30M (thinking).
  - Prefill is cheap relative to decode for a 3B-active MoE, but multi-turn contexts are re-prefilled each turn when hybrid prefix caching is disabled or flaky.
- **Aggregate decode throughput.** For an A3B MoE at 4-bit with 16–32 concurrent sequences, plan for about 400–1000 tok/s aggregate on a 4090 under WSL2. This is a guess: batched MoE decode scales well because only 3B parameters are active, but the WSL2 penalty and Marlin W4A16 kernels cut into it.
  - 7.5M tokens / 500 tok/s ≈ 4 h.
  - 25M tokens / 500 tok/s ≈ 14 h.
  - For 2k trajectories, scale by 0.4: about 1.5–6 h.
- **Training compute.** 5k trajectories × about 6k tokens ≈ 30M tokens per epoch.
  - Compute ≈ 6 × 2e9 × 3e7 ≈ 3.6e17 FLOPs, plus about 33% recompute for gradient checkpointing ≈ 4.8e17.
  - The 4090 peaks around 165 TFLOPS dense bf16. At a realistic 25–40% MFU, that is about 1–2.5 h per epoch. Allow up to about 4 h for padding waste, Windows/WSL overhead and DeltaNet kernels.
  - 2–3 epochs ≈ 3–10 h. Packing or padding-free (needs FlashAttention 2/3) materially reduces the padding waste.
- Pilot first: generate 50 trajectories and measure vLLM's logged "Avg generation throughput", then train 100 steps and read tokens/s from the TRL logs. Extrapolate from those numbers instead of the estimates above.

### Gaps
- No published aggregate (multi-request) throughput for Qwen3.6-35B-A3B 4-bit on vLLM on a single 4090.
- No published Qwen3.5-2B LoRA training throughput (tokens/s) on a 4090.
- The WSL2 overhead estimate comes from a single anecdote and needs local measurement.
