# Distilling multi-agent and agentic (tool-using) trajectories into small LMs (≤4B), as of Nov 2026

Context for the report writer: the target project distills an orchestrator that spawns parallel sub-agents through a tool call. The teacher is a Qwen ~30–35B MoE, the student is a Qwen ~2B, training is SFT+LoRA on one RTX 4090, and the time budget is about 3 days. The closest published analogue is ParaManager (Apr 2026). It used a Qwen3-30B-A3B-Instruct-2507 teacher, a Qwen3-4B student and parallel tool calls, with 4,470 filtered SFT trajectories followed by RL (details below).

Research date: 2026-11-13. Most numbers below are self-reported by the paper authors. Several items come only from abstracts or secondary pages and are flagged as such.

## Q1. Key papers: recipes and results

### Takeaway
Every successful recipe follows the same pattern. A strong teacher (or teacher system) generates tool-using trajectories. Only correct ones are kept, they are converted into one model's multi-turn format with tool observations masked from the loss, and the student is SFT'd. The strongest results add RL afterwards, either outcome-based (DAPO/GRPO) or with parallelism-shaping rewards. Pure SFT works for ≤4B students when the data is small and clean: about 2k trajectories in Agent Distillation and 4.5k in ParaManager.

### Cited Findings

**Chain-of-Agents / Agent Foundation Models (AFM). OPPO, arXiv 2508.13167, submitted 6 Aug 2025**
- One model simulates a multi-agent system end to end by dynamically activating "tool agents" and "role-playing agents". It is trained by multi-agent distillation from trajectories of a stronger multi-agent framework (OAgents), followed by agentic RL on verifiable tasks. Weights, code and data are open (github.com/OPPO-PersonalAI/Agent_Foundation_Models; HF org PersonalAILab). — [arXiv abs](https://arxiv.org/abs/2508.13167v1)
- Base models: Qwen2.5-3B/7B/32B-Instruct for the web agent and Qwen2.5-Coder-7B/32B-Instruct for the code agent. — [arXiv HTML](https://arxiv.org/html/2508.13167)
- Web SFT data: 16,433 trajectories (8,826 multi-hop QA and 7,607 web agent), averaging 4.35 and 7.29 hops. Code SFT is about 47k traces according to §4.2.1 but 59,929 in Table 9; the paper does not reconcile the two. — [arXiv HTML](https://arxiv.org/html/2508.13167)
- Conversion: each agent activation in OAgents is logged with the reasoning state before it and the agent's output, then serialized into one sequence with the tags `<think>`, `<tools>`, `<observation>`, `<reflection>` and `<answer>`. Sub-agent work is therefore **inlined** into a single trajectory, not trained as separate samples. — [arXiv HTML](https://arxiv.org/html/2508.13167)
- The filter has four stages:
  1. Drop trajectories with fewer than 5 agent-tool interactions.
  2. Drop incorrect answers, redundant tool inputs and instruction-following failures.
  3. Downsample non-reflective trajectories, and drop them entirely for math and code.
  4. Upsample trajectories that correct a low-credibility first answer.
  
  — [arXiv HTML](https://arxiv.org/html/2508.13167)
- Loss masks observation tokens ("tool call outputs are masked"). Web SFT settings: batch 256, 2.5 epochs, LR 1.4e-5, cosine schedule, LLaMA-Factory. Code SFT settings: 2 epochs, LR 3e-5 for 7B. — [arXiv HTML](https://arxiv.org/html/2508.13167)
- RL uses DAPO in VeRL with 8 rollouts per prompt and a binary reward from an LLM judge (Qwen2.5-72B) for web tasks. The code reward is correctness × format. — [arXiv HTML](https://arxiv.org/html/2508.13167)
- Results (Pass@1) for AFM-RL-32B: GAIA 55.3, BrowseComp 11.1, HLE 18.0, LiveCodeBench v5 47.9, AIME25 59.8. For AFM-RL-7B: GAIA 40.8, HLE 15.6. The paper claims an 84.6% reduction in inference token cost versus multi-agent systems. In the text I read, the evidence for this figure was only a qualitative comparison on 10 GAIA tasks, so treat the number as weakly supported. — [arXiv HTML](https://arxiv.org/html/2508.13167)
- Lesson on tool-format generalization: a web agent trained only on search tools fails to emit correctly formatted calls for unseen tools (exact JSON, code fences), which causes parser errors and task aborts. The code-trained model, which learned strict formatting, generalized better. — [arXiv HTML](https://arxiv.org/html/2508.13167)

**MAGDi. Chen, Saha, Stengel-Eskin, Bansal; arXiv 2402.01620, Feb 2024; ICML 2024**
- Teacher LLMs hold multi-round debates that form a "multi-agent interaction graph": each node is one agent's reasoning in one round, and each edge is a refinement. The student is augmented with a GCN and trained with three objectives (next-token prediction, a contrastive loss, and a graph objective). It also learns from *incorrect* teacher reasoning. — [ICML proceedings](https://proceedings.mlr.press/v235/chen24ah.html); [arXiv](https://arxiv.org/abs/2402.01620v1); code [github.com/dinobby/magdi](https://github.com/dinobby/magdi)
- Reported to beat single-teacher and multi-teacher distillation on 7 commonsense and math benchmarks, with "an order of magnitude higher efficiency" than its teachers. — [ICML proceedings](https://proceedings.mlr.press/v235/chen24ah.html)
- (Not verified in detail: exact student sizes and numbers. I took them only from the abstract and search summaries.)

**Agent Distillation ("Distilling LLM Agent into Small Models with Retrieval and Code Tools"). Kang et al. (KAIST et al.); arXiv 2505.17612, 23 May 2025, v2 5 Nov 2025; NeurIPS 2025**
- The teacher is Qwen2.5-32B-Instruct. Students are Qwen2.5-Instruct at 0.5B/1.5B/3B/7B, plus Llama-3.2-1B and Phi-4-mini (3.8B). The tools are Wikipedia retrieval and a code interpreter, in a CodeAct-style setup. — [arXiv HTML v2](https://arxiv.org/html/2505.17612v2)
- Data: 1,000 HotpotQA and 2,000 MATH questions, with one teacher trajectory per question and incorrect trajectories removed. This leaves **about 2,000 trajectories**. — [arXiv HTML v2](https://arxiv.org/html/2505.17612v2)
- Training: observations excluded from the loss; **LoRA rank 64 on all linear layers, 2 epochs, batch 8, LR 2e-4**. Inference uses greedy decoding with at most 5 steps. — [arXiv HTML v2](https://arxiv.org/html/2505.17612v2)
- Average scores, CoT-distill → agent-distill → agent+FTP+SAG:

  | Student | CoT-distill | Agent-distill | Agent + FTP + SAG |
  |---|---|---|---|
  | 0.5B | 13.64 | 19.24 | 21.90 |
  | 1.5B | 21.28 | 28.06 | 30.55 |
  | 3B | 27.72 | 33.60 | 36.60 |
  | 7B | 33.54 | 39.85 | 42.68 |

  The 32B teacher scores 46.00 with agent prompting. — [arXiv HTML v2](https://arxiv.org/html/2505.17612v2)
- Prompting small models as agents *without* distillation fails: 1.80 (0.5B), 7.94 (1.5B), 21.20 (3B), and 36.54 for 7B. — [arXiv HTML v2](https://arxiv.org/html/2505.17612v2)
- "First-thought prefix" (FTP), which seeds teacher trajectories with the teacher's CoT first step, adds about 1 point. It also reduces retrieval calls and "can sometimes degrade performance". "Self-consistent action generation" (SAG) samples N=8 actions at T=0.4 and keeps one that executes consistently. It mainly reduces code parse and execution errors, especially at 0.5B. — [arXiv HTML v2](https://arxiv.org/html/2505.17612v2)
- Code: github.com/Nardien/agent-distillation. — [alphaxiv](https://alphaxiv.org/abs/2505.17612)

**ToolOrchestra / Orchestrator-8B. NVIDIA + HKU; arXiv 2511.21689, Nov 2025**
- Qwen3-8B is trained **with RL only (GRPO)**; no SFT stage is described. It learns when to call stronger LLMs (e.g., GPT-5) and tools. The reward combines a binary outcome judged by GPT-5, an efficiency term (negative cost and latency) and a user-preference term. — [arXiv HTML](https://arxiv.org/html/2511.21689)
- Data: GeneralThought-430K (filtered) plus the synthetic ToolScale set (10 domains; about 4k tasks by summing Table 5). Training uses 16 H100s, rollout batch 8, at most 50 turns, and LR 1e-6. Sample filters: drop groups with reward std < 0.1, format-inconsistent samples and invalid outputs. Task synthesis filters: drop tasks unsolved at pass@8 and tasks solvable without any action. — [arXiv HTML](https://arxiv.org/html/2511.21689)
- Results versus GPT-5: HLE 37.1 vs 35.1, FRAMES 76.3 vs 74.0, τ²-Bench 80.2 vs 77.7, at about 30% of the cost and 2.5× faster. — [arXiv HTML](https://arxiv.org/html/2511.21689); [NVIDIA project page](https://research.nvidia.com/labs/lpr/ToolOrchestra)
- Pilot finding relevant to orchestrators: a *prompted* Qwen3-8B orchestrator delegated to GPT-5 in 73% of cases ("self-enhancement bias"). GPT-5 as orchestrator called GPT-5 or GPT-5-mini 98% of the time. In other words, prompted orchestrators have strong delegation biases. — [arXiv HTML](https://arxiv.org/html/2511.21689)
- Weights (huggingface.co/nvidia/Orchestrator-8B) are under a non-commercial license; code is Apache-2.0. — [VentureBeat](https://venturebeat.com/ai/nvidias-new-ai-framework-trains-an-8b-model-to-manage-tools-like-a-pro)

**APR: Adaptive Parallel Reasoning. Pan, Li, Lian et al. (UC Berkeley); arXiv 2504.15466, Apr 2025, rev. Aug 2025**
- `spawn()`/`join()` operations let a parent thread launch child threads. The children explore in parallel and return results, and the parent continues decoding from them. — [arXiv HTML](https://arxiv.org/html/2504.15466v2)
- The model is a **228M non-embedding (≈293M total) Llama-2-architecture model trained from scratch** on Countdown. SFT uses 500k symbolic-solver demonstrations (a hybrid BFS/DFS solver whose nodes are delegated to children). RL then uses GRPO (group size 5, 150 steps, 2 GPUs). — [arXiv HTML](https://arxiv.org/html/2504.15466v2)
- Results: 83.4% vs 60.0% (serialized SoS+) at a 4k context, and 75.2% vs 57.3% at about 5,000 ms latency. RL raised APR from 75.5% to 83.4%. RL mainly *increased the number of child threads* (6.1 → 8.2 on average), i.e., wider search. With the thread count held fixed, accuracy barely changed (83.2 → 83.3). — [arXiv HTML](https://arxiv.org/html/2504.15466v2)
- Serving uses SGLang (continuous batching plus radix-attention prefix sharing across threads). — [arXiv HTML](https://arxiv.org/html/2504.15466v2)

**Kimi K2.5 Agent Swarm / PARL. Moonshot AI; released 27 Jan 2026; tech report arXiv 2602.02276**
- A trainable orchestrator dynamically creates and delegates to **frozen** sub-agents. "During training, sub-agents are frozen and their execution trajectories are excluded from the optimization objective". Only the orchestrator is updated by RL. — [Kimi K2.5 tech report](https://arxiv.org/html/2602.02276v1)
- The reward is r = λ₁·r_parallel + λ₂·r_finish + r_perf, with λ₁ and λ₂ annealed to 0 over training. r_parallel (an instantiation reward) counters **"serial collapse"**, where the orchestrator defaults to single-agent execution. r_finish (the sub-agent finish rate) counters **"spurious parallelism"**, where it spawns many sub-agents without meaningful decomposition. — [Kimi K2.5 tech report](https://arxiv.org/html/2602.02276v1); [Kimi blog](https://www.kimi.com/blog/kimi-k2-5)
- The metric is "Critical Steps" = Σ_t (S_main(t) + max_i S_sub,i(t)): spawning helps only if it shortens the critical path. — [Kimi K2.5 tech report](https://arxiv.org/html/2602.02276v1)
- "We first train the orchestrator using small-size subagents before transitioning to larger models." No swarm-specific SFT cold start is described. Swarm behavior is induced by synthetic wide-search and deep-search prompts that do *not* explicitly instruct parallelization. Only task-relevant sub-agent outputs, not full traces, are routed back to the orchestrator. — [Kimi K2.5 tech report](https://arxiv.org/html/2602.02276v1)
- Results, single-agent → swarm: BrowseComp 60.6 → 78.4, WideSearch Item-F1 72.7 → 79.0, in-house Swarm Bench 41.6 → 58.3. Swarms use up to 100 sub-agents and 1,500 tool calls/steps, with up to 4.5× lower latency. — [Kimi K2.5 tech report](https://arxiv.org/html/2602.02276v1); [Kimi blog](https://www.kimi.com/blog/kimi-k2-5)
- The report gives no tool names for spawning or assigning (no `create_subagent` appears in the text I read). A third-party paper (arXiv 2605.02801) notes that full PARL algorithmic details are not public. — [Kimi K2.5 tech report](https://arxiv.org/html/2602.02276v1); [arXiv 2605.02801](https://arxiv.org/pdf/2605.02801)

**Newer directly relevant work (2026)**
- **ParaManager, "Small Model as Master Orchestrator" (arXiv 2604.17009, Apr 2026).** The student is **Qwen3-4B-Instruct-2507**. The SFT teacher is **Qwen3-30B-A3B-Instruct-2507** (the same family and scale as this project's teacher), run in their parallel orchestration framework on DAPO-Math. — [arXiv HTML](https://arxiv.org/html/2604.17009v1)
  - Data: 8 samples per instance, at most 1 kept, correct final answer only, recoveries from tool errors prioritized, and a tool-balance constraint. This gives **4,470 SFT instances**. — [arXiv HTML](https://arxiv.org/html/2604.17009v1)
  - Format and loss: each round is a *set* of 1–4 tool calls (parallel) in `<reasoning>` and `<tool_call>` JSON tags. Invalid calls return a PARSE_ERR observation, and the loss covers only orchestrator tokens. — [arXiv HTML](https://arxiv.org/html/2604.17009v1)
  - RL: GRPO with G=8, no KL, DAPO-style clipping. The reward is task + format + diversity + efficiency. The diversity term requires ≥1.25 calls/round and ≥3 unique tools; the efficiency term targets 12,288 tokens and 8 cost units. — [arXiv HTML](https://arxiv.org/html/2604.17009v1)
  - Results: SFT-only averages 63.43 and SFT+RL averages 70.48 (+7.05). This beats GPT-OSS-20B majority vote (66.88) and ToolOrchestra-style serial orchestration (66.21). The forced-serial variant scores 66.48. — [arXiv HTML](https://arxiv.org/html/2604.17009v1)
- **Mos'ka (NeurIPS 2026 workshop "Managing Agents that Manage Agents").** A small orchestrator works only through two tools: `spawn_subagent` (background) and `wait` (block until the next report). It is distilled by **SFT on successful teacher traces with sub-agents frozen**. The policy reasons about critical-path latency, greedily schedules unblocked subtasks, and revises its plan as reports arrive. — [NeurIPS 2026 page](https://neurips.cc/virtual/2026/169875)
  - Results: the student comes within 3.2 pp of the teacher on Workplace Assistant, using about half the orchestrator tokens. With E2B workers, pass@1 rises from 13.6% to 55.1% versus a direct agent. With 26B-A4B workers, GAIA2-Search rises from 22.9 to 29.0 but GAIA2-Execution *drops* from 22.3 to 11.0. Student size and trace count are not given on the page (abstract only; unverified details). — [NeurIPS 2026 page](https://neurips.cc/virtual/2026/169875)
- **AgentArk (arXiv 2602.03955, Feb 2026)** distills multi-agent debate into a single agent. I did not read it in detail. — [arXiv HTML](https://arxiv.org/html/2602.03955v1)

### Inferences
- ParaManager is the closest blueprint for this project: the same teacher family, a ≤4B Qwen3 student, parallel calls per round, about 4.5k filtered trajectories, and orchestrator-only loss. A 2B student is about half its size, so expect lower SFT-only performance than its 63.43.
- Two approaches exist for including sub-agents. One inlines the sub-agent work into one sequence (CoA/AFM, where sub-agents are simulated by the same model). The other freezes sub-agents and trains only orchestrator tokens, with sub-agent outputs as observations (PARL, Mos'ka, ParaManager). For an orchestrator that spawns real parallel sub-agents, the second approach is what all 2026 orchestrator works use. Sub-agent trajectories could additionally be trained as separate samples if the same small model will also serve as the sub-agent, but no source I found evaluates that choice directly.

### Gaps
- MAGDi student sizes and numbers were not verified from the full text.
- The CoA 84.6% cost reduction is weakly evidenced in the portion of the paper I read.
- Kimi PARL hyperparameters and the exact spawn tool schema are not public.
- I could not confirm Mos'ka's student and teacher sizes or trace counts (abstract only).
- I did not evaluate AgentArk's recipe.

## Q2. Practical lessons: filtering, data volume, format, masking, data mixing, on-policy vs SFT, small-model failure modes

### Takeaway
Keep only correct teacher trajectories, and prefer ones with error recovery and real (non-trivial) tool use. A few thousand trajectories are enough for LoRA SFT at ≤4B. Mask tool and observation tokens. Make the tool-call format strict and include parse-error recoveries. The dominant small-model failures are malformed calls, tool avoidance or serial collapse, over-spawning and loss of instruction following. On-policy distillation or RL fixes them more cheaply than more SFT.

### Cited Findings

**Filtering**
- The standard filter is correctness-based rejection sampling:
  - Agent Distillation drops wrong trajectories. — [arXiv 2505.17612](https://arxiv.org/html/2505.17612v2)
  - ParaManager keeps ≤1 of 8 samples per instance, correct only, and prioritizes error-recovery. — [arXiv 2604.17009](https://arxiv.org/html/2604.17009v1)
  - Mos'ka uses "successful teacher traces". — [NeurIPS 2026](https://neurips.cc/virtual/2026/169875)
- Beyond correctness, CoA/AFM drops trajectories with fewer than 5 tool interactions, redundant tool inputs or instruction-following failures. It also upsamples reflective and error-correcting trajectories. — [arXiv 2508.13167](https://arxiv.org/html/2508.13167)
- ParaManager adds a tool-balance constraint so no single tool dominates. — [arXiv 2604.17009](https://arxiv.org/html/2604.17009v1)
- ToolOrchestra drops tasks solvable without any action and tasks unsolved at pass@8 (too hard), and removes low-variance groups for RL. — [arXiv 2511.21689](https://arxiv.org/html/2511.21689)
- LLM-judge filtering and rewards: CoA uses Qwen2.5-72B as a binary judge for web answers, and ToolOrchestra uses GPT-5. — [arXiv 2508.13167](https://arxiv.org/html/2508.13167); [arXiv 2511.21689](https://arxiv.org/html/2511.21689)

**Data volume**

| Recipe | SFT data | Model(s) |
|---|---|---|
| Agent Distillation | ~2,000 trajectories | 0.5B–7B, LoRA |
| ParaManager | 4,470 | 4B |
| CoA web | 16,433 | 3B–32B |
| CoA code | 47k–60k | — |
| APR | 500k synthetic demos | 228M model trained from scratch |

Sources: [2505.17612](https://arxiv.org/html/2505.17612v2); [2604.17009](https://arxiv.org/html/2604.17009v1); [2508.13167](https://arxiv.org/html/2508.13167); [2504.15466](https://arxiv.org/html/2504.15466v2)

**Format**
- Multi-turn interaction with tagged reasoning plus a JSON tool call. Tool outputs come back as observations, under the user role in ToolOrchestra. — [2604.17009](https://arxiv.org/html/2604.17009v1); [2511.21689](https://arxiv.org/html/2511.21689)
- Parallel spawning is expressed in two ways:
  - Multiple calls in one turn (ParaManager, 1–4 per round). — [2604.17009](https://arxiv.org/html/2604.17009v1)
  - Asynchronous `spawn_subagent` + `wait` (Mos'ka). — [NeurIPS 2026](https://neurips.cc/virtual/2026/169875)
- Training only on narrow tool formats generalizes poorly to new tools. Strict-format training (exact JSON, code fences) generalized better in CoA §5.2. — [2508.13167](https://arxiv.org/html/2508.13167)

**Loss masking**
- Observation and tool-output tokens are masked in CoA (t ∉ 𝒪), Agent Distillation (Eq. 4) and ParaManager (Eq. 9, orchestrator tokens only). — [2508.13167](https://arxiv.org/html/2508.13167); [2505.17612](https://arxiv.org/html/2505.17612v2); [2604.17009](https://arxiv.org/html/2604.17009v1)
- In PARL, sub-agent trajectories are excluded from the objective. — [2602.02276](https://arxiv.org/html/2602.02276v1)

**Mixing in non-agentic data**
- The Thinking Machines experiment: SFT of Qwen3-8B on 100% domain documents dropped IF-eval from 85% to 45%. A 70/30 document/chat mix kept it at 79%, and subsequent on-policy distillation restored it to 83%. — [Thinking Machines blog, 27 Oct 2025](https://thinkingmachines.ai/blog/on-policy-distillation/)
- ToolOrchestra mixes GeneralThought-430K with ToolScale for RL. — [2511.21689](https://arxiv.org/html/2511.21689)

**On-policy distillation vs SFT**
- On-policy distillation samples from the student and minimizes per-token reverse KL to the teacher. It reached 70% AIME'24 in about 150 steps, versus 60% for SFT on 400k prompts, at roughly 9–30× less compute than SFT and about 1/10 the cost of RL (1,800 vs 17,920 GPU hours). — [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- With LoRA rank 32, the gap to full fine-tuning was 13% after SFT but only 6% after on-policy distillation. — [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- The blog notes that plain SFT on the model's own samples also degraded instruction following. — [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- (Note: per-token reverse KL needs teacher logprobs over the student's tokens. Qwen teacher and student share a tokenizer, which makes this feasible. The blog content I read did not state a tokenizer requirement explicitly.)

**Small-model failure modes**
- *Format and parse errors:* undistilled small models produce mostly unusable agent output (1.80 avg at 0.5B, 7.94 at 1.5B). Distilled small students still emit unparseable or failing code; SAG reduces this. — [2505.17612](https://arxiv.org/html/2505.17612v2)
- *Format brittleness:* models fail on unseen tool formats. — [2508.13167](https://arxiv.org/html/2508.13167)
- *Under-use of tools:* small students under-use retrieval and over-rely on the first retrieved doc. FTP further reduces retrieval calls and pushes the model toward hallucinated internal knowledge. — [2505.17612](https://arxiv.org/html/2505.17612v2)
- *Tool avoidance:* RL without an SFT warm start produces mostly parse and execution errors, sparse rewards and collapse toward answering in one turn without tools. — [2604.17009](https://arxiv.org/html/2604.17009v1)
- *Mode collapse:* without a diversity reward, the policy collapses onto one dominant tool path. — [2604.17009](https://arxiv.org/html/2604.17009v1)
- *Serial collapse and spurious parallelism (over-spawning):* — [Kimi K2.5 report](https://arxiv.org/html/2602.02276v1)
- *Delegation bias:* a prompted 8B orchestrator delegates 73% of calls to the strongest model. — [2511.21689](https://arxiv.org/html/2511.21689)
- *Task-dependent harm:* orchestration can hurt some task types (GAIA2-Execution 22.3 → 11.0 in Mos'ka). — [NeurIPS 2026](https://neurips.cc/virtual/2026/169875)

### Inferences
- For a 2B student on one 4090, a realistic data target is about 2k–5k correct, filtered orchestrator trajectories. The Agent Distillation LoRA recipe (r=64, all linear layers, LR 2e-4, 2 epochs) is a proven starting point at 0.5B–3B.
- Include parse-error and recovery turns (ParaManager-style PARSE_ERR observations) so the student learns to repair malformed spawn calls.
- Mix in some general chat or instruction data, on the order of 20–30%, to protect instruction following. The 70/30 ratio comes from a non-agentic experiment, so it is only an analogy.
- If time allows after SFT, a short on-policy distillation pass from the same-tokenizer Qwen teacher is the cheapest documented fix for SFT degradation and for the LoRA gap.
- Measure over-spawning and serial collapse directly: spawn count per task and critical-path steps.
- (Loops and repeated identical calls were not documented quantitatively in the sources I read. CoA's "redundant tool inputs" filter is the nearest evidence.)

### Gaps
- No source gives an ablation on how many trajectories are needed at ~2B for orchestration specifically.
- No direct ablation exists for inlining sub-agent traces versus training them as separate samples.
- Agent Distillation's full-FT vs LoRA comparison (App. D.5) was not retrieved.
- No source quantified looping rates.

## Q3. Evidence that SFT on teacher traces degrades small models, and how to avoid it

### Takeaway
There is solid evidence of a "learnability gap" at ≤3B. Small models learn worse from long or strong-teacher traces than from shorter or weaker-teacher ones. SFT on narrow data also erodes instruction following. The documented mitigations are:
- Mix complexities and teachers (Mix Distillation).
- Keep traces short and clean.
- Mix in general data.
- Follow SFT with on-policy distillation or RL.

### Cited Findings
- "Small Model Learnability Gap": models ≤3B do not consistently benefit from long-CoT data or from distillation from much larger teachers. The fix, Mix Distillation, blends long and short CoT (or strong- and weak-teacher responses) at a 1:4 ratio. With Mix-Long, Qwen2.5-3B-Instruct gained more than 8 points on MATH and AMC over long-CoT-only training, and Mix-Large gained more than 7 points on MATH, AIME and AMC. Base models show a larger gap than Instruct models. — [Li et al., arXiv 2502.12143, Feb 2025, ICML 2025](https://arxiv.org/html/2502.12143v3); [project page](https://small-model-gap.github.io)
- Naive agent distillation degraded the 3B student on most MATH difficulty levels relative to CoT, and agent students lag CoT on MATH500 at 3B and 7B. FTP partially fixed levels 4 and 5. — [arXiv 2505.17612](https://arxiv.org/html/2505.17612v2)
- Domain SFT on Qwen3-8B dropped IF-eval from 85% to 45%. Data mixing (70/30) plus on-policy distillation recovered it to 83%. — [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- The learnability gap is absent when the trajectories are short and agentic. Agent-distilled 0.5B–3B students beat CoT-distilled students 2–4× their size (0.5B agent 21.90 vs 1.5B CoT 21.28; 3B agent 36.60 vs 7B CoT 33.54). — [arXiv 2505.17612](https://arxiv.org/html/2505.17612v2)
- SFT-only orchestrators leave a large gap that RL closes: ParaManager-4B scored 63.43 after SFT and 70.48 after RL. — [arXiv 2604.17009](https://arxiv.org/html/2604.17009v1)

### Inferences
- With a Qwen ~30B MoE teacher, prefer the teacher's non-thinking or Instruct-mode traces, or truncate and summarize long thinking, over long reasoning traces for a 2B student. This follows the Mix-Long finding; it is an inference, not tested on orchestration.
- Start from the Instruct checkpoint of the 2B model rather than the base, since base models showed a larger gap.

### Gaps
- No study I found measures the learnability gap specifically for orchestration or tool-call traces at ~2B.

## Q4. Work distilling parallel tool-calling or parallel sub-agent behavior into small models

### Takeaway
As of Nov 2026 there is a small but direct literature. Two papers distill a parallel orchestrator into a small student:
- ParaManager: Qwen3-30B-A3B teacher to a Qwen3-4B student, SFT then RL, with 1–4 parallel calls per round.
- Mos'ka: SFT-only distillation of a `spawn_subagent`/`wait` orchestrator, with frozen workers.

APR shows spawn/join can be learned by a ~228M model trained from scratch. TinyAgent (1.1B) shows LLMCompiler-style parallel function-call planning at the edge. Kimi PARL documents the key failure modes (serial collapse, spurious parallelism) and fixes them with annealed auxiliary rewards rather than SFT.

### Cited Findings
- ParaManager (Qwen3-4B, Apr 2026):
  - A forced-serial variant scored 66.48 vs 70.48 for the parallel model.
  - The parallelism reward requires ≥1.25 calls per round on average.
  - Without SFT initialization, RL fails.
  
  — [arXiv 2604.17009](https://arxiv.org/html/2604.17009v1)
- Mos'ka (NeurIPS 2026 workshop): the SFT-distilled small orchestrator comes within 3.2 pp of the teacher at about half the orchestrator tokens. Its learned policy schedules unblocked subtasks greedily based on critical-path latency. — [NeurIPS 2026](https://neurips.cc/virtual/2026/169875)
- APR: SFT on 500k solver demonstrations, then GRPO. RL mainly increased the number of spawned children (6.1 → 8.2). — [arXiv 2504.15466](https://arxiv.org/html/2504.15466v2)
- Kimi K2.5 PARL: no swarm SFT cold start is described. r_parallel and r_finish are annealed to 0. The orchestrator was first trained with small sub-agents before moving to larger ones. — [arXiv 2602.02276](https://arxiv.org/html/2602.02276v1)
- TinyAgent (Berkeley, arXiv 2409.00608, EMNLP 2024 demo): TinyAgent-1.1B and 7B were fine-tuned on a curated function-calling dataset using the LLMCompiler parallel-planning format, with tool retrieval and quantization for the edge. The 1.1B model is reported to surpass GPT-4-Turbo on their task, at a reported 80.06% success for 1.1B and 84.95% for 7B. These success rates come from a secondary summary; verify against the paper. — [arXiv 2409.00608](https://arxiv.org/abs/2409.00608); [BAIR blog](https://bair.berkeley.edu/blog/2024/05/29/tiny-agent)
- LLMCompiler (the parallel function-calling planner, executor and task-fetching unit) reports up to 1.35× latency gain over OpenAI parallel function calling at similar accuracy. This is from a secondary summary. — [search summary via catalyzex/TinyAgent](https://arxiv.org/html/2409.00608v2)

### Inferences
- Pure SFT can transfer *some* parallel spawning to a small model (Mos'ka, ParaManager-SFT). However, every RL-based work found that parallelism needed explicit reward shaping. This suggests SFT students may under-spawn (serial collapse) or over-spawn, and that the SFT data should contain clear, consistent parallel decompositions.
- For a 3-day SFT-only budget, the main levers are:
  - Consistent spawn patterns in the data.
  - Balanced spawn counts (filter out degenerate 1-spawn and many-spawn traces where they don't fit the task).
  - Strict call schemas.
  - Logging spawn-count distributions against the teacher during evaluation.

### Gaps
- I found no work that distills parallel orchestration into a model of ~2B specifically. The smallest documented orchestrator students are 4B (ParaManager) and an unspecified "small" model (Mos'ka).
- TinyAgent's training data, and whether it was teacher-generated, were not verified.
