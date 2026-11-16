# LLM Task Planning as DAGs of Parallel Subtasks and the Math of Multi-Agent LLM Orchestration (as of 2026-11-13)

Legend: [VERIFIED] means fetched from the primary source in this session. [RECONSTRUCTED] means standard math written by the researcher with no primary source fetched; flag it on slides. [UNVERIFIED] means it comes from secondary coverage only.

## Q1. Plan-as-DAG systems: DAG formats, dependency handling, reported speedups and accuracy

### Takeaway
LLMCompiler (ICML 2024) is the canonical "compile a question into a DAG" design. A Planner emits tasks whose arguments use `$k` placeholders for earlier outputs. A non-LLM Task Fetching Unit greedily dispatches ready tasks and substitutes values. An Executor runs the tasks concurrently. Against ReAct it reports up to 3.7x lower latency, up to 6.7x lower cost and about 9% higher accuracy, and it is only about 1.1–1.4x faster than OpenAI parallel function calling. Native parallel-reasoning models (Multiverse, APR, Parallel-R1, NPR, Hogwild!) move the fork/join inside a single model's decoding. Kimi K2.5 Agent Swarm learns the orchestration with RL (PARL) and reports up to 4.5x latency reduction.

### Cited Findings

**LLMCompiler: "An LLM Compiler for Parallel Function Calling"**
- Authors: Kim, Moon, Tabrizi, Lee, Mahoney, Keutzer, Gholami. ICML 2024. arXiv 2312.04511: v1 7 Dec 2023, v2 6 Feb 2024, v3 5 Jun 2024. Headline: up to 3.7x latency speedup, up to 6.7x cost savings and up to ~9% accuracy improvement vs ReAct. [VERIFIED] — [arXiv abs](https://arxiv.org/abs/2312.04511)
- Its three components [VERIFIED] — [arXiv HTML](https://arxiv.org/html/2312.04511):
  - **Planner:** emits a DAG of tasks with arguments and dependencies. Placeholder variables like `$1` and `$2` stand for the outputs of earlier tasks.
  - **Task Fetching Unit:** a greedy policy fetches tasks as soon as they are ready and replaces placeholders (e.g. `$1` in Task 3) with the actual outputs. It is a simple fetch-and-queue mechanism with no LLM.
  - **Executor:** runs dispatched tasks asynchronously and concurrently. Each task has its own memory. Tools can be functions, APIs or other LLM agents.
- **Dynamic replanning:** the Executor's intermediate results go back to the Planner, which emits a new task set. This loops until the final answer is reached. For Game of 24 the tools are thought_proposer, state_evaluator and top_k_select. [VERIFIED] — [arXiv HTML](https://arxiv.org/html/2312.04511)
- **Streaming planner** (`--stream`): tasks are dispatched to the Task Fetching Unit and Executor as they are parsed, instead of waiting for the whole plan. [VERIFIED] — [GitHub SqueezeAILab/LLMCompiler](https://github.com/SqueezeAILab/LLMCompiler)
- **Latency model** (Appendix E.1, as summarized from the paper's HTML; the original symbol typography may differ slightly) [VERIFIED] — [arXiv HTML](https://arxiv.org/html/2312.04511):
  - ReAct: $T^R=\sum_i\big(T^R_P(P_i)+T_E(E_i)\big)$
  - LLMCompiler: $T^C=\sum_i T^C_P(P_i)+\max_k T_E(E_k)$
  - With streaming: $T^{SC}=\sum_i T^C_P(P_i)+T_E(E_N)$, with $T^{SC}\le T^C$
  - Speedup: $\gamma=T^R/T^C$. Bounds: $\gamma_{\max}\approx N$ (number of tasks); $\gamma_{\min}\approx 1$ when planning latency dominates.
- **Table 1 (GPT; speedup vs ReAct† = ReAct with an anti-looping prompt)** [VERIFIED] — [arXiv HTML](https://arxiv.org/html/2312.04511):

  | Benchmark | ReAct† (acc / latency) | OAI parallel FC (acc / latency / speedup) | LLMCompiler (acc / latency / speedup) |
  |---|---|---|---|
  | HotpotQA | 62.47 / 7.12 s | 62.05 / 4.42 s / 1.61x | 62.00 / 3.95 s / 1.80x |
  | Movie Rec. | 72.47 / 20.47 s | 77.00 / 7.42 s / 2.76x | 77.13 / 5.47 s / 3.74x |
  | ParallelQA | 89.09 / 35.90 s (plain ReAct) | 87.32 / 19.29 s / 1.86x | 89.38 / 16.69 s / 2.15x |

  - Game of 24 (GPT-4): ToT 74.00% at 241.2 s vs LLMCompiler 75.33% at 83.6 s (2.89x).
  - LLaMA-2 (ReAct†): HotpotQA 54.40 → 57.83 (1.40x); Movie Rec. 70.60 → 77.80 (2.82x).
  - Cost (Table 2): 3.37x lower on HotpotQA, 6.73x on Movie Rec., 4.65x on ParallelQA.
- **vs OpenAI parallel function calling**, computed by the researcher from Table 1: 1.12x (HotpotQA), 1.36x (Movie Rec.), 1.16x (ParallelQA). [VERIFIED numbers; ratio computed] — [arXiv HTML](https://arxiv.org/html/2312.04511)
- **Possible inconsistencies in the source table** (as extracted):
  - ParallelQA LLaMA-2 lists ReAct at 15.47 s and LLMCompiler at 26.20 s but a 2.27x speedup.
  - Game of 24 LLaMA-2 is 2.01x in the text but 2.09x in the table.

  Check the PDF before putting these on slides. — [arXiv HTML](https://arxiv.org/html/2312.04511)
- **LLMCompiler vs ReWOO:** ReWOO decouples reasoning from observation to save tokens. LLMCompiler adds parallel execution (lower latency as well as cost) and dynamic replanning. [VERIFIED] — [arXiv HTML](https://arxiv.org/html/2312.04511)

**ReWOO: "Decoupling Reasoning from Observations for Efficient Augmented Language Models"**
- Xu et al., arXiv 2305.18323, 23 May 2023. 5x token efficiency and 4% accuracy improvement on HotpotQA. Offloads reasoning from 175B GPT-3.5 to 7B LLaMA. [VERIFIED] — [arXiv](https://arxiv.org/abs/2305.18323)
- The Planner/Worker/Solver design with `#E1`-style evidence variables is from memory and was not confirmed on the abstract page. [UNVERIFIED]

**Graph of Thoughts**
- Besta et al., arXiv 2308.09687 (18 Aug 2023; v4 6 Feb 2024), AAAI 2024. Sorting quality +62% over ToT, cost reduced by more than 31%. [VERIFIED] — [arXiv](https://arxiv.org/abs/2308.09687)

**Skeleton-of-Thought**
- arXiv 2307.15337, Microsoft Research / Tsinghua. The LLM first writes an answer skeleton, then expands the points in parallel (by API calls or batched decoding).
- Speedups of up to 2.39x. Microsoft Research reports >2x on 8 of 12 models, with a range of 1.13x–2.39x. The arXiv v1 abstract says 11 LLMs, so the model count differs between sources. — [arXiv](https://arxiv.org/abs/2307.15337v1); [Microsoft Research](https://www.microsoft.com/en-us/research/?p=983583)

**Native parallel reasoning (fork/join inside one model)**
- **Multiverse** (Yang, An, Liu, T. Chen, B. Chen; arXiv 2506.09991, 11 Jun 2025) [VERIFIED] — [arXiv](https://arxiv.org/abs/2506.09991)
  - Uses MapReduce stages: Map (adaptive decomposition), Process (parallel subtasks) and Reduce (lossless merge).
  - Fine-tuned ~3 h on 1K examples. Multiverse-32B scores 54% on AIME24 and 46% on AIME25.
  - Up to 2x speedup across batch sizes, and +1.87% on average over AR LLMs at the same context length.
- **Hogwild! Inference** (Rodionov et al., Alistarh; arXiv 2504.06261, 8 Apr 2025; v4 17 Nov 2025; NeurIPS 2025) [VERIFIED] — [arXiv](https://arxiv.org/abs/2504.06261)
  - Several instances of the same LLM share a concurrently updated KV cache ("concurrent attention"). Workers decide among themselves how to split the work. RoPE avoids recomputation. No fine-tuning is needed.
  - The abstract gives no headline numbers.
- **APR (Adaptive Parallel Reasoning)** (arXiv 2504.15466) — [arXiv HTML](https://arxiv.org/html/2504.15466v2), via search snippet
  - Uses `spawn()`/`join()` operations for multi-threaded inference.
  - Countdown: 83.4% vs 60.0% at a 4k context.
- **Parallel-R1** (Zheng et al., Tencent; arXiv 2509.07980, 9 Sep 2025) [VERIFIED] — [arXiv](https://arxiv.org/abs/2509.07980)
  - RL for parallel thinking. +8.4% accuracy over a sequential-thinking model trained with RL, and +42.9% over the baseline on AIME25.
  - Parallel thinking is described as a "mid-training exploration scaffold".
- **Native Parallel Reasoner (NPR)** (arXiv 2512.07461; Qwen3-4B; teacher-free self-distilled RL) — [arXiv](https://arxiv.org/pdf/2512.07461); [arXiv HTML v2](https://arxiv.org/html/2512.07461v2), via search snippet
  - Up to 24.5% performance gains and up to 4.6x speedup (AIME25 4.6x, HMMT25 4.1x, AMC23 2.9x).
  - Reports a 100% genuine-parallel trigger rate. Multiverse-32B sometimes falls back to AR decoding.

**Kimi K2.5 Agent Swarm**
- "Kimi K2.5: Visual Agentic Intelligence", Kimi Team, arXiv 2602.02276. v1 2 Feb 2026; v2 7 Aug 2026. [VERIFIED] — [arXiv](https://arxiv.org/abs/2602.02276)
  - "a self-directed parallel agent orchestration framework", "reduces latency by up to 4.5×".
  - The body says it cuts the time to reach target performance by 3×–4.5× on WideSearch.
- **Results (Table 6, Agent Swarm vs K2.5 single agent)** [VERIFIED] — [arXiv HTML](https://arxiv.org/html/2602.02276)
  - BrowseComp 78.4 vs 60.6
  - WideSearch Item-F1 79.0 vs 72.7
  - In-house Swarm Bench 58.3 vs 41.6
- The model has "interfaces for sub-agent creation and task delegation". The paper does not name the tools. There are no predefined subagents or roles. [VERIFIED] — [arXiv HTML](https://arxiv.org/html/2602.02276)
- Up to 100 sub-agents and up to 1,500 tool calls (one Kimi page says "steps") — vendor blog. These figures were not found in the portion of the paper that was read. — [Kimi blog](https://www.kimi.com/blog/kimi-k2-5)
- Agent Swarm is a beta on Kimi.com. — [Kimi blog](https://www.kimi.com/blog/kimi-k2-5)
- Independent testers report sub-agents drifting into inconsistent definitions of shared concepts. — [HF blog (mlabonne)](https://huggingface.co/blog/mlabonne/kimik25)

### Inferences
- DAG formats fall into a spectrum:
  - Explicit text plans with variable references: LLMCompiler `$k`, ReWOO `#E`.
  - Provider-native parallel tool calls: several calls in one assistant turn, with dependencies only across turns, which gives a level-by-level (BFS) schedule.
  - Skeleton-then-expand: SoT, a single fan-out level.
  - In-model fork/join tokens: Multiverse Map/Process/Reduce, APR spawn/join, NPR.
  - Learned orchestrator tool calls that spawn sub-agents: Kimi.
- For a MedCalc-style orchestrator, LLMCompiler's split (an LLM planner plus a deterministic scheduler doing placeholder substitution) is the most directly reusable pattern.
- The LLMCompiler latency formula and Kimi's CriticalSteps are the same idea: a sum over sequential stages of the max over parallel branches, i.e. the critical path.
- LLMCompiler's gain over OpenAI parallel function calling is modest (1.1–1.4x). Most of the gain over ReAct comes from parallelism itself, not from the specific DAG syntax.

### Gaps
- Exact OpenAI/Qwen parallel-function-calling formats and documentation: not fetched in this session.
- "Flash" / async function calling papers: not searched (budget).
- The exact LLMCompiler planner grammar (e.g. `join()` as a final task) is in Appendix H, which was truncated in the fetch. Check the repo's `configs/*/prompts.py`.
- APR speedup numbers and Hogwild! quantitative results were not obtained.

## Q2. Mathematical formalisms: Dec-POMDP, options/HRL, critical steps, PARL reward, GRPO, error propagation, Condorcet

### Takeaway
Kimi K2.5 gives exact formulas for the PARL reward ($\lambda_1 r_{\text{parallel}}+\lambda_2 r_{\text{finish}}+r_{\text{perf}}$, with both $\lambda$s annealed to 0) and for CriticalSteps, plus its own token-clipped policy objective, which is GRPO-like but not called GRPO. Liu et al. 2025 (MAGRPO) gives an explicit Dec-POMDP tuple for LLM collaboration. The options view, the DAG error product and the Condorcet theorem are textbook math that the researcher reconstructed, not taken from fetched papers.

### Cited Findings

**PARL (Kimi K2.5, Section 3)** [VERIFIED] — [arXiv HTML](https://arxiv.org/html/2602.02276)
- **Reward:**
  $$r_{\mathrm{PARL}}(x,y)=\lambda_1\cdot r_{\text{parallel}}+\lambda_2\cdot r_{\text{finish}}+r_{\text{perf}}(x,y)$$
  - $r_{\text{parallel}}$ is the "instantiation reward", $r_{\text{finish}}$ the "sub-agent finish rate" and $r_{\text{perf}}$ the task-level outcome.
  - "the hyperparameters λ1 and λ2 are annealed to zero over the course of training". The schedule is not specified.
  - No closed-form definitions are given for $r_{\text{parallel}}$ or $r_{\text{finish}}$.
- **Failure modes:**
  - **Serial collapse:** "a local optimum where the orchestrator defaults to single-agent execution". $r_{\text{parallel}}$ is there to counter it.
  - **Spurious parallelism:** "a reward-hacking behavior in which the orchestrator increases parallel metrics dramatically by spawning many subagents" without meaningful decomposition. "By rewarding completed subtasks, r_finish enforces feasibility."
- **Critical steps:**
  $$\text{CriticalSteps}=\sum_{t=1}^{T}\left(S_{\mathrm{main}}^{(t)}+\max_i S_{\mathrm{sub},i}^{(t)}\right)$$
  - Stage $t$: the main agent takes $S^{(t)}_{\text{main}}$ steps (typically 1), and subagent $i$ in that parallel group takes $S^{(t)}_{\text{sub},i}$.
  - "The duration of stage t is governed by the longest-running subagent within that cohort."
  - The metric is modeled on the critical path of a computation graph. It favors balanced decomposition over maximal concurrency.
  - It is used as a resource constraint / latency-oriented measure, as opposed to total steps.
- **Training setup:**
  - "a trainable orchestrator and frozen subagents instantiated from fixed intermediate policy checkpoints".
  - Subagent trajectories are excluded from the objective; only the orchestrator is updated by RL. The stated rationale is to avoid credit-assignment ambiguity and training instability.
  - Curriculum: "we first train the orchestrator using small-size subagents before transitioning to larger models."
- **Policy objective (Section 4.4.2)**, as extracted (the per-token ratio is $\pi_\theta/\pi_{\text{old}}$):
  $$L_{\mathrm{RL}}(\theta)=\mathbb{E}_{x\sim\mathcal{D}}\left[\frac{1}{N}\sum_{j=1}^{K}\sum_{i=1}^{|y_j|}\mathrm{Clip}\left(\frac{\pi_\theta}{\pi_{\mathrm{old}}},\alpha,\beta\right)\left(r(x,y_j)-\bar{r}(x)\right)-\tau\left(\log\frac{\pi_\theta}{\pi_{\mathrm{old}}}\right)^2\right]$$
  - A "token-level clipping mechanism designed to mitigate the off-policy divergence", which departs from K1.5.
  - $\alpha,\beta,\tau>0$ are hyperparameters with no values given. The paper does not call it GRPO.
  - The baseline $\bar r(x)$ is a group mean over $K$ samples, which makes it GRPO-like (researcher's inference).
  - Re-check the exact typesetting against the PDF; the WebFetch rendering may have dropped the token conditioning arguments.

**Dec-POMDP for LLM collaboration: Liu, Liang, Lyu, Amato, "LLM Collaboration With Multi-Agent Reinforcement Learning"** (arXiv 2508.04652, 6 Aug 2025; AAAI version indexed) [VERIFIED] — [arXiv HTML v1](https://arxiv.org/html/2508.04652v1)
- Tuple: $\langle \mathcal{I},\mathcal{S},\{\mathcal{O}_i\},\{\mathcal{A}_i\},R,T,H\rangle$, with $n$ LLM agents, a joint reward and horizon $H$.
- MAGRPO advantage:
  $$\widehat A^{(g)}_t=\frac{R^{(g)}_t-\frac1G\sum_{g=1}^G R^{(g)}_t}{\sigma(R^{(g)}_t)},\qquad R^{(g)}_t=\sum_{\tau=t}^{H-1} r^{(g)}_\tau$$
- MAGRPO objective: a clipped surrogate $\min(\rho\hat A,\ \varepsilon\text{-clip}(\rho)\hat A)$ averaged over batch and group, with per-agent ratio $\rho^{(g)}_{i,t}=\pi_{\theta_i}(a\mid h)/\pi_{\theta_i,\text{old}}(a\mid h)$. The source rendering was slightly garbled.
- Other Dec-POMDP formalizations — [arXiv 2512.24609](https://arxiv.org/pdf/2512.24609v1); [DynaHMRC arXiv 2606.14882](https://arxiv.org/pdf/2606.14882), via search snippets only:
  - arXiv 2512.24609: Dec-POMDP with CTDE and GRPO.
  - DynaHMRC: heterogeneous multi-robot LLM agents.

**Standard formulas, not fetched this session** [RECONSTRUCTED]
- **GRPO** (Shao et al. 2024, DeepSeekMath, arXiv 2402.03300), standard form:
  $$J(\theta)=\mathbb{E}\Big[\tfrac1G\sum_{i=1}^G\tfrac1{|o_i|}\sum_t \min\big(\rho_{i,t}\hat A_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}\big)-\beta D_{KL}[\pi_\theta\|\pi_{\text{ref}}]\Big]$$
  with $\hat A_{i,t}=(r_i-\mathrm{mean}(\mathbf r))/\mathrm{std}(\mathbf r)$ for outcome supervision. Verify against arXiv 2402.03300 before the slides.
- **Options / HRL view** (Sutton, Precup & Singh 1999):
  - An option is $\omega=\langle \mathcal I_\omega,\pi_\omega,\beta_\omega\rangle$.
  - The orchestrator is a policy over options. A sub-agent call is an option: a frozen $\pi_\omega$ that runs until termination $\beta_\omega$ and returns a summary.
  - This gives a semi-MDP at the orchestrator level. That is exactly Kimi's setup (frozen subagents, RL only on the orchestrator). The mapping is the researcher's own; no paper fetched states it.
- **Error propagation in a DAG**, under independence and with no verification:
  $$P(\text{correct})=\prod_{v\in V} p_v$$
  For a chain of depth $d$ with uniform $p$, this is $p^d$. It motivates verification nodes. It is empirically consistent with Kim et al.'s finding that centralized verification contains error amplification at 4.4x vs 17.2x for independent agents (see Q3).
- **Condorcet jury theorem:** with $n$ (odd) independent voters, each correct with $p>1/2$, majority accuracy is
  $$P_n=\sum_{k=\lceil n/2\rceil}^{n}\binom{n}{k}p^k(1-p)^{n-k}$$
  It increases monotonically toward 1 as $n\to\infty$. Correlated LLM errors violate the independence assumption.

### Inferences
- **MedCalc analogue:** for a DAG with stages $t$, $\text{latency}\approx\sum_t(\text{plan}_t+\max_{i\in t}\text{exec}_i)$. This is the same structure as the LLMCompiler and Kimi formulas. Parallelism helps only to the extent that per-level width × balance is large.
- PARL's two auxiliary terms are shaping rewards that are later annealed away. Present them as a curriculum or exploration bonus, not as part of the final objective.

### Gaps
- The exact Kimi annealing schedule, clip values, and functional forms of $r_{\text{parallel}}$/$r_{\text{finish}}$ are not specified in the text read; about 63k characters of the HTML were unread.
- No fetched paper formalizes the orchestrator–subagent relationship explicitly as options/semi-MDP.
- GRPO and Condorcet were not verified against a source in this session.

## Q3. Empirical results: when multi-agent systems help (Kim et al. 2025 / Nature MI 2026, Gao et al. 2025, Tang et al. 2025)

### Takeaway
Kim et al. ("Towards a Science of Scaling Agent Systems", arXiv 2512.08296) appears to be the arXiv version of the Nature Machine Intelligence 2026 paper "Capable language models can outgrow the benefits of collaboration": same 20 authors and same headline numbers. Its main findings:
- Gains range from +80.8% (decomposable finance tasks, centralized) to −70.0% (sequential planning, independent).
- Coordination stops helping once single-agent accuracy passes about 45%.
- Errors amplify 17.2x with independent agents vs 4.4x with centralized ones.

Gao et al. find the MAS advantage shrinks with frontier models. Tang et al. find MAS gains grow with task depth and width, especially depth.

### Cited Findings

**Kim et al., "Towards a Science of Scaling Agent Systems"** [VERIFIED] — [arXiv abs](https://arxiv.org/abs/2512.08296); [arXiv HTML](https://arxiv.org/html/2512.08296)
- arXiv 2512.08296: v1 9 Dec 2025, v3 8 Apr 2026. Authors: Yubin Kim, Ken Gu, Chanwoo Park, … Daniel McDuff, Xin Liu (20 authors).
- **Design:** 260 configurations, 6 benchmarks, 5 architectures and 3 LLM families. Task prompts, tools and compute budget are held fixed.
- **Benchmarks:** BrowseComp-Plus, Finance-Agent, PlanCraft, WorkBench, SWE-bench Verified, Terminal-Bench.
- **Architectures:**
  - SAS: a single agent.
  - Independent: parallel agents with no communication, plus an aggregator.
  - Decentralized: all-to-all debate.
  - Centralized: an orchestrator with sub-agents.
  - Hybrid: an orchestrator plus limited peer communication.
- **Models:** GPT-5-nano, GPT-5-mini, GPT-5; Gemini-2.0 Flash, Gemini-2.5 Flash, Gemini-2.5 Pro; Claude Sonnet 3.7, 4 and 4.5. Sonnet 3.7 was excluded from SWE-bench and Terminal-Bench.
- **Deltas vs SAS:**
  - Finance-Agent, Centralized: +80.8%
  - PlanCraft, Independent: −70.0%
  - BrowseComp-Plus, Decentralized: +9.2%
  - WorkBench, Decentralized: +5.6%
  - SWE-bench Verified: −2.1% to −14.9%
  - Terminal-Bench: Independent +1.7%, Centralized −19.2%
- **Error amplification** (trace-level, Table 5): SAS 1.0, Centralized 4.4, Hybrid 5.1, Decentralized 7.8, Independent 17.2. The task-level figure is ~1.1–1.3x.
- **Coordination overhead vs SAS:** Independent 58%, Centralized 285%, Decentralized 263%, Hybrid 515%.
- **Predictive model (Eq. 1):** 20 parameters. Selected standardized coefficients:

  | Term | Coefficient | p |
  |---|---|---|
  | Intercept | 0.430 | |
  | Intelligence | 0.126 | 0.008 |
  | log(1+T) tools | 0.166 | <0.001 |
  | Single-agent baseline | 0.250 | 0.001 |
  | Baseline × log(1+agents) | −0.236 | 0.004 |
  | Efficiency × tools | −0.096 | 0.002 (95% CI [−0.154, −0.037]) |
  | Redundancy × agents | 0.024 | 0.034 |

- **Fit:** CV R² = 0.373 (0.413 with the Agentic Capability Index).
- **Architecture selection on held-out configurations:** 87% correct, vs 20% random and 54% for capability-only models.
- **Capability-saturation threshold:** about 45% single-agent accuracy (decision boundary ≈0.45 after denormalization).

**Nature Machine Intelligence version** — [UW Ubicomp lab](https://ubicomplab.cs.washington.edu/publications/Zhihan_NatureMI2026/); [MIT Media Lab](https://www.media.mit.edu/publications/capable-language-models-can-outgrow-the-benefits-of-collaboration/)
- "Capable language models can outgrow the benefits of collaboration", same 20 authors. Search snippet: Nature Machine Intelligence (2026) 8, 1157–1172, DOI 10.1038/s42256-026-01268-y.
- Abstract numbers:
  - Same 260/6/5/3 setup, R² 0.373/0.413 and 87%.
  - The capability-saturation threshold predicted the coordination effect in 94% of validation configurations on SWE-bench Verified and Terminal-Bench.
  - Single-agent baseline performance is "the most robust predictor".
  - "Baseline-scaled error amplification" holds under cluster-robust inference ($p_{\text{robust}}=0.030$).
  - The threshold is presented as a practical selection rule, not a universal scaling law.
- Bioengineer.org news coverage was July 2026, which suggests publication around then. [UNVERIFIED date] — [Bioengineer.org](https://bioengineer.org/ai-language-models-may-surpass-collaboration-benefits-as-they-scale/)

**Gao et al., "Single-agent or Multi-agent Systems? Why Not Both?"** [VERIFIED] — [arXiv](https://arxiv.org/abs/2505.18286)
- arXiv 2505.18286, 23 May 2025. Authors: Mingyan Gao, Yanzi Li, Banruo Liu, Yifan Yu, Phillip Wang, Ching-Yu Lin, Fan Lai.
- The MAS advantage over SAS diminishes as LLMs improve (examples: OpenAI-o3, Gemini-2.5-Pro).
- The paper proposes efficient identification of the error-prone agent.
- Hybrid "request cascading" between MAS and SAS improves accuracy by 1.1–12% and cuts deployment cost by up to 20%.

**Tang et al., "On the Importance of Task Complexity in Evaluating LLM-Based Multi-Agent Systems"** [VERIFIED] — [arXiv](https://arxiv.org/abs/2510.04311)
- arXiv 2510.04311, 5 Oct 2025. Authors: Bohan Tang, Huidong Liang, Keyue Jiang, Xiaowen Dong.
- Tasks are characterized by depth (reasoning length) and width (capability diversity), with a theoretical analysis of multi-agent debate.
- The MAS advantage grows with both depth and width, more so with depth.
- The abstract gives no numbers.

**Related work, not fetched** — [arXiv 2604.07821](https://arxiv.org/pdf/2604.07821); [arXiv 2511.02687](https://arxiv.org/html/2511.02687v1), via search snippets only
- "More Capable, Less Cooperative?" (arXiv 2604.07821; ICML 2026 per snippet) reports a "capability-cooperation inversion".
- "The Collaboration Gap" (arXiv 2511.02687) finds that models strong alone often degrade when they must collaborate.

### Inferences
- **Design rule for the orchestrator:** use a centralized orchestrator with verification rather than independent voting. Parallelize only genuinely decomposable (wide) subproblems, and keep strictly sequential chains in one agent.
  - Kim et al.: PlanCraft −70% and Finance +80.8%.
  - Error amplification is 4.4x with an orchestrator vs 17.2x with independent agents.
- If the base model already exceeds ~45% single-agent accuracy on the task, expect diminishing or negative returns from adding agents. Latency gains from parallel DAG execution (LLMCompiler, Kimi) may then be the main benefit rather than accuracy.
- Tang et al. (MAS helps more with depth) and Kim et al. (sequential planning hurts) are in tension. Tang et al. study debate on reasoning depth, while Kim et al. measure multi-agent decomposition of sequential planning. The setups differ, so the results are not directly contradictory, but the tension should be flagged on slides.

### Gaps
- The exact Nature MI publication date and full text were not retrieved; volume, pages and DOI come from a search-engine snippet only.
- It is unconfirmed whether the Nature MI paper's numbers differ from arXiv v3 (e.g. whether the 45% threshold is restated).
- Gao et al. per-benchmark numbers and Tang et al. quantitative results were not retrieved (abstracts only).
