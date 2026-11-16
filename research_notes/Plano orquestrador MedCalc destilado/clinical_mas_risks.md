# Risks, failure modes and lessons for clinical multi-agent LLM systems (as of Nov 2026)

Context: a ~2B LLM orchestrator splits MedCalc-Bench rule-based score calculation into parallel per-variable extraction sub-agents and then aggregates their outputs. Research cut-off is 2026-11-13. Where a figure comes from a search snippet or secondary summary rather than the full text, it is flagged **[unverified]**.

## 1. "The Optimization Paradox in Clinical AI Multi-Agent Systems" (arXiv 2506.06574)

### Takeaway
A "Best of Breed" pipeline built from the best component at each stage had the best component metrics but the worst diagnostic accuracy (67.7% vs 77.4% for the top multi-agent system). The cause was the handoffs between agents: an upstream agent under-gathered and the downstream agent hallucinated. The lesson is to validate the whole system end to end, not only its components.

### Cited Findings
- Paper: Bedi, Mlauzi, Shin, Koyejo, Shah (Stanford). v1 posted 6 Jun 2025, v2 on 12 Jun 2025. Categories cs.AI and cs.MA. — [arXiv abs](https://arxiv.org/abs/2506.06574)
- Data: 2,400 MIMIC-CDM cases (appendicitis 957, cholecystitis 648, pancreatitis 538, diverticulitis 257). After 20 pilot cases were removed, the rest were split 1,190 dev / 1,190 held-out test. — [arXiv HTML v2](https://arxiv.org/html/2506.06574v2)
- The task has three stages: information gathering (ordering tests), interpretation (labelling labs high/normal/low) and differential diagnosis. Eight backbones were tested: GPT-4o, GPT-4.1, Claude-3.5-Sonnet, Gemini-1.5-Pro, Gemini-2.0-Flash, Llama-3.3-70B, o3-mini and DeepSeek-R1. — [arXiv HTML v2](https://arxiv.org/html/2506.06574v2)
- Best components chosen on dev: gathering GPT-4o (coverage-to-test ratio 0.107), interpretation GPT-4.1 (85.4% lab accuracy), diagnosis Gemini-2.0-Flash (78% top-1). — [arXiv HTML v2](https://arxiv.org/html/2506.06574v2)
- Test set results:
  - Best single-agent system: 75.63%.
  - Best of 26 multi-agent systems: 77.39%.
  - Best of Breed: 67.65%, which is 9.75 points below the top MAS (95% CI 7.11–12.38, McNemar p<0.0001) and 7.98 points below the top single agent (p<0.0001). — [arXiv HTML v2](https://arxiv.org/html/2506.06574v2)
- Best of Breed beat the other multi-agent systems on information gathering (80% win rate), interpretation (76%), computational cost (80%) and clinical resource cost (84%), but won only **8%** on diagnosis. Its lab interpretation was 85.50% vs 82.81% for the top MAS, and it used 3.62 tools per patient vs 6.47. — [arXiv HTML v2](https://arxiv.org/html/2506.06574v2)
- Across all systems, the process metric (coverage-to-test ratio) did not correlate with diagnostic accuracy (Spearman ρ = −0.057, p = 0.748). — [arXiv HTML v2](https://arxiv.org/html/2506.06574v2)
- How Best of Breed failed:
  - The gathering agent stopped early in 7.06% of cases.
  - The diagnosis agent ordered unauthorized tests in 13.87% of cases and hallucinated test results in 13.87% (165 cases), vs 0.42% hallucination for the top MAS, roughly 33 times more.
  - In the 174 cases the top MAS got right and Best of Breed got wrong, the causes were 46.55% hallucinated results, 36.21% insufficient gathering and 17.24% other. — [arXiv HTML v2](https://arxiv.org/html/2506.06574v2)
- Effect of mixing model families:
  - Median accuracy was 74.29% for homogeneous backbones, 74.75% for mixed and 71.22% for heterogeneous.
  - The effect sizes are large (d = 1.41 and 1.17) but not significant after Bonferroni correction.
  - The authors only hypothesize that differing "communication styles" lose information at handoffs; they did not test this. — [arXiv HTML v2](https://arxiv.org/html/2506.06574v2)
- Stated lessons: component metrics poorly predict integrated performance, end-to-end validation is required, and regulators should ask for system-level evidence. Stated limitations: single center (BIDMC), four pathologies, simple sequential orchestration, and process metrics that were not tied to outcomes. — [arXiv HTML v2](https://arxiv.org/html/2506.06574v2)

### Inferences
- What this means here: high per-variable extraction accuracy does not guarantee correct total scores. The headline metric should be exact-match on the final score, with per-variable accuracy reported alongside, plus an analysis of how per-variable errors combine. Errors can cancel out or compound.
- The failure pattern of "upstream omits, downstream fills the gap" maps directly to this design. If a sub-agent returns nothing or "not mentioned", the aggregator or orchestrator may invent a value. The aggregator should be deterministic code, not an LLM, so it cannot hallucinate inputs.
- Using the same model family for every agent (here, the same 2B base) may reduce handoff incompatibility. This rests on weak evidence (not significant).

### Gaps
- The paper does not test small models (~2B) or parallel (as opposed to sequential) decomposition.

## 2. Clinical multi-agent systems: gains and failures

### Takeaway
Clinical multi-agent systems report modest-to-large gains over single agents on frontier or ~8B+ backbones. The documented failures are groupthink in debate, redundant specialist outputs, latency and token cost, and poor LLM self-assessment of case complexity. None of the works found evaluates per-criterion score extraction with a ~2B orchestrator.

### Cited Findings
**MDAgents** (Kim et al., MIT Media Lab, NeurIPS 2024)
- It automatically assigns a solo or group collaboration structure based on how complex the LLM judges the question to be. It was best on 7 of 10 medical benchmarks, with up to a 4.2% gain over the previous best methods. In ablation, moderator review plus external medical knowledge gave an average +11.8% accuracy. — [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/90d1fc07f46e31387978b88e7e057a31-Abstract.html); [arXiv 2404.15155](https://arxiv.org/pdf/2404.15155)
- Agreement between LLM and physician complexity ratings was weak (correlations −0.090 to 0.110 in one arXiv version) **[unverified, from search snippet]**. — [arXiv 2404.15155](https://arxiv.org/pdf/2404.15155)
- In Aegle's evaluation, MDAgents scored 25.73% diagnostic accuracy on ClinicalBench under an evidence-elicitation setting, barely above the base model's 25.60%. — [Aegle arXiv 2604.08927](https://arxiv.org/html/2604.08927)

**MoodAngels** (Xiao et al., arXiv 2506.03750, v1 4 Jun 2025, v2 11 Oct 2025, journal reference NeurIPS 2025)
- A retrieval-augmented multi-agent framework for diagnosing mood disorders, combining "granular-scale analysis of clinical assessments" with a structured verification step. Its baseline agent is reported +12.3% accuracy over GPT-4o on real cases; the gain of the full MAS is not quantified in the abstract. It also releases MoodSyn, 1,173 synthetic psychiatric cases. — [arXiv abs](https://arxiv.org/abs/2506.03750)
- From the abstract alone, I could not confirm whether its agents map to individual DSM criteria.

**Aegle** ("Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents", Chen et al., ACL 2026 Findings; arXiv v3 23 Apr 2026)
- Architecture:
  - A graph-based orchestrator that only routes and does no medical reasoning.
  - Parallel specialists that propose state updates in isolation.
  - A "write-then-speak" aggregator that merges proposals into a structured SOAP state.
  - Evidence and reasoning are decoupled: diagnosis may only run on frozen, finalized features. — [arXiv HTML](https://arxiv.org/html/2604.08927); [ACL Anthology](https://aclanthology.org/2026.findings-acl.789/)
- Results, all on a DeepSeek-V3.2 backbone:
  - ClinicalBench (1,500 cases, 24 departments) diagnostic accuracy: 46.93% vs CoT 39.60, MedAgents 39.20, MDAgents 25.73, base 25.60.
  - IDEA documentation score: 72.78 vs MDAgents 68.41.
  - On average Aegle activated 2.416 specialists per case, vs 3.702 for MDAgents and 4.968 for MedAgents.
  - Ablation: removing the structured state caused the largest drop. — [arXiv HTML](https://arxiv.org/html/2604.08927)
- Stated limitations: added latency, a growing context and token cost, and redundant or overlapping specialist outputs that "dilute the aggregator's signal". Evaluation used gpt-4o-mini as the judge, with a small physician correlation study. — [arXiv HTML](https://arxiv.org/html/2604.08927)

**CARE-AD** (npj Digital Medicine, 2025)
- Multi-agent prediction of Alzheimer's disease from longitudinal EHR notes: a data-extraction agent pulls AD-related signs and symptoms, specialist agents assess them, and an AD-specialist agent synthesizes a risk prediction. Evaluated on 1,000 AD cases and 3,627 controls, it was more accurate than single-model approaches. The authors say prospective validation is still needed. — [npj Digit Med](https://www.nature.com/articles/s41746-025-01940-4); [PMC12375024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12375024/)
- A search summary says the public code uses fine-tuned LLaMA-3.1-8B and five virtual specialists **[unverified]**. — [GitHub alicelee1/ad_care](https://github.com/alicelee1/ad_care)

**SOAP-note problem detection** (Lee, Wang, Yang; arXiv 2508.21803, 29 Aug 2025; ACM-BCB 2025 poster)
- A Manager agent assembles a dynamic team of specialists, who debate hierarchically until they reach consensus. It uses only the S and O sections of 420 MIMIC-III notes. It beat a single-agent baseline on CHF, AKI and sepsis; the abstract gives no metric values. Qualitative failure: it is "occasionally susceptible to groupthink". — [arXiv abs](https://arxiv.org/abs/2508.21803)

### Inferences
- Aegle's design choices transfer directly to this project:
  - The orchestrator only routes.
  - Specialists work in isolation.
  - The aggregator merges structured state, not free text.
  - Evidence is frozen before computation.
  - In MedCalc terms: sub-agents emit typed JSON per variable, and code computes the score.
- Groupthink and debate failures do not arise if sub-agents never see each other's outputs. Parallel, isolated extraction avoids this failure class but gives up cross-checking.

### Gaps
- I found no clinical MAS paper that uses one agent per score criterion for MedCalc-style calculation, and none with a ≤3B orchestrator.
- I could not verify the exact CARE-AD metrics (AUROC or accuracy values) or MoodAngels' per-criterion design from the full text.

## 3. General multi-agent failure taxonomies, error propagation, information loss, overhead

### Takeaway
MAST shows that multi-agent failures come mostly from system design and specification (about 44%) and from weak or wrong verification (about 23%), not only from model capability. A scaling study found that independent parallel agents amplify errors up to 17.2×, while a centralized orchestrator limits this to about 4.4×. Multi-agent setups also hurt sequential-reasoning tasks.

### Cited Findings
- MAST ("Why Do Multi-Agent LLM Systems Fail?"): Cemri, Pan, Yang, ... Zaharia, Gonzalez, Stoica. arXiv 2503.13657, v1 17 Mar 2025, v3 26 Oct 2025.
  - Built from 150 traces with expert annotators (κ = 0.88).
  - Resulting dataset: more than 1,600 annotated traces across 7 MAS frameworks.
  - 14 failure modes in 3 categories.
  - Includes a validated LLM-as-judge annotator. — [arXiv abs](https://arxiv.org/abs/2503.13657)
- MAST failure-mode frequencies, as given in the paper's HTML v3:
  - FC1, system design: disobey task spec 11.8%; disobey role spec 1.5%; step repetition 15.7%; loss of conversation history 2.8%; unaware of termination conditions 12.4%.
  - FC2, inter-agent misalignment: conversation reset 2.2%; fail to ask for clarification 6.8%; task derailment 7.4%; information withholding 0.85%; ignored other agent's input 1.9%; reasoning–action mismatch 13.2%.
  - FC3, task verification: premature termination 6.2%; no or incomplete verification 8.2%; incorrect verification 9.1%. — [arXiv HTML v3](https://arxiv.org/html/2503.13657v3)
  - Category sums (computed by me, not stated in the paper): FC1 ≈ 44.2%, FC2 ≈ 32.4%, FC3 ≈ 23.5%.
- MAST interventions: on ChatDev, better role specification or workflow gave +9.4% task success, and a high-level objective-verification step gave up to +15.6% on ProgramDev. The authors conclude that simple fixes are insufficient. Correlations between categories are low (0.17–0.32). — [arXiv HTML v3](https://arxiv.org/html/2503.13657v3)
- "Towards a Science of Scaling Agent Systems" (Yubin Kim, Xin Liu et al., Google Research and MIT; arXiv 2512.08296; Google blog 28 Jan 2026):
  - Scope: 180 configurations, 5 architectures (single, independent, centralized, decentralized, hybrid), 3 model families (GPT, Gemini, Claude), 4 benchmarks.
  - Error amplification relative to a single agent: independent up to 17.2×, centralized 4.4×.
  - Centralized coordination gave +80.9% on parallelizable Finance-Agent.
  - Every MAS variant lost 39–70% on sequential PlanCraft.
  - A predictive model (R² = 0.513) picks the best architecture for 87% of unseen configurations. — [Google Research blog](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/); [arXiv 2512.08296](https://arxiv.org/html/2512.08296v2)
- The blog states no numeric token or coordination overhead; it describes overhead only qualitatively as a "tool-coordination trade-off". — [Google Research blog](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
- Aegle names longer contexts, token cost and latency as the main costs of decomposition, and avoids over-activation through dynamic routing (2.4 vs 3.7–5.0 agents per case). — [Aegle arXiv](https://arxiv.org/html/2604.08927)
- In the Optimization Paradox paper, information loss at handoffs between model families is hypothesized but untested. — [arXiv HTML v2](https://arxiv.org/html/2506.06574v2)

### Inferences
- Per-variable extraction fits the "parallelizable" regime where centralized multi-agent systems help. The final arithmetic is sequential and should be done in code.
- Map the MAST categories onto this pipeline:
  - "Disobey task spec" means a malformed or invalid value, e.g. outside the allowed categories.
  - "Premature termination" means a variable is never dispatched.
  - "No or incorrect verification" means no schema or range checks.
  - "Information withholding" means a sub-agent sees only a note fragment and misses context.
- Instrument these failure modes with schema validation, a check that every required variable for the calculator was dispatched, and range checks.
- For this project, token overhead scales roughly with the number of variables times note length, because each sub-agent re-reads the note. Measure tokens and latency per case against a single-pass extraction baseline. This is an inference; I found no published figure for this setting.

### Gaps
- I found no quantified study of information loss in summaries passed between agents in clinical settings, and no MAST-style analysis restricted to ~2B models. MAST's open-model appendix was not read.

## 4. Pitfalls of LLM clinical information extraction

### Takeaway
On MedCalc-type tasks, extraction and clinical-interpretation errors are a large share of failures, about 27–30% of GPT-4 errors. Recurring sub-types are unit confusion, missing variables, and misreading qualifiers ("trace" read as "no"). Negation is the most frequent hallucination type in clinical note generation. MedCalc-Bench's own ground truth assumes that an unmentioned criterion is absent and scores 0 points. The extraction design must reproduce this convention explicitly, or the system will be scored as wrong.

### Cited Findings
- MedCalc-Bench (Khandekar et al., arXiv 2406.12036, NeurIPS 2024 Datasets & Benchmarks):
  - 55 calculators (19 rule-based, 36 equation-based) and 1,047 test instances.
  - Tolerance: rule-based scores must match exactly; lab, physical and dosage equations allow ±5%.
  - Best result: GPT-4 one-shot CoT at 50.91%.
  - GPT-4 one-shot accuracy on rule-based subtypes: Risk 33.75%, Severity 27.50%, Diagnosis 53.33%. — [arXiv HTML v4](https://arxiv.org/html/2406.12036v4)
- MedCalc-Bench error breakdown for GPT-4: zero-shot CoT was 57% knowledge, 30% extraction, 13% arithmetic; one-shot CoT was 40% knowledge, 27% extraction, 33% arithmetic. The authors note models "struggle significantly as the number of attributes increase". — [arXiv HTML v4](https://arxiv.org/html/2406.12036v4)
- Convention for missing values in MedCalc-Bench:
  - Extraction marks absent values as "Not Mentioned".
  - Ground-truth explanation templates state that a missing parameter "is missing from the patient note and so we assume it is absent from the patient" and add no points (Appendix A.4).
  - The authors did not manually verify the GPT-4 parameter extractions for the training set (Appendix C.1). — [arXiv HTML v4](https://arxiv.org/html/2406.12036v4)
- "From Scores to Steps" (Wang, Xia, ... Hong Yu, Zonghai Yao; arXiv 2509.16584; repo named EMNLP-2025-MedRaC; ACL Anthology lists 2025.emnlp-main.548):
  - Step-wise grading drops GPT-4o from 62.7% to 43.6%.
  - Expert review removed 108 of 1,048 MedCalc-Bench cases (10.3%) as corrupted. Examples:
    - QTc Bazett: seconds vs milliseconds mismatch.
    - APACHE II: point-threshold error.
    - Caprini: every female given a point.
    - Estimated Due Date: mistyped equation.
    - 45 rows with reversed limits.
  - Proposed MedRaC (RAG plus code) raises accuracy from 16.35% to 53.19% without fine-tuning. — [arXiv HTML](https://arxiv.org/html/2509.16584); [ACL Anthology](https://aclanthology.org/2025.emnlp-main.548)
- "From Scores to Steps" error counts for Llama-3.1-8B, CoT vs MedRaC:

  | Error type | CoT | MedRaC |
  |---|---|---|
  | Formula misselection | 757 | 170 |
  | Incorrect variable extraction | 301 | 237 |
  | Clinical misinterpretation | 305 | 223 |
  | Unit conversion errors | 204 | 83 |
  | Missing variables | 308 | 131 |
  | Arithmetic | 426 | 74 |

  - Self-Refine raised missing-variable errors to 716.
  - Examples: creatinine in µmol/L treated as mg/dL; "trace ascites" read as "no ascites". — [arXiv HTML](https://arxiv.org/html/2509.16584)
- Asgari et al., "A framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation", npj Digital Medicine 8:274, 13 May 2025 (preprint Sep 2024):
  - 18 configurations and 12,999 clinician-annotated sentences: hallucination rate 1.47%, omission rate 3.45%.
  - Negations were the largest share of hallucinations, about 30% (56 cases), and the authors call them the most concerning type.
  - Prompt and workflow iteration reduced major errors. — [PMC12075489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12075489/); [medRxiv](https://www.medrxiv.org/content/10.1101/2024.09.12.24313556v2.full.pdf)
  - The 56 / 30% negation figure comes from a search snippet of the medRxiv version **[verify against final version]**.
- CLINES (medRxiv, Dec 2025), an LLM extraction agent:
  - It explicitly handles negation, experiencer, dosage, lab values and temporal alignment.
  - It concludes that the residual hallucination rate "mandates human verification" for safety-critical use. — [medRxiv](https://www.medrxiv.org/content/10.64898/2025.12.01.25341355v2.full) **[seen in search snippet only]**
- A Findings of EMNLP 2025 paper reports that in multi-day longitudinal summaries, errors accumulate into temporal drift and hallucination, and that RAG only partly helps. — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1128.pdf) **[snippet only]**

### Inferences
- **Absent criteria.** The project must decide whether "not mentioned" maps to 0 points, as MedCalc-Bench ground truth assumes. Keep "explicitly negated" (0 points, high confidence) separate from "not mentioned" (0 points by convention, flagged). Report how often the final score depends on defaults. Clinically, defaulting to 0 systematically underestimates risk scores, and the risk/ethics section should say so.
- **Per-variable sub-agents and qualifiers.** Each agent sees a narrow question ("Is CHF present?"). This raises the risk of:
  - matching history-of conditions vs current ones (temporality);
  - missing negations stated elsewhere in the note;
  - missing family-history or experiencer cues.
  
  Per-variable prompts should require a supporting evidence span, plus fields for negation, temporality and experiencer.
- **Units.** Unit conversion (mg/dL vs µmol/L) and thresholds should be handled deterministically in code, not by the 2B model.
- **Evaluation set.** Use the corrected 940-case subset, or flag the known-corrupted calculators. Otherwise label noise will contaminate the error analysis.

### Gaps
- I found no study that quantifies negation and temporality error rates specifically for small (≤3B) LLMs extracting score criteria. Rates by error type for MedCalc rule-based calculators alone were not available.

## 5. Evaluation and ethics guidance for medical LLM research

### Takeaway
TRIPOD-LLM (Nature Medicine, Jan 2025) is the reporting standard to cite. It has 19 main items and 50 subitems, is modular by task type, and emphasizes transparency, human oversight and task-specific performance reporting. The MedCalc literature also supports stating explicitly that a benchmark calculator is not a validated clinical tool.

### Cited Findings
- TRIPOD-LLM ("The TRIPOD-LLM reporting guideline for studies using large language models"), Nature Medicine 31(1):60–69, January 2025, doi 10.1038/s41591-024-03425-5:
  - 19 main items and 50 subitems; 14 main items and 32 subitems apply to every design.
  - Developed through an expedited Delphi process.
  - A "living" guideline with a web checklist tool (tripod-llm.vercel.app). — [PubMed 39779929](https://pubmed.ncbi.nlm.nih.gov/39779929/); [PMC12104976](https://pmc.ncbi.nlm.nih.gov/articles/12104976)
- MedCalc-Bench authors: "more work needs to be done in improving LLMs for them to be reliable clinical calculators" (Appendix C.3). The paper contains no explicit "not for clinical use" statement. — [arXiv HTML v4](https://arxiv.org/html/2406.12036v4)
- Optimization Paradox: regulators and procurers should require system-level, end-to-end evidence rather than component accuracy. — [arXiv HTML v2](https://arxiv.org/html/2506.06574v2)
- Asgari et al. combine an error taxonomy with a clinical-harm grading (major vs minor errors), offered as a template for safety-oriented evaluation. — [PMC12075489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12075489/)
- "From Scores to Steps" argues that grading only the final answer (±5%) hides faulty reasoning, and proposes step-wise attribution of the first error. — [arXiv HTML](https://arxiv.org/html/2509.16584)

### Inferences
Suggested evaluation design:
1. **Primary metric:** end-to-end exact-match score accuracy on the cleaned test set.
2. **Secondary metrics:**
   - per-variable extraction accuracy;
   - first-error attribution (routing or decomposition, extraction, aggregation);
   - rate of default-to-absent;
   - share of invalid or malformed sub-agent outputs, mapped to MAST modes;
   - tokens and latency vs a single-agent 2B baseline and a teacher model.
3. **Statistics:** paired comparisons with McNemar's test, as in the Optimization Paradox paper.
4. **Required controls:** a single-agent baseline of the same 2B model, and the same pipeline with a deterministic aggregator vs an LLM aggregator.
5. **Ethics statement:** research prototype only, not for clinical use; benchmark notes partly synthetic; known label errors; under-scoring bias when absent criteria default to 0; reported with a TRIPOD-LLM checklist.

### Gaps
- I did not retrieve the item-level wording of TRIPOD-LLM. Get it from the PMC full text if exact checklist items are needed.
- I found no guidance specific to multi-agent clinical LLM systems; existing guidelines treat the LLM system as a single model.
