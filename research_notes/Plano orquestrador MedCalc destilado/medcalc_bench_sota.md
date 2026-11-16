# MedCalc-Bench: State of the Art, Baselines and Dataset Details (as of Nov 2026)

Notes compiled 2026-11-13. Every number has an inline source. Where secondary sources (aggregators, third-party tables) are used, they are flagged. Results across papers use DIFFERENT dataset variants (v1.0 "Original" 1,047 test rows; v1.2 / Verified 1,100 test rows; "Cleaned" 940 rows; "recomputed" labels) and are NOT directly comparable.

## Q1. Dataset details: versions, splits, columns, note types, categories, evaluation protocol

### Takeaway
MedCalc-Bench has 55 calculators (19 rule-based, 36 equation-based). The current recommended release is MedCalc-Bench Verified (HF `nsk7153/MedCalc-Bench-Verified`, derived from v1.2): 10,538 train / 1,100 test (20 per calculator) / 55 one-shot rows. Accuracy is final-answer only: +/-5% tolerance for decimal equation outputs, exact match otherwise. "Relevant Entities" is a per-row dict of parameter -> value (with [value, unit] for numerics). It was GPT-4-extracted and clinician-verified only for the test set (not for train), and it was revised in later versions. Neither the Verified release nor v1.2 publishes a changelog with label-change counts.

### Cited Findings
**Versions**
- v1.0 (HF `ncbi/MedCalc-Bench-v1.0`) is "kept for reproducibility purposes only". The card says "Because of the number of changes, we find this dataset to be highly unreliable" and lists three fixes made in later versions: (1) 12 calculator implementations were fixed; (2) the Relevant Entities were revised to better match the patient note; (3) notes that fit their calculator poorly were replaced. No v1.1 details or dates are given. — [HF ncbi/MedCalc-Bench-v1.0](https://huggingface.co/datasets/ncbi/MedCalc-Bench-v1.0)
- `ncbi/MedCalc-Bench` duplicates v1.2: 10,543 train and 1,100 test rows (11,643 total), CC-BY-SA-4.0. It points to MedCalc-Bench Verified as the latest version. — [HF ncbi/MedCalc-Bench](https://huggingface.co/datasets/ncbi/MedCalc-Bench)
- MedCalc-Bench Verified: GitHub `nikhilk7153/MedCalc-Bench-Verified`, HF `nsk7153/MedCalc-Bench-Verified`. It is "modified from MedCalc-Bench-v1.2" and "audited on multiple occasions". Splits are train 10,538, test 1,100 and one_shot 55. The card has no changelog, no count of changed labels and no release date. It asks users to cite the version they used (example format "v.1.0.6"). The official ncbi repo README says "The first version of MedCalc-Bench Verified is an update from v1.2". — [HF Verified card](https://huggingface.co/datasets/nsk7153/MedCalc-Bench-Verified); [GitHub Verified](https://github.com/nikhilk7153/MedCalc-Bench-Verified); [GitHub ncbi-nlp/MedCalc-Bench](https://github.com/ncbi-nlp/MedCalc-Bench)
- Original paper: arXiv 2406.12036 (v4 HTML is dated 30 Jun 2024; NeurIPS 2024 Datasets & Benchmarks, Oral). It reports 1,047 test instances: equation-based 667 (Lab 327, Physical 240, Date 60, Dosage 40) and rule-based 380 (Risk 240, Severity 80, Diagnosis 60). — [arXiv 2406.12036](https://arxiv.org/html/2406.12036)
- Train set in the paper (Appendix C.1) is 10,053 instances covering only 40 of the 55 calculators. The 15 calculators with no training notes are Child-Pugh, RCRI, GBS, HAS-BLED, HOMA-IR, CCI, FENa, Framingham Risk, Centor, MELD Na, PSI, GCS, APACHE II, SOFA and Caprini. Train GPT-4 parameter extractions were NOT manually verified, unlike the test set. Train counts are very imbalanced, e.g. MAP 948, Maintenance Fluids 889, LDL 66. — [arXiv 2406.12036 App. C](https://arxiv.org/html/2406.12036)
- Size discrepancies across sources: train 10,053 (paper) vs 10,543 (v1.2) vs 10,538 (Verified) vs 9,765 / 38 subtasks (as used by MedCalc-R1); test 1,047 (v1.0) vs 1,100 (v1.2 / Verified) vs 1,048 (MedRaC, MedCalc-R1). — [paper](https://arxiv.org/html/2406.12036); [HF v1.2](https://huggingface.co/datasets/ncbi/MedCalc-Bench); [HF Verified](https://huggingface.co/datasets/nsk7153/MedCalc-Bench-Verified); [MedCalc-R1](https://arxiv.org/html/2608.08623v1); [MedRaC](https://arxiv.org/html/2509.16584)

**Columns**
- The 14 columns are Row Number, Calculator ID, Calculator Name, Category, Output Type, Note ID, Note Type, Patient Note, Question, Relevant Entities, Ground Truth Answer, Lower Limit, Upper Limit and Ground Truth Explanation. — [GitHub ncbi-nlp](https://github.com/ncbi-nlp/MedCalc-Bench); [HF Verified](https://huggingface.co/datasets/nsk7153/MedCalc-Bench-Verified)
- Output types are decimal, integer, date (MM/DD/YY), and time in (weeks, days). — [HF ncbi/MedCalc-Bench](https://huggingface.co/datasets/ncbi/MedCalc-Bench)
- Relevant Entities is a dict mapping parameter name to extracted value, e.g. sex, weight, height, creatinine, age. In the paper's parsing (App. A.4), IDs were mapped to Python variable names, numerics were split into [value, unit] pairs, "True"/"False" strings became booleans, and categoricals were matched to MDCalc's option lists. Missing non-numeric values became "Not Mentioned" and are assumed false by the template. — [arXiv 2406.12036 App. A/B](https://arxiv.org/html/2406.12036)
- Ground Truth Explanation is produced by a per-calculator Python template function that takes the parameters and emits step-by-step text plus the answer. The fetched paper's own Glasgow-Blatchford example has an inconsistency: the text says 3 points while the code adds 6. — [arXiv 2406.12036](https://arxiv.org/html/2406.12036)
- Notes were edited to add irrelevant numeric and descriptive distractor values, specifically to test correct extraction. — [arXiv 2406.12036](https://arxiv.org/html/2406.12036)

**Note types**
- There are three note types. Extracted notes come from Open-Patients (PMC-Patients, MedQA-USMLE, TREC CDS/CT). Template notes are generated by Python functions. Clinician-synthesized notes (later labeled "Synthetic" / "Handwritten") were LLM-generated and then clinician-edited. Per the paper: 34 calculators use Open-Patients notes, 10 equation-based calculators use 20 template notes each, and 11 rule-based calculators use 20 clinician notes each. The appendix lists 11 and 10 (inconsistent). The paper caps each calculator at 20 notes. — [arXiv 2406.12036](https://arxiv.org/html/2406.12036); [HF Verified](https://huggingface.co/datasets/nsk7153/MedCalc-Bench-Verified)

**Categories and evaluation**
- Equation-based categories are lab, physical, date and dosage. Rule-based categories are risk, severity and diagnosis. — [GitHub](https://github.com/ncbi-nlp/MedCalc-Bench)
- Tolerance: for decimal equation outputs, Lower/Upper Limit = 95%/105% of ground truth; otherwise both equal the ground truth (exact match for rule-based scores and dates). Example: GT 141.0403 gives the range [133.98828, 148.09232]. — [HF ncbi/MedCalc-Bench](https://huggingface.co/datasets/ncbi/MedCalc-Bench); [paper](https://arxiv.org/html/2406.12036)
- The official harness is `python run.py --model <m> --prompt {direct_answer|zero_shot|one_shot_cot}`. It writes a per-row "Result" of Correct/Incorrect and reports overall accuracy with std. A code-interpreter script exists for GPT-3.5/4 only. — [GitHub](https://github.com/ncbi-nlp/MedCalc-Bench)
- MedRaC found 45 negative-value rows with reversed lower and upper limits, which breaks naive range checks. — [arXiv 2509.16584](https://arxiv.org/html/2509.16584)

### Inferences
- For a DAG / sub-agent design, Relevant Entities gives ready-made per-variable supervision targets (variable name -> value + unit), and the Python template functions give the reference calculate(). On train, though, entities are unverified GPT-4 extractions, so expect noise. 15 calculators have zero train coverage, so a train/test calculator split happens naturally.
- Report results on Verified (1,100 rows), and also on a physician-relabeled subset if possible (see Q4). State the exact version, because headline numbers differ by many points across variants.

### Gaps
- No public changelog for v1.1, v1.2 or Verified, and no published count of how many labels changed between versions. The only quantified fix is "12 calculator implementations" (v1.0 to later). The Verified version tag and release date could not be found.
- The exact per-note-type instance counts in the 1,100-row test set were not found (likely derivable by loading the dataset).

## Q2. Reported results: original paper and 2025-2026 follow-ups (CoT, code, tools, RL, small models, fine-tuning)

### Takeaway
Original GPT-4 one-shot CoT reached 50.9% (v1.0). Approaches that beat plain CoT all inject formula knowledge (open-book spec, formula RAG, SFT on explanations) and/or offload arithmetic to code. Small models gain most from SFT warm-up on the train explanations followed by GRPO: a 3B model goes from about 10-12% to about 51-52%. Reported highs: about 74% (DeepSeek-R1, Orig), 71.4% (Qwen3-8B GRPO on recomputed labels), and 82-85.5% (open-book prompting of GLM-4.6V/4.7 on Verified).

### Cited Findings
**Original paper (arXiv 2406.12036, v1.0 test 1,047; mean accuracy %)**
- GPT-4: direct 20.82, zero-shot CoT 37.92, one-shot CoT 50.91. Llama3-70B: 20.82 / 35.53 / 39.45. GPT-3.5: 18.82 / 23.69 / 34.86. Mixtral-8x7B: 14.23 / 22.35 / 29.23. Llama3-8B: 13.09 / 16.43 / 27.13. Mistral-7B: 9.84 / 10.79 / 16.05. Meditron-70B: 6.21 / 0.38 / 26.27. PMC-LLaMA-13B: 0 / 0 / 6.97. — [arXiv 2406.12036 Table 2](https://arxiv.org/html/2406.12036)
- GPT-4 one-shot per category: Lab 51.68, Physical 77.50, Date 46.67, Dosage 37.50, Risk 33.75, Severity 27.50, Diagnosis 53.33. Rule-based (risk/severity) is the hardest. — [same](https://arxiv.org/html/2406.12036)
- Code interpreter (App. D, zero-shot CoT to code prompt, up to 20 retries): GPT-4 37.92 -> 48.51 overall (Lab 26.30 -> 49.85, Physical 71.25 -> 81.66, but Dosage 40.0 -> 32.5, Date 48.33 -> 43.33, Diagnosis 28.33 -> 23.33); GPT-3.5 23.69 -> 30.29. Code helps equation-type lab/physical tasks, not rule-based ones. — [arXiv 2406.12036 App. D](https://arxiv.org/html/2406.12036)
- Fine-tuning on the train split: Mistral-7B 10.79 -> 49.19; Llama-2-7B 1.53 -> 45.75. — [arXiv 2406.12036 App. C, Table 14](https://arxiv.org/html/2406.12036)

**RL / GRPO on MedCalc**
- EHRMind (Lin, Wu, Sun; arXiv 2505.24105, 30 May 2025) used LLaMA-3.2-3B-Instruct. SFT warm-up used about 2,010 train examples with the official explanations as traces, followed by GRPO (12 rollouts, KL 0.001, LR 1e-6, 2 epochs, VeRL). Overall: base 3B 9.74; RLVR-only 41.26; SFT-only 37.82; SFT+RLVR 51.96. Baselines: GPT-4 37.92, Claude-3.5-Sonnet 41.18, o3-mini 46.42, DeepSeek-R1 48.13. SFT+RLVR per category: Lab 55.66, Phys 81.25, Date 53.33, Dosage 80.00, Risk 22.08, Severity 18.75, Diagnosis 63.33. The per-category column mapping comes from the fetched HTML and is flagged by the extractor as needing a PDF check. — [arXiv 2505.24105](https://arxiv.org/html/2505.24105v1)
- MedCalc-R1 (Wang et al., HIT; arXiv 2608.08623, 9 Aug 2026, "work in progress") used Qwen2.5-Instruct 1.5B/3B with SFT followed by GRPO (rollout 5, 5 epochs, 2,048 context). The reward is format + knowledge + answer. The knowledge reward is +1/-1 when a frozen Qwen2.5-14B verifier validates the model's stated formula. The answer reward is hard (+2 in tolerance, -3 outside) plus soft exp(-|v-v*|/tau). Averages: Qwen2.5-1.5B 4.82 -> SFT 33.68 -> R1 42.36; Qwen2.5-3B 12.49 -> SFT 40.65 -> R1 51.34; Qwen2.5-7B base 23.37; 32B base 39.03; GPT-4o 40.36; o1-mini 67.84; DeepSeek-R1 73.95. Ablation (3B): without knowledge reward 46.85; without hybrid reward 46.28; without both 39.60. — [arXiv 2608.08623](https://arxiv.org/html/2608.08623v1)
- Scalable Stewardship (Ye, Tawfik, Goodell, Kotha, Buyyounouski, Bayati; arXiv 2512.19691, v1 Dec 2025, v3 13 Apr 2026): a controlled RL experiment where training on recomputed labels beat training on the original labels by +13.5 pp (95% CI 10.6-16.6) on physician-labeled instances. — [arXiv 2512.19691](https://arxiv.org/abs/2512.19691). A third-party table (the audit paper) attributes "Qwen3-8B GRPO on recomputed labels = 71.4%" to this line of work. That figure comes from a secondary table and was not verified against the primary source. — [arXiv 2603.02222](https://arxiv.org/html/2603.02222v1)
- MedCalc-Eval / MedCalc-Env (Mao et al., arXiv 2510.27267, 31 Oct 2025) is a new benchmark of 700+ tasks plus an RL environment on InternBootcamp, with Qwen2.5-32B fine-tuned. No MedCalc-Bench numbers appear on the abstract page. — [arXiv 2510.27267](https://arxiv.org/abs/2510.27267)

**Tool / agent / knowledge-injection approaches**
- MedRaC (Wang et al., Yale/UMass; arXiv 2509.16584, EMNLP 2025 main) is training-free: formula RAG over MDCalc plus Python code execution. On the cleaned 940-row set (Rule / Calc accuracy), Qwen3-4B one-shot scores 49.85 / 59.57 and MedRaC 45.72 / 68.72. Qwen3-8B: one-shot 58.70 / 62.90, MedRaC 46.61 / 74.54. Qwen3-14B: 60.77 / 67.05 vs 50.44 / 78.37. Phi-4-mini: 12.09 / 16.47 vs 35.10 / 68.39. LLaMA3.2-3B failed to output executable code. GPT-4o: 62.24 / 54.24 vs 51.03 / 64.39. Pattern: MedRaC strongly improves equation calculators and HURTS rule-based ones relative to one-shot. — [arXiv 2509.16584](https://arxiv.org/html/2509.16584)
- Open-book prompting (Krohn-Grimberghe; arXiv 2603.02222, 10 Feb 2026, single author, not peer-reviewed) appends the calculator spec (formula, parameter definitions, scoring rules, units) to the prompt, on the Verified test set. GLM-4.6V: vanilla 51.9% -> open-book 81.5% -> +guidance 82.0% (n=1,100). GLM-4.7: 36.0% -> 85.5% (n=275 subsample). GPT-5.2-Thinking with tools solved 140/198 of GLM's residual failures. The estimated ceiling is 94.7-97.4%. — [arXiv 2603.02222](https://arxiv.org/html/2603.02222v1)
- RiskAgent (Liu et al.; arXiv 2503.03802) is multi-agent, with Decider, Executor and Reviewer LLM agents over hundreds of validated clinical tools. RiskAgent-GPT-4o reached 67.71% on the MedCalc-Bench Original test as an external tool-learning evaluation. — [arXiv 2503.03802](https://arxiv.org/html/2503.03802v2) (number confirmed via search snippet and the [audit table](https://arxiv.org/html/2603.02222v1); the primary PDF was not opened)
- Other numbers in the audit's comparison table, all secondary and unverified: Claude Opus 4.5 extended thinking + tools 61.3% ("company eval", a VENDOR claim); HELM/MedHELM direct prompting with DeepSeek-R1 / Gemini 2.5 Pro 34.8% on Verified. — [arXiv 2603.02222](https://arxiv.org/html/2603.02222v1)
- MedCalc-Pro (Zhao et al., ECUST; arXiv 2607.02879, 3 Jul 2026) is a single-agent pipeline: query rewriting, retrieval and reranking, tool selection, then execution with parameter validation (completeness/type/enum/unit checks plus an evidence audit, with backtracking to extraction). It covers single, multi and nested calculators (2,268 cases, 77 calculators; the multi set is built from MedCalc-Bench). Single-calculator, gpt-oss-120b: Extract 53.56, Score 80.15. Single-calculator score is +4.77 over MedRaC. No models under 32B were evaluated. — [arXiv 2607.02879](https://arxiv.org/html/2607.02879v1)
- AgentMD (Jin et al., Nat Commun 2025; arXiv 2402.13225) curates more than 2,000 calculators (RiskCalcs) and applies them. It was evaluated on RiskQA and 698 ED notes, not MedCalc-Bench. One search-aggregator snippet claimed "86.73% on MedCalc-Bench" for AgentMD. This could NOT be verified and should be treated as unreliable. — [NLM seminar listing](https://dirseminar.nlm.nih.gov/seminar/5661/detail/); [arXiv 2402.13225](https://arxiv.org/pdf/2402.13225)

### Inferences
- Plain CoT is weak for small models: Qwen2.5-1.5B 4.8%, Qwen2.5-3B 12.5%, LLaMA-3.2-3B 9.7%. A ~2B orchestrator without formula knowledge or tools will start near 5-15%.
- The highest-leverage components for the planned system are: (a) providing the calculator spec / variable list (open-book: +30 pp on a mid model); (b) a deterministic calculate() (removes arithmetic errors, about +10 pp on equation calculators); (c) SFT on train explanations before any GRPO (RL alone cannot learn missing formulas). The proposed DAG + Python design is essentially "open-book + code", which is the strongest known recipe. Its remaining bottleneck will be extraction, especially for rule-based scores with many criteria.
- A realistic target for a 2B system with a calculate() tool is above the 51-52% of 3B RL models on Orig, and plausibly 60-75% on Verified. This is an inference, not a reported result.

### Gaps
- No reported result for Qwen3-1.7B or Qwen3.5-2B specifically on MedCalc-Bench. The smallest Qwen3 with published numbers is Qwen3-4B (MedRaC).
- No official leaderboard. Papers With Code / wizwand only mirror individual paper numbers.
- MedCalc-R1's Table 1 column mapping was garbled in the HTML extraction (8 values per row for 10 headers). Per-category values should be checked in the PDF.
- The AgentMD MedCalc-Bench number and the Claude Opus 4.5 61.3% were not verified at a primary source.

## Q3. Error analysis: extraction vs formula/rule knowledge vs arithmetic

### Takeaway
In vanilla prompting the dominant error is missing or misapplied formula/rule knowledge (40-67% of errors, and about 30 pp of accuracy). Arithmetic is next (13-39% of errors, 5-13 pp). Once knowledge and arithmetic are supplied (spec/RAG + code), parameter extraction and clinical interpretation become the residual bottleneck and are the hardest to improve. This is exactly the sub-task the proposed sub-agents target.

### Cited Findings
- Original paper error typing (GPT-4 classifier, 89% accurate on 200 manual checks), zero-shot CoT -> one-shot CoT. GPT-4: knowledge (A) 57% -> 40%, extraction (B) 30% -> 27%, computation (C) 13% -> 33%. Llama3-70B: A 67% -> 34%, B 16% -> 20%, C 17% -> 39%. GPT-3.5: A 50% -> 36%, B 31% -> 30%, C 17% -> 31%. — [arXiv 2406.12036 Table 4](https://arxiv.org/html/2406.12036)
- MedRaC step-wise grading (formula / extraction / arithmetic / final). Without RAG, formula is the first error in 71.96% of failures; with RAG, 20.78%. Formula conditional correctness goes 46.49% -> 92.66%. Code cut calculation first-error attribution from 31.88% to 3.23%. LLaMA3.1-8B error counts, CoT -> MedRaC: formula 757 -> 170 (-77.5%), arithmetic 426 -> 74 (-82.6%), missing variables 308 -> 131, incorrect extraction 301 -> 237 (only -21.3%), clinical misinterpretation 305 -> 223 (-26.9%). — [arXiv 2509.16584](https://arxiv.org/html/2509.16584)
- Step-wise grading with stricter MDCalc decimal rules dropped GPT-4o from 62.7% to 43.6%: right final answers often come from wrong steps. LLM-judge vs expert agreement was 90.2% on formula, 78.3% on extraction, 88.1% on calculation and 97.8% on final answer. — [arXiv 2509.16584](https://arxiv.org/html/2509.16584)
- The open-book audit decomposes accuracy as formula recall about 30 pp, arithmetic precision about 5-13 pp, and parameter extraction about 3-5 pp (the "irreducible core"; derivation not shown). Residual errors include choosing the wrong reference encounter (which visit/time point) and formula-version conflicts. — [arXiv 2603.02222](https://arxiv.org/html/2603.02222v1)
- EHRMind found that RL fixes "misapplied knowledge" but not "missing knowledge". Base Pass@12 predicts RL gain (R^2 = 0.91). Dosage Pass@12 was 3.42%, and RL-only Dosage stayed at 7.5% while SFT+RL reached 80%. Unseen-calculator Risk/Severity items lag o3-mini. — [arXiv 2505.24105](https://arxiv.org/html/2505.24105v1)
- MedCalc-R1 found a "knowledge-reasoning gap": formula recall rose from 12.02 (base 3B) to 53.91 (SFT) and 57.92 (R1), while final accuracy was 12.49, 40.65 and 51.34. Severity stays low, around 9-11%, due to hierarchical constraints. Small models are unstable on Date. — [arXiv 2608.08623](https://arxiv.org/html/2608.08623v1)
- MedCalc-Pro: with tools, single-calculator errors are dominated by Param Value (468) and Enum Mapping (386). Score falls from 68.91% (Easy, few parameters) to 42.81% (Hard, many parameters). Parameter errors propagate through nested dependency chains. — [arXiv 2607.02879](https://arxiv.org/html/2607.02879v1)

### Inferences
- With calculate() and an explicit variable list (DAG), knowledge and arithmetic errors largely vanish. Performance will then hinge on per-variable extraction accuracy, mapping values to the calculator's enum categories, unit conversion, and picking the right encounter/time point. Many-criteria rule-based scores (APACHE II, SOFA, Caprini, CCI, HAS-BLED) are where a parallel per-variable sub-agent design should help most, and where joint error compounding hurts most: accuracy is roughly the product of per-variable accuracies.
- Relevant Entities allows measuring per-variable extraction accuracy directly, a metric the original benchmark lacks.

### Gaps
- No paper reports extraction-only accuracy against Relevant Entities for small models. MedCalc-Pro's "Extract" metric is on its own benchmark with models of 32B and up.

## Q4. Known label-quality problems and criticisms

### Takeaway
Label quality is a serious problem. Independent audits estimate that more than 25% of the original/v1.x test labels are wrong or incomputable. Several calculator implementations had bugs that propagate to whole calculators. Verified fixes "a lot of issues" but publishes no changelog, and audits through 2026 still find errors in it.

### Cited Findings
- Scalable Stewardship (Stanford; arXiv 2512.19691): "at least 27% of test labels are likely erroneous or incomputable". On a 50-instance physician-validated subset, recomputed labels matched physicians 74% of the time (95% CI 60-84) vs 20% (95% CI 11-33) for the original labels. Original labels underestimate frontier LLM accuracy by 16-23 pp. — [arXiv 2512.19691](https://arxiv.org/abs/2512.19691)
- Companion repo `junzeye/validate-medcalc-labels` releases 887 physician-adjudicated labels (4 Stanford physicians) out of the original 1,047, and reports a divergence of more than 25%. A search snippet said 279/1,047 (26.6%) were flagged, with error types of feature extraction, scoring logic, and unanswerable from the note (proposed label "N/A"). The snippet was not opened, so it is secondary. — [GitHub validate-medcalc-labels](https://github.com/junzeye/validate-medcalc-labels)
- MedRaC cleaning (2 clinicians, double-blind) removed 108/1,048 (10.3%), leaving 940. Problems: all Estimated Due Date rows (mistyped gestational-age equation); all APACHE II rows (wrong +4 threshold); all QTc Bazett rows (s vs ms); all Caprini rows (wrong point for all females); one FeverPAIN row; 45 negative-value rows with reversed limits. — [arXiv 2509.16584](https://arxiv.org/html/2509.16584)
- The open-book audit (Feb 2026) found 20 calculator-implementation errors ("over 20" in the text). Examples: CKD-EPI 2021 male coefficient; MELD-Na vs MELD 3.0; HOMA-IR units; Child-Pugh bilirubin molecular weight; MME fentanyl patch factor (0.13 vs 2.4); HEART atherosclerosis logic; SIRS missing bands; SOFA FiO2/vasopressors; GCS "not testable"; CCI key typo `liver_diease` that zeroes liver points; RCRI key typo; Framingham argument order; Glasgow-Blatchford BUN boundary; CURB-65 BUN threshold; FIB-4 platelet scaling; QTc Framingham. It also argues the benchmark "predominantly measures formula memorization and arithmetic precision rather than clinical reasoning". — [arXiv 2603.02222](https://arxiv.org/html/2603.02222v1)
- `hellomate2/labelgate` (an individual, unreviewed project) argues that Verified training labels pass a formula re-check but contradict their own notes, and proposes an 84-row fix list. — [GitHub labelgate](https://github.com/hellomate2/labelgate)
- Training on noisy original labels measurably harms RL: -13.5 pp vs recomputed labels. — [arXiv 2512.19691](https://arxiv.org/abs/2512.19691)

### Inferences
- For a distilled/RL training set built from the train split, use Verified labels at minimum. Consider recomputing labels by running the (fixed) calculate() on Relevant Entities, and filtering rows where the recomputed value disagrees with the stored label. Exclude or flag calculators known to be buggy, such as APACHE II, Caprini, QTc Bazett and due-date. Evaluating on both Verified and the 887-row physician-adjudicated set would make results robust to label noise.

### Gaps
- The exact relationship between the Stanford 887-row set and Verified is unclear: which errors Verified already fixes. No source quantifies the residual error rate in Verified.

## Q5. Multi-agent approaches on MedCalc-Bench or medical calculators

### Takeaway
Only RiskAgent is an explicit multi-agent system evaluated on MedCalc-Bench (67.71% Orig, GPT-4o). Other strong systems (MedRaC, MedCalc-Pro, AgentMD, MeNTi) are single-agent tool pipelines. No published work decomposes a calculator into a variable DAG with parallel per-variable extraction sub-agents, and none uses a ~2B orchestrator, so the proposed project appears novel.

### Cited Findings
- RiskAgent: Decider, Executor and Reviewer LLM agents with hundreds of validated evidence-based tools; RiskAgent-GPT-4o 67.71% on MedCalc-Bench Original. — [arXiv 2503.03802](https://arxiv.org/html/2503.03802v2)
- MedCalc-Pro: a single agent with dependency-aware (nested) calculator execution, where sub-calculators run first. This is the closest analogue to a DAG. Its parameter validation (completeness/type/enum/unit) plus evidence audit with backtracking improved nested extraction 68.78 -> 88.48. It beats MeNTi on nested tasks by +32.27 (sub) and +14.51 (parent). — [arXiv 2607.02879](https://arxiv.org/html/2607.02879v1)
- AgentMD: a tool-learning agent over more than 2,000 auto-curated calculators (RiskCalcs). It reports over 85% accuracy on expert quality checks and over 90% unit-test pass rates, and was evaluated on RiskQA and ED notes. — [NLM seminar](https://dirseminar.nlm.nih.gov/seminar/5661/detail/); [arXiv 2402.13225](https://arxiv.org/pdf/2402.13225)
- MedRaC: retrieval plus code agent, training-free. — [arXiv 2509.16584](https://arxiv.org/html/2509.16584)
- Other related benchmarks: CMedCalc-Bench (1,143), OpenMedCalc (287), CalcQA (100), MedCalc-Eval (700+ tasks). — [arXiv 2607.02879 Table 1](https://arxiv.org/html/2607.02879v1); [arXiv 2510.27267](https://arxiv.org/abs/2510.27267)

### Inferences
- Useful design ideas to borrow: MedCalc-Pro's per-parameter validation (type/enum/unit/evidence span) as the aggregation check before calculate(); its finding that extraction degrades with parameter count, which supports parallel per-variable sub-agents; and an evidence-span requirement to fight hallucinated values.
- Baselines to report against: plain CoT and one-shot CoT for the same small model; open-book (spec in prompt, single call); open-book + code (MedRaC-like); SFT on train explanations; and RL numbers from EHRMind (3B, 51.96) and MedCalc-R1 (3B, 51.34; 1.5B, 42.36).

### Gaps
- No multi-agent result with small models (≤8B) on MedCalc-Bench was found. MedCalc-Pro's MeNTi baseline numbers and the MedMCP-Calc details were not retrieved.
