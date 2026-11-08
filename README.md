# LLM Rerankers Under Stress: Dependability of LLM-Assisted Recommendation

**What happens when you put an LLM into a production-style recommender pipeline, and what do you do when it breaks?**

This is my final year project (BSc Computer Science, University of Birmingham, **First Class**). It stress-tests three open-weight LLMs as zero-shot rerankers on top of a SASRec recommender, measures how and when they fail, and builds a fallback mechanism that keeps the quality gains without the failures.

This repository is the foundation of **[RouteRec: Strict Evaluation of Recommender-Agent Selection and Aggregation](https://arxiv.org/abs/2607.09908)** (Zhou, **Kalmykov**, Feng). The pipeline and conclusions from these preliminary dependability experiments were used to build the router framework in the paper. I co-authored it as 2nd author, and it was accepted at the **SIGIR 2026 AgentSearch workshop**, ranked joint-1st by average reviewer score.

📄 [Final report (PDF)](research-pdfs/Final-Report.pdf)

## Key findings

| | Finding |
|---|---|
| 🛡️ **Tiered fallback works** | A two-tier failure taxonomy (recoverable vs fatal) lets the pipeline repair fixable LLM output such as duplicates, instead of throwing it away. This recovers quality that a naive "fall back on any error" policy discards. |
| 📈 **Ranking quality up** | Qwen2-7B reranking lifted **NDCG@10 by 8.5%** (0.0655 → 0.0711) and **MRR by 21.5%** (0.0362 → 0.0440) over the SASRec baseline, with HR@10 unchanged. |
| 🎯 **Prompt wording dominates** | Rewording the prompt is **~100× more destabilising** than reordering the candidate list, which contradicts the literature's focus on position bias. |
| 🧩 **Instruction tuning beats size** | Valid-output rate: Qwen2-7B-Instruct **100%**, Nemotron-Mini-4B **45%**, gpt-oss-20b **0%**. Format compliance comes from instruction-tuning quality, not parameter count. |
| ⚖️ **Diversity prompts backfire** | Aggressive diversity prompting cut popularity bias but *reduced* genre diversity: the LLM confuses "diverse" with "obscure". Apparent fairness gains came from degrading the advantaged group. |
| ❌ **A documented negative result** | Self-consistency across repeated LLM calls showed no significant correlation with quality, so it does not work as a reliability signal for reranking. |

## Pipeline

```
MovieLens-1M ──► SASRec (trained once, frozen) ──► top-20 candidates
                                                        │
                                       LLM listwise reranker (zero-shot)
                                       Qwen2-7B · Nemotron-4B · gpt-oss-20b
                                                        │
                                  output validation + failure classification
                                     ├─ valid ───────────► top-10
                                     ├─ recoverable ─ repair ─► top-10
                                     └─ fatal ─ fall back to SASRec ─► top-10
```

Four configurations compared: **SASRec baseline**, **naive LLM**, **hard fallback**, and **tiered fallback**, across temperatures, prompt variants, candidate orderings and diversity prompts.

**Evaluation:** HR@10, NDCG@10, MRR (accuracy) · ARP, PopRatio, Gini, coverage (popularity bias) · ILD, genre coverage (diversity) · per-gender HR/NDCG gaps (fairness).

## Tech stack

Python · PyTorch · Hugging Face Transformers · SASRec · Slurm · Apptainer containers · GPU jobs on two HPC clusters (Birmingham **BlueBear** and **Isambard**)

## Repository guide

| Path | Contents |
|---|---|
| [`python/`](python/) | Main code: candidate generation, LLM reranking, fallback, evaluation, perturbation and consistency analysis |
| [`key-logs/`](key-logs/) | **Start here for results.** Curated job logs for each milestone, e.g. [final evaluation](key-logs/final-evaluation_40628363.out) |
| [`sh-jobs/`](sh-jobs/) | Slurm job scripts for each pipeline stage |
| [`research-pdfs/`](research-pdfs/) | Final report, original proposal, project specification, and two literature reviews (recommender bias, recommender dependability) |
| [`primary-references/`](primary-references/) | Key papers from the ~60 reviewed |
| [`logs/`](logs/) | Full raw job logs |
| [`old-python/`](old-python/) | Earlier snapshot of the code, kept for history |
| [`ml-1m/`](ml-1m/) | MovieLens-1M dataset ([GroupLens](https://grouplens.org/datasets/movielens/1m/)) |
| [`baseline_apptainer.def`](baseline_apptainer.def) | Container definition for the HPC environment |

The SASRec baseline builds on a public SASRec implementation; the reranking, fallback and evaluation code is my own.
