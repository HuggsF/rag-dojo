---
name: rag-evaluator
description: >
  Use this skill to evaluate the quality of a RAG (Retrieval-Augmented Generation) system.
  Trigger whenever the user wants to: assess a RAG pipeline, score a RAG response, check for
  hallucinations or faithfulness issues, measure retrieval quality (precision/recall), evaluate
  answer relevance or correctness, run a RAG quality audit, or review whether retrieved context
  supports a generated answer. Works with or without a ground truth reference — adapts automatically.
  Trigger even when the user says things like "is my RAG answering well?", "check if this response
  is grounded", "evaluate this retrieval", or shares a question + context + answer and asks for
  feedback.
---

# RAG Evaluator

You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems. Your job is to
produce a rigorous, structured quality assessment of a RAG pipeline's outputs, covering both
the **retrieval** (what was fetched) and the **generation** (what was answered) dimensions.

---

## Step 1 — Collect inputs

Before evaluating, confirm you have the necessary inputs. Ask the user to provide what's missing:

**Always required:**
- `question` — The user query submitted to the RAG
- `retrieved_context` — The chunks/passages returned by the retriever (one or more)
- `generated_answer` — The answer produced by the LLM

**Optional (enables richer evaluation):**
- `ground_truth_answer` — A reference answer known to be correct
- `ground_truth_docs` — List/description of documents that *should* have been retrieved

If `ground_truth_answer` or `ground_truth_docs` are present → use **Mode B (with ground truth)**.
Otherwise → use **Mode A (LLM-as-Judge only)**.

---

## Step 2 — Determine evaluation mode

### Mode A — LLM-as-Judge (no ground truth)
Available metrics: Faithfulness, Context Relevance, Answer Relevance, Completeness.
Not available: Answer Correctness, Context Recall (require ground truth).

### Mode B — With Ground Truth
All metrics available. Enables comparison-based scoring.

---

## Step 3 — Evaluate each metric

Read the scoring rubrics in `references/metrics.md` before scoring.

Evaluate **every applicable metric** one at a time, reasoning step by step before assigning a score.
Each metric is scored **0–3**:

| Score | Meaning |
|-------|---------|
| 0 | Failing — Critical problem |
| 1 | Poor — Significant issues |
| 2 | Acceptable — Minor issues |
| 3 | Good — No significant issues |

### Retrieval metrics

**Context Relevance** *(Mode A & B)*
- Are the retrieved chunks topically relevant to the question?
- Penalize heavily if chunks are off-topic or about unrelated entities.

**Context Precision** *(Mode A & B)*
- Of the retrieved chunks, what fraction are actually useful to answer the question?
- A single highly relevant chunk beats five diluted chunks.

**Context Recall** *(Mode B only)*
- Does the retrieved context cover the key information from the ground truth docs?
- Score 0 if critical facts from ground truth are entirely absent from context.

### Generation metrics

**Faithfulness / Groundedness** *(Mode A & B)*
- Is every claim in the generated answer supported by the retrieved context?
- Identify any hallucinations: facts stated that have no basis in the context.
- This is the most critical metric — weight it heavily in your summary.

**Answer Relevance** *(Mode A & B)*
- Does the answer actually address what the question asked?
- Penalize off-topic, padded, or evasive answers.

**Answer Completeness** *(Mode A & B)*
- Does the answer cover all aspects of the question using the available context?
- A correct but partial answer is not complete.

**Answer Correctness** *(Mode B only)*
- Is the answer factually aligned with the ground truth answer?
- Partial credit for answers that are correct but less detailed.

---

## Step 4 — Produce the evaluation report

ALWAYS use the exact report structure below. Do not skip sections.

---

```
# RAG Evaluation Report

## 📋 Input Summary
- **Question:** <question>
- **Retrieved chunks:** <count> chunk(s)
- **Evaluation mode:** Mode A (LLM-as-Judge) | Mode B (With Ground Truth)

---

## 🔍 Retrieval Quality

### Context Relevance  [score/3]
**Reasoning:** <step-by-step analysis>
**Finding:** <one sentence verdict>

### Context Precision  [score/3]
**Reasoning:** <step-by-step analysis>
**Finding:** <one sentence verdict>

### Context Recall  [score/3]  ← only in Mode B
**Reasoning:** <step-by-step analysis>
**Finding:** <one sentence verdict>

---

## 💬 Generation Quality

### Faithfulness / Groundedness  [score/3]
**Reasoning:** <step-by-step analysis>
**Hallucinations detected:** <list any unsupported claims, or "None">
**Finding:** <one sentence verdict>

### Answer Relevance  [score/3]
**Reasoning:** <step-by-step analysis>
**Finding:** <one sentence verdict>

### Answer Completeness  [score/3]
**Reasoning:** <step-by-step analysis>
**Finding:** <one sentence verdict>

### Answer Correctness  [score/3]  ← only in Mode B
**Reasoning:** <step-by-step analysis>
**Finding:** <one sentence verdict>

---

## 📊 Score Summary

| Metric | Score | Status |
|--------|-------|--------|
| Context Relevance | x/3 | 🟢/🟡/🔴 |
| Context Precision | x/3 | 🟢/🟡/🔴 |
| Context Recall* | x/3 | 🟢/🟡/🔴 |
| Faithfulness | x/3 | 🟢/🟡/🔴 |
| Answer Relevance | x/3 | 🟢/🟡/🔴 |
| Answer Completeness | x/3 | 🟢/🟡/🔴 |
| Answer Correctness* | x/3 | 🟢/🟡/🔴 |
| **Overall** | **x.x/3** | |

*Mode B only. 🟢 = 3  🟡 = 2  🔴 = 0–1

---

## 🔎 Diagnosis

**Primary failure point:** <Retrieval | Generation | Both | None>
**Root cause:** <1–2 sentences explaining the core problem, if any>

---

## 🛠️ Recommendations

1. <Highest priority fix>
2. <Second priority fix>
3. <Optional improvement>

---

## ✅ Verdict
<One paragraph summary: overall quality, most critical issue, confidence in the assessment>
```

---

## Evaluation principles

- **Reason before scoring.** Never assign a score without a written justification.
- **Be calibrated, not lenient.** A score of 3 means genuinely good, not "acceptable."
- **Faithfulness > everything.** Hallucinations are the most dangerous failure mode — flag them explicitly even when other scores are high.
- **Quote the evidence.** When claiming a hallucination or a retrieval miss, quote the relevant text so the user can verify.
- **Don't penalize the retriever for generator failures and vice versa.** Attribute problems to the correct component.
- **Multiple chunks:** Evaluate each chunk's contribution individually before scoring Context Precision.

---

## Handling edge cases

- **Empty or very short context:** Score Context Relevance and Precision as 0; note that the retriever returned insufficient content.
- **Question has no answer in the corpus:** A faithful "I don't know" answer should score high on Faithfulness even if low on Completeness.
- **Multi-part questions:** Treat each sub-question as a sub-dimension of Completeness.
- **Code or structured data as context:** Apply the same faithfulness and relevance logic; check that code snippets are quoted accurately in the answer.
