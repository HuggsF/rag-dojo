# RAG Metrics — Detailed Scoring Rubrics

Use this file when you need more granular guidance on a specific metric score.

---

## Context Relevance

Measures: Are the retrieved passages topically related to the question?

| Score | Criteria |
|-------|----------|
| 3 | All retrieved chunks are directly about the topic of the question. No irrelevant passages. |
| 2 | Most chunks are relevant; 1 chunk is tangentially related but not harmful. |
| 1 | Mix of relevant and irrelevant chunks; retriever is adding noise. |
| 0 | Most or all chunks are off-topic or about entirely different subjects. |

**What to check:**
- Named entities: does the context mention the same person/product/event as the question?
- Topic match: same domain, same concept?
- Time relevance: if question is about a specific period, is the context from that period?

---

## Context Precision

Measures: Of what was retrieved, how much was actually useful?

| Score | Criteria |
|-------|----------|
| 3 | Every retrieved chunk contributes meaningful information to answering the question. |
| 2 | Most chunks are useful; 1–2 are filler but don't mislead. |
| 1 | Less than half the chunks are useful; significant noise. |
| 0 | Retrieved context is mostly irrelevant despite being topically related (e.g., right document, wrong section). |

**What to check:**
- Would removing a chunk change the answer? If not, it contributes nothing → lower precision.
- Are chunks redundant (same information 3 times)?

---

## Context Recall

Requires: `ground_truth_docs` or `ground_truth_answer`.

Measures: Were the key facts that should have been retrieved actually retrieved?

| Score | Criteria |
|-------|----------|
| 3 | All key facts from the ground truth are present in the retrieved context. |
| 2 | Most key facts are present; 1 minor fact is missing. |
| 1 | Several important facts from ground truth are absent from context. |
| 0 | The retrieved context is missing the core information needed to answer correctly. |

**What to check:**
- List the key facts in the ground truth answer. Check each one against the context.
- Missing facts → the retriever failed to find the right document.

---

## Faithfulness / Groundedness

Measures: Is every claim in the answer supported by the retrieved context?

| Score | Criteria |
|-------|----------|
| 3 | Every factual claim in the answer can be traced to a specific statement in the context. |
| 2 | Minor claims are slightly embellished but not dangerously wrong; no explicit contradictions. |
| 1 | One or more factual claims go beyond what the context says, or slightly contradict it. |
| 0 | The answer contains explicit hallucinations: facts stated with confidence that are absent from or contradicted by the context. |

**Hallucination types to flag:**
- **Fabricated facts:** numbers, names, dates not in context
- **Over-generalization:** context says "some" but answer says "all"
- **Contradiction:** answer says X but context says not-X
- **Interpolation:** answer infers a conclusion the context doesn't support

---

## Answer Relevance

Measures: Does the answer address what was actually asked?

| Score | Criteria |
|-------|----------|
| 3 | Answer directly and specifically addresses the question. No padding or tangents. |
| 2 | Answer addresses the question but includes some unnecessary information. |
| 1 | Answer is partially responsive; misses the core ask or answers a related but different question. |
| 0 | Answer does not address the question (evasion, refusal without reason, total topic mismatch). |

---

## Answer Completeness

Measures: Does the answer cover all parts of the question using available context?

| Score | Criteria |
|-------|----------|
| 3 | All aspects of the question are addressed using the available context. |
| 2 | Most aspects covered; 1 minor sub-question left unanswered despite context being available. |
| 1 | Significant parts of the question are left unanswered even though the context contains the information. |
| 0 | Answer is a fragment — the question had multiple parts and only one (or none) was answered. |

**Note:** Don't penalize incompleteness if the context itself doesn't contain the missing information. That's a retrieval failure, not a generation failure.

---

## Answer Correctness

Requires: `ground_truth_answer`.

Measures: Is the answer factually aligned with the known correct answer?

| Score | Criteria |
|-------|----------|
| 3 | Answer is factually equivalent to ground truth (different wording is fine). |
| 2 | Answer is mostly correct; minor omissions or slight imprecision but no wrong facts. |
| 1 | Answer contains the right gist but has at least one clearly wrong fact. |
| 0 | Answer is factually wrong or contradicts the ground truth on the main point. |
