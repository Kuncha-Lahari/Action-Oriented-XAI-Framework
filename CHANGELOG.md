# Updated project changes

## Counterfactual
- Added a human-actionability allow-list.
- Added conservative domain caps for education years and hours/week.
- Rejects `Without-pay` as a target employment category.
- Synchronizes the duplicated Adult `education` / `education_num` representation.
- Reports rejection reasons separately.
- Added an evaluation-friendly `initial_total_cfs` parameter.

## Causal SCM
- PC discovery includes `income`.
- Mutable features are marked usable only when connected to the outcome.
- Cyclic oriented graphs are handled through a temporary ordering DAG.
- Cycle-breaking edges are reported; the original discovered graph remains unchanged.

## RAG
- Uses `applies_to` resource metadata for feature-specific retrieval.
- Falls back to the full corpus only when no resources are tagged for a feature.
- Adds feature-specific query wording.
- Caches the resource index within a Python process.

## LLM
- Ollama remains free/local using Llama 3.1 8B.
- Prompt now includes resource descriptions and evidence IDs.
- Generated URLs and `[R#]` evidence IDs are checked against retrieved evidence.
- Known unrealistic workclass phrasing is rejected.
- Reduced default output/context budgets and added Ollama `keep_alive` for faster repeat calls.
- Python controls the final probability and counterfactual values.

## Evaluation
- Added average and median probability gain.
- Added actionability-valid rate.
- Supports boundary or random selection.
- Respects `--eval-cases`.
- Saves detailed per-case CSV and aggregate JSON.
- Evaluation DiCE budget is smaller by default for tractable repeated runs.

## UI
- Added `app.py` for manual candidate input.
- The UI reuses the same trained model, SCM, DiCE, RAG, and LLM/template backend.
- The current command-line pipeline remains unchanged for reproducible experiments.


## Review fixes (2026-09-22)
- Corrected O*NET v2 occupation keyword search endpoint to `/online/search`.
- Loaded `.env` automatically when `python-dotenv` is available.
- Fixed LLM Phase 3 change descriptions to use decoded feature values instead of encoded category IDs.
- Tightened evidence-ID grounding validation to match the exact top-2 resources exposed to the LLM prompt.
- Linked the manual UI's `education_num` field to the selected `education` value to prevent internally inconsistent profiles.
- Clarified that undirected PC connectivity is a structural eligibility signal rather than proof of an identified directed causal effect.


## Search corrections (2026-09-24)
- Counterfactual retries now continue when DiCE returns only valid near-miss candidates;
  the loop stops only after a candidate crosses the decision threshold after all repairs and filters.
- Education counterfactuals that decrease `education_num` are rejected as non-actionable downgrades.
- `Never-worked` is excluded as a target workclass for human-facing action plans, alongside `Without-pay`.
- Fixed LLM text-response Phase 3 decoding so categorical changes use the active `DataLoaderResult` rather than raw encoded IDs.
