# Action-Oriented XAI Framework — Updated Project

This version implements the remaining project-quality suggestions around
counterfactual realism, causal-graph cycle handling, retrieval grounding,
LLM grounding, evaluation, runtime controls, and manual candidate input.

## Core pipeline

1. UCI Adult Income loading and encoding
2. Model comparison: Logistic Regression, Random Forest, XGBoost
3. Outcome-aware PC causal discovery + SCM representation
4. DiCE counterfactual generation
5. SCM + actionability + domain-consistency filtering
6. RAG retrieval with feature-aware FAISS ranking and optional O*NET
7. Llama 3.1 8B / template action-plan synthesis
8. Evaluation and export

## Main improvements

- The PC discovery graph now includes `income`, so counterfactual search uses
  features that are connected to the actual prediction target.
- Cycles in the oriented PC graph are handled explicitly for action ordering.
  The discovered graph itself is not rewritten for validity checks.
- Counterfactuals are filtered for redundant-field consistency, conservative
  numeric change caps, education downgrades, and unsuitable destination categories.
- DiCE retries continue until a validated candidate actually crosses the decision
  threshold; validated near-misses no longer stop the search early.
- The RAG search uses the existing `applies_to` metadata in `resources.json`
  instead of ranking the whole resource corpus blindly.
- Ollama prompts contain resource descriptions and evidence IDs.
- Generated URLs and evidence IDs are checked against the supplied evidence;
  failed grounding checks fall back to the deterministic template.
- Evaluation supports boundary-focused or reproducible random test cases,
  exports per-case CSV and aggregate JSON, and reports probability gain and
  actionability.
- Evaluation uses a smaller DiCE retry budget by default so experiments are
  tractable; increase it for a final high-budget run.
- `app.py` provides manual candidate entry through Streamlit while reusing the
  same backend pipeline.

## Run from PowerShell

Activate your Python 3.12 virtual environment first.

### Fast single-case run

```powershell
python main.py --backend ollama --skip-eval
```

### Evaluation run

```powershell
python main.py --backend template --eval-cases 10 --selection-mode random
```

For a stronger final experiment:

```powershell
python main.py --backend template --eval-cases 20 --selection-mode random --eval-max-iterations 2 --eval-total-cfs 5
```

### Free local LLM

Make sure Ollama is running and `llama3.1:8b` is installed:

```powershell
ollama list
python main.py --backend ollama --skip-eval
```

If Ollama does not finish before the timeout, the pipeline safely falls back
to the deterministic template.

### Interactive UI

Install Streamlit if needed:

```powershell
pip install streamlit
```

Then:

```powershell
streamlit run app.py
```

## O*NET

O*NET is optional. Set `ONET_API_KEY` before running to enable live occupation
lookups. The project also loads values from a local `.env` file when `python-dotenv`
is installed. The rest of the RAG pipeline works without it.

Example in the current PowerShell session:

```powershell
$env:ONET_API_KEY="YOUR_KEY"
```

## Outputs

Evaluation results are written to:

- `outputs/evaluation_case_results.csv`
- `outputs/evaluation_report.json`

These files are the values to use when reporting quantitative evaluation in
the paper. Do not substitute planned metrics for measured values.

## Important interpretation

A counterfactual is a model-level alternative profile that crosses the model's
decision boundary. It is not a guarantee that a person can or should make
every represented feature change in the real world. The RAG layer grounds
practical exploration in retrieved evidence, while the LLM only synthesizes
that evidence into readable steps.


## Methodological note

The PC stage uses label-encoded categorical variables and an encoded binary income target. The current eligibility mask uses undirected graph connectivity to the income node because PC returns a CPDAG with partially oriented edges. This is a structural connectivity constraint, not proof that every retained edge is an identified causal effect; the distinction should be stated explicitly in the paper.
