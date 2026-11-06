import pandas as pd
from loader import load_adult_income_data
from predictor import train_and_select_model
from causal_scm import discover_causal_structure
from evaluator import _select_denied_test_rows, _evaluate_one_method
from config import CONFIG

loader = load_adult_income_data()
predictor = train_and_select_model(loader)
scm = discover_causal_structure(loader)

# Same 50 cases the CF quality used
rows = _select_denied_test_rows(
    predictor, loader, n_cases=50,
    selection_mode="random", random_seed=42,
)

print(f"Selected {len(rows)} cases")
print(f"Test indices: {[r[0] for r in rows]}")

# Run main_pipeline only
record = _evaluate_one_method(
    method="main_pipeline",
    case_rows=rows,
    predictor_result=predictor,
    scm_result=scm,
    loader_result=loader,
    dice_total_cfs=10,
    wachter_n_steps=200,
    ustun_max_changes=3,
)

print(f"\nMain pipeline success: {record.n_successful}/{record.n_cases}")