import numpy as np
from loader import load_adult_income_data
from predictor import train_and_select_model
from causal_scm import discover_causal_structure
from counterfactual import generate_counterfactuals

loader = load_adult_income_data()
predictor = train_and_select_model(loader)
scm = discover_causal_structure(loader)

X_test = loader.X_test
probas = predictor.best_model.predict_proba(X_test)[:, predictor.positive_class_label]
denied = np.where(probas < 0.45)[0]  # denied and clearly below boundary
rng = np.random.default_rng(42)
sample = rng.choice(denied, size=20, replace=False)

with_success = without_success = 0
with_features = set()
without_features = set()
different_candidates = 0

for idx in sample:
    row = X_test.iloc[int(idx)]
    prob = float(probas[int(idx)])

    r1 = generate_counterfactuals(
        predictor, scm, loader, query_row=row, query_probability=prob,
        scm_mask=scm.bool_mask,
    )
    r2 = generate_counterfactuals(
        predictor, scm, loader, query_row=row, query_probability=prob,
        scm_mask=scm.mutable_only_mask,
    )

    if r1.success: with_success += 1
    if r2.success: without_success += 1
    with_features.update(r1.search_feature_set)
    without_features.update(r2.search_feature_set)

    # Did the winning candidates differ?
    if r1.candidates and r2.candidates:
        c1 = set(r1.candidates[0].changed_features.keys())
        c2 = set(r2.candidates[0].changed_features.keys())
        if c1 != c2:
            different_candidates += 1

print(f"With SCM:    success={with_success}/20")
print(f"Without SCM: success={without_success}/20")
print(f"With SCM feature set: {sorted(with_features)}")
print(f"Without SCM feature set: {sorted(without_features)}")
print(f"Cases with different winning candidates: {different_candidates}/20")