import pandas as pd
import numpy as np

a = pd.read_csv("outputs/det_c/evaluation_case_results.csv")
b = pd.read_csv("outputs/det_d/evaluation_case_results.csv")

print("Same length:", len(a) == len(b))
print("Same test indices:", (a["test_index"] == b["test_index"]).all())

# Handle NaN explicitly — NaN == NaN is False, so we can't compare directly.
def _all_equal(s1, s2):
    both_nan = s1.isna() & s2.isna()
    both_present = s1.notna() & s2.notna()
    equal_values = both_present & (s1 == s2)
    return (both_nan | equal_values).all()

print("Same success:", _all_equal(a["success"], b["success"]))
print("Same best probabilities:", _all_equal(a["best_probability"], b["best_probability"]))
print("Same changed features:", _all_equal(a["changed_features"], b["changed_features"]))

# If any of those are False, show the offending rows.
for col in ["success", "best_probability", "changed_features"]:
    if not _all_equal(a[col], b[col]):
        both_nan = a[col].isna() & b[col].isna()
        both_present = a[col].notna() & b[col].notna()
        equal_values = both_present & (a[col] == b[col])
        diff_mask = ~(both_nan | equal_values)
        print(f"\nRows where {col} differs:")
        print(a.loc[diff_mask, ["case_number", "test_index", col]])
        print(b.loc[diff_mask, ["case_number", "test_index", col]])