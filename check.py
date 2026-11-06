import pandas as pd

a = pd.read_csv("outputs/det_c/evaluation_case_results.csv")
b = pd.read_csv("outputs/det_d/evaluation_case_results.csv")

print("=== Cases where success differs ===")
diff = a["success"] != b["success"]
print(a.loc[diff, ["case_number", "test_index", "success", "num_candidates", "best_probability"]])
print()
print(b.loc[diff, ["case_number", "test_index", "success", "num_candidates", "best_probability"]])
print()

print("=== Cases where best_probability differs ===")
diff2 = (a["best_probability"].fillna(-1) - b["best_probability"].fillna(-1)).abs() > 1e-6
print(a.loc[diff2, ["case_number", "test_index", "success", "best_probability", "changed_features"]])
print()
print(b.loc[diff2, ["case_number", "test_index", "success", "best_probability", "changed_features"]])