"""
check_scm.py
Throwaway diagnostic: compare chisq vs kci causal discovery.
Run once, look at the output, decide which CI test to keep, then delete.
"""

from loader import load_adult_income_data
from causal_scm import discover_causal_structure, print_scm_summary

loader = load_adult_income_data()

# --- Run 1: chisq (what you already ran) ---
print("\n" + "=" * 70)
print("RUN 1: chisq")
print("=" * 70)
scm_chisq = discover_causal_structure(
    loader,
    ci_test="chisq",
    max_sample_rows=2000,
)
print_scm_summary(scm_chisq)

# --- Run 2: kci (mixed-data correct, but slow) ---
print("\n" + "=" * 70)
print("RUN 2: kci")
print("=" * 70)
scm_kci = discover_causal_structure(
    loader,
    ci_test="kci",
    max_sample_rows=500,   # kci is O(n^3); keep this small
)
print_scm_summary(scm_kci)

# --- Compare bool_masks (this is what matters downstream) ---
print("\n" + "=" * 70)
print("bool_mask comparison (mutable + causally connected to income)")
print("=" * 70)

chisq_mask = {
    name
    for name, flag in zip(scm_chisq.feature_names, scm_chisq.bool_mask)
    if flag
}
kci_mask = {
    name
    for name, flag in zip(scm_kci.feature_names, scm_kci.bool_mask)
    if flag
}

print(f"chisq bool_mask ({len(chisq_mask)}): {sorted(chisq_mask)}")
print(f"kci   bool_mask ({len(kci_mask)}): {sorted(kci_mask)}")
print(f"Only in chisq: {sorted(chisq_mask - kci_mask)}")
print(f"Only in kci:   {sorted(kci_mask - chisq_mask)}")
print(f"Agreed on:     {sorted(chisq_mask & kci_mask)}")