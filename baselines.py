"""
baselines.py
Module 9: Counterfactual Baseline Methods

Provides three established baselines plus a post-hoc-filtered DiCE
baseline that applies the main pipeline's constraint stack to vanilla
DiCE output. The post-hoc baseline isolates the contribution of the
constraint-aware search from the contribution of the filter.
"""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import dice_ml
from dice_ml import Dice

from config import CONFIG
from counterfactual import INTEGER_VALUED_FEATURES
from loader import (
    DISALLOWED_CATEGORICAL_VALUES,
    NUMERIC_CHANGE_LIMITS,
    SEARCHABLE_FEATURES,
    DataLoaderResult,
)
from predictor import PredictorResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("baselines")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BaselineCandidate:
    changed_features: Dict[str, Tuple[float, float]]
    predicted_probability: float
    num_changes: int
    proximity_l1: float
    method: str


@dataclass
class BaselineResult:
    method: str
    original_features: pd.Series
    original_probability: float
    candidates: List[BaselineCandidate] = field(default_factory=list)
    success: bool = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _positive_label(predictor_result: PredictorResult) -> int:
    return int(predictor_result.positive_class_label)


def _predict_probability(model, row: pd.Series, positive_label: int) -> float:
    return float(
        model.predict_proba(pd.DataFrame([row]))[0, positive_label]
    )


def _extract_changes(
    original: pd.Series,
    candidate: pd.Series,
    loader_result: DataLoaderResult,
    tolerance: float = 1e-6,
) -> Dict[str, Tuple[float, float]]:
    changes: Dict[str, Tuple[float, float]] = {}
    for feature in original.index:
        if feature not in candidate.index:
            continue
        old_value = float(original[feature])
        new_value = float(candidate[feature])
        if feature in loader_result.label_encoders:
            n_classes = len(loader_result.label_encoders[feature].classes_)
            new_value = float(np.clip(round(new_value), 0, n_classes - 1))
        elif feature in INTEGER_VALUED_FEATURES:
            new_value = float(round(new_value))
        if abs(old_value - new_value) > tolerance:
            changes[feature] = (old_value, new_value)
    return changes


def _proximity_l1(changes: Dict[str, Tuple[float, float]]) -> float:
    if not changes:
        return float("nan")
    return float(np.mean([abs(new - old) for old, new in changes.values()]))


def _default_query_row(predictor_result: PredictorResult) -> pd.Series:
    return predictor_result.test_case_features


def _default_query_probability(predictor_result: PredictorResult) -> float:
    return float(predictor_result.test_case_probability)


def _seed_random_state() -> None:
    seed = int(CONFIG.predictor.random_state)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Constraint violation checker
# ---------------------------------------------------------------------------

def check_constraint_violations(
    changes: Dict[str, Tuple[float, float]],
    loader_result: DataLoaderResult,
    scm_mask: Optional[np.ndarray] = None,
    scm_feature_names: Optional[List[str]] = None,
) -> List[str]:
    """Return list of constraint tags this candidate violates."""
    if not changes:
        return []

    violations: List[str] = []
    searchable_set = set(SEARCHABLE_FEATURES)

    scm_idx_map: Dict[str, int] = {}
    if scm_mask is not None and scm_feature_names is not None:
        scm_idx_map = {name: i for i, name in enumerate(scm_feature_names)}

    for feature, (old_value, new_value) in changes.items():
        old_f = float(old_value)
        new_f = float(new_value)

        if feature not in searchable_set:
            violations.append("non_searchable")

        if scm_idx_map and feature in scm_idx_map:
            idx = scm_idx_map[feature]
            if not bool(scm_mask[idx]):
                violations.append("scm_disconnected")

        rules = NUMERIC_CHANGE_LIMITS.get(feature, {})

        if "max_delta" in rules:
            if abs(new_f - old_f) > float(rules["max_delta"]):
                violations.append(f"delta_cap:{feature}")

        if "min_value" in rules and "max_value" in rules:
            if not (
                float(rules["min_value"]) <= new_f <= float(rules["max_value"])
            ):
                violations.append(f"range_cap:{feature}")

        direction = rules.get("direction")
        if direction == "increase" and new_f < old_f:
            violations.append(f"direction_decrease:{feature}")
        if direction == "decrease" and new_f > old_f:
            violations.append(f"direction_increase:{feature}")

        if feature == "education_num" and new_f < old_f:
            violations.append("education_downgrade")

        if (
            feature in DISALLOWED_CATEGORICAL_VALUES
            and feature in loader_result.label_encoders
        ):
            encoder = loader_result.label_encoders[feature]
            try:
                proposed_label = str(
                    encoder.inverse_transform([int(round(new_f))])[0]
                )
            except Exception:
                proposed_label = None
            if (
                proposed_label is not None
                and proposed_label in DISALLOWED_CATEGORICAL_VALUES[feature]
            ):
                violations.append(
                    f"disallowed_category:{feature}:{proposed_label}"
                )

    return violations


# ---------------------------------------------------------------------------
# Baseline 1: vanilla DiCE
# ---------------------------------------------------------------------------

def _build_vanilla_dice(
    loader_result: DataLoaderResult,
    predictor_result: PredictorResult,
) -> Dice:
    all_continuous_features = list(loader_result.X_train.columns)
    combined_df = pd.concat(
        [loader_result.X_train, loader_result.y_train.rename("income")],
        axis=1,
    )
    data_interface = dice_ml.Data(
        dataframe=combined_df,
        continuous_features=all_continuous_features,
        outcome_name="income",
    )
    model_interface = dice_ml.Model(
        model=predictor_result.best_model,
        backend="sklearn",
    )
    return Dice(data_interface, model_interface, method="random")


def generate_dice_baseline(
    predictor_result: PredictorResult,
    loader_result: DataLoaderResult,
    query_row: Optional[pd.Series] = None,
    query_probability: Optional[float] = None,
    total_cfs: int = 10,
    decision_threshold: Optional[float] = None,
) -> BaselineResult:
    """Vanilla DiCE: no SCM, no realism filter, no direction check."""
    if decision_threshold is None:
        decision_threshold = float(CONFIG.predictor.decision_threshold)

    original = (
        query_row if query_row is not None
        else _default_query_row(predictor_result)
    )
    original_probability = (
        query_probability if query_probability is not None
        else _default_query_probability(predictor_result)
    )

    vary_features = list(loader_result.searchable_features)
    if not vary_features:
        logger.warning("Vanilla DiCE: no searchable features available.")
        return BaselineResult(
            method="vanilla_dice",
            original_features=original,
            original_probability=original_probability,
            candidates=[],
            success=False,
        )

    explainer = _build_vanilla_dice(loader_result, predictor_result)
    _seed_random_state()

    try:
        dice_result = explainer.generate_counterfactuals(
            pd.DataFrame([original]),
            total_CFs=total_cfs,
            desired_class="opposite",
            features_to_vary=vary_features,
        )
        cf_df = dice_result.cf_examples_list[0].final_cfs_df
    except Exception as exc:
        logger.warning("Vanilla DiCE failed: %s", exc)
        return BaselineResult(
            method="vanilla_dice",
            original_features=original,
            original_probability=original_probability,
            candidates=[],
            success=False,
        )

    candidates: List[BaselineCandidate] = []
    seen: set = set()

    for _, row in cf_df.iterrows():
        cleaned = row.drop(labels=["income"], errors="ignore")
        changes = _extract_changes(original, cleaned, loader_result)
        if not changes:
            continue
        signature = tuple(
            sorted(
                (feature, round(float(old), 4), round(float(new), 4))
                for feature, (old, new) in changes.items()
            )
        )
        if signature in seen:
            continue
        seen.add(signature)

        modified = original.copy()
        for feature, (_, new_value) in changes.items():
            modified[feature] = new_value
        probability = _predict_probability(
            predictor_result.best_model, modified,
            _positive_label(predictor_result),
        )

        candidates.append(
            BaselineCandidate(
                changed_features=changes,
                predicted_probability=probability,
                num_changes=len(changes),
                proximity_l1=_proximity_l1(changes),
                method="vanilla_dice",
            )
        )

    crossing = [
        c for c in candidates if c.predicted_probability >= decision_threshold
    ]

    logger.info(
        "Vanilla DiCE: %d candidate(s), %d crossing threshold %.2f.",
        len(candidates), len(crossing), decision_threshold,
    )

    return BaselineResult(
        method="vanilla_dice",
        original_features=original,
        original_probability=original_probability,
        candidates=sorted(
            candidates,
            key=lambda c: (
                c.predicted_probability < decision_threshold,
                c.proximity_l1,
            ),
        ),
        success=len(crossing) > 0,
    )


# ---------------------------------------------------------------------------
# Baseline 2: post-hoc-filtered DiCE
# ---------------------------------------------------------------------------

def generate_post_hoc_filtered_dice(
    predictor_result: PredictorResult,
    loader_result: DataLoaderResult,
    scm_result,
    query_row: Optional[pd.Series] = None,
    query_probability: Optional[float] = None,
    total_cfs: int = 10,
    decision_threshold: Optional[float] = None,
) -> BaselineResult:
    """Vanilla DiCE followed by the main pipeline's constraint stack.

    This isolates the contribution of the constraint-aware *search* from
    the contribution of the *filter*. If post-hoc filtering of vanilla
    DiCE produces the same success rate as the main pipeline, the search
    contributes nothing and the filter is doing all the work. If it
    produces substantially lower success, the constrained search is
    finding candidates that vanilla DiCE cannot.
    """
    if decision_threshold is None:
        decision_threshold = float(CONFIG.predictor.decision_threshold)

    raw_result = generate_dice_baseline(
        predictor_result, loader_result,
        query_row=query_row,
        query_probability=query_probability,
        total_cfs=total_cfs,
        decision_threshold=decision_threshold,
    )

    # Reuse the exact predicates the main pipeline uses.
    from counterfactual import _is_scm_valid, _is_realistic_change

    surviving: List[BaselineCandidate] = []
    for candidate in raw_result.candidates:
        if not _is_scm_valid(
            candidate.changed_features,
            scm_result,
            scm_result.bool_mask,
        ):
            continue
        realistic, _ = _is_realistic_change(
            candidate.changed_features, loader_result
        )
        if not realistic:
            continue
        surviving.append(candidate)

    crossing = [
        c for c in surviving
        if c.predicted_probability >= decision_threshold
    ]

    logger.info(
        "Post-hoc-filtered DiCE: %d raw candidate(s), %d survived "
        "constraints, %d crossing threshold %.2f.",
        len(raw_result.candidates), len(surviving),
        len(crossing), decision_threshold,
    )

    return BaselineResult(
        method="post_hoc_filtered_dice",
        original_features=raw_result.original_features,
        original_probability=raw_result.original_probability,
        candidates=surviving,
        success=len(crossing) > 0,
    )


# ---------------------------------------------------------------------------
# Baseline 3: Wachter et al.
# ---------------------------------------------------------------------------

def generate_wachter_baseline(
    predictor_result: PredictorResult,
    loader_result: DataLoaderResult,
    query_row: Optional[pd.Series] = None,
    query_probability: Optional[float] = None,
    n_steps: int = 200,
    learning_rate: float = 0.5,
    distance_weight: float = 0.1,
    fd_epsilon: float = 0.05,
    decision_threshold: Optional[float] = None,
) -> BaselineResult:
    if decision_threshold is None:
        decision_threshold = float(CONFIG.predictor.decision_threshold)

    original = (
        query_row if query_row is not None
        else _default_query_row(predictor_result)
    )
    original_probability = (
        query_probability if query_probability is not None
        else _default_query_probability(predictor_result)
    )

    feature_names = list(loader_result.X_train.columns)
    searchable_set = set(loader_result.searchable_features)
    actionable_mask = np.array(
        [f in searchable_set for f in feature_names], dtype=bool
    )

    if not actionable_mask.any():
        return BaselineResult(
            method="wachter",
            original_features=original,
            original_probability=original_probability,
            candidates=[],
            success=False,
        )

    model = predictor_result.best_model
    positive_label = _positive_label(predictor_result)

    means = loader_result.X_train.mean().to_numpy(
        dtype=np.float64, copy=True
    )
    stds = loader_result.X_train.std().to_numpy(
        dtype=np.float64, copy=True
    )
    stds[stds == 0] = 1.0

    original_vec = original.to_numpy(dtype=np.float64)
    original_normalized = (original_vec - means) / stds
    x = original_normalized.copy()
    target_probability = min(decision_threshold + 0.1, 0.95)

    def predict_probability(normalized_vec: np.ndarray) -> float:
        denorm = (normalized_vec * stds + means).reshape(1, -1)
        df = pd.DataFrame(denorm, columns=feature_names)
        return float(model.predict_proba(df)[0, positive_label])

    best_x = x.copy()
    best_loss = float("inf")

    for step in range(n_steps):
        current_probability = predict_probability(x)
        distance = float(np.sum((x - original_normalized) ** 2))
        probability_gap = max(0.0, target_probability - current_probability)
        loss = distance_weight * distance + probability_gap

        if loss < best_loss:
            best_loss = loss
            best_x = x.copy()

        if probability_gap == 0.0:
            x = x - learning_rate * distance_weight * (
                x - original_normalized
            )
            x[~actionable_mask] = original_normalized[~actionable_mask]
            continue

        gradient = np.zeros_like(x)
        for i in range(len(feature_names)):
            if not actionable_mask[i]:
                continue
            x_plus = x.copy()
            x_plus[i] += fd_epsilon
            x_minus = x.copy()
            x_minus[i] -= fd_epsilon
            prob_plus = predict_probability(x_plus)
            prob_minus = predict_probability(x_minus)
            gradient[i] = (prob_plus - prob_minus) / (2.0 * fd_epsilon)

        ascent = gradient * actionable_mask
        x = x + learning_rate * ascent
        x = x - learning_rate * distance_weight * (
            x - original_normalized
        )
        x[~actionable_mask] = original_normalized[~actionable_mask]

    final_probability = predict_probability(best_x)
    final_denorm = best_x * stds + means

    candidate_row = pd.Series(final_denorm, index=feature_names)
    changes = _extract_changes(original, candidate_row, loader_result)

    if not changes:
        logger.info(
            "Wachter: no change found (probability %.3f -> %.3f).",
            original_probability, final_probability,
        )
        return BaselineResult(
            method="wachter",
            original_features=original,
            original_probability=original_probability,
            candidates=[],
            success=False,
        )

    modified = original.copy()
    for feature, (_, new_value) in changes.items():
        modified[feature] = new_value
    real_probability = _predict_probability(
        model, modified, positive_label
    )

    candidate = BaselineCandidate(
        changed_features=changes,
        predicted_probability=real_probability,
        num_changes=len(changes),
        proximity_l1=_proximity_l1(changes),
        method="wachter",
    )

    logger.info(
        "Wachter: probability %.3f -> %.3f, changes=%d.",
        original_probability, real_probability, len(changes),
    )

    return BaselineResult(
        method="wachter",
        original_features=original,
        original_probability=original_probability,
        candidates=[candidate],
        success=real_probability >= decision_threshold,
    )


# ---------------------------------------------------------------------------
# Baseline 4: Ustun et al.
# ---------------------------------------------------------------------------

def _extract_linear_coefficients(
    pipeline,
    feature_names: List[str],
) -> Optional[Tuple[np.ndarray, float]]:
    if not hasattr(pipeline, "named_steps"):
        return None
    steps = pipeline.named_steps
    if "scaler" not in steps or "model" not in steps:
        return None
    scaler = steps["scaler"]
    model = steps["model"]
    if not hasattr(model, "coef_"):
        return None
    coef_scaled = model.coef_[0]
    intercept_scaled = float(model.intercept_[0])
    scale = scaler.scale_
    mean = scaler.mean_
    coef_raw = coef_scaled / scale
    intercept_raw = intercept_scaled - float(
        np.sum(coef_scaled * mean / scale)
    )
    return coef_raw.astype(np.float64), intercept_raw


def generate_ustun_baseline(
    predictor_result: PredictorResult,
    loader_result: DataLoaderResult,
    query_row: Optional[pd.Series] = None,
    query_probability: Optional[float] = None,
    max_changes: int = 3,
    decision_threshold: Optional[float] = None,
) -> BaselineResult:
    if decision_threshold is None:
        decision_threshold = float(CONFIG.predictor.decision_threshold)

    original = (
        query_row if query_row is not None
        else _default_query_row(predictor_result)
    )
    original_probability = (
        query_probability if query_probability is not None
        else _default_query_probability(predictor_result)
    )

    linear_model = predictor_result.trained_models.get("LogisticRegression")
    if linear_model is None:
        logger.warning(
            "Ustun baseline: LogisticRegression not present in trained_models."
        )
        return BaselineResult(
            method="ustun",
            original_features=original,
            original_probability=original_probability,
            candidates=[],
            success=False,
        )

    feature_names = list(loader_result.X_train.columns)
    coefs = _extract_linear_coefficients(linear_model, feature_names)
    if coefs is None:
        logger.warning(
            "Ustun baseline: could not extract linear coefficients."
        )
        return BaselineResult(
            method="ustun",
            original_features=original,
            original_probability=original_probability,
            candidates=[],
            success=False,
        )

    coef_raw, intercept_raw = coefs
    searchable_set = set(loader_result.searchable_features)
    actionable_indices = [
        i for i, f in enumerate(feature_names) if f in searchable_set
    ]
    if not actionable_indices:
        return BaselineResult(
            method="ustun",
            original_features=original,
            original_probability=original_probability,
            candidates=[],
            success=False,
        )

    value_grids: Dict[int, List[float]] = {}
    for i in actionable_indices:
        feature = feature_names[i]
        original_value = float(original[feature])
        if feature in loader_result.label_encoders:
            n_classes = len(loader_result.label_encoders[feature].classes_)
            values = [float(c) for c in range(n_classes)]
            values = [v for v in values if v != original_value]
        elif feature == "education_num":
            values = [
                float(v)
                for v in range(int(original_value) - 3, int(original_value) + 4)
                if 3 <= v <= 16 and v != original_value
            ]
        elif feature == "hours_per_week":
            values = [
                float(v)
                for v in range(
                    int(original_value) - 12, int(original_value) + 13, 4
                )
                if 20 <= v <= 60 and v != original_value
            ]
        else:
            values = []
        value_grids[i] = values

    x_orig = original.to_numpy(dtype=np.float64)
    score_orig = float(coef_raw @ x_orig + intercept_raw)

    def sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def evaluate_indices(change_indices):
        best_prob = -1.0
        best_changes = None
        value_lists = [value_grids[i] for i in change_indices]
        if any(not vl for vl in value_lists):
            return -1.0, None
        for combo in itertools.product(*value_lists):
            delta_score = 0.0
            changes: Dict[str, Tuple[float, float]] = {}
            for idx, value in zip(change_indices, combo):
                feature = feature_names[idx]
                old_value = float(original[feature])
                delta_score += float(coef_raw[idx]) * (value - old_value)
                changes[feature] = (old_value, float(value))
            new_score = score_orig + delta_score
            new_prob = sigmoid(new_score)
            if new_prob > best_prob:
                best_prob = new_prob
                best_changes = changes
        return best_prob, best_changes

    chosen_changes = None
    chosen_probability = None

    for k in range(1, max_changes + 1):
        best_prob_this_round = -1.0
        best_changes_this_round = None
        for combo in itertools.combinations(actionable_indices, k):
            prob, changes = evaluate_indices(list(combo))
            if changes is None:
                continue
            if prob > best_prob_this_round:
                best_prob_this_round = prob
                best_changes_this_round = changes
        if best_changes_this_round is None:
            continue
        if best_prob_this_round >= decision_threshold:
            chosen_changes = best_changes_this_round
            chosen_probability = best_prob_this_round
            break
        if (
            chosen_probability is None
            or best_prob_this_round > chosen_probability
        ):
            chosen_changes = best_changes_this_round
            chosen_probability = best_prob_this_round

    if not chosen_changes:
        logger.info(
            "Ustun: no candidate found within max_changes=%d.", max_changes
        )
        return BaselineResult(
            method="ustun",
            original_features=original,
            original_probability=original_probability,
            candidates=[],
            success=False,
        )

    modified = original.copy()
    for feature, (_, new_value) in chosen_changes.items():
        modified[feature] = new_value
    real_probability = _predict_probability(
        linear_model, modified, _positive_label(predictor_result)
    )

    candidate = BaselineCandidate(
        changed_features=chosen_changes,
        predicted_probability=real_probability,
        num_changes=len(chosen_changes),
        proximity_l1=_proximity_l1(chosen_changes),
        method="ustun",
    )

    logger.info(
        "Ustun: probability %.3f -> %.3f, changes=%d.",
        original_probability, real_probability, len(chosen_changes),
    )

    return BaselineResult(
        method="ustun",
        original_features=original,
        original_probability=original_probability,
        candidates=[candidate],
        success=real_probability >= decision_threshold,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_baseline_summary(
    result: BaselineResult,
    loader_result: Optional[DataLoaderResult] = None,
) -> None:
    print(f"Baseline: {result.method}")
    print(f"  Original probability: {result.original_probability:.3f}")
    print(f"  Success: {result.success}")
    print(f"  Candidates: {len(result.candidates)}")
    for i, candidate in enumerate(result.candidates[:3], start=1):
        print(
            f"    {i}. prob={candidate.predicted_probability:.3f} "
            f"changes={candidate.num_changes} "
            f"proximity={candidate.proximity_l1:.3f}"
        )
        for feature, (old, new) in candidate.changed_features.items():
            if (
                loader_result is not None
                and feature in loader_result.label_encoders
            ):
                encoder = loader_result.label_encoders[feature]
                try:
                    old_text = encoder.inverse_transform([int(round(old))])[0]
                    new_text = encoder.inverse_transform([int(round(new))])[0]
                    print(f"       {feature}: {old_text} -> {new_text}")
                except (ValueError, IndexError, TypeError):
                    print(f"       {feature}: {old:.2f} -> {new:.2f}")
            else:
                print(f"       {feature}: {old:.2f} -> {new:.2f}")