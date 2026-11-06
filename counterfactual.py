"""
counterfactual.py
Module 4 of 8: Iterative Counterfactual Generation

Design notes
------------
- Feasibility scoring is computed on range-normalized deltas.
- capital_gain and capital_loss have delta caps in NUMERIC_CHANGE_LIMITS.
- The search space is SEARCHABLE_FEATURES (loader.py) intersected with
  the SCM mask. The final action plan uses RECOMMENDABLE_FEATURES only.
- The global RNG is seeded once at the start of generate_counterfactuals.
  It is NOT reseeded inside the iteration loop: reseeding would reset the
  RNG to the same starting state on every iteration, which makes the
  "widening pool" step produce identical candidates each time. Seeding
  once lets the RNG advance naturally, so each iteration generates a
  genuinely new candidate pool while remaining reproducible across runs.
"""

from __future__ import annotations

import heapq
import logging
import random as _py_random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice

from causal_scm import CausalSCMResult, rank_features_by_causal_priority
from config import CONFIG
from loader import (
    DISALLOWED_CATEGORICAL_VALUES,
    NUMERIC_CHANGE_LIMITS,
    RECOMMENDABLE_FEATURES,
    DataLoaderResult,
)
from predictor import PredictorResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("counterfactual")

MAX_ITERATIONS = int(CONFIG.counterfactual.max_iterations)
INITIAL_TOTAL_CFS = int(CONFIG.counterfactual.initial_total_cfs)
CFS_GROWTH_FACTOR = int(CONFIG.counterfactual.cfs_growth_factor)
SPARSITY_PENALTY_WEIGHT = float(CONFIG.counterfactual.sparsity_penalty_weight)
CAUSAL_DISTANCE_WEIGHT = float(CONFIG.counterfactual.causal_distance_weight)
TARGET_MARGIN = float(CONFIG.counterfactual.target_margin)
INTEGER_VALUED_FEATURES = set(CONFIG.counterfactual.integer_valued_features)


@dataclass
class CounterfactualCandidate:
    changed_features: Dict[str, Tuple[float, float]]
    feasibility_score: float
    num_changes: int
    proximity_l1: float
    predicted_probability: float
    causal_distance: float = float("nan")
    base_feasibility_score: float = float("nan")

    @property
    def recommendable(self) -> bool:
        return all(
            f in RECOMMENDABLE_FEATURES for f in self.changed_features
        )


@dataclass
class CounterfactualResult:
    original_features: pd.Series
    original_probability: float
    candidates: List[CounterfactualCandidate]
    iterations_used: int
    scm_rejected_count: int
    search_feature_set: List[str]
    success: bool
    rejection_counts: Dict[str, int] = field(default_factory=dict)
    validated_candidate_changes: List[
        Dict[str, Tuple[float, float]]
    ] = field(default_factory=list)
    effective_threshold: float = float("nan")
    target_margin_used: float = float("nan")


def _build_dice_explainer(
    loader_result: DataLoaderResult,
    predictor_result: PredictorResult,
    search_features: List[str],
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


def _extract_candidate_changes(
    original: pd.Series,
    counterfactual_row: pd.Series,
    loader_result: DataLoaderResult,
    tolerance: float = 1e-6,
) -> Dict[str, Tuple[float, float]]:
    changes: Dict[str, Tuple[float, float]] = {}

    for feature in original.index:
        if feature not in counterfactual_row.index:
            continue

        original_value = float(original[feature])
        new_value = float(counterfactual_row[feature])

        if feature in loader_result.label_encoders:
            num_categories = len(
                loader_result.label_encoders[feature].classes_
            )
            new_value = float(
                np.clip(round(new_value), 0, num_categories - 1)
            )
        elif feature in INTEGER_VALUED_FEATURES:
            new_value = float(round(new_value))

        if abs(original_value - new_value) > tolerance:
            changes[feature] = (original_value, new_value)

    return changes


def _is_scm_valid(
    changes: Dict[str, Tuple[float, float]],
    scm_result: CausalSCMResult,
    scm_mask: np.ndarray,
) -> bool:
    if not changes:
        return False
    name_to_idx = {name: i for i, name in enumerate(scm_result.feature_names)}
    for feature in changes:
        idx = name_to_idx.get(feature)
        if idx is None:
            return False
        if not bool(scm_mask[idx]):
            return False
    return True


def _is_realistic_change(
    changes: Dict[str, Tuple[float, float]],
    loader_result: DataLoaderResult,
) -> Tuple[bool, str]:
    if not changes:
        return False, "empty"

    searchable = set(loader_result.searchable_features)

    for feature, (old_value, new_value) in changes.items():
        if feature not in searchable:
            return False, "non_searchable_feature"

        rules = NUMERIC_CHANGE_LIMITS.get(feature, {})
        old_f = float(old_value)
        new_f = float(new_value)

        if "max_delta" in rules:
            if abs(new_f - old_f) > float(rules["max_delta"]):
                return False, f"delta_cap:{feature}"

        if "min_value" in rules and "max_value" in rules:
            if not (
                float(rules["min_value"]) <= new_f <= float(rules["max_value"])
            ):
                return False, f"range_cap:{feature}"

        direction = rules.get("direction")
        if direction == "increase" and new_f < old_f:
            return False, f"direction_decrease:{feature}"
        if direction == "decrease" and new_f > old_f:
            return False, f"direction_increase:{feature}"

        if feature == "education_num" and new_f < old_f:
            return False, "education_downgrade"

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
                return False, f"invalid_category:{feature}"
            if proposed_label in DISALLOWED_CATEGORICAL_VALUES[feature]:
                return False, f"disallowed_category:{feature}:{proposed_label}"

    return True, ""


def build_realistic_delta_caps(
    loader_result: DataLoaderResult,
) -> Dict[str, Dict[str, float]]:
    return dict(NUMERIC_CHANGE_LIMITS)


def _feature_range_for_scoring(
    feature: str,
    loader_result: Optional[DataLoaderResult],
) -> float:
    """Return the normalization range used for feasibility scoring."""
    if loader_result is None:
        return 1.0

    if feature in loader_result.label_encoders:
        n_classes = len(loader_result.label_encoders[feature].classes_)
        return max(n_classes - 1, 1)

    if feature not in loader_result.X_train.columns:
        return 1.0

    column = loader_result.X_train[feature]
    value_range = float(column.max() - column.min())
    return value_range if value_range > 0 else 1.0


def _feasibility_score(
    changes: Dict[str, Tuple[float, float]],
    loader_result: Optional[DataLoaderResult] = None,
) -> Tuple[float, float]:
    """Compute (feasibility_score, proximity_l1) on normalized deltas."""
    if not changes:
        return float("inf"), float("inf")

    normalized_terms: List[float] = []
    for feature, (old, new) in changes.items():
        feature_range = _feature_range_for_scoring(feature, loader_result)
        normalized_terms.append(abs(new - old) / feature_range)

    proximity = float(np.mean(normalized_terms))
    score = proximity + SPARSITY_PENALTY_WEIGHT * len(changes)
    return score, proximity


def _build_causal_distance_map(
    scm_result: Optional[CausalSCMResult],
    target_feature: str = "income",
) -> Dict[str, int]:
    if scm_result is None:
        return {}
    try:
        return dict(
            rank_features_by_causal_priority(scm_result, target_feature)
        )
    except Exception:
        return {}


def _rank_candidates(
    raw_candidates: List[Dict[str, Tuple[float, float]]],
    model,
    original: pd.Series,
    positive_class_label: int,
    loader_result: Optional[DataLoaderResult] = None,
    scm_result: Optional[CausalSCMResult] = None,
    causal_aware: bool = True,
    causal_distance_weight: float = CAUSAL_DISTANCE_WEIGHT,
    prefer_recommendable: bool = True,
) -> List[CounterfactualCandidate]:
    heap: List[
        Tuple[float, int, Dict[str, Tuple[float, float]], float, float]
    ] = []

    causal_distance_map = _build_causal_distance_map(scm_result)
    recommendable_set = set(RECOMMENDABLE_FEATURES)

    for i, changes in enumerate(raw_candidates):
        base_score, proximity = _feasibility_score(changes, loader_result)

        causal_distance = float("nan")
        causal_penalty = 0.0

        if causal_distance_map and changes:
            distances = [
                float(causal_distance_map[feature])
                for feature in changes
                if feature in causal_distance_map
            ]
            if distances:
                causal_distance = float(np.mean(distances))
                if causal_aware:
                    causal_penalty = (
                        float(causal_distance_weight) * causal_distance
                    )

        recommendability_penalty = 0.0
        if prefer_recommendable and changes:
            if not all(f in recommendable_set for f in changes):
                recommendability_penalty = 0.5

        final_score = float(
            base_score + causal_penalty + recommendability_penalty
        )

        heapq.heappush(
            heap,
            (final_score, i, changes, base_score, causal_distance),
        )

    ranked: List[CounterfactualCandidate] = []
    seen_signatures: set = set()

    while heap:
        score, _, changes, base_score, causal_distance = heapq.heappop(heap)

        signature = tuple(
            sorted(
                (
                    feature,
                    round(float(old), 4),
                    round(float(new), 4),
                )
                for feature, (old, new) in changes.items()
            )
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        modified = original.copy()
        for feature, (_, new_value) in changes.items():
            modified[feature] = new_value

        probability = float(
            model.predict_proba(pd.DataFrame([modified]))[
                0, positive_class_label
            ]
        )

        _, proximity = _feasibility_score(changes, loader_result)

        ranked.append(
            CounterfactualCandidate(
                changed_features=changes,
                feasibility_score=float(score),
                num_changes=len(changes),
                proximity_l1=proximity,
                predicted_probability=probability,
                causal_distance=causal_distance,
                base_feasibility_score=float(base_score),
            )
        )

    return ranked


def generate_counterfactuals(
    predictor_result: PredictorResult,
    scm_result: CausalSCMResult,
    loader_result: DataLoaderResult,
    decision_threshold: Optional[float] = None,
    target_margin: Optional[float] = None,
    max_iterations: int = MAX_ITERATIONS,
    query_row: Optional[pd.Series] = None,
    query_probability: Optional[float] = None,
    initial_total_cfs: int = INITIAL_TOTAL_CFS,
    causal_aware_ranking: bool = True,
    causal_distance_weight: float = CAUSAL_DISTANCE_WEIGHT,
    prefer_recommendable: bool = True,
    scm_mask: Optional[np.ndarray] = None,
) -> CounterfactualResult:
    if decision_threshold is None:
        decision_threshold = float(CONFIG.predictor.decision_threshold)
    if target_margin is None:
        target_margin = TARGET_MARGIN

    effective_threshold = float(decision_threshold) + float(target_margin)

    if scm_mask is None:
        scm_mask = scm_result.bool_mask
    else:
        scm_mask = np.asarray(scm_mask, dtype=bool)
        if scm_mask.shape != scm_result.bool_mask.shape:
            raise ValueError(
                f"scm_mask has shape {scm_mask.shape}; expected "
                f"{scm_result.bool_mask.shape}."
            )

    if query_row is not None and query_probability is None:
        raise ValueError(
            "query_probability must be provided alongside a custom query_row."
        )

    original = (
        query_row if query_row is not None
        else predictor_result.test_case_features
    )
    original_probability = (
        query_probability if query_row is not None
        else predictor_result.test_case_probability
    )

    searchable_set = set(loader_result.searchable_features)
    search_features = [
        name
        for name, usable in zip(scm_result.feature_names, scm_mask)
        if (usable and name in searchable_set)
    ]

    if not search_features:
        logger.warning("No searchable + causally-connected features available.")
        return CounterfactualResult(
            original_features=original,
            original_probability=original_probability,
            candidates=[],
            iterations_used=0,
            scm_rejected_count=0,
            search_feature_set=[],
            success=False,
            validated_candidate_changes=[],
            effective_threshold=effective_threshold,
            target_margin_used=float(target_margin),
        )

    logger.info(
        "Counterfactual search restricted to: %s "
        "(effective threshold %.3f = %.3f + margin %.3f)",
        search_features, effective_threshold,
        decision_threshold, target_margin,
    )

    explainer = _build_dice_explainer(
        loader_result, predictor_result, search_features
    )
    query_instance = pd.DataFrame([original])

    # Seed the global RNGs ONCE at the start of the search. Do not reseed
    # inside the loop — reseeding would reset the RNG to the same starting
    # state on every iteration, causing the "widening" step to produce the
    # same candidates each time. Seeding once lets numpy's RNG advance
    # naturally across iterations, which is what makes the widening real.
    _rng_seed = int(CONFIG.predictor.random_state)
    np.random.seed(_rng_seed)
    _py_random.seed(_rng_seed)

    valid_candidates: List[Dict[str, Tuple[float, float]]] = []
    seen_signatures: set = set()
    rejection_counts: Dict[str, int] = {"scm": 0, "realism": 0}

    iterations_used = 0
    total_cfs = max(1, int(initial_total_cfs))

    for iteration in range(1, max_iterations + 1):
        iterations_used = iteration
        logger.info(
            "Iteration %d: requesting %d candidates from DiCE...",
            iteration, total_cfs,
        )

        try:
            dice_result = explainer.generate_counterfactuals(
                query_instance,
                total_CFs=total_cfs,
                desired_class="opposite",
                features_to_vary=search_features,
            )
            cf_df = dice_result.cf_examples_list[0].final_cfs_df
        except Exception as exc:
            logger.warning("Iteration %d: DiCE raised %s", iteration, exc)
            cf_df = None

        if cf_df is not None:
            for _, row in cf_df.iterrows():
                changes = _extract_candidate_changes(
                    original,
                    row.drop(labels=["income"], errors="ignore"),
                    loader_result,
                )

                if not _is_scm_valid(changes, scm_result, scm_mask):
                    rejection_counts["scm"] += 1
                    continue

                realistic, _ = _is_realistic_change(changes, loader_result)
                if not realistic:
                    rejection_counts["realism"] += 1
                    continue

                signature = tuple(
                    sorted(
                        (
                            feature,
                            round(float(old), 4),
                            round(float(new), 4),
                        )
                        for feature, (old, new) in changes.items()
                    )
                )
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    valid_candidates.append(changes)

        ranked_this_iteration = _rank_candidates(
            valid_candidates,
            predictor_result.best_model,
            original,
            predictor_result.positive_class_label,
            loader_result=loader_result,
            scm_result=scm_result,
            causal_aware=causal_aware_ranking,
            causal_distance_weight=causal_distance_weight,
            prefer_recommendable=prefer_recommendable,
        )

        flipped_this_iteration = [
            c
            for c in ranked_this_iteration
            if c.predicted_probability >= effective_threshold
        ]

        if flipped_this_iteration:
            logger.info(
                "Iteration %d: %d valid candidate(s), %d above effective "
                "threshold %.3f; stopping.",
                iteration, len(ranked_this_iteration),
                len(flipped_this_iteration), effective_threshold,
            )
            break

        if ranked_this_iteration:
            best_probability = max(
                c.predicted_probability for c in ranked_this_iteration
            )
            logger.info(
                "Iteration %d: %d valid near-miss candidate(s); "
                "best probability %.3f is still below %.3f. Widening pool.",
                iteration, len(ranked_this_iteration),
                best_probability, effective_threshold,
            )
        else:
            logger.info(
                "Iteration %d: no valid candidates yet; widening pool.",
                iteration,
            )

        total_cfs *= CFS_GROWTH_FACTOR

    ranked = _rank_candidates(
        valid_candidates,
        predictor_result.best_model,
        original,
        predictor_result.positive_class_label,
        loader_result=loader_result,
        scm_result=scm_result,
        causal_aware=causal_aware_ranking,
        causal_distance_weight=causal_distance_weight,
        prefer_recommendable=prefer_recommendable,
    )

    flipped = [
        c for c in ranked if c.predicted_probability >= effective_threshold
    ]
    success = len(flipped) > 0

    if success:
        logger.info(
            "Search succeeded: %d candidate(s) at or above %.3f after %d "
            "iteration(s); rejections=%s.",
            len(flipped), effective_threshold, iterations_used,
            rejection_counts,
        )
    else:
        logger.warning(
            "Search failed: no candidate reached %.3f after %d "
            "iteration(s); rejections=%s.",
            effective_threshold, iterations_used, rejection_counts,
        )

    return CounterfactualResult(
        original_features=original,
        original_probability=original_probability,
        candidates=(flipped if success else ranked),
        iterations_used=iterations_used,
        scm_rejected_count=int(
            rejection_counts["scm"] + rejection_counts["realism"]
        ),
        search_feature_set=search_features,
        success=success,
        rejection_counts=rejection_counts,
        validated_candidate_changes=[
            dict(changes) for changes in valid_candidates
        ],
        effective_threshold=effective_threshold,
        target_margin_used=float(target_margin),
    )


def print_counterfactual_summary(
    result: CounterfactualResult,
    loader_result: Optional[DataLoaderResult] = None,
) -> None:
    print(f"Search feature set: {result.search_feature_set}")
    print(f"Iterations used: {result.iterations_used}")
    print(f"Rejected candidates: {result.scm_rejected_count}")
    if result.rejection_counts:
        print(f"Rejection breakdown: {result.rejection_counts}")
    print(f"Search succeeded: {result.success}")
    print(
        "Ranking mode: "
        f"{'SCM distance-aware' if result.candidates and not np.isnan(result.candidates[0].causal_distance) else 'proximity + sparsity'}"
    )
    print(
        "Original predicted probability of favorable outcome: "
        f"{result.original_probability:.3f}"
    )
    if not np.isnan(result.effective_threshold):
        print(
            f"Effective success threshold: {result.effective_threshold:.3f} "
            f"(margin {result.target_margin_used:.3f} above 0.500)"
        )
    if not result.success and result.candidates:
        best_near_miss = max(
            c.predicted_probability for c in result.candidates
        )
        print(f"Best validated near-miss probability: {best_near_miss:.3f}")

    print(f"\nTop candidates ({len(result.candidates)}):")

    for i, candidate in enumerate(result.candidates[:5], start=1):
        causal_text = (
            f", graph_distance={candidate.causal_distance:.2f}"
            if not np.isnan(candidate.causal_distance)
            else ""
        )
        rec_text = "" if candidate.recommendable else " [non-recommendable]"
        print(
            f"\n  Candidate {i}{rec_text} "
            f"(score={candidate.feasibility_score:.3f}"
            f", base={candidate.base_feasibility_score:.3f}"
            f"{causal_text}, "
            f"new probability={candidate.predicted_probability:.3f}):"
        )
        for feature, (old, new) in candidate.changed_features.items():
            if loader_result and feature in loader_result.label_encoders:
                encoder = loader_result.label_encoders[feature]
                old_label = encoder.inverse_transform([int(round(old))])[0]
                new_label = encoder.inverse_transform([int(round(new))])[0]
                print(f"    {feature}: {old_label} -> {new_label}")
            elif feature == "education_num":
                print(
                    f"    education_num (years): "
                    f"{int(round(old))} -> {int(round(new))}"
                )
            else:
                print(f"    {feature}: {old:.2f} -> {new:.2f}")


if __name__ == "__main__":
    from loader import load_adult_income_data
    from predictor import train_and_select_model
    from causal_scm import discover_causal_structure

    loader_result = load_adult_income_data()
    predictor_result = train_and_select_model(loader_result)
    scm_result = discover_causal_structure(loader_result)

    cf_result = generate_counterfactuals(
        predictor_result, scm_result, loader_result
    )
    print_counterfactual_summary(cf_result, loader_result)