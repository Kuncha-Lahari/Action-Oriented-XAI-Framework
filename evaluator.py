"""
evaluator.py
Module 7: Evaluation

Includes:
- CF quality (main pipeline only)
- SCM ranking ablation (proximity-only vs SCM-aware)
- Four-method baseline comparison including post-hoc-filtered DiCE
- Recommendable success rate for all methods
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from baselines import (
    BaselineResult,
    check_constraint_violations,
    generate_dice_baseline,
    generate_post_hoc_filtered_dice,
    generate_ustun_baseline,
    generate_wachter_baseline,
)
from causal_scm import CausalSCMResult
from config import CONFIG
from counterfactual import (
    _rank_candidates,
    generate_counterfactuals,
)
from loader import DataLoaderResult, RECOMMENDABLE_FEATURES
from predictor import ModelScore, PredictorResult


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("evaluator")


DEFAULT_N_CASES = int(CONFIG.evaluation.default_n_cases)
DEFAULT_ABLATION_TOTAL_CFS = int(CONFIG.evaluation.default_ablation_total_cfs)
DEFAULT_RANDOM_SEED = int(CONFIG.evaluation.default_random_seed)
DEFAULT_COMPARISON_CASES = int(
    CONFIG.evaluation.baseline_comparison.default_n_cases
)
DEFAULT_COMPARISON_DICE_CFS = int(
    CONFIG.evaluation.baseline_comparison.dice_total_cfs
)
DEFAULT_COMPARISON_WACHTER_STEPS = int(
    CONFIG.evaluation.baseline_comparison.wachter_n_steps
)
DEFAULT_COMPARISON_USTUN_MAX = int(
    CONFIG.evaluation.baseline_comparison.ustun_max_changes
)

EVALUATION_MAX_ITERATIONS = 4
EVALUATION_INITIAL_TOTAL_CFS = 5

SelectionMode = Literal["boundary", "random"]


@dataclass
class CFCaseResult:
    case_number: int
    test_index: int
    original_features: pd.Series
    original_probability: float
    distance_to_boundary: float
    success: bool
    num_candidates: int
    best_probability: float
    probability_gain: float
    normalized_proximity_l1: float
    num_features_changed: int
    actionability_valid: bool
    selected_causal_distance: float
    changed_features: List[str] = field(default_factory=list)
    validated_candidate_changes: List[Dict] = field(default_factory=list)


@dataclass
class CFQualityMetrics:
    n_cases_evaluated: int
    n_cases_valid: int
    validity_rate: float
    avg_proximity_l1: float
    avg_sparsity: float
    avg_probability_gain: float
    median_probability_gain: float
    actionability_rate: float
    avg_distance_to_boundary: float
    selection_mode: str
    random_seed: Optional[int] = None
    causal_aware_ranking: bool = True
    avg_selected_causal_distance: float = float("nan")
    case_results: List[CFCaseResult] = field(default_factory=list)


@dataclass
class AblationResult:
    n_cases: int
    candidate_pool_cases_compared: int
    threshold_crossing_cases_compared: int
    proximity_only_avg_proximity: float
    scm_distance_aware_avg_proximity: float
    proximity_only_avg_sparsity: float
    scm_distance_aware_avg_sparsity: float
    proximity_only_avg_probability_gain: float
    scm_distance_aware_avg_probability_gain: float
    proximity_only_avg_base_score: float
    scm_distance_aware_avg_base_score: float
    proximity_only_avg_causal_distance: float
    scm_distance_aware_avg_causal_distance: float
    selected_candidate_changed_count: int
    selected_candidate_changed_rate: float


@dataclass
class MethodComparisonRecord:
    method: str
    n_cases: int
    n_successful: int
    success_rate: float
    recommendable_success_rate: float
    avg_proximity: float
    avg_sparsity: float
    avg_probability_gain: float
    median_probability_gain: float
    avg_violations_per_candidate: float
    fraction_candidates_with_any_violation: float
    violation_type_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class BaselineComparisonResult:
    n_cases: int
    selection_mode: str
    random_seed: Optional[int]
    records: List[MethodComparisonRecord] = field(default_factory=list)


@dataclass
class EvaluationReport:
    model_scoreboard: List[ModelScore]
    best_model_name: str
    cf_quality: CFQualityMetrics
    ablation: AblationResult
    baseline_comparison: Optional[BaselineComparisonResult] = None


def _select_denied_test_rows(
    predictor_result: PredictorResult,
    loader_result: DataLoaderResult,
    n_cases: int,
    selection_mode: SelectionMode = "boundary",
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> List[tuple]:
    if n_cases <= 0:
        return []

    X_test = loader_result.X_test
    threshold = float(CONFIG.predictor.decision_threshold)
    probabilities = predictor_result.best_model.predict_proba(X_test)[
        :, predictor_result.positive_class_label
    ]
    denied_indices = np.where(probabilities < threshold)[0]

    if len(denied_indices) == 0:
        logger.warning("No denied cases were found in the test set.")
        return []

    if selection_mode == "boundary":
        order = np.argsort(-probabilities[denied_indices])
        selected = denied_indices[order][:n_cases]
    elif selection_mode == "random":
        rng = np.random.default_rng(random_seed)
        sample_size = min(n_cases, len(denied_indices))
        selected = rng.choice(
            denied_indices, size=sample_size, replace=False
        )
        selected = np.sort(selected)
    else:
        raise ValueError(
            f"Unknown selection_mode={selection_mode!r}. "
            "Use 'boundary' or 'random'."
        )

    rows = []
    for idx in selected:
        probability = float(probabilities[idx])
        distance = abs(threshold - probability)
        rows.append((int(idx), X_test.iloc[idx], probability, distance))
    return rows


def _feature_range(feature: str, loader_result: DataLoaderResult) -> float:
    if feature in loader_result.label_encoders:
        num_categories = len(
            loader_result.label_encoders[feature].classes_
        )
        return max(num_categories - 1, 1)
    column = loader_result.X_train[feature]
    value_range = float(column.max() - column.min())
    return value_range if value_range > 0 else 1.0


def _normalized_proximity(
    changes: dict,
    loader_result: DataLoaderResult,
) -> float:
    if not changes:
        return float("nan")
    terms = []
    for feature, (old, new) in changes.items():
        feature_range = _feature_range(feature, loader_result)
        try:
            change = abs(float(new) - float(old))
        except (TypeError, ValueError):
            continue
        terms.append(change / feature_range)
    return float(np.mean(terms)) if terms else float("nan")


def evaluate_cf_quality(
    predictor_result: PredictorResult,
    scm_result: CausalSCMResult,
    loader_result: DataLoaderResult,
    n_cases: int = DEFAULT_N_CASES,
    selection_mode: SelectionMode = "boundary",
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_cf_iterations: int = EVALUATION_MAX_ITERATIONS,
    initial_total_cfs: int = EVALUATION_INITIAL_TOTAL_CFS,
    causal_aware_ranking: bool = True,
) -> CFQualityMetrics:
    rows = _select_denied_test_rows(
        predictor_result, loader_result, n_cases,
        selection_mode=selection_mode, random_seed=random_seed,
    )
    logger.info(
        "Evaluating counterfactual quality across %d case(s) "
        "(selection=%s)...", len(rows), selection_mode,
    )

    proximities: List[float] = []
    sparsities: List[int] = []
    distances: List[float] = []
    probability_gains: List[float] = []
    selected_causal_distances: List[float] = []
    n_valid = 0
    case_results: List[CFCaseResult] = []

    for case_number, (test_index, row, probability, distance) in enumerate(
        rows, start=1
    ):
        result = generate_counterfactuals(
            predictor_result, scm_result, loader_result,
            query_row=row, query_probability=probability,
            max_iterations=max_cf_iterations,
            initial_total_cfs=initial_total_cfs,
            causal_aware_ranking=causal_aware_ranking,
        )
        success = bool(result.success and result.candidates)
        best_probability = float("nan")
        num_candidates = len(result.candidates)
        normalized_proximity = float("nan")
        probability_gain = float("nan")
        actionability_valid = False
        num_changes = 0
        changed_features: List[str] = []

        if success:
            best = result.candidates[0]
            best_probability = float(best.predicted_probability)
            probability_gain = best_probability - probability
            actionability_valid = all(
                f in loader_result.recommendable_features
                for f in best.changed_features
            )
            normalized_proximity = _normalized_proximity(
                best.changed_features, loader_result
            )
            num_changes = int(best.num_changes)
            changed_features = sorted(best.changed_features.keys())

            if not np.isnan(normalized_proximity):
                proximities.append(normalized_proximity)
            sparsities.append(num_changes)
            probability_gains.append(probability_gain)
            if not np.isnan(best.causal_distance):
                selected_causal_distances.append(
                    float(best.causal_distance)
                )
            n_valid += 1

        distances.append(distance)
        case_results.append(
            CFCaseResult(
                case_number=case_number,
                test_index=test_index,
                original_features=row.copy(),
                original_probability=probability,
                distance_to_boundary=distance,
                success=success,
                num_candidates=num_candidates,
                best_probability=best_probability,
                probability_gain=probability_gain,
                normalized_proximity_l1=normalized_proximity,
                num_features_changed=num_changes,
                actionability_valid=actionability_valid,
                selected_causal_distance=(
                    float(best.causal_distance)
                    if (success and not np.isnan(best.causal_distance))
                    else float("nan")
                ),
                changed_features=changed_features,
                validated_candidate_changes=[
                    dict(changes)
                    for changes in result.validated_candidate_changes
                ],
            )
        )

    validity_rate = n_valid / len(rows) if rows else 0.0
    avg_proximity = float(np.mean(proximities)) if proximities else float("nan")
    avg_sparsity = float(np.mean(sparsities)) if sparsities else float("nan")
    avg_gain = float(np.mean(probability_gains)) if probability_gains else float("nan")
    median_gain = float(np.median(probability_gains)) if probability_gains else float("nan")
    actionability_rate = (
        float(
            sum(
                1 for case in case_results
                if case.success and case.actionability_valid
            ) / n_valid
        ) if n_valid else 0.0
    )
    avg_distance = float(np.mean(distances)) if distances else float("nan")

    return CFQualityMetrics(
        n_cases_evaluated=len(rows),
        n_cases_valid=n_valid,
        validity_rate=validity_rate,
        avg_proximity_l1=avg_proximity,
        avg_sparsity=avg_sparsity,
        avg_probability_gain=avg_gain,
        median_probability_gain=median_gain,
        actionability_rate=actionability_rate,
        avg_distance_to_boundary=avg_distance,
        selection_mode=selection_mode,
        random_seed=(random_seed if selection_mode == "random" else None),
        causal_aware_ranking=causal_aware_ranking,
        avg_selected_causal_distance=(
            float(np.mean(selected_causal_distances))
            if selected_causal_distances else float("nan")
        ),
        case_results=case_results,
    )


def _candidate_signature(changes: Dict[str, tuple]) -> tuple:
    return tuple(
        sorted(
            (f, round(float(o), 4), round(float(n), 4))
            for f, (o, n) in changes.items()
        )
    )


def run_ablation_study(
    predictor_result: PredictorResult,
    scm_result: CausalSCMResult,
    loader_result: DataLoaderResult,
    n_cases: int = DEFAULT_N_CASES,
    total_cfs: int = EVALUATION_INITIAL_TOTAL_CFS,
    selection_mode: SelectionMode = "boundary",
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_cf_iterations: int = EVALUATION_MAX_ITERATIONS,
    case_results: Optional[List[CFCaseResult]] = None,
) -> AblationResult:
    if case_results is None:
        rows = _select_denied_test_rows(
            predictor_result, loader_result, n_cases,
            selection_mode=selection_mode, random_seed=random_seed,
        )
        generated_cases: List[CFCaseResult] = []
        for case_number, (test_index, row, probability, distance) in enumerate(
            rows, start=1
        ):
            result = generate_counterfactuals(
                predictor_result, scm_result, loader_result,
                query_row=row, query_probability=probability,
                max_iterations=max_cf_iterations,
                initial_total_cfs=total_cfs,
                causal_aware_ranking=True,
            )
            selected = result.candidates[0] if result.candidates else None
            generated_cases.append(
                CFCaseResult(
                    case_number=case_number,
                    test_index=test_index,
                    original_features=row.copy(),
                    original_probability=probability,
                    distance_to_boundary=distance,
                    success=bool(result.success),
                    num_candidates=len(result.candidates),
                    best_probability=(
                        float(selected.predicted_probability)
                        if selected is not None else float("nan")
                    ),
                    probability_gain=(
                        float(selected.predicted_probability - probability)
                        if selected is not None else float("nan")
                    ),
                    normalized_proximity_l1=(
                        _normalized_proximity(
                            selected.changed_features, loader_result
                        )
                        if selected is not None else float("nan")
                    ),
                    num_features_changed=(
                        int(selected.num_changes)
                        if selected is not None else 0
                    ),
                    actionability_valid=(
                        all(
                            f in loader_result.recommendable_features
                            for f in selected.changed_features
                        ) if selected is not None else False
                    ),
                    selected_causal_distance=(
                        float(selected.causal_distance)
                        if selected is not None else float("nan")
                    ),
                    changed_features=(
                        sorted(selected.changed_features.keys())
                        if selected is not None else []
                    ),
                    validated_candidate_changes=[
                        dict(changes)
                        for changes in result.validated_candidate_changes
                    ],
                )
            )
        case_results = generated_cases
    else:
        case_results = list(case_results[: max(0, n_cases)])

    proximity_values: List[float] = []
    causal_proximity_values: List[float] = []
    proximity_sparsities: List[float] = []
    causal_sparsities: List[float] = []
    proximity_gains: List[float] = []
    causal_gains: List[float] = []
    proximity_base_scores: List[float] = []
    causal_base_scores: List[float] = []
    proximity_causal_distances: List[float] = []
    causal_causal_distances: List[float] = []

    candidate_pool_cases_compared = 0
    threshold_crossing_cases_compared = 0
    selected_candidate_changed_count = 0

    threshold = float(CONFIG.predictor.decision_threshold)

    for case in case_results:
        raw_changes = [
            dict(changes) for changes in case.validated_candidate_changes
        ]
        if not raw_changes:
            continue
        proximity_ranked = _rank_candidates(
            raw_changes, predictor_result.best_model, case.original_features,
            predictor_result.positive_class_label,
            loader_result=loader_result, scm_result=scm_result,
            causal_aware=False,
        )
        causal_ranked = _rank_candidates(
            raw_changes, predictor_result.best_model, case.original_features,
            predictor_result.positive_class_label,
            loader_result=loader_result, scm_result=scm_result,
            causal_aware=True,
        )
        if not proximity_ranked or not causal_ranked:
            continue
        candidate_pool_cases_compared += 1
        if any(c.predicted_probability >= threshold for c in proximity_ranked):
            threshold_crossing_cases_compared += 1
        baseline = proximity_ranked[0]
        causal = causal_ranked[0]
        if (
            _candidate_signature(baseline.changed_features)
            != _candidate_signature(causal.changed_features)
        ):
            selected_candidate_changed_count += 1
        proximity_values.append(float(baseline.proximity_l1))
        causal_proximity_values.append(float(causal.proximity_l1))
        proximity_sparsities.append(float(baseline.num_changes))
        causal_sparsities.append(float(causal.num_changes))
        probability = float(case.original_probability)
        proximity_gains.append(float(baseline.predicted_probability - probability))
        causal_gains.append(float(causal.predicted_probability - probability))
        proximity_base_scores.append(float(baseline.base_feasibility_score))
        causal_base_scores.append(float(causal.base_feasibility_score))
        if not np.isnan(baseline.causal_distance):
            proximity_causal_distances.append(float(baseline.causal_distance))
        if not np.isnan(causal.causal_distance):
            causal_causal_distances.append(float(causal.causal_distance))

    changed_rate = (
        selected_candidate_changed_count / candidate_pool_cases_compared
        if candidate_pool_cases_compared else 0.0
    )

    def _mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return AblationResult(
        n_cases=len(case_results),
        candidate_pool_cases_compared=candidate_pool_cases_compared,
        threshold_crossing_cases_compared=threshold_crossing_cases_compared,
        proximity_only_avg_proximity=_mean(proximity_values),
        scm_distance_aware_avg_proximity=_mean(causal_proximity_values),
        proximity_only_avg_sparsity=_mean(proximity_sparsities),
        scm_distance_aware_avg_sparsity=_mean(causal_sparsities),
        proximity_only_avg_probability_gain=_mean(proximity_gains),
        scm_distance_aware_avg_probability_gain=_mean(causal_gains),
        proximity_only_avg_base_score=_mean(proximity_base_scores),
        scm_distance_aware_avg_base_score=_mean(causal_base_scores),
        proximity_only_avg_causal_distance=_mean(proximity_causal_distances),
        scm_distance_aware_avg_causal_distance=_mean(causal_causal_distances),
        selected_candidate_changed_count=selected_candidate_changed_count,
        selected_candidate_changed_rate=changed_rate,
    )


def _evaluate_one_method(
    method: str,
    case_rows: List[tuple],
    predictor_result: PredictorResult,
    scm_result: CausalSCMResult,
    loader_result: DataLoaderResult,
    dice_total_cfs: int,
    wachter_n_steps: int,
    ustun_max_changes: int,
) -> MethodComparisonRecord:
    threshold = float(CONFIG.predictor.decision_threshold)
    recommendable_set = set(RECOMMENDABLE_FEATURES)

    n_cases = len(case_rows)
    n_successful = 0
    n_recommendable_successful = 0
    proximities: List[float] = []
    sparsities: List[float] = []
    gains: List[float] = []

    total_candidates = 0
    total_violations = 0
    candidates_with_violation = 0
    violation_type_counts: Dict[str, int] = {}

    for case_number, (test_index, row, probability, _) in enumerate(
        case_rows, start=1
    ):
        if method == "main_pipeline":
            result = generate_counterfactuals(
                predictor_result, scm_result, loader_result,
                query_row=row, query_probability=probability,
                max_iterations=EVALUATION_MAX_ITERATIONS,
                initial_total_cfs=EVALUATION_INITIAL_TOTAL_CFS,
            )
            candidates = result.candidates
        elif method == "vanilla_dice":
            result = generate_dice_baseline(
                predictor_result, loader_result,
                query_row=row, query_probability=probability,
                total_cfs=dice_total_cfs,
            )
            candidates = result.candidates
        elif method == "post_hoc_filtered_dice":
            result = generate_post_hoc_filtered_dice(
                predictor_result, loader_result, scm_result,
                query_row=row, query_probability=probability,
                total_cfs=dice_total_cfs,
            )
            candidates = result.candidates
        elif method == "wachter":
            result = generate_wachter_baseline(
                predictor_result, loader_result,
                query_row=row, query_probability=probability,
                n_steps=wachter_n_steps,
            )
            candidates = result.candidates
        elif method == "ustun":
            result = generate_ustun_baseline(
                predictor_result, loader_result,
                query_row=row, query_probability=probability,
                max_changes=ustun_max_changes,
            )
            candidates = result.candidates
        else:
            raise ValueError(f"Unknown method: {method}")

        crossing = [
            c for c in candidates if c.predicted_probability >= threshold
        ]
        if crossing:
            n_successful += 1
            selected = crossing[0]
            changed_set = set(selected.changed_features.keys())
            if changed_set.issubset(recommendable_set):
                n_recommendable_successful += 1
            gains.append(selected.predicted_probability - probability)
            sparsities.append(float(selected.num_changes))
            prox = _normalized_proximity(
                selected.changed_features, loader_result
            )
            if not np.isnan(prox):
                proximities.append(prox)

        for candidate in candidates:
            total_candidates += 1
            violations = check_constraint_violations(
                candidate.changed_features, loader_result,
                scm_mask=scm_result.bool_mask,
                scm_feature_names=scm_result.feature_names,
            )
            if violations:
                candidates_with_violation += 1
                total_violations += len(violations)
                for tag in violations:
                    key = tag.split(":")[0]
                    violation_type_counts[key] = (
                        violation_type_counts.get(key, 0) + 1
                    )

        logger.info(
            "Method %s case %d/%d: success=%s candidates=%d",
            method, case_number, n_cases, bool(crossing), len(candidates),
        )

    success_rate = n_successful / n_cases if n_cases else 0.0
    recommendable_rate = (
        n_recommendable_successful / n_cases if n_cases else 0.0
    )
    avg_prox = float(np.mean(proximities)) if proximities else float("nan")
    avg_spar = float(np.mean(sparsities)) if sparsities else float("nan")
    avg_gain = float(np.mean(gains)) if gains else float("nan")
    median_gain = float(np.median(gains)) if gains else float("nan")
    avg_violations_per_candidate = (
        total_violations / total_candidates if total_candidates else 0.0
    )
    fraction_with_violation = (
        candidates_with_violation / total_candidates
        if total_candidates else 0.0
    )

    return MethodComparisonRecord(
        method=method,
        n_cases=n_cases,
        n_successful=n_successful,
        success_rate=success_rate,
        recommendable_success_rate=recommendable_rate,
        avg_proximity=avg_prox,
        avg_sparsity=avg_spar,
        avg_probability_gain=avg_gain,
        median_probability_gain=median_gain,
        avg_violations_per_candidate=avg_violations_per_candidate,
        fraction_candidates_with_any_violation=fraction_with_violation,
        violation_type_counts=violation_type_counts,
    )


def run_baseline_comparison(
    predictor_result: PredictorResult,
    scm_result: CausalSCMResult,
    loader_result: DataLoaderResult,
    n_cases: int = DEFAULT_COMPARISON_CASES,
    selection_mode: SelectionMode = "boundary",
    random_seed: int = DEFAULT_RANDOM_SEED,
    dice_total_cfs: int = DEFAULT_COMPARISON_DICE_CFS,
    wachter_n_steps: int = DEFAULT_COMPARISON_WACHTER_STEPS,
    ustun_max_changes: int = DEFAULT_COMPARISON_USTUN_MAX,
) -> BaselineComparisonResult:
    case_rows = _select_denied_test_rows(
        predictor_result, loader_result, n_cases,
        selection_mode=selection_mode, random_seed=random_seed,
    )
    logger.info(
        "Running baseline comparison across %d case(s) with methods: "
        "main_pipeline, vanilla_dice, post_hoc_filtered_dice, "
        "wachter, ustun.", len(case_rows),
    )

    records = []
    for method in (
        "main_pipeline", "vanilla_dice",
        "post_hoc_filtered_dice", "wachter", "ustun",
    ):
        logger.info("--- Evaluating method: %s ---", method)
        record = _evaluate_one_method(
            method=method,
            case_rows=case_rows,
            predictor_result=predictor_result,
            scm_result=scm_result,
            loader_result=loader_result,
            dice_total_cfs=dice_total_cfs,
            wachter_n_steps=wachter_n_steps,
            ustun_max_changes=ustun_max_changes,
        )
        records.append(record)

    return BaselineComparisonResult(
        n_cases=len(case_rows),
        selection_mode=selection_mode,
        random_seed=(random_seed if selection_mode == "random" else None),
        records=records,
    )


def run_full_evaluation(
    predictor_result: PredictorResult,
    scm_result: CausalSCMResult,
    loader_result: DataLoaderResult,
    n_cases: int = DEFAULT_N_CASES,
    selection_mode: SelectionMode = "boundary",
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_cf_iterations: int = EVALUATION_MAX_ITERATIONS,
    initial_total_cfs: int = EVALUATION_INITIAL_TOTAL_CFS,
    causal_aware_ranking: bool = True,
    include_baseline_comparison: bool = False,
    comparison_n_cases: int = DEFAULT_COMPARISON_CASES,
) -> EvaluationReport:
    cf_quality = evaluate_cf_quality(
        predictor_result, scm_result, loader_result,
        n_cases=n_cases,
        selection_mode=selection_mode,
        random_seed=random_seed,
        max_cf_iterations=max_cf_iterations,
        initial_total_cfs=initial_total_cfs,
        causal_aware_ranking=causal_aware_ranking,
    )
    ablation = run_ablation_study(
        predictor_result, scm_result, loader_result,
        n_cases=n_cases,
        total_cfs=initial_total_cfs,
        selection_mode=selection_mode,
        random_seed=random_seed,
        max_cf_iterations=max_cf_iterations,
        case_results=cf_quality.case_results,
    )
    comparison = None
    if include_baseline_comparison:
        comparison = run_baseline_comparison(
            predictor_result, scm_result, loader_result,
            n_cases=comparison_n_cases,
            selection_mode=selection_mode,
            random_seed=random_seed,
        )
    return EvaluationReport(
        model_scoreboard=predictor_result.scoreboard,
        best_model_name=predictor_result.best_model_name,
        cf_quality=cf_quality,
        ablation=ablation,
        baseline_comparison=comparison,
    )


def save_case_results(
    report: EvaluationReport,
    output_path: str = "evaluation_case_results.csv",
) -> Path:
    rows = []
    for case in report.cf_quality.case_results:
        rows.append({
            "case_number": case.case_number,
            "test_index": case.test_index,
            "selection_mode": report.cf_quality.selection_mode,
            "original_probability": case.original_probability,
            "distance_to_boundary": case.distance_to_boundary,
            "success": case.success,
            "num_candidates": case.num_candidates,
            "num_validated_candidates": len(case.validated_candidate_changes),
            "best_probability": case.best_probability,
            "probability_gain": case.probability_gain,
            "normalized_proximity_l1": case.normalized_proximity_l1,
            "num_features_changed": case.num_features_changed,
            "actionability_valid": case.actionability_valid,
            "causal_aware_ranking": report.cf_quality.causal_aware_ranking,
            "selected_causal_distance": case.selected_causal_distance,
            "changed_features": ", ".join(case.changed_features),
        })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    logger.info("Saved per-case evaluation results to %s", output)
    return output


def print_evaluation_report(report: EvaluationReport) -> None:
    print("Model comparison")
    print(f"{'Model':<20}{'AUC':<10}{'Accuracy':<10}")
    print("-" * 40)
    for score in report.model_scoreboard:
        marker = " <- best" if score.name == report.best_model_name else ""
        print(
            f"{score.name:<20}{score.auc:<10.3f}"
            f"{score.accuracy:<10.3f}{marker}"
        )

    cf = report.cf_quality
    print(
        f"\nCounterfactual quality ({cf.n_cases_evaluated} cases; "
        f"selection={cf.selection_mode})"
    )
    print(
        f"  Success rate: {cf.validity_rate:.1%} "
        f"({cf.n_cases_valid}/{cf.n_cases_evaluated})"
    )
    print(f"  Avg proximity (normalized L1): {cf.avg_proximity_l1:.3f}")
    print(f"  Avg sparsity (features changed): {cf.avg_sparsity:.2f}")
    print(
        f"  Avg probability gain: {cf.avg_probability_gain:.3f} "
        f"(median={cf.median_probability_gain:.3f})"
    )
    print(f"  Actionability-valid rate: {cf.actionability_rate:.1%}")
    print(f"  Avg distance to boundary: {cf.avg_distance_to_boundary:.3f}")

    ab = report.ablation
    print(f"\nSCM distance-aware ranking ablation ({ab.n_cases} cases)")
    print(
        f"  Cases with validated candidates: "
        f"{ab.candidate_pool_cases_compared}/{ab.n_cases}"
    )
    print(
        f"  Cases with threshold-crossing candidates: "
        f"{ab.threshold_crossing_cases_compared}/{ab.n_cases}"
    )
    print(
        f"  Candidate selection changed: "
        f"{ab.selected_candidate_changed_count}/"
        f"{ab.candidate_pool_cases_compared} "
        f"({ab.selected_candidate_changed_rate:.1%})"
    )
    print("\n  Selected-candidate metrics:")
    print(f"    {'Metric':<28}{'Proximity-only':>18}{'SCM-aware':>18}")
    print("    " + "-" * 64)
    print(
        f"    {'Avg normalized proximity':<28}"
        f"{ab.proximity_only_avg_proximity:>18.3f}"
        f"{ab.scm_distance_aware_avg_proximity:>18.3f}"
    )
    print(
        f"    {'Avg sparsity':<28}"
        f"{ab.proximity_only_avg_sparsity:>18.2f}"
        f"{ab.scm_distance_aware_avg_sparsity:>18.2f}"
    )
    print(
        f"    {'Avg probability gain':<28}"
        f"{ab.proximity_only_avg_probability_gain:>18.3f}"
        f"{ab.scm_distance_aware_avg_probability_gain:>18.3f}"
    )
    print(
        f"    {'Avg base feasibility score':<28}"
        f"{ab.proximity_only_avg_base_score:>18.3f}"
        f"{ab.scm_distance_aware_avg_base_score:>18.3f}"
    )
    print(
        f"    {'Avg graph distance to income':<28}"
        f"{ab.proximity_only_avg_causal_distance:>18.2f}"
        f"{ab.scm_distance_aware_avg_causal_distance:>18.2f}"
    )

    if report.baseline_comparison is not None:
        bc = report.baseline_comparison
        print(
            f"\nBaseline comparison ({bc.n_cases} cases; "
            f"selection={bc.selection_mode})"
        )
        print(
            f"    {'Method':<24}{'Success':>10}{'Recomm':>10}{'Prox':>10}"
            f"{'Spar':>8}{'Gain':>10}{'Viol/cand':>12}{'Viol frac':>12}"
        )
        print("    " + "-" * 96)
        for rec in bc.records:
            print(
                f"    {rec.method:<24}"
                f"{rec.success_rate:>10.1%}"
                f"{rec.recommendable_success_rate:>10.1%}"
                f"{rec.avg_proximity:>10.3f}"
                f"{rec.avg_sparsity:>8.2f}"
                f"{rec.avg_probability_gain:>10.3f}"
                f"{rec.avg_violations_per_candidate:>12.2f}"
                f"{rec.fraction_candidates_with_any_violation:>12.1%}"
            )
        print("\n    Violation breakdown by type (total counts per method):")
        for rec in bc.records:
            if rec.violation_type_counts:
                print(f"      {rec.method}: {rec.violation_type_counts}")


def save_evaluation_report(
    report: EvaluationReport,
    output_path: str = "evaluation_report.json",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict = {
        "best_model_name": report.best_model_name,
        "model_scoreboard": [
            {
                "name": score.name, "auc": score.auc,
                "accuracy": score.accuracy, "precision": score.precision,
                "recall": score.recall, "f1": score.f1,
            }
            for score in report.model_scoreboard
        ],
        "counterfactual_quality": {
            "n_cases_evaluated": report.cf_quality.n_cases_evaluated,
            "n_cases_valid": report.cf_quality.n_cases_valid,
            "validity_rate": report.cf_quality.validity_rate,
            "avg_proximity_l1": report.cf_quality.avg_proximity_l1,
            "avg_sparsity": report.cf_quality.avg_sparsity,
            "avg_probability_gain": report.cf_quality.avg_probability_gain,
            "median_probability_gain": report.cf_quality.median_probability_gain,
            "actionability_rate": report.cf_quality.actionability_rate,
            "avg_distance_to_boundary": report.cf_quality.avg_distance_to_boundary,
            "selection_mode": report.cf_quality.selection_mode,
            "random_seed": report.cf_quality.random_seed,
            "causal_aware_ranking": report.cf_quality.causal_aware_ranking,
            "avg_selected_causal_distance": report.cf_quality.avg_selected_causal_distance,
        },
        "scm_ranking_ablation": {
            "n_cases": report.ablation.n_cases,
            "candidate_pool_cases_compared": report.ablation.candidate_pool_cases_compared,
            "threshold_crossing_cases_compared": report.ablation.threshold_crossing_cases_compared,
            "selected_candidate_changed_count": report.ablation.selected_candidate_changed_count,
            "selected_candidate_changed_rate": report.ablation.selected_candidate_changed_rate,
            "proximity_only_avg_proximity": report.ablation.proximity_only_avg_proximity,
            "scm_distance_aware_avg_proximity": report.ablation.scm_distance_aware_avg_proximity,
            "proximity_only_avg_sparsity": report.ablation.proximity_only_avg_sparsity,
            "scm_distance_aware_avg_sparsity": report.ablation.scm_distance_aware_avg_sparsity,
            "proximity_only_avg_probability_gain": report.ablation.proximity_only_avg_probability_gain,
            "scm_distance_aware_avg_probability_gain": report.ablation.scm_distance_aware_avg_probability_gain,
            "proximity_only_avg_base_score": report.ablation.proximity_only_avg_base_score,
            "scm_distance_aware_avg_base_score": report.ablation.scm_distance_aware_avg_base_score,
            "proximity_only_avg_causal_distance": report.ablation.proximity_only_avg_causal_distance,
            "scm_distance_aware_avg_causal_distance": report.ablation.scm_distance_aware_avg_causal_distance,
        },
    }

    if report.baseline_comparison is not None:
        bc = report.baseline_comparison
        payload["baseline_comparison"] = {
            "n_cases": bc.n_cases,
            "selection_mode": bc.selection_mode,
            "random_seed": bc.random_seed,
            "records": [
                {
                    "method": rec.method,
                    "n_cases": rec.n_cases,
                    "n_successful": rec.n_successful,
                    "success_rate": rec.success_rate,
                    "recommendable_success_rate": rec.recommendable_success_rate,
                    "avg_proximity": rec.avg_proximity,
                    "avg_sparsity": rec.avg_sparsity,
                    "avg_probability_gain": rec.avg_probability_gain,
                    "median_probability_gain": rec.median_probability_gain,
                    "avg_violations_per_candidate": rec.avg_violations_per_candidate,
                    "fraction_candidates_with_any_violation": rec.fraction_candidates_with_any_violation,
                    "violation_type_counts": rec.violation_type_counts,
                }
                for rec in bc.records
            ],
        }

    output.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Saved evaluation report to %s", output)
    return output