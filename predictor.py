"""
predictor.py
Module 2 of 8: Model Training and Selection

Purpose
-------
Trains three candidate classifiers and selects the best one by test AUC.
Exposes an auto-selected boundary test case for the counterfactual module.

Design notes
------------
- "Best" is selection by AUC on the held-out test split, not accuracy.
- The auto-selected test case must have a predicted probability that is
  at least 'boundary_exclusion_margin' below the decision threshold
  (default 0.05) and at least 'boundary_min_probability' above zero
  (default 0.30). This avoids selecting rows that sit exactly on the
  decision threshold -- which produces degenerate counterfactuals such
  as "work one hour less" -- while also avoiding denied cases so deep
  that no small actionable change could flip them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import CONFIG
from loader import DataLoaderResult, RANDOM_SEED_DEFAULT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("predictor")


@dataclass
class ModelScore:
    """AUC/accuracy for a single candidate model."""

    name: str
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class PredictorResult:
    """Container returned by train_and_select_model."""

    best_model: object
    best_model_name: str
    scoreboard: List[ModelScore]
    positive_class_label: int
    test_case_index: int
    test_case_features: pd.Series
    test_case_probability: float
    trained_models: Dict[str, object] = field(default_factory=dict)


def _build_candidates(random_state: int) -> Dict[str, object]:
    """Instantiate the three candidate models using config hyperparameters.

    n_jobs is set to 1 and XGBoost uses tree_method="exact" so that the
    models are bit-for-bit identical across runs given the same seed.
    With n_jobs=-1 (the previous setting), parallel tree construction
    introduced non-determinism: the same seed produced slightly different
    models, which in turn changed the denied-case probabilities, which in
    turn changed the random test-case sample, which in turn changed the
    reported success rate. Reproducibility of the reported numbers depends
    on this being single-threaded.
    """
    lr_cfg = CONFIG.predictor.logistic_regression
    rf_cfg = CONFIG.predictor.random_forest
    xgb_cfg = CONFIG.predictor.xgboost

    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=int(lr_cfg.max_iter),
                random_state=random_state,
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=int(rf_cfg.n_estimators),
            max_depth=int(rf_cfg.max_depth),
            random_state=random_state,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=int(xgb_cfg.n_estimators),
            max_depth=int(xgb_cfg.max_depth),
            learning_rate=float(xgb_cfg.learning_rate),
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=1,
            tree_method="exact",
        ),
    }

def _find_boundary_case(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    positive_class_label: int,
    decision_threshold: Optional[float] = None,
    exclusion_margin: Optional[float] = None,
    min_probability: Optional[float] = None,
) -> tuple[int, float]:
    """Select a denied case in the "informative" band below the boundary.

    The selection targets rows with probability in
    [min_probability, decision_threshold - exclusion_margin). Within that
    band, the row closest to the boundary is preferred, because it is
    the most likely to have a small valid counterfactual. Rows exactly
    on the threshold are excluded to avoid degenerate one-unit flips.

    Returns (row_index_within_X_test, predicted_probability).
    Raises ValueError if no row falls in the target band.
    """
    if decision_threshold is None:
        decision_threshold = float(CONFIG.predictor.decision_threshold)
    if exclusion_margin is None:
        exclusion_margin = float(
            CONFIG.predictor.boundary_exclusion_margin
        )
    if min_probability is None:
        min_probability = float(CONFIG.predictor.boundary_min_probability)

    probabilities = model.predict_proba(X_test)[:, positive_class_label]

    upper = decision_threshold - exclusion_margin
    lower = min_probability

    in_band = np.where(
        (probabilities >= lower) & (probabilities < upper)
    )[0]

    if len(in_band) == 0:
        # Fallback: relax the lower bound to 0 and try again.
        in_band = np.where(probabilities < upper)[0]
        if len(in_band) == 0:
            raise ValueError(
                "No denied test case below the boundary band "
                f"[{lower:.2f}, {upper:.2f}); cannot select a case."
            )
        logger.warning(
            "No boundary case in [%.2f, %.2f); falling back to "
            "highest-probability denied row below %.2f.",
            lower, upper, upper,
        )

    best_row = int(in_band[np.argmax(probabilities[in_band])])
    return best_row, float(probabilities[best_row])


def train_and_select_model(
    loader_result: DataLoaderResult,
    random_state: Optional[int] = None,
) -> PredictorResult:
    """Train LR, RF, and XGBoost; select the best by test AUC; pick a boundary test case."""
    if random_state is None:
        random_state = int(CONFIG.predictor.random_state)

    X_train, y_train = loader_result.X_train, loader_result.y_train
    X_test, y_test = loader_result.X_test, loader_result.y_test

    income_encoder = loader_result.label_encoders["income"]
    positive_class_label = int(np.where(income_encoder.classes_ == ">50K")[0][0])

    candidates = _build_candidates(random_state)
    trained_models: Dict[str, object] = {}
    scoreboard: List[ModelScore] = []

    for name, model in candidates.items():
        logger.info("Training %s...", name)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, positive_class_label]
        predictions = model.predict(X_test)

        auc = roc_auc_score(y_test, probabilities)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, pos_label=positive_class_label, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, pos_label=positive_class_label, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, pos_label=positive_class_label, zero_division=0
        )

        trained_models[name] = model
        scoreboard.append(
            ModelScore(
                name=name,
                auc=auc,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
        logger.info(
            "%s: AUC=%.3f Accuracy=%.3f Precision=%.3f Recall=%.3f F1=%.3f",
            name, auc, accuracy, precision, recall, f1,
        )

    scoreboard.sort(key=lambda s: s.auc, reverse=True)
    best_name = scoreboard[0].name
    best_model = trained_models[best_name]
    logger.info("Best model selected: %s (AUC=%.3f)", best_name, scoreboard[0].auc)

    test_row_idx, test_probability = _find_boundary_case(
        best_model, X_test, y_test, positive_class_label
    )
    test_case_features = X_test.iloc[test_row_idx]

    logger.info(
        "Boundary test case selected: row=%d predicted_prob_high_income=%.3f",
        test_row_idx, test_probability,
    )

    return PredictorResult(
        best_model=best_model,
        best_model_name=best_name,
        scoreboard=scoreboard,
        positive_class_label=positive_class_label,
        test_case_index=test_row_idx,
        test_case_features=test_case_features,
        test_case_probability=test_probability,
        trained_models=trained_models,
    )


def print_scoreboard(result: PredictorResult) -> None:
    """Print a ranked comparison table of the candidate models."""
    print(
        f"{'Model':<20}{'AUC':<10}{'Accuracy':<10}"
        f"{'Precision':<11}{'Recall':<10}{'F1':<10}"
    )
    print("-" * 71)
    for score in result.scoreboard:
        marker = " <- best" if score.name == result.best_model_name else ""
        print(
            f"{score.name:<20}{score.auc:<10.3f}{score.accuracy:<10.3f}"
            f"{score.precision:<11.3f}{score.recall:<10.3f}{score.f1:<10.3f}{marker}"
        )


if __name__ == "__main__":
    from loader import load_adult_income_data

    loader_result = load_adult_income_data()
    predictor_result = train_and_select_model(loader_result)

    print(f"\nData source: {loader_result.source}")
    print_scoreboard(predictor_result)
    print(f"\nBest model: {predictor_result.best_model_name}")
    print(f"Selected boundary test case (row {predictor_result.test_case_index}):")
    print(f"Predicted probability of >50K: {predictor_result.test_case_probability:.3f}")
    print(predictor_result.test_case_features)