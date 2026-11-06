"""
loader.py
Module 1 of 8: Data Loading

Design notes
------------
- 'education' is dropped in favor of 'education_num'.
- Two feature tiers:
    SEARCHABLE_FEATURES    - what the counterfactual search may vary
    RECOMMENDABLE_FEATURES - what may appear in a final action plan
- NUMERIC_CHANGE_LIMITS carries:
    max_delta - maximum single-step change for the feature
    min/max_value - allowed range for the feature
    direction - 'increase' or 'decrease', a hard constraint on the sign
  capital_gain and capital_loss now have delta caps, so the search
  cannot propose a $97,000 capital-gain jump. This keeps the ranking
  meaningful and the resulting advice realistic.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("loader")

UCI_ADULT_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    "income",
]

DROPPED_FEATURES = {"education"}

MUTABLE_FEATURES = [
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

SEARCHABLE_FEATURES = [
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

RECOMMENDABLE_FEATURES = [
    "education_num",
    "occupation",
    "workclass",
    "hours_per_week",
]

NUMERIC_CHANGE_LIMITS = {
    "education_num": {
        "max_delta": 3.0,
        "min_value": 3.0,
        "max_value": 16.0,
        "direction": "increase",
    },
    "hours_per_week": {
        "max_delta": 12.0,
        "min_value": 20.0,
        "max_value": 60.0,
        "direction": "increase",
    },
    "capital_gain": {
        "max_delta": 5000.0,
        "direction": "increase",
    },
    "capital_loss": {
        "max_delta": 1000.0,
        "direction": "decrease",
    },
}

DISALLOWED_CATEGORICAL_VALUES = {
    "workclass": {"Without-pay", "Never-worked"},
}

FEATURE_DISPLAY_NAMES = {
    "education_num": "years of education",
    "occupation": "occupation",
    "workclass": "employment type",
    "hours_per_week": "hours worked per week",
    "capital_gain": "capital gain",
    "capital_loss": "capital loss",
    "marital_status": "marital status",
}

RANDOM_SEED_DEFAULT = int(CONFIG.data.random_state)


@dataclass
class DataLoaderResult:
    """Container returned by load_adult_income_data."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    full_df: pd.DataFrame
    feature_names: List[str]
    mutable_features: List[str]
    immutable_features: List[str]
    label_encoders: Dict[str, LabelEncoder] = field(default_factory=dict)
    source: str = "unknown"

    @property
    def searchable_features(self) -> List[str]:
        return [f for f in SEARCHABLE_FEATURES if f in self.feature_names]

    @property
    def recommendable_features(self) -> List[str]:
        return [f for f in RECOMMENDABLE_FEATURES if f in self.feature_names]

    def education_label_for_years(self, years: float) -> Optional[str]:
        encoder = self.label_encoders.get("education")
        if encoder is None or "education" not in self.full_df.columns:
            return None
        grouped = (
            self.full_df.groupby("education")["education_num"]
            .median().round().astype(int)
        )
        if grouped.empty:
            return None
        best_code = min(
            grouped.index,
            key=lambda code: abs(int(grouped[code]) - int(round(years))),
        )
        return str(encoder.inverse_transform([int(best_code)])[0])


def _try_download_uci_adult(timeout_seconds: int = 10) -> Optional[pd.DataFrame]:
    try:
        logger.info("Attempting to download UCI Adult Income dataset...")
        request = urllib.request.Request(
            UCI_ADULT_DATA_URL,
            headers={"User-Agent": "xai-framework-loader/1.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_bytes = response.read()
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            names=COLUMN_NAMES,
            sep=r",\s*",
            engine="python",
            na_values="?",
        )
        df = df.dropna().reset_index(drop=True)
        logger.info("Downloaded UCI dataset: %d rows.", len(df))
        return df
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        logger.warning(
            "UCI download failed (%s). Falling back to synthetic data.", exc
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "UCI dataset parsing failed (%s). Falling back to synthetic data.", exc
        )
        return None


def _generate_synthetic_adult_data(
    n_rows: int = 6000, random_state: int = RANDOM_SEED_DEFAULT
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 75, size=n_rows).astype(np.float64)
    education_num = rng.integers(3, 17, size=n_rows).astype(np.float64)
    hours_per_week = rng.integers(10, 80, size=n_rows).astype(np.float64)
    capital_gain = np.where(
        rng.random(n_rows) < 0.1, rng.integers(0, 20000, size=n_rows), 0
    ).astype(np.float64)
    capital_loss = np.where(
        rng.random(n_rows) < 0.05, rng.integers(0, 3000, size=n_rows), 0
    ).astype(np.float64)

    workclass_options = [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay",
    ]
    workclass = rng.choice(
        workclass_options, size=n_rows,
        p=[0.55, 0.10, 0.05, 0.08, 0.10, 0.10, 0.02],
    )

    edu_bins = np.digitize(education_num, bins=[6, 9, 13, 15, 16])
    edu_bin_to_label = {
        0: "9th", 1: "11th", 2: "HS-grad",
        3: "Some-college", 4: "Bachelors", 5: "Doctorate",
    }
    education = np.array([edu_bin_to_label[b] for b in edu_bins])

    marital_status_options = [
        "Married-civ-spouse", "Never-married", "Divorced", "Widowed", "Separated",
    ]
    marital_status = rng.choice(
        marital_status_options, size=n_rows,
        p=[0.45, 0.32, 0.15, 0.05, 0.03],
    )

    occupation_options = [
        "Exec-managerial", "Prof-specialty", "Tech-support", "Sales",
        "Craft-repair", "Adm-clerical", "Machine-op-inspct", "Other-service",
    ]
    occupation = rng.choice(occupation_options, size=n_rows)

    relationship_options = [
        "Husband", "Wife", "Own-child", "Not-in-family", "Unmarried",
    ]
    relationship = rng.choice(relationship_options, size=n_rows)

    race_options = [
        "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other",
    ]
    race = rng.choice(race_options, size=n_rows, p=[0.75, 0.10, 0.08, 0.04, 0.03])
    sex = rng.choice(["Male", "Female"], size=n_rows)
    native_country = rng.choice(
        ["United-States", "Mexico", "India", "Philippines", "Other"],
        size=n_rows, p=[0.85, 0.04, 0.03, 0.03, 0.05],
    )
    fnlwgt = rng.integers(20000, 300000, size=n_rows).astype(np.float64)

    occupation_premium = np.where(
        np.isin(occupation, ["Exec-managerial", "Prof-specialty"]), 1.2, 0.0
    )
    age_effect = -0.01 * np.abs(age - 45)

    z = (
        0.28 * (education_num - 9)
        + 0.045 * (hours_per_week - 40)
        + 0.00012 * capital_gain
        - 0.0004 * capital_loss
        + occupation_premium
        + age_effect
        - 1.7
    )
    probability_high_income = 1.0 / (1.0 + np.exp(-z))
    noise = rng.normal(0, 0.05, size=n_rows)
    income_label = np.where(probability_high_income + noise > 0.5, ">50K", "<=50K")

    df = pd.DataFrame({
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country,
        "income": income_label,
    })
    logger.info(
        "Generated synthetic dataset: %d rows, positive rate=%.3f",
        len(df), (df["income"] == ">50K").mean(),
    )
    return df


def _encode_and_split(
    df: pd.DataFrame, test_size: float, random_state: int
) -> DataLoaderResult:
    df = df.copy()

    numeric_columns = [c for c in df.columns if c not in CATEGORICAL_COLUMNS]
    for col in numeric_columns:
        df[col] = df[col].astype(np.float64)

    label_encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        df[col] = df[col].astype(np.float64)
        label_encoders[col] = encoder

    feature_columns = [
        c for c in df.columns
        if c != "income" and c not in DROPPED_FEATURES
    ]
    X = df[feature_columns]
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    immutable_features = [c for c in feature_columns if c not in MUTABLE_FEATURES]

    return DataLoaderResult(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        full_df=df,
        feature_names=feature_columns,
        mutable_features=[c for c in MUTABLE_FEATURES if c in feature_columns],
        immutable_features=immutable_features,
        label_encoders=label_encoders,
    )


def load_adult_income_data(
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    force_synthetic: Optional[bool] = None,
) -> DataLoaderResult:
    if test_size is None:
        test_size = float(CONFIG.data.test_size)
    if random_state is None:
        random_state = int(CONFIG.data.random_state)
    if force_synthetic is None:
        force_synthetic = bool(CONFIG.data.force_synthetic)

    df = None
    source = "synthetic"

    if not force_synthetic:
        df = _try_download_uci_adult()
        if df is not None:
            source = "uci"

    if df is None:
        df = _generate_synthetic_adult_data(random_state=random_state)

    result = _encode_and_split(df, test_size=test_size, random_state=random_state)
    result.source = source
    logger.info(
        "Data ready. source=%s train_rows=%d test_rows=%d "
        "mutable_features=%d immutable_features=%d",
        result.source, len(result.X_train), len(result.X_test),
        len(result.mutable_features), len(result.immutable_features),
    )
    return result


if __name__ == "__main__":
    result = load_adult_income_data()
    print(f"Source: {result.source}")
    print(f"Train shape: {result.X_train.shape}, Test shape: {result.X_test.shape}")
    print(f"Feature names ({len(result.feature_names)}): {result.feature_names}")
    print(f"Mutable features ({len(result.mutable_features)}): {result.mutable_features}")
    print(f"Searchable features ({len(result.searchable_features)}): {result.searchable_features}")
    print(f"Recommendable features ({len(result.recommendable_features)}): {result.recommendable_features}")
    print(f"Immutable features ({len(result.immutable_features)}): {result.immutable_features}")
    print()
    print(result.full_df.head())