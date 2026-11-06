"""
rag_retrieval.py
Module 5 of 8: RAG Retrieval (O*NET API + FAISS Semantic Search)

Design notes
------------
- Retrieval only runs for features in RECOMMENDABLE_FEATURES. A
  counterfactual may include non-recommendable changes (marital_status,
  capital_loss) that are valid for the model but inappropriate for a
  final action plan. Those features are silently skipped here so they
  never reach the LLM prompt or the user-facing output.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import faiss
import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import CONFIG
from counterfactual import CounterfactualCandidate
from loader import DataLoaderResult


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("rag_retrieval")


ONET_BASE_URL = str(CONFIG.rag.onet_base_url)
ONET_TIMEOUT_SECONDS = int(CONFIG.rag.onet_timeout_seconds)
DEFAULT_RESOURCES_PATH = str(CONFIG.rag.resources_path)
DEFAULT_TOP_K = int(CONFIG.rag.default_top_k)
DEFAULT_MIN_SIMILARITY = float(CONFIG.rag.min_similarity)
SENTENCE_TRANSFORMER_MODEL = str(CONFIG.rag.sentence_transformer_model)


@dataclass
class ResourceMatch:
    """A single retrieved resource with its similarity score."""

    resource_id: str
    title: str
    provider: str
    resource_type: str
    description: str
    url: str
    similarity_score: float


@dataclass
class OccupationDetails:
    """Live occupation data retrieved from the O*NET API."""

    onet_code: str
    title: str
    description: str
    bright_outlook: bool


@dataclass
class FeatureRetrieval:
    """Retrieval results for a single changed feature."""

    feature: str
    query_text: str
    resources: List[ResourceMatch]
    occupation_details: Optional[OccupationDetails] = None


@dataclass
class RAGRetrievalResult:
    """Container returned by retrieve_resources_for_candidate."""

    candidate: CounterfactualCandidate
    feature_retrievals: List[FeatureRetrieval] = field(default_factory=list)


class _TfidfEmbeddingBackend:
    """Fallback embedding backend using TF-IDF."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "TfidfEmbeddingBackend.fit() must be called before encode()."
            )
        vectors = self._vectorizer.transform(texts).toarray()
        return vectors.astype(np.float32)


class _SentenceTransformerBackend:
    """Primary embedding backend using sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer

        if model_name is None:
            model_name = SENTENCE_TRANSFORMER_MODEL
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]) -> None:
        pass

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts), dtype=np.float32)


def _build_embedding_backend(corpus: List[str]):
    """Try sentence-transformers first; fall back to TF-IDF."""
    try:
        backend = _SentenceTransformerBackend(
            model_name=SENTENCE_TRANSFORMER_MODEL
        )
        backend.fit(corpus)
        logger.info("Using sentence-transformers embedding backend.")
        return backend, "sentence-transformers"
    except Exception as exc:
        logger.warning(
            "sentence-transformers unavailable (%s). "
            "Falling back to TF-IDF embeddings.",
            exc,
        )
        backend = _TfidfEmbeddingBackend()
        backend.fit(corpus)
        return backend, "tfidf"


@dataclass
class ResourceIndex:
    """A FAISS index over the curated resource set."""

    resources: List[dict]
    faiss_index: faiss.Index
    embedding_backend: object
    backend_name: str
    vectors: np.ndarray

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        applies_to_feature: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> List[ResourceMatch]:
        """Search semantically, preferring resources tagged for the feature."""
        if min_similarity is None:
            min_similarity = DEFAULT_MIN_SIMILARITY

        if top_k <= 0 or not self.resources:
            return []

        query_vector = _l2_normalize(
            self.embedding_backend.encode([query])
        )[0]

        candidate_indices: Optional[List[int]] = None
        if applies_to_feature:
            tagged = [
                i
                for i, resource in enumerate(self.resources)
                if applies_to_feature in resource.get("applies_to", [])
            ]
            if tagged:
                candidate_indices = tagged

        if candidate_indices is None:
            candidate_indices = list(range(len(self.resources)))

        subset_vectors = self.vectors[candidate_indices]
        similarities = subset_vectors @ query_vector
        order = np.argsort(-similarities)

        matches: List[ResourceMatch] = []
        for local_idx in order[:top_k]:
            idx = candidate_indices[int(local_idx)]
            score = float(similarities[int(local_idx)])
            if score < min_similarity:
                continue
            resource = self.resources[idx]
            matches.append(
                ResourceMatch(
                    resource_id=resource["id"],
                    title=resource["title"],
                    provider=resource["provider"],
                    resource_type=resource["type"],
                    description=resource["description"],
                    url=resource["url"],
                    similarity_score=score,
                )
            )
        return matches


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize rows to unit L2 norm."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


@lru_cache(maxsize=2)
def build_resource_index(
    resources_path: str = DEFAULT_RESOURCES_PATH,
) -> ResourceIndex:
    """Load resources.json and build a FAISS semantic-search index."""
    with open(resources_path, "r", encoding="utf-8") as f:
        resources = json.load(f)

    corpus = [
        f"{resource['title']}. {resource['description']}"
        for resource in resources
    ]

    backend, backend_name = _build_embedding_backend(corpus)
    vectors = backend.encode(corpus)
    vectors = _l2_normalize(vectors)
    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    logger.info(
        "Resource index ready. backend=%s resources=%d dimension=%d",
        backend_name,
        len(resources),
        dimension,
    )

    return ResourceIndex(
        resources=resources,
        faiss_index=index,
        embedding_backend=backend,
        backend_name=backend_name,
        vectors=vectors,
    )


class ONetClient:
    """Thin wrapper over the O*NET API v2."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("ONET_API_KEY")
        self._enabled = bool(self.api_key)

        if not self._enabled:
            logger.info(
                "ONET_API_KEY not set; O*NET lookups will be skipped. "
                "Set ONET_API_KEY to enable them."
            )

    def lookup_occupation(
        self,
        occupation_title: str,
    ) -> Optional[OccupationDetails]:
        """Search O*NET for an occupation and retrieve occupation details."""
        if not self._enabled:
            return None

        headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
        }

        try:
            search_response = requests.get(
                f"{ONET_BASE_URL}/online/search",
                params={"keyword": occupation_title, "start": 1, "end": 1},
                headers=headers,
                timeout=ONET_TIMEOUT_SECONDS,
            )
            search_response.raise_for_status()

            results = search_response.json().get("occupation", [])
            if not results:
                logger.warning(
                    "O*NET search returned no results for '%s'.",
                    occupation_title,
                )
                return None

            top_match = results[0]
            onet_code = top_match["code"]

            occupation_response = requests.get(
                f"{ONET_BASE_URL}/online/occupations/{onet_code}/",
                headers=headers,
                timeout=ONET_TIMEOUT_SECONDS,
            )
            occupation_response.raise_for_status()
            occupation_data = occupation_response.json()

            title = occupation_data.get(
                "title", top_match.get("title", occupation_title)
            )
            description = occupation_data.get("description", "")
            tags = occupation_data.get("tags", {})
            bright_outlook = bool(tags.get("bright_outlook", False))

            return OccupationDetails(
                onet_code=onet_code,
                title=title,
                description=description,
                bright_outlook=bright_outlook,
            )

        except requests.exceptions.RequestException as exc:
            logger.warning(
                "O*NET lookup failed for '%s': %s", occupation_title, exc
            )
            return None
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "O*NET response parsing failed for '%s': %s",
                occupation_title,
                exc,
            )
            return None


def _decode_value(
    feature: str,
    value,
    loader_result: DataLoaderResult,
) -> str:
    """Decode label-encoded categorical values for human-readable queries."""
    if feature in loader_result.label_encoders:
        encoder = loader_result.label_encoders[feature]
        try:
            return str(
                encoder.inverse_transform([int(round(value))])[0]
            )
        except (TypeError, ValueError, IndexError):
            pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.1f}"
    return str(value)


def _build_query_text(
    feature: str,
    old_value,
    new_value,
    loader_result: DataLoaderResult,
) -> str:
    """Build a feature-specific retrieval query with explicit action semantics."""
    old_text = _decode_value(feature, old_value, loader_result)
    new_text = _decode_value(feature, new_value, loader_result)

    if feature == "education_num":
        old_years = int(round(float(old_value)))
        new_years = int(round(float(new_value)))
        old_label = loader_result.education_label_for_years(float(old_value))
        new_label = loader_result.education_label_for_years(float(new_value))
        label_text = ""
        if old_label and new_label:
            label_text = (
                f" (roughly {old_label} to {new_label})"
            )
        return (
            f"practical training, education, certification, and "
            f"apprenticeship options to increase years of education "
            f"from {old_years} to {new_years}{label_text}"
        )
    if feature == "occupation":
        return (
            f"career exploration and job resources for transitioning from "
            f"{old_text} toward {new_text}; required skills, training, and jobs"
        )
    if feature == "workclass":
        return (
            f"legitimate employment options and job-search resources for moving "
            f"from {old_text} toward {new_text}; self-employment and employment planning"
        )
    if feature == "hours_per_week":
        direction = "increase" if float(new_value) > float(old_value) else "change"
        return (
            f"practical ways to {direction} paid working hours from {old_text} "
            f"to {new_text}, including part-time, freelance, or additional work"
        )

    return (
        f"practical resources related to changing {feature} "
        f"from {old_text} to {new_text}"
    )


def retrieve_resources_for_candidate(
    candidate: CounterfactualCandidate,
    loader_result: DataLoaderResult,
    resource_index: ResourceIndex,
    onet_client: ONetClient,
    top_k: int = DEFAULT_TOP_K,
) -> RAGRetrievalResult:
    """Retrieve resources for each recommendable feature change.

    Non-recommendable features (marital_status, capital_loss) are
    skipped: they may be valid counterfactual changes for the model but
    they do not belong in a user-facing career action plan.
    """
    recommendable = set(loader_result.recommendable_features)
    feature_retrievals: List[FeatureRetrieval] = []

    for feature, (old_value, new_value) in candidate.changed_features.items():
        if feature not in recommendable:
            logger.info(
                "Skipping retrieval for non-recommendable feature '%s'.",
                feature,
            )
            continue

        query_text = _build_query_text(
            feature, old_value, new_value, loader_result
        )

        resources = resource_index.search(
            query_text,
            top_k=top_k,
            applies_to_feature=feature,
            min_similarity=DEFAULT_MIN_SIMILARITY,
        )

        occupation_details = None
        if (
            feature == "occupation"
            and "occupation" in loader_result.label_encoders
        ):
            encoder = loader_result.label_encoders["occupation"]
            new_occupation_label = encoder.inverse_transform(
                [int(round(new_value))]
            )[0]
            occupation_details = onet_client.lookup_occupation(
                new_occupation_label
            )

        feature_retrievals.append(
            FeatureRetrieval(
                feature=feature,
                query_text=query_text,
                resources=resources,
                occupation_details=occupation_details,
            )
        )

    logger.info(
        "RAG complete: retrieved_features=%s backend=%s",
        [fr.feature for fr in feature_retrievals],
        resource_index.backend_name,
    )

    return RAGRetrievalResult(
        candidate=candidate,
        feature_retrievals=feature_retrievals,
    )


def print_retrieval_summary(result: RAGRetrievalResult) -> None:
    """Print a readable summary of RAG retrieval results."""
    for feature_retrieval in result.feature_retrievals:
        print(f"\nFeature change: {feature_retrieval.query_text}")

        if feature_retrieval.occupation_details:
            occupation = feature_retrieval.occupation_details
            print(
                f"  O*NET: {occupation.title} ({occupation.onet_code})"
            )
            print(f"    bright_outlook={occupation.bright_outlook}")
            if occupation.description:
                print(f"    description: {occupation.description}")

        for resource in feature_retrieval.resources:
            print(
                f"  [{resource.similarity_score:.3f}] "
                f"{resource.title} "
                f"({resource.provider}) "
                f"- {resource.url}"
            )


if __name__ == "__main__":
    from causal_scm import discover_causal_structure
    from counterfactual import generate_counterfactuals
    from loader import load_adult_income_data
    from predictor import train_and_select_model

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL MODULE TEST")
    print("=" * 70)

    loader_result = load_adult_income_data()
    predictor_result = train_and_select_model(loader_result)
    scm_result = discover_causal_structure(loader_result)
    cf_result = generate_counterfactuals(
        predictor_result, scm_result, loader_result
    )

    resource_index = build_resource_index()
    onet_client = ONetClient()

    if cf_result.candidates:
        rag_result = retrieve_resources_for_candidate(
            cf_result.candidates[0],
            loader_result,
            resource_index,
            onet_client,
        )
        print_retrieval_summary(rag_result)
    else:
        print("No counterfactual candidates available.")