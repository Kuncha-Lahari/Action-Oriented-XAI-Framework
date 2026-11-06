"""
causal_scm.py
Module 3 of 8: Causal Structural Model (SCM) Discovery

Purpose
-------
Runs the PC Algorithm on the training data to discover a causal graph
between features -- including the target ('income') -- rather than
assuming or hand-coding one. The output is consumed by module 4
(counterfactual generation): a candidate counterfactual is only
considered valid if the changes it proposes are consistent with the
discovered causal structure AND the feature being changed is actually
connected to the outcome in the discovered graph.

Public interface
-----------------
discover_causal_structure(loader_result, ci_test="kci", ...)
    Runs PC discovery over features + income jointly, builds the
    adjacency matrix, intersects it with the mutable-feature mask
    from module 1 AND with graph-reachability to income, and
    returns a CausalSCMResult that module 4 can query directly.

rank_features_by_causal_priority(scm_result, target_feature)
    Uses a heapq min-heap to rank a feature's causal neighbors by
    graph distance to a target.

Design notes
------------
Data types
~~~~~~~~~~
The Adult dataset mixes discrete categoricals and continuous numerics.
Applying a single Gaussian CI test (fisherz) to label-encoded
categoricals is statistically invalid. This module therefore selects
the conditional-independence test based on the data type of the
variables involved.

Three CI-test strategies are supported via the ``ci_test`` argument:

  "kci" (default, paper-facing)
      Kernel Conditional Independence test. Handles mixed discrete/
      continuous data natively. Validated against ``chisq`` in a
      controlled comparison: both agreed on all six actionable
      features, and kci additionally recovered
      ``capital_gain -- income``. Trade-off: O(n^3) per test, so
      ``max_sample_rows`` is set conservatively (default 500).

  "chisq"
      Continuous features are quantile-discretized, then chi-square
      is applied uniformly. Fast, but prone to spurious edges
      between high-cardinality categoricals, and loses information
      from binning continuous columns.

  "fisherz"
      Naive Gaussian test on the raw (label-encoded) data. Retained
      only for backward compatibility and ablation. Not recommended.

Configuration
~~~~~~~~~~~~~
All defaults (ci_test, n_bins, max_sample_rows, random_state,
alpha_schedule) now come from config.yaml via config.CONFIG.
Explicit arguments to discover_causal_structure still override.

Methodological caveat
~~~~~~~~~~~~~~~~~~~~~
This module treats label-encoded categoricals as discrete and
continuous features as continuous-after-discretization (for chisq)
or as genuinely mixed (for kci). The resulting graph is a *structural
connectivity* graph, not an identified causal-effect graph.
"""

from __future__ import annotations

import heapq
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc

from config import CONFIG
from loader import CATEGORICAL_COLUMNS, DataLoaderResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("causal_scm")

# ---------------------------------------------------------------------------
# Configuration-backed module constants
# ---------------------------------------------------------------------------

ALPHA_RETRY_SCHEDULE = list(CONFIG.scm.alpha_schedule)
MAX_PC_SAMPLE_ROWS = int(CONFIG.scm.max_sample_rows)
TARGET_FEATURE = "income"
DEFAULT_CI_TEST = str(CONFIG.scm.ci_test)
DEFAULT_N_BINS = int(CONFIG.scm.n_bins)
SUPPORTED_CI_TESTS = ("kci", "chisq", "fisherz")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CausalSCMResult:
    """Container returned by discover_causal_structure."""

    feature_names: List[str]
    adjacency_matrix: np.ndarray
    directed_graph: nx.DiGraph
    bool_mask: np.ndarray
    mutable_only_mask: np.ndarray
    alpha_used: float
    edge_count: int
    ci_test_used: str = DEFAULT_CI_TEST
    discrete_mask: Optional[np.ndarray] = None
    topological_order: List[str] = field(default_factory=list)
    cycle_breaks: List[Tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature-type classification
# ---------------------------------------------------------------------------

def _classify_features(
    feature_names: List[str],
) -> np.ndarray:
    """Return a boolean mask: True if the feature is discrete (categorical).

    The outcome ``income`` is discrete. Every column listed in
    ``CATEGORICAL_COLUMNS`` (from loader.py) is discrete after label
    encoding. Everything else is treated as continuous.
    """
    categorical_set = set(CATEGORICAL_COLUMNS)
    discrete = np.zeros(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        if name in categorical_set or name == TARGET_FEATURE:
            discrete[i] = True
    return discrete


# ---------------------------------------------------------------------------
# Continuous-feature discretization
# ---------------------------------------------------------------------------

def _discretize_continuous(
    data: np.ndarray,
    discrete_mask: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> np.ndarray:
    """Quantile-discretize continuous columns; leave discrete columns unchanged.

    Uses equal-frequency binning so each bin contains roughly the same
    number of samples. This matters for chi-square CI tests: an
    equal-width binning of a heavy-tailed feature like capital_gain
    would put 99% of rows in one bin, making the test degenerate.
    """
    out = np.array(data, dtype=float, copy=True)
    for j in range(out.shape[1]):
        if discrete_mask[j]:
            continue
        column = out[:, j]
        if np.all(column == column[0]):
            out[:, j] = 0.0
            continue
        try:
            edges = np.quantile(
                column,
                np.linspace(0.0, 1.0, n_bins + 1)[1:-1],
            )
            edges = np.unique(edges)
            if edges.size == 0:
                out[:, j] = 0.0
            else:
                out[:, j] = np.digitize(column, edges).astype(float)
        except Exception:
            out[:, j] = 0.0
    return out


# ---------------------------------------------------------------------------
# PC Algorithm execution with retry
# ---------------------------------------------------------------------------

def _prepare_pc_input(
    data: np.ndarray,
    discrete_mask: np.ndarray,
    ci_test: str,
    n_bins: int,
) -> Tuple[np.ndarray, str]:
    """Return (data_matrix_for_pc, causal_learn_indep_test_name)."""
    if ci_test == "chisq":
        discretized = _discretize_continuous(data, discrete_mask, n_bins)
        return discretized.astype(np.float64), "chisq"
    if ci_test == "kci":
        return data.astype(np.float64), "kci"
    if ci_test == "fisherz":
        return data.astype(np.float64), "fisherz"
    raise ValueError(
        f"Unknown ci_test={ci_test!r}. "
        f"Supported: {SUPPORTED_CI_TESTS}."
    )


def _run_pc_with_retry(
    data: np.ndarray,
    feature_names: List[str],
    discrete_mask: np.ndarray,
    ci_test: str = DEFAULT_CI_TEST,
    n_bins: int = DEFAULT_N_BINS,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run PC Algorithm, retrying at looser alpha if no edges are found.

    Returns (undirected_adjacency, directed_adjacency, alpha_used).
    Both adjacency matrices are n x n numpy arrays aligned to feature_names.
    """
    pc_input, indep_test_name = _prepare_pc_input(
        data, discrete_mask, ci_test, n_bins
    )
    n = len(feature_names)

    for alpha in ALPHA_RETRY_SCHEDULE:
        logger.info(
            "Running PC Algorithm at alpha=%.2f (indep_test=%s)...",
            alpha,
            indep_test_name,
        )
        try:
            cg = pc(
                pc_input,
                alpha=alpha,
                indep_test=indep_test_name,
                show_progress=False,
            )
        except Exception as exc:
            logger.warning(
                "PC Algorithm failed at alpha=%.2f with indep_test=%s: %s",
                alpha,
                indep_test_name,
                exc,
            )
            continue

        raw_graph = cg.G.graph

        undirected = np.zeros((n, n), dtype=int)
        directed = np.zeros((n, n), dtype=int)

        # causal-learn edge marks: -1 = tail, 1 = arrowhead, 2 = circle.
        # graph[i, j] == -1 and graph[j, i] == 1 means i -> j.
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if raw_graph[i, j] != 0 or raw_graph[j, i] != 0:
                    undirected[i, j] = 1
                    undirected[j, i] = 1
                if raw_graph[i, j] == -1 and raw_graph[j, i] == 1:
                    directed[i, j] = 1

        edge_count = int(undirected.sum() / 2)
        logger.info(
            "alpha=%.2f produced %d undirected edges.",
            alpha,
            edge_count,
        )

        if edge_count > 0:
            return undirected, directed, alpha

    logger.warning(
        "PC Algorithm found 0 edges even at the loosest alpha (%.2f). "
        "Returning an empty graph; module 4 will have no causal "
        "constraints to apply.",
        ALPHA_RETRY_SCHEDULE[-1],
    )
    return (
        np.zeros((n, n), dtype=int),
        np.zeros((n, n), dtype=int),
        ALPHA_RETRY_SCHEDULE[-1],
    )


# ---------------------------------------------------------------------------
# Graph construction and ordering
# ---------------------------------------------------------------------------

def _build_directed_graph(
    directed_adjacency: np.ndarray,
    feature_names: List[str],
) -> nx.DiGraph:
    """Build a NetworkX DiGraph from the oriented edges PC discovered."""
    graph = nx.DiGraph()
    graph.add_nodes_from(feature_names)
    n = len(feature_names)
    for i in range(n):
        for j in range(n):
            if directed_adjacency[i, j] == 1:
                graph.add_edge(feature_names[i], feature_names[j])
    return graph


def _safe_topological_order(
    graph: nx.DiGraph,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Return a safe ordering plus edges removed only for ordering when cycles exist.

    The PC graph itself is never mutated for causal-validity checks. If
    the oriented subgraph is cyclic, we construct a temporary DAG used
    only for sequencing actions. Because PC does not expose edge-strength
    scores in this interface, cycle edges are broken deterministically
    and recorded in ``cycle_breaks`` so the final report remains explicit
    about the limitation.
    """
    if graph.number_of_edges() == 0:
        return list(graph.nodes), []

    dag = graph.copy()
    cycle_breaks: List[Tuple[str, str]] = []

    while not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        if not cycles:
            break
        cycle = min(
            [tuple(c) for c in cycles],
            key=lambda c: (len(c), c),
        )
        edge = (cycle[-1], cycle[0])
        if dag.has_edge(*edge):
            dag.remove_edge(*edge)
            cycle_breaks.append(edge)
        else:
            edge = (cycle[0], cycle[1])
            dag.remove_edge(*edge)
            cycle_breaks.append(edge)

    order = (
        list(nx.topological_sort(dag))
        if nx.is_directed_acyclic_graph(dag)
        else []
    )
    if cycle_breaks:
        logger.warning(
            "Directed PC graph contained cycles; removed %d edge(s) only "
            "for action ordering. The original discovered graph remains "
            "unchanged.",
            len(cycle_breaks),
        )
    return order, cycle_breaks


def _bfs_reachable(
    adjacency: np.ndarray,
    start_idx: int,
) -> Set[int]:
    """BFS the undirected adjacency matrix from start_idx.

    Returns the set of node indices reachable from start_idx via any
    path (direct or indirect), excluding start_idx itself.
    """
    visited: Set[int] = {start_idx}
    queue = deque([start_idx])
    while queue:
        node = queue.popleft()
        neighbors = np.nonzero(adjacency[node])[0]
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    visited.discard(start_idx)
    return visited


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def discover_causal_structure(
    loader_result: DataLoaderResult,
    max_sample_rows: int = MAX_PC_SAMPLE_ROWS,
    random_state: Optional[int] = None,
    ci_test: str = DEFAULT_CI_TEST,
    n_bins: int = DEFAULT_N_BINS,
) -> CausalSCMResult:
    """Discover a causal graph over training features + income using PC.

    Parameters
    ----------
    loader_result : DataLoaderResult
        Output of loader.load_adult_income_data(). X_train and y_train
        are combined into a single matrix so the target participates
        in discovery.
    max_sample_rows : int
        PC Algorithm's runtime grows quickly with row count, and kci
        grows cubically per test. Training sets larger than this are
        subsampled for the discovery step only.
    random_state : int, optional
        Seed for the subsampling step. Defaults to config.scm.random_state.
    ci_test : str
        Conditional-independence test to use. One of:
          - "kci"     (default) kernel CI test; mixed-data correct, slow.
          - "chisq"   discretize continuous, chi-square test; fast, but
                      prone to spurious categorical-categorical edges.
          - "fisherz" naive Gaussian test on raw encoded data.
                      Retained for backward compatibility only.
    n_bins : int
        Number of quantile bins used when ``ci_test="chisq"``.

    Returns
    -------
    CausalSCMResult
    """
    if ci_test not in SUPPORTED_CI_TESTS:
        raise ValueError(
            f"Unknown ci_test={ci_test!r}. Supported: {SUPPORTED_CI_TESTS}."
        )

    if random_state is None:
        random_state = int(CONFIG.scm.random_state)

    if TARGET_FEATURE in loader_result.X_train.columns:
        raise ValueError(
            f"'{TARGET_FEATURE}' should not already be a column of "
            "X_train; it is appended from y_train for discovery."
        )

    X = loader_result.X_train.copy()
    X[TARGET_FEATURE] = np.asarray(loader_result.y_train)
    feature_names = list(X.columns)

    if len(X) > max_sample_rows:
        X = X.sample(n=max_sample_rows, random_state=random_state)
        logger.info(
            "Subsampled training data to %d rows for PC Algorithm.",
            max_sample_rows,
        )

    data_matrix = X.to_numpy().astype(np.float64)

    discrete_mask = _classify_features(feature_names)
    logger.info(
        "Feature type classification: %d discrete / %d continuous.",
        int(discrete_mask.sum()),
        int((~discrete_mask).sum()),
    )

    undirected_adj, directed_adj, alpha_used = _run_pc_with_retry(
        data_matrix,
        feature_names,
        discrete_mask,
        ci_test=ci_test,
        n_bins=n_bins,
    )

    directed_graph = _build_directed_graph(directed_adj, feature_names)
    topological_order, cycle_breaks = _safe_topological_order(directed_graph)

    income_idx = feature_names.index(TARGET_FEATURE)
    reachable_to_income = _bfs_reachable(undirected_adj, income_idx)

    mutable_set = set(loader_result.mutable_features)
    bool_mask = np.zeros(len(feature_names), dtype=bool)
    mutable_only_mask = np.zeros(len(feature_names), dtype=bool)
    for idx, name in enumerate(feature_names):
        if name == TARGET_FEATURE:
            continue
        is_mutable = name in mutable_set
        mutable_only_mask[idx] = is_mutable
        is_connected_to_income = idx in reachable_to_income
        bool_mask[idx] = is_mutable and is_connected_to_income

    edge_count = int(undirected_adj.sum() / 2)

    logger.info(
        "Causal SCM ready. ci_test=%s alpha=%.2f edges=%d "
        "usable_features=%d/%d (mutable_only=%d)",
        ci_test,
        alpha_used,
        edge_count,
        int(bool_mask.sum()),
        len(feature_names),
        int(mutable_only_mask.sum()),
    )

    return CausalSCMResult(
        feature_names=feature_names,
        adjacency_matrix=undirected_adj,
        directed_graph=directed_graph,
        bool_mask=bool_mask,
        mutable_only_mask=mutable_only_mask,
        alpha_used=alpha_used,
        edge_count=edge_count,
        ci_test_used=ci_test,
        discrete_mask=discrete_mask,
        topological_order=topological_order,
        cycle_breaks=cycle_breaks,
    )


# ---------------------------------------------------------------------------
# Feature ranking by graph distance
# ---------------------------------------------------------------------------

def rank_features_by_causal_priority(
    scm_result: CausalSCMResult,
    target_feature: str,
) -> List[Tuple[str, int]]:
    """Rank all features by shortest causal-graph distance to target_feature.

    Uses a heapq min-heap to run a Dijkstra-style shortest-path search
    over the undirected adjacency matrix (all edge weights are 1, so
    this is equivalent to BFS, but the heap generalizes cleanly if
    module 4 later wants to weight edges by discovered edge strength).

    Returns a list of (feature_name, distance) tuples sorted by distance,
    excluding target_feature itself. Features with no path to the target
    are omitted.
    """
    if target_feature not in scm_result.feature_names:
        raise ValueError(f"Unknown feature: {target_feature}")

    n = len(scm_result.feature_names)
    name_to_idx = {name: i for i, name in enumerate(scm_result.feature_names)}
    start = name_to_idx[target_feature]

    distances: Dict[int, float] = {i: float("inf") for i in range(n)}
    distances[start] = 0
    heap: List[Tuple[int, int]] = [(0, start)]
    visited: Set[int] = set()

    while heap:
        dist, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)

        neighbors = np.nonzero(scm_result.adjacency_matrix[node])[0]
        for neighbor in neighbors:
            neighbor = int(neighbor)
            new_dist = dist + 1
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))

    ranked = [
        (scm_result.feature_names[i], int(distances[i]))
        for i in range(n)
        if i != start and distances[i] != float("inf")
    ]
    ranked.sort(key=lambda pair: pair[1])
    return ranked


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_scm_summary(scm_result: CausalSCMResult) -> None:
    """Print a concise summary of the discovered causal SCM, including the edge list."""
    print("Causal SCM summary")
    print("-" * 60)
    feature_names = scm_result.feature_names
    adjacency = scm_result.adjacency_matrix

    print(f"CI test used: {scm_result.ci_test_used}")
    print(f"Alpha used: {scm_result.alpha_used:.2f}")
    print(f"Total features: {len(feature_names)}")
    print(f"Total edges discovered: {scm_result.edge_count}")

    print("\nDiscovered causal edges:")
    edges = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if adjacency[i, j] != 0 or adjacency[j, i] != 0:
                edges.append((feature_names[i], feature_names[j]))
    if edges:
        for source, target in edges:
            print(f"  {source} -- {target}")
    else:
        print("  No edges discovered.")

    print("\nMutable features:")
    mutable_features = [
        feature_names[i]
        for i, is_mutable in enumerate(scm_result.mutable_only_mask)
        if is_mutable
    ]
    print(f"  {mutable_features}")

    print(f"\nMutable AND causally connected to '{TARGET_FEATURE}':")
    scm_features = [
        feature_names[i]
        for i, is_usable in enumerate(scm_result.bool_mask)
        if is_usable
    ]
    print(f"  {scm_features}")

    print(f"\nCausal priority order (distance to '{TARGET_FEATURE}'):")
    if TARGET_FEATURE in feature_names:
        priority = rank_features_by_causal_priority(scm_result, TARGET_FEATURE)
        print(f"  {priority}")
    else:
        print(f"  '{TARGET_FEATURE}' not found in feature_names; skipping.")

    if scm_result.topological_order:
        print(f"\nAction ordering basis: {scm_result.topological_order}")
    else:
        print("\nNo usable action ordering was obtained from the oriented graph.")

    if scm_result.cycle_breaks:
        print("\nCycle handling (ordering only; discovered graph unchanged):")
        for source, target in scm_result.cycle_breaks:
            print(f"  removed ordering edge {source} -> {target}")


if __name__ == "__main__":
    from loader import load_adult_income_data

    loader_result = load_adult_income_data()
    scm_result = discover_causal_structure(loader_result)
    print_scm_summary(scm_result)

    if scm_result.edge_count > 0:
        print(f"\nFeatures ranked by graph distance to '{TARGET_FEATURE}':")
        for name, dist in rank_features_by_causal_priority(
            scm_result, TARGET_FEATURE
        ):
            print(f"  {name}: distance={dist}")