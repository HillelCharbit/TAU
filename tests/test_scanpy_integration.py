"""Tests and comparison suite for the TAU–Scanpy integration.

Run fast integration tests:
    pytest -q tests/test_scanpy_integration.py

Run the full TAU vs Leiden vs Louvain comparison with visible TAU progress:
    RUN_TAU_COMPARISONS=1 TAU_TEST_VERBOSE=1 \
        pytest -s -q tests/test_scanpy_integration.py

Optional comparison controls:
    TAU_COMPARISON_POPULATION=20
    TAU_COMPARISON_GENERATIONS=5
    TAU_COMPARISON_OUT=tau_scanpy_comparison.csv

The comparison suite deliberately does not assert that one clustering method
must always outperform another. Quality and runtime comparisons are reported,
while correctness tests assert API behavior, reproducibility, graph handling,
and valid metrics.
"""

from __future__ import annotations

import os
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from tau_community_detection.tl import (
    _connectivities_to_igraph,
    _get_connectivities,
    tau,
)

SEED = 42

TAU_TEST_VERBOSE = os.getenv("TAU_TEST_VERBOSE", "0") == "1"
RUN_COMPARISONS = os.getenv("RUN_TAU_COMPARISONS", "0") == "1"

FAST_TAU_KWARGS = {
    "population_size": 8,
    "max_generations": 2,
    "n_iterations": 2,
    "worker_count": 1,
}

COMPARISON_TAU_KWARGS = {
    "population_size": int(os.getenv("TAU_COMPARISON_POPULATION", "20")),
    "max_generations": int(os.getenv("TAU_COMPARISON_GENERATIONS", "5")),
    "n_iterations": 2,
    "worker_count": 1,
}


@pytest.fixture
def adata_blobs() -> AnnData:
    """Deterministic four-class dataset with one shared Scanpy neighbor graph."""
    x, truth = make_blobs(
        n_samples=240,
        n_features=10,
        centers=4,
        cluster_std=0.85,
        random_state=SEED,
    )
    adata = AnnData(x.astype(np.float32))
    adata.obs["truth"] = pd.Categorical(truth.astype(str))
    sc.pp.neighbors(adata, n_neighbors=12, random_state=SEED)
    return adata


def _labels(adata: AnnData, key: str) -> np.ndarray:
    return adata.obs[key].astype(str).to_numpy()


def _independent_metrics(
    adata: AnnData,
    key: str,
    *,
    runtime_seconds: float,
    method: str,
    seed: int,
    resolution: float,
) -> dict[str, float | int | str]:
    """Calculate all methods' metrics from the same AnnData graph."""
    labels = _labels(adata, key)
    truth = _labels(adata, "truth")

    return {
        "method": method,
        "seed": seed,
        "resolution": resolution,
        "n_clusters": int(pd.Series(labels).nunique()),
        "modularity": float(
            sc.metrics.modularity(
                adata,
                labels=key,
                mode="calculate",
            )
        ),
        "ari_truth": float(adjusted_rand_score(truth, labels)),
        "nmi_truth": float(normalized_mutual_info_score(truth, labels)),
        "runtime_seconds": float(runtime_seconds),
    }


def _run_tau(
    adata: AnnData,
    *,
    key: str,
    seed: int = SEED,
    resolution: float = 1.0,
    verbose: bool = False,
    comparison: bool = False,
    adjacency=None,
    use_weights: bool = True,
) -> float:
    kwargs = COMPARISON_TAU_KWARGS if comparison else FAST_TAU_KWARGS
    start = time.perf_counter()
    tau(
        adata,
        key_added=key,
        resolution=resolution,
        random_state=seed,
        adjacency=adjacency,
        use_weights=use_weights,
        verbose=verbose,
        **kwargs,
    )
    return time.perf_counter() - start


def _run_leiden(
    adata: AnnData,
    *,
    key: str,
    seed: int,
    resolution: float,
    adjacency,
    use_weights: bool = True,
) -> float:
    """Run Scanpy Leiden with igraph, matching TAU's Leiden backend."""
    start = time.perf_counter()
    sc.tl.leiden(
        adata,
        key_added=key,
        resolution=resolution,
        random_state=seed,
        adjacency=adjacency,
        directed=False,
        use_weights=use_weights,
        n_iterations=2,
        flavor="igraph",
    )
    return time.perf_counter() - start


def _run_louvain(
    adata: AnnData,
    *,
    key: str,
    seed: int,
    adjacency,
    use_weights: bool = True,
) -> float:
    """Run Scanpy's built-in igraph Louvain implementation."""

    start = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sc.tl.louvain(
            adata,
            key_added=key,
            random_state=seed,
            adjacency=adjacency,
            flavor="igraph",
            directed=False,
            use_weights=use_weights,
        )

    return time.perf_counter() - start

# ---------------------------------------------------------------------------
# Graph retrieval
# ---------------------------------------------------------------------------


def test_get_connectivities_default(adata_blobs: AnnData) -> None:
    result = _get_connectivities(adata_blobs)

    assert result is adata_blobs.obsp["connectivities"]
    assert result.shape == (adata_blobs.n_obs, adata_blobs.n_obs)
    assert result.nnz > 0


def test_get_connectivities_from_obsp(adata_blobs: AnnData) -> None:
    adata_blobs.obsp["custom_graph"] = adata_blobs.obsp[
        "connectivities"
    ].copy()

    result = _get_connectivities(adata_blobs, obsp="custom_graph")

    assert result is adata_blobs.obsp["custom_graph"]


def test_get_connectivities_from_neighbors_key(adata_blobs: AnnData) -> None:
    adata_blobs.obsp["custom_connectivities"] = adata_blobs.obsp[
        "connectivities"
    ].copy()
    adata_blobs.uns["custom_neighbors"] = {
        "connectivities_key": "custom_connectivities"
    }

    result = _get_connectivities(
        adata_blobs,
        neighbors_key="custom_neighbors",
    )

    assert result is adata_blobs.obsp["custom_connectivities"]


def test_get_connectivities_rejects_two_graph_selectors(
    adata_blobs: AnnData,
) -> None:
    with pytest.raises(ValueError, match="Specify only one"):
        _get_connectivities(
            adata_blobs,
            neighbors_key="neighbors",
            obsp="connectivities",
        )


def test_get_connectivities_missing_graph() -> None:
    adata = AnnData(np.zeros((5, 2), dtype=np.float32))

    with pytest.raises(KeyError, match="Run `scanpy.pp.neighbors"):
        _get_connectivities(adata)


def test_get_connectivities_invalid_neighbors_key(
    adata_blobs: AnnData,
) -> None:
    with pytest.raises(KeyError, match="No connectivity graph found"):
        _get_connectivities(
            adata_blobs,
            neighbors_key="does_not_exist",
        )


# ---------------------------------------------------------------------------
# Scanpy matrix -> igraph conversion
# ---------------------------------------------------------------------------


def test_connectivities_to_igraph_preserves_graph() -> None:
    matrix = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.2, 0.0, 0.0],
                [0.2, 0.0, 0.8, 0.0],
                [0.0, 0.8, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
    )

    graph = _connectivities_to_igraph(matrix)

    assert graph.vcount() == 4
    assert graph.ecount() == 2
    assert graph.is_directed() is False
    assert graph.degree(3) == 0
    assert graph.vs["name"] == ["0", "1", "2", "3"]
    assert "weight" in graph.es.attributes()
    assert sorted(graph.es["weight"]) == pytest.approx([0.2, 0.8])


def test_connectivities_to_igraph_can_ignore_weights() -> None:
    matrix = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.4],
                [0.4, 0.0],
            ],
            dtype=float,
        )
    )

    graph = _connectivities_to_igraph(matrix, use_weights=False)

    assert graph.vcount() == 2
    assert graph.ecount() == 1
    assert "weight" not in graph.es.attributes()


def test_connectivities_to_igraph_accepts_dense_matrix() -> None:
    matrix = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ]
    )

    graph = _connectivities_to_igraph(matrix)

    assert graph.vcount() == 3
    assert graph.ecount() == 2


def test_connectivities_to_igraph_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="must be square"):
        _connectivities_to_igraph(np.ones((3, 4)))


def test_scanpy_graph_edge_count_matches_igraph(
    adata_blobs: AnnData,
) -> None:
    matrix = _get_connectivities(adata_blobs)
    graph = _connectivities_to_igraph(matrix)

    assert graph.vcount() == adata_blobs.n_obs
    assert graph.ecount() == sparse.triu(matrix, k=1).nnz


# ---------------------------------------------------------------------------
# Public tau() behavior
# ---------------------------------------------------------------------------


def test_tau_inplace_contract_and_metadata(
    adata_blobs: AnnData,
) -> None:
    result = tau(
        adata_blobs,
        key_added="tau_test",
        random_state=SEED,
        verbose=TAU_TEST_VERBOSE,
        **FAST_TAU_KWARGS,
    )

    assert result is None
    assert "tau_test" in adata_blobs.obs
    assert isinstance(adata_blobs.obs["tau_test"].dtype, pd.CategoricalDtype)
    assert len(adata_blobs.obs["tau_test"]) == adata_blobs.n_obs
    assert not adata_blobs.obs["tau_test"].isna().any()

    stored = adata_blobs.uns["tau_test"]
    assert set(stored) == {"params", "modularity"}
    assert stored["params"]["resolution"] == 1.0
    assert stored["params"]["random_state"] == SEED
    assert stored["params"]["worker_count"] == 1
    assert np.isfinite(stored["modularity"])


def test_tau_stored_modularity_matches_independent_metric(
    adata_blobs: AnnData,
) -> None:
    tau(
        adata_blobs,
        key_added="tau_modularity",
        random_state=SEED,
        verbose=TAU_TEST_VERBOSE,
        **FAST_TAU_KWARGS,
    )

    stored = float(adata_blobs.uns["tau_modularity"]["modularity"])
    independent = float(
        sc.metrics.modularity(
            adata_blobs,
            labels="tau_modularity",
            mode="calculate",
        )
    )

    assert stored == pytest.approx(independent, abs=1e-8)


def test_tau_copy_contract(adata_blobs: AnnData) -> None:
    copied = tau(
        adata_blobs,
        key_added="tau_copy",
        random_state=SEED,
        copy=True,
        verbose=TAU_TEST_VERBOSE,
        **FAST_TAU_KWARGS,
    )

    assert copied is not None
    assert copied is not adata_blobs
    assert "tau_copy" in copied.obs
    assert "tau_copy" in copied.uns
    assert "tau_copy" not in adata_blobs.obs
    assert "tau_copy" not in adata_blobs.uns


def test_tau_accepts_explicit_adjacency(
    adata_blobs: AnnData,
) -> None:
    adjacency = adata_blobs.obsp["connectivities"].copy()
    bare = AnnData(adata_blobs.X.copy())

    tau(
        bare,
        key_added="tau_explicit",
        adjacency=adjacency,
        random_state=SEED,
        verbose=TAU_TEST_VERBOSE,
        **FAST_TAU_KWARGS,
    )

    assert "tau_explicit" in bare.obs
    assert len(bare.obs["tau_explicit"]) == bare.n_obs


def test_tau_rejects_adjacency_with_graph_selector(
    adata_blobs: AnnData,
) -> None:
    adjacency = adata_blobs.obsp["connectivities"]

    with pytest.raises(ValueError, match="Do not specify"):
        tau(
            adata_blobs,
            adjacency=adjacency,
            obsp="connectivities",
            **FAST_TAU_KWARGS,
        )


def test_tau_uses_custom_obsp(adata_blobs: AnnData) -> None:
    adata_blobs.obsp["tau_graph"] = adata_blobs.obsp[
        "connectivities"
    ].copy()

    tau(
        adata_blobs,
        key_added="tau_custom_obsp",
        obsp="tau_graph",
        random_state=SEED,
        verbose=TAU_TEST_VERBOSE,
        **FAST_TAU_KWARGS,
    )

    assert "tau_custom_obsp" in adata_blobs.obs


def test_tau_uses_custom_neighbors_key(adata_blobs: AnnData) -> None:
    adata_blobs.obsp["tau_connectivities"] = adata_blobs.obsp[
        "connectivities"
    ].copy()
    adata_blobs.uns["tau_neighbors"] = {
        "connectivities_key": "tau_connectivities"
    }

    tau(
        adata_blobs,
        key_added="tau_custom_neighbors",
        neighbors_key="tau_neighbors",
        random_state=SEED,
        verbose=TAU_TEST_VERBOSE,
        **FAST_TAU_KWARGS,
    )

    assert "tau_custom_neighbors" in adata_blobs.obs


def test_tau_unweighted_mode(adata_blobs: AnnData) -> None:
    tau(
        adata_blobs,
        key_added="tau_unweighted",
        use_weights=False,
        random_state=SEED,
        verbose=TAU_TEST_VERBOSE,
        **FAST_TAU_KWARGS,
    )

    assert "tau_unweighted" in adata_blobs.obs
    assert (
        adata_blobs.uns["tau_unweighted"]["params"]["use_weights"] is False
    )


def test_tau_reproducible_with_fixed_seed_and_one_worker(
    adata_blobs: AnnData,
) -> None:
    first = adata_blobs.copy()
    second = adata_blobs.copy()

    tau(
        first,
        key_added="tau",
        random_state=SEED,
        verbose=TAU_TEST_VERBOSE,
        **FAST_TAU_KWARGS,
    )
    tau(
        second,
        key_added="tau",
        random_state=SEED,
        verbose=TAU_TEST_VERBOSE,
        **FAST_TAU_KWARGS,
    )

    np.testing.assert_array_equal(
        _labels(first, "tau"),
        _labels(second, "tau"),
    )


def test_tau_handles_isolated_vertices() -> None:
    adjacency = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
    )
    adata = AnnData(np.zeros((8, 2), dtype=np.float32))

    tau(
        adata,
        key_added="tau_isolates",
        adjacency=adjacency,
        random_state=SEED,
        verbose=TAU_TEST_VERBOSE,
        **FAST_TAU_KWARGS,
    )

    assert len(adata.obs["tau_isolates"]) == 8
    assert not adata.obs["tau_isolates"].isna().any()


@pytest.mark.parametrize("resolution", [0.5, 1.0, 1.5])
def test_tau_multiple_resolutions(
    adata_blobs: AnnData,
    resolution: float,
) -> None:
    key = f"tau_r_{resolution}"

    tau(
        adata_blobs,
        key_added=key,
        resolution=resolution,
        random_state=SEED,
        verbose=TAU_TEST_VERBOSE,
        **FAST_TAU_KWARGS,
    )

    assert key in adata_blobs.obs
    assert adata_blobs.uns[key]["params"]["resolution"] == resolution
    assert adata_blobs.obs[key].nunique() >= 1


# ---------------------------------------------------------------------------
# Direct comparison with Scanpy's built-in methods
# ---------------------------------------------------------------------------


def test_tau_leiden_louvain_use_the_same_graph(
    adata_blobs: AnnData,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One fast comparison that always runs when Louvain is installed."""
    adjacency = adata_blobs.obsp["connectivities"].copy()

    tau_runtime = _run_tau(
        adata_blobs,
        key="tau_compare",
        seed=SEED,
        resolution=1.0,
        verbose=TAU_TEST_VERBOSE,
        adjacency=adjacency,
    )
    leiden_runtime = _run_leiden(
        adata_blobs,
        key="leiden_compare",
        seed=SEED,
        resolution=1.0,
        adjacency=adjacency,
    )
    louvain_runtime = _run_louvain(
    adata_blobs,
    key="louvain_compare",
    seed=SEED,
    adjacency=adjacency,
)

    rows = [
        _independent_metrics(
            adata_blobs,
            "tau_compare",
            runtime_seconds=tau_runtime,
            method="TAU",
            seed=SEED,
            resolution=1.0,
        ),
        _independent_metrics(
            adata_blobs,
            "leiden_compare",
            runtime_seconds=leiden_runtime,
            method="Leiden",
            seed=SEED,
            resolution=1.0,
        ),
        _independent_metrics(
            adata_blobs,
            "louvain_compare",
            runtime_seconds=louvain_runtime,
            method="Louvain",
            seed=SEED,
            resolution=1.0,
        ),
    ]
    report = pd.DataFrame(rows)

    for key in ("tau_compare", "leiden_compare", "louvain_compare"):
        assert isinstance(adata_blobs.obs[key].dtype, pd.CategoricalDtype)
        assert len(adata_blobs.obs[key]) == adata_blobs.n_obs
        assert not adata_blobs.obs[key].isna().any()

    assert np.isfinite(report["modularity"]).all()
    assert np.isfinite(report["ari_truth"]).all()
    assert np.isfinite(report["nmi_truth"]).all()
    assert (report["runtime_seconds"] >= 0).all()
    assert (report["n_clusters"] >= 1).all()

    if TAU_TEST_VERBOSE:
        with capsys.disabled():
            print("\nFast TAU / Leiden / Louvain comparison")
            print(report.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


@pytest.mark.skipif(
    not RUN_COMPARISONS,
    reason="Set RUN_TAU_COMPARISONS=1 to run the full comparison matrix.",
)

@pytest.mark.skipif(
    not RUN_COMPARISONS,
    reason="Set RUN_TAU_COMPARISONS=1 to run the full comparison matrix.",
)
def test_full_tau_leiden_louvain_comparison(
    adata_blobs: AnnData,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compare quality, runtime, resolution behavior, and seed stability."""

    seeds = [0, 1, 2]
    resolutions = [0.5, 1.0, 1.5]

    rows: list[dict[str, float | int | str]] = []
    memberships: dict[tuple[str, float, int], np.ndarray] = {}

    # Every method receives this exact same weighted, undirected graph.
    adjacency = adata_blobs.obsp["connectivities"].copy()

    # TAU and Leiden both support the resolution parameter.
    for resolution in resolutions:
        for seed in seeds:
            tau_key = f"tau_r{resolution}_s{seed}"
            tau_runtime = _run_tau(
                adata_blobs,
                key=tau_key,
                seed=seed,
                resolution=resolution,
                verbose=TAU_TEST_VERBOSE,
                comparison=True,
                adjacency=adjacency,
            )
            rows.append(
                _independent_metrics(
                    adata_blobs,
                    tau_key,
                    runtime_seconds=tau_runtime,
                    method="TAU",
                    seed=seed,
                    resolution=resolution,
                )
            )
            memberships[("TAU", resolution, seed)] = _labels(
                adata_blobs,
                tau_key,
            )

            leiden_key = f"leiden_r{resolution}_s{seed}"
            leiden_runtime = _run_leiden(
                adata_blobs,
                key=leiden_key,
                seed=seed,
                resolution=resolution,
                adjacency=adjacency,
            )
            rows.append(
                _independent_metrics(
                    adata_blobs,
                    leiden_key,
                    runtime_seconds=leiden_runtime,
                    method="Leiden",
                    seed=seed,
                    resolution=resolution,
                )
            )
            memberships[("Leiden", resolution, seed)] = _labels(
                adata_blobs,
                leiden_key,
            )

    # Scanpy's igraph Louvain backend does not support resolution.
    # Run it once per seed and record it as the default-resolution baseline.
    for seed in seeds:
        louvain_key = f"louvain_s{seed}"
        louvain_runtime = _run_louvain(
            adata_blobs,
            key=louvain_key,
            seed=seed,
            adjacency=adjacency,
        )
        rows.append(
            _independent_metrics(
                adata_blobs,
                louvain_key,
                runtime_seconds=louvain_runtime,
                method="Louvain",
                seed=seed,
                resolution=1.0,
            )
        )
        memberships[("Louvain", 1.0, seed)] = _labels(
            adata_blobs,
            louvain_key,
        )

    report = pd.DataFrame(rows).sort_values(
        ["resolution", "seed", "method"]
    )

    summary = (
        report.groupby(["method", "resolution"], sort=False)
        .agg(
            modularity_mean=("modularity", "mean"),
            modularity_std=("modularity", "std"),
            ari_mean=("ari_truth", "mean"),
            ari_std=("ari_truth", "std"),
            nmi_mean=("nmi_truth", "mean"),
            clusters_mean=("n_clusters", "mean"),
            runtime_mean=("runtime_seconds", "mean"),
        )
        .reset_index()
    )

    stability_rows: list[dict[str, float | str]] = []

    comparison_settings = [
        ("TAU", resolution) for resolution in resolutions
    ] + [
        ("Leiden", resolution) for resolution in resolutions
    ] + [
        ("Louvain", 1.0)
    ]

    for method, resolution in comparison_settings:
        pairwise_ari = [
            adjusted_rand_score(
                memberships[(method, resolution, first_seed)],
                memberships[(method, resolution, second_seed)],
            )
            for first_seed, second_seed in combinations(seeds, 2)
        ]

        stability_rows.append(
            {
                "method": method,
                "resolution": resolution,
                "mean_pairwise_ari": float(np.mean(pairwise_ari)),
                "min_pairwise_ari": float(np.min(pairwise_ari)),
            }
        )

    stability = pd.DataFrame(stability_rows)

    expected_runs = len(seeds) * (2 * len(resolutions) + 1)

    assert len(report) == expected_runs
    assert np.isfinite(
        report[
            [
                "modularity",
                "ari_truth",
                "nmi_truth",
                "runtime_seconds",
            ]
        ].to_numpy()
    ).all()
    assert report["n_clusters"].ge(1).all()
    assert stability["mean_pairwise_ari"].between(-1, 1).all()
    assert stability["min_pairwise_ari"].between(-1, 1).all()

    output_path = os.getenv("TAU_COMPARISON_OUT")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)

    with capsys.disabled():
        print("\nRaw comparison results")
        print(
            report.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        print("\nMean results across seeds")
        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        print("\nSeed stability")
        print(
            stability.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        if output_path:
            print(
                "\nSaved raw comparison results to:",
                Path(output_path).resolve(),
            )