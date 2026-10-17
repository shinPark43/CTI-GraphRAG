"""Propagation math and the held-out-report guard on a toy graph."""

import numpy as np

from ctiattr.graph_embed import SnapshotGraph, adjacency, propagate

# T1 - T2 (hier); beh:0 from report R1 grounds T1; T3 isolated
G = SnapshotGraph(ids=["T1", "T2", "beh:0", "T3"], kind=["tech", "tech", "beh", "tech"], texts=[""] * 4,
                  edges={"hier": [(0, 1)], "report": [(2, 0)]}, report_of={2: "R1"})
X = np.eye(4, dtype=np.float32)


def test_alpha_zero_is_the_text_embedding():
    H = propagate(X, adjacency(G, ("hier", "report")), alpha=0.0, hops=2)
    assert np.allclose(H, X)


def test_one_hop_mixes_neighbor_mean():
    H = propagate(X, adjacency(G, ("hier", "report")), alpha=0.5, hops=1)
    # T1 neighbors: T2, beh:0 -> 0.5*x_T1 + 0.5*mean(x_T2, x_beh), then L2-normalized
    expect = np.array([0.5, 0.25, 0.25, 0.0])
    assert np.allclose(H[0], expect / np.linalg.norm(expect))
    assert np.allclose(H[3], X[3])  # isolated node keeps its own vector


def test_own_report_edges_are_removed():
    A = adjacency(G, ("hier", "report"), exclude_report="R1")
    assert A[0, 2] == 0 and A[2, 0] == 0 and A[0, 1] == 1
    H = propagate(X, A, alpha=0.5, hops=2)
    assert H[0, 2] == 0  # no trace of the held-out report's behavior in T1
