import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

MODULES = Path(__file__).parents[1] / "modules"
sys.path.insert(0, str(MODULES))

import spherical_surface


def test_sample_raster_includes_global_boundaries():
    graph = nx.Graph()
    graph.add_node(0, pos=(-180, 90))
    graph.add_node(1, pos=(180, -90))
    spherical_surface.sample_raster(graph, np.array([[1, 2], [3, 4]]), "value")
    assert graph.nodes[0]["value"] == 1
    assert graph.nodes[1]["value"] == 4


def test_extract_fault_graph_connects_periodic_seam_and_uses_lon_lat():
    strain = np.zeros((12, 24))
    strain[5, [22, 23, 0, 1]] = 1
    graph, mask, skeleton = spherical_surface.extract_fault_graph(
        strain, threshold=0.5, minimum_component_size=2
    )
    assert mask.sum() == 4
    assert skeleton.sum() == 4
    assert graph.number_of_nodes() == 4
    assert len(set(nx.get_node_attributes(graph, "fault").values())) == 1
    seam_edges = [
        (u, v)
        for u, v in graph.edges
        if abs(graph.nodes[u]["pos"][0] - graph.nodes[v]["pos"][0]) > 180
    ]
    assert seam_edges
    assert max(nx.get_edge_attributes(graph, "length").values()) < 2000


def test_read_surface_archive(tmp_path):
    path = tmp_path / "surface.npz"
    strain = np.arange(8, dtype=np.float32).reshape(2, 4)
    velocity = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    np.savez_compressed(
        path,
        strain=strain,
        velocity=velocity,
        output_number=np.int32(12),
        time_years=np.float64(3.5e6),
        grid_shape=np.array([2, 4], dtype=np.int32),
    )
    loaded = spherical_surface.read_surface_archive(path)
    np.testing.assert_allclose(loaded["strain"], strain)
    np.testing.assert_allclose(loaded["velocity"], velocity)
    assert loaded["output_number"] == 12
    assert loaded["time_years"] == pytest.approx(3.5e6)
    assert loaded["grid_shape"] == (2, 4)
