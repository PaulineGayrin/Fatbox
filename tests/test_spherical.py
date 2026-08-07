import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

MODULES = Path(__file__).parents[1] / "modules"
sys.path.insert(0, str(MODULES))

import spherical


def test_latlon_cartesian_round_trip_vectorized():
    lat = np.array([-90.0, -30.0, 0.0, 45.0, 90.0])
    lon = np.array([-180.0, -120.0, 0.0, 179.0, 45.0])
    xyz = spherical.latlon_to_cartesian(lat, lon)
    actual_lat, actual_lon = spherical.cartesian_to_latlon(*xyz)
    np.testing.assert_allclose(actual_lat, lat, atol=1e-12)
    np.testing.assert_allclose(
        (actual_lon - lon + 180.0) % 360.0 - 180.0, 0.0, atol=1e-12
    )


def test_haversine_quarter_circumference_and_antipodes():
    radius = spherical.EARTH_RADIUS_KM
    assert spherical.haversine_distance(0, 0, 0, 90) == pytest.approx(
        math.pi * radius / 2
    )
    assert spherical.haversine_distance(0, 0, 0, 180) == pytest.approx(
        math.pi * radius
    )


def test_haversine_takes_short_path_across_dateline():
    distance = spherical.haversine_distance(0, 179, 0, -179)
    assert distance == pytest.approx(
        math.radians(2) * spherical.EARTH_RADIUS_KM
    )


def test_haversine_matches_cartesian_central_angles_randomized():
    random = np.random.default_rng(3989)
    lat1 = random.uniform(-90, 90, 1000)
    lon1 = random.uniform(-180, 180, 1000)
    lat2 = random.uniform(-90, 90, 1000)
    lon2 = random.uniform(-180, 180, 1000)
    xyz1 = np.stack(spherical.latlon_to_cartesian(lat1, lon1, radius=1), axis=1)
    xyz2 = np.stack(spherical.latlon_to_cartesian(lat2, lon2, radius=1), axis=1)
    expected = np.arccos(np.clip(np.sum(xyz1 * xyz2, axis=1), -1, 1))
    actual = spherical.angular_distance(lat1, lon1, lat2, lon2)
    np.testing.assert_allclose(actual, expected, atol=2e-14)


@pytest.mark.parametrize(
    "start, target, expected",
    [
        ((0, 0), (0, 10), 0.0),
        ((0, 0), (10, 0), 90.0),
        ((0, 0), (0, -10), 0.0),
        ((0, 0), (-10, 0), 90.0),
        ((179, 0), (-179, 0), 90.0),
    ],
)
def test_strike_cardinal_and_dateline(start, target, expected):
    assert spherical.calculate_strike(start, target) == pytest.approx(expected)


def test_destination_crosses_dateline_and_remains_on_sphere():
    lon, lat = spherical.destination_point(0, 179, 90, 2)
    assert lon == pytest.approx(-179)
    assert lat == pytest.approx(0, abs=1e-12)
    lon, lat = spherical.destination_point(89, 0, 0, 2)
    assert -180 <= lon < 180
    assert lat == pytest.approx(89)
    assert abs(lon) == pytest.approx(180)


def test_connect_across_both_dateline_coordinate_conventions():
    graph = nx.Graph()
    graph.add_node("a", pos=(-179.5, 10.0))
    graph.add_node("b", pos=(179.5, 10.0))
    spherical.connect_across_dateline(graph, threshold=1.1)
    assert graph.has_edge("a", "b")

    graph = nx.Graph()
    graph.add_node("a", pos=(0.5, 10.0))
    graph.add_node("b", pos=(359.5, 10.0))
    spherical.connect_across_dateline(graph, threshold=1.1)
    assert graph.has_edge("a", "b")


def test_pickup_points_are_perpendicular_and_wrap():
    graph = nx.path_graph(3)
    positions = {0: (178.0, 0.0), 1: (179.0, 0.0), 2: (-180.0, 0.0)}
    nx.set_node_attributes(graph, positions, "pos")
    spherical.calculate_direction(graph, cutoff=1)
    pickups = spherical.calculate_pickup_points(graph, factor=1)
    lon_pos, lat_pos = pickups.nodes[1, 1]["pos"]
    lon_neg, lat_neg = pickups.nodes[1, 2]["pos"]
    assert lon_pos == pytest.approx(179.0)
    assert lon_neg == pytest.approx(179.0)
    assert sorted([lat_pos, lat_neg]) == pytest.approx([-1.0, 1.0])


def test_zero_velocity_is_valid_for_slip_and_nan_is_missing():
    graph = nx.Graph()
    graph.add_node(0)
    samples = nx.Graph()
    for side, velocity in ((1, (0, 0, 0)), (2, (3, 4, 0))):
        samples.add_node(
            (0, side), v_x=velocity[0], v_y=velocity[1], v_z=velocity[2]
        )
    spherical.calculate_slip_rate_sphere(graph, samples, dim=3)
    assert graph.nodes[0]["slip_rate"] == pytest.approx(5)

    samples.nodes[0, 1]["v_x"] = np.nan
    spherical.calculate_slip_rate_sphere(graph, samples, dim=3)
    assert math.isnan(graph.nodes[0]["slip_rate"])


def test_cartesian_slip_is_projected_to_local_geological_components():
    graph = nx.Graph()
    graph.add_node(0, bearing=0)
    samples = nx.Graph()
    samples.add_node((0, 0), pos=(0, 0))
    samples.add_node((0, 1), v_x=3, v_y=4, v_z=12)
    samples.add_node((0, 2), v_x=0, v_y=0, v_z=0)
    spherical.calculate_slip_sphere(graph, samples, dt=1, dim=3)
    assert graph.nodes[0]["slip_east"] == pytest.approx(4)
    assert graph.nodes[0]["slip_north"] == pytest.approx(12)
    assert graph.nodes[0]["slip_up"] == pytest.approx(3)
    spherical.write_slip_to_displacement(graph, dim=3)
    assert graph.nodes[0]["heave"] == pytest.approx(4)
    assert graph.nodes[0]["lateral"] == pytest.approx(12)
    assert graph.nodes[0]["throw"] == pytest.approx(3)
    assert graph.nodes[0]["displacement"] == pytest.approx(13)


def test_fault_correlation_supports_geodesic_metric():
    first = nx.Graph()
    first.add_node(0, pos=(179.9, 0), fault=1)
    second = nx.Graph()
    second.add_node(0, pos=(-179.9, 0), fault=7)
    correlations, forward, backward = spherical.correlation_slow(
        first, second, R=25, metric="great_circle"
    )
    assert correlations == {(1, 7)}
    assert forward[0, 0] == pytest.approx(backward[0, 0])
    assert forward[0, 0] < 25


def _isolated_fault_graph(points):
    graph = nx.Graph()
    for local_fault, position in enumerate(points):
        graph.add_node(local_fault, pos=position, fault=local_fault)
    return graph


def test_progressive_tracking_recovers_a_fault_after_one_missing_output():
    graphs = [
        _isolated_fault_graph([(0, 0), (100, 0)]),
        _isolated_fault_graph([(1, 0), (50, 0)]),
        _isolated_fault_graph([(2, 0), (101, 0), (51, 0)]),
    ]
    tracked, history = spherical.track_fault_sequence(
        graphs, R=300, metric="great_circle", lookback=2
    )
    first_labels = [tracked[0].nodes[node]["fault"] for node in tracked[0]]
    final_labels = [tracked[2].nodes[node]["fault"] for node in tracked[2]]
    assert first_labels == [0, 1]
    assert final_labels == [0, 1, 2]
    assert tracked[2].nodes[1]["match_lag"] == 2
    assert history[2]["matched"] == 3
    assert history[2]["new"] == 0
    assert all("local_fault" in data for _, data in tracked[2].nodes(data=True))
    assert all("local_fault" not in data for _, data in graphs[2].nodes(data=True))


def test_progressive_tracking_records_splits_and_merges_as_families():
    first = _isolated_fault_graph([(0, 0), (4, 0)])
    second = _isolated_fault_graph([(1, 0), (1.5, 0)])
    tracked, history = spherical.track_fault_sequence(
        [first, second], R=500, metric="great_circle"
    )
    assert tracked[1].nodes[0]["fault"] == 0
    assert tracked[1].nodes[1]["fault"] == 0
    assert tracked[1].nodes[0]["family"] == (0, 1)
    assert history[1]["parents"][0] == [0, 1]


def test_progressive_tracking_validates_arguments():
    with pytest.raises(ValueError):
        spherical.track_fault_sequence([], R=-1)
    with pytest.raises(ValueError):
        spherical.track_fault_sequence([], R=1, lookback=0)


def test_southern_hemisphere_nodes_are_processed_without_filtering():
    graph = nx.path_graph(3)
    nx.set_node_attributes(
        graph, {0: (-20, -60), 1: (-19, -60), 2: (-18, -60)}, "pos"
    )
    spherical.calculate_direction(graph, cutoff=1)
    samples = spherical.calculate_pickup_points(graph, factor=1)
    for node in graph:
        samples.nodes[node, 1].update(v_x=1, v_y=2, v_z=3)
        samples.nodes[node, 2].update(v_x=0, v_y=0, v_z=0)
    spherical.calculate_slip_rate_sphere(graph, samples)
    assert all(np.isfinite(graph.nodes[node]["slip_rate"]) for node in graph)


def test_input_validation():
    with pytest.raises(ValueError):
        spherical.destination_point(0, 0, 0, float("nan"))
    with pytest.raises(ValueError):
        spherical.calculate_slip_sphere(nx.Graph(), nx.Graph(), -1, 3)
