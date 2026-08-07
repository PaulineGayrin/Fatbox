"""Small, dependency-light helpers for fault networks on a sphere.

Coordinates stored in graphs are ``(longitude, latitude)`` pairs in degrees.
Functions accepting latitude and longitude as separate arguments keep that
order explicit in their signatures.  Distances are kilometres unless a
different radius is supplied.

Prepared-raster loading, skeleton extraction and sampling belong in
:mod:`spherical_surface`, not in this numerical core.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import networkx as nx
import numpy as np

EARTH_RADIUS_M = 6_371_000.0
EARTH_RADIUS_KM = EARTH_RADIUS_M / 1000.0


def _wrap_longitude(longitude):
    wrapped = (np.asarray(longitude, dtype=float) + 180.0) % 360.0 - 180.0
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def _as_lon_lat(position: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(position, dtype=float)
    if values.shape == (2,):
        lon, lat = values
    elif values.shape == (3,):
        radius = np.linalg.norm(values)
        if radius == 0:
            raise ValueError("Cartesian position cannot be the origin")
        x, y, z = values
        lon = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arcsin(np.clip(z / radius, -1.0, 1.0)))
    else:
        raise ValueError("position must contain (lon, lat) or (x, y, z)")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"latitude must be in [-90, 90], got {lat}")
    return float(_wrap_longitude(lon)), float(lat)


def latlon_to_cartesian(lat, lon, radius=EARTH_RADIUS_M):
    """Convert latitude/longitude in degrees to Earth-centred Cartesian data."""
    lat_rad = np.radians(np.asarray(lat, dtype=float))
    lon_rad = np.radians(np.asarray(lon, dtype=float))
    radius = np.asarray(radius, dtype=float)
    return (
        radius * np.cos(lat_rad) * np.cos(lon_rad),
        radius * np.cos(lat_rad) * np.sin(lon_rad),
        radius * np.sin(lat_rad),
    )


def cartesian_to_latlon(x, y, z):
    """Convert Earth-centred Cartesian data to ``(latitude, longitude)``."""
    x, y, z = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )
    radius = np.sqrt(x * x + y * y + z * z)
    if np.any(radius == 0):
        raise ValueError("Cartesian position cannot be the origin")
    lat = np.degrees(np.arcsin(np.clip(z / radius, -1.0, 1.0)))
    lon = _wrap_longitude(np.degrees(np.arctan2(y, x)))
    return lat, lon


def angular_distance(lat1, lon1, lat2, lon2):
    """Return the robust great-circle central angle in radians."""
    lat1, lon1, lat2, lon2 = np.broadcast_arrays(
        np.radians(np.asarray(lat1, dtype=float)),
        np.radians(np.asarray(lon1, dtype=float)),
        np.radians(np.asarray(lat2, dtype=float)),
        np.radians(np.asarray(lon2, dtype=float)),
    )
    dlat = lat2 - lat1
    dlon = (lon2 - lon1 + np.pi) % (2 * np.pi) - np.pi
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    angle = 2 * np.arctan2(
        np.sqrt(np.clip(a, 0, 1)), np.sqrt(np.clip(1 - a, 0, 1))
    )
    return float(angle) if angle.ndim == 0 else angle


def haversine_distance(lat1, lon1, lat2, lon2, radius=EARTH_RADIUS_KM):
    """Return great-circle distance, in kilometres with the default radius."""
    return np.asarray(radius) * angular_distance(lat1, lon1, lat2, lon2)


def initial_bearing(lat1, lon1, lat2, lon2):
    """Return initial bearing clockwise from north in ``[0, 360)`` degrees."""
    phi1, phi2 = np.radians([lat1, lat2])
    delta_lon = np.radians(_wrap_longitude(lon2 - lon1))
    x = np.sin(delta_lon) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - (
        np.sin(phi1) * np.cos(phi2) * np.cos(delta_lon)
    )
    if np.isclose(x, 0) and np.isclose(y, 0):
        raise ValueError("bearing is undefined for coincident or antipodal points")
    return float(np.degrees(np.arctan2(x, y)) % 360)


def destination_point(lat, lon, bearing, angular_distance_degrees):
    """Follow a great circle and return ``(longitude, latitude)`` in degrees."""
    parameters = np.asarray([lat, lon, bearing, angular_distance_degrees])
    if not np.all(np.isfinite(parameters)):
        raise ValueError("destination-point parameters must be finite")
    if not -90 <= float(lat) <= 90:
        raise ValueError("latitude must be in [-90, 90]")
    phi1, lam1, theta, delta = np.radians(parameters.astype(float))
    sin_phi2 = (
        np.sin(phi1) * np.cos(delta)
        + np.cos(phi1) * np.sin(delta) * np.cos(theta)
    )
    phi2 = np.arcsin(np.clip(sin_phi2, -1, 1))
    lam2 = lam1 + np.arctan2(
        np.sin(theta) * np.sin(delta) * np.cos(phi1),
        np.cos(delta) - np.sin(phi1) * np.sin(phi2),
    )
    return float(_wrap_longitude(np.degrees(lam2))), float(np.degrees(phi2))


def calculate_strike(pos1, pos2):
    """Return undirected geodesic strike in ``[0, 180)`` degrees."""
    lon1, lat1 = _as_lon_lat(pos1)
    lon2, lat2 = _as_lon_lat(pos2)
    return initial_bearing(lat1, lon1, lat2, lon2) % 180


def connect_across_dateline(graph, threshold=2.0, position="pos"):
    """Connect graph nodes that are neighbours across a longitude-grid seam.

    Both ``[-180, 180]`` and ``[0, 360]`` longitude conventions are accepted.
    ``threshold`` is in coordinate-grid degrees, not kilometres.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    positions = nx.get_node_attributes(graph, position)
    if len(positions) != graph.number_of_nodes():
        raise KeyError(f"every node must define the {position!r} attribute")
    nodes = list(graph)
    coordinates = np.asarray([positions[node][:2] for node in nodes], dtype=float)
    if coordinates.size == 0:
        return graph
    longitudes = coordinates[:, 0]
    zero_to_360 = np.all((longitudes >= 0) & (longitudes <= 360))
    lower, upper = (0.0, 360.0) if zero_to_360 else (-180.0, 180.0)
    left = np.flatnonzero(longitudes <= lower + threshold)
    right = np.flatnonzero(longitudes >= upper - threshold)
    for i in left:
        for j in right:
            wrapped_dx = abs(coordinates[i, 0] + 360 - coordinates[j, 0])
            dy = coordinates[i, 1] - coordinates[j, 1]
            if math.hypot(wrapped_dx, dy) <= threshold:
                graph.add_edge(nodes[i], nodes[j])
    return graph


def calculate_direction(graph, cutoff, geographic_positions=None, normalize=True):
    """Estimate local graph direction and store east/north components.

    No latitude filter is applied: Northern and Southern Hemisphere nodes are
    treated identically.  ``geographic_positions`` may override node ``pos``.
    """
    if cutoff < 1:
        raise ValueError("cutoff must be at least 1")
    positions = geographic_positions or nx.get_node_attributes(graph, "pos")
    for node in graph:
        distances = nx.single_source_shortest_path_length(
            graph, node, cutoff=cutoff
        )
        furthest = max(distances.values())
        candidates = [
            other for other, distance in distances.items()
            if distance == furthest and other != node
        ]
        if not candidates:
            graph.nodes[node].update(dx=0.0, dy=0.0, bearing=np.nan)
            continue
        node0, node1 = (node, candidates[0]) if len(candidates) == 1 else candidates[:2]
        lon0, lat0 = _as_lon_lat(positions[node0])
        lon1, lat1 = _as_lon_lat(positions[node1])
        bearing = initial_bearing(lat0, lon0, lat1, lon1)
        scale = angular_distance(lat0, lon0, lat1, lon1) if not normalize else 1.0
        graph.nodes[node].update(
            dx=math.sin(math.radians(bearing)) * scale,
            dy=math.cos(math.radians(bearing)) * scale,
            bearing=bearing,
        )
    return graph


def calculate_pickup_points(graph, factor, geographic_positions=None):
    """Create great-circle samples ``factor`` degrees either side of a trace."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    positions = geographic_positions or nx.get_node_attributes(graph, "pos")
    samples = nx.Graph()
    for node in graph:
        lon, lat = _as_lon_lat(positions[node])
        bearing = graph.nodes[node].get("bearing")
        if bearing is None or not np.isfinite(bearing):
            dx = float(graph.nodes[node]["dx"])
            dy = float(graph.nodes[node]["dy"])
            bearing = math.degrees(math.atan2(dx, dy)) % 360
        positive = destination_point(lat, lon, bearing + 90, factor)
        negative = destination_point(lat, lon, bearing - 90, factor)
        for side, pos, component in (
            (0, (lon, lat), -1), (1, positive, -2), (2, negative, -3)
        ):
            samples.add_node((node, side), pos=pos, component=component)
        samples.add_edge((node, 2), (node, 1))
    return samples


def _fault_labels(graph):
    labels = nx.get_node_attributes(graph, "fault")
    if len(labels) != graph.number_of_nodes():
        raise KeyError("every node must define the 'fault' attribute")
    return sorted(set(labels.values()))


def _fault_points(graph, labels):
    return [
        [data["pos"] for _, data in graph.nodes(data=True) if data["fault"] == label]
        for label in labels
    ]


def _pairwise_point_distance(points_a, points_b, metric):
    a, b = np.asarray(points_a, dtype=float), np.asarray(points_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] < 2 or b.shape[1] < 2:
        raise ValueError("point sets must be non-empty coordinate-pair arrays")
    if metric == "euclidean":
        return np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=2)
    if metric == "great_circle":
        return haversine_distance(
            a[:, None, 1], a[:, None, 0], b[None, :, 1], b[None, :, 0]
        )
    raise ValueError("metric must be 'euclidean' or 'great_circle'")


def compute_similarity(set_a, set_b, metric="euclidean"):
    """Return directed mean nearest-neighbour distance from A to B."""
    distances = _pairwise_point_distance(set_a, set_b, metric)
    return float(np.mean(np.min(distances, axis=1)))


def correlation_slow(graph0, graph1, R, metric="euclidean"):
    """Correlate labelled faults using directed mean point distances.

    A pair is returned when either direction is strictly closer than ``R``.
    The two distance matrices follow ``graph0 x graph1`` and
    ``graph1 x graph0`` ordering respectively.
    """
    if R < 0:
        raise ValueError("R must be non-negative")
    labels0, labels1 = _fault_labels(graph0), _fault_labels(graph1)
    points0 = _fault_points(graph0, labels0)
    points1 = _fault_points(graph1, labels1)
    forward = np.empty((len(labels0), len(labels1)))
    backward = np.empty((len(labels1), len(labels0)))
    correlations = set()
    for i, left in enumerate(points0):
        for j, right in enumerate(points1):
            forward[i, j] = compute_similarity(left, right, metric)
            backward[j, i] = compute_similarity(right, left, metric)
            if forward[i, j] < R or backward[j, i] < R:
                correlations.add((labels0[i], labels1[j]))
    return correlations, forward, backward


def track_fault_sequence(graphs, R, metric="great_circle", lookback=2):
    """Assign persistent fault labels across an ordered graph sequence.

    Parameters
    ----------
    graphs : sequence of networkx.Graph
        Independently extracted graphs in chronological order. Every node must
        define ``pos`` and its timestep-local ``fault`` label.
    R : float
        Maximum directed mean distance accepted as a possible match. With the
        default great-circle metric this is expressed in kilometres.
    metric : {"great_circle", "euclidean"}
        Distance used by :func:`correlation_slow`.
    lookback : int
        Number of preceding timesteps considered. ``2`` allows a structure
        missing from one output to recover its earlier persistent identity.

    Returns
    -------
    tracked_graphs : list of networkx.Graph
        Copies of the input graphs. ``local_fault`` retains the extraction
        label and ``fault`` contains the persistent label. ``family`` records
        every plausible parent label; the closest parent supplies ``fault``.
    history : list of dict
        JSON-compatible diagnostics for the initial graph and every temporal
        transition, including mappings, new labels, parent candidates,
        balanced two-direction distances and look-back lags.

    Notes
    -----
    A split may give several local components the same persistent label. A
    merge records all plausible parent labels in ``family``. Candidate
    acceptance retains the either-direction rule of :func:`correlation_slow`,
    while the primary identity minimises the worse of the two directions. This
    prevents a small fragment contained in a large network from taking over
    that network merely because its one-way distance is zero. Inputs are never
    modified.
    """
    if R < 0:
        raise ValueError("R must be non-negative")
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    tracked = [graph.copy() for graph in graphs]
    if not tracked:
        return [], []

    initial_local_labels = _fault_labels(tracked[0])
    initial_mapping = {
        local_label: persistent_label
        for persistent_label, local_label in enumerate(initial_local_labels)
    }
    for node, data in tracked[0].nodes(data=True):
        local_label = data["fault"]
        persistent_label = initial_mapping[local_label]
        data.update(
            local_fault=local_label,
            fault=persistent_label,
            family=(persistent_label,),
            match_distance=0.0,
            match_lag=0,
        )
    next_label = len(initial_mapping)
    history = [{
        "index": 0,
        "mapping": dict(initial_mapping),
        "parents": {
            local: [persistent] for local, persistent in initial_mapping.items()
        },
        "matched": 0,
        "new": len(initial_mapping),
        "candidates": [],
    }]

    for index in range(1, len(tracked)):
        current = tracked[index]
        local_labels = _fault_labels(current)
        for _, data in current.nodes(data=True):
            data["local_fault"] = data["fault"]

        candidates = {local_label: [] for local_label in local_labels}
        candidate_records = []
        for lag in range(1, min(lookback, index) + 1):
            reference = tracked[index - lag]
            correlations, forward, backward = correlation_slow(
                reference, current, R=R, metric=metric
            )
            reference_labels = _fault_labels(reference)
            reference_index = {
                label: position for position, label in enumerate(reference_labels)
            }
            local_index = {
                label: position for position, label in enumerate(local_labels)
            }
            for persistent_label, local_label in sorted(correlations):
                i = reference_index[persistent_label]
                j = local_index[local_label]
                directed_distance = float(min(forward[i, j], backward[j, i]))
                distance = float(max(forward[i, j], backward[j, i]))
                candidate = {
                    "fault": persistent_label,
                    "distance": distance,
                    "directed_distance": directed_distance,
                    "lag": lag,
                }
                candidates[local_label].append(candidate)
                candidate_records.append({
                    "local_fault": local_label,
                    **candidate,
                })

        mapping = {}
        parents = {}
        selected = {}
        new_labels = []
        for local_label in local_labels:
            by_parent = {}
            for candidate in candidates[local_label]:
                parent = candidate["fault"]
                previous = by_parent.get(parent)
                key = (candidate["distance"], candidate["lag"])
                if previous is None or key < (
                    previous["distance"], previous["lag"]
                ):
                    by_parent[parent] = candidate
            ordered = sorted(
                by_parent.values(),
                key=lambda item: (item["distance"], item["lag"], item["fault"]),
            )
            parents[local_label] = [item["fault"] for item in ordered]
            if ordered:
                choice = ordered[0]
                mapping[local_label] = choice["fault"]
                selected[local_label] = choice
            else:
                mapping[local_label] = next_label
                parents[local_label] = [next_label]
                selected[local_label] = {
                    "fault": next_label,
                    "distance": None,
                    "lag": None,
                }
                new_labels.append(next_label)
                next_label += 1

        for _, data in current.nodes(data=True):
            local_label = data["local_fault"]
            choice = selected[local_label]
            data.update(
                fault=mapping[local_label],
                family=tuple(parents[local_label]),
                match_distance=choice["distance"],
                match_lag=choice["lag"],
            )
        history.append({
            "index": index,
            "mapping": mapping,
            "parents": parents,
            "matched": len(local_labels) - len(new_labels),
            "new": len(new_labels),
            "new_labels": new_labels,
            "candidates": candidate_records,
        })
    return tracked, history


def _velocity_difference(samples, node, dim):
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    axes = ("x", "z") if dim == 2 else ("x", "y", "z")
    positive = np.asarray(
        [samples.nodes[node, 1].get(f"v_{axis}", np.nan) for axis in axes]
    )
    negative = np.asarray(
        [samples.nodes[node, 2].get(f"v_{axis}", np.nan) for axis in axes]
    )
    return axes, positive - negative


def _ecef_to_enu(vector, lon, lat):
    longitude, latitude = np.radians([lon, lat])
    east = np.array([-np.sin(longitude), np.cos(longitude), 0])
    north = np.array([
        -np.sin(latitude) * np.cos(longitude),
        -np.sin(latitude) * np.sin(longitude),
        np.cos(latitude),
    ])
    up = np.array([
        np.cos(latitude) * np.cos(longitude),
        np.cos(latitude) * np.sin(longitude),
        np.sin(latitude),
    ])
    return np.array([east @ vector, north @ vector, up @ vector])


def _write_velocity_jump(graph, samples, dim, dt=None):
    prefix = "slip_rate" if dt is None else "slip"
    scale = 1.0 if dt is None else dt
    for node in graph:
        axes, difference = _velocity_difference(samples, node, dim)
        values = difference * scale
        for axis, value in zip(axes, values):
            graph.nodes[node][f"{prefix}_{axis}"] = float(abs(value))
        if dim == 3 and (node, 0) in samples:
            lon, lat = _as_lon_lat(samples.nodes[node, 0]["pos"])
            for axis, value in zip(("east", "north", "up"), _ecef_to_enu(values, lon, lat)):
                graph.nodes[node][f"{prefix}_{axis}"] = float(value)
        graph.nodes[node][prefix] = (
            float(np.linalg.norm(values)) if np.all(np.isfinite(values)) else np.nan
        )
    return graph


def calculate_slip_rate_sphere(graph, samples, dim=3):
    """Store the velocity jump across each trace node."""
    return _write_velocity_jump(graph, samples, dim)


def calculate_slip_sphere(graph, samples, dt, dim=3):
    """Integrate the velocity jump over a non-negative interval ``dt``."""
    if dt < 0:
        raise ValueError("dt must be non-negative")
    return _write_velocity_jump(graph, samples, dim, dt)


def write_slip_to_displacement(graph, dim=3):
    """Convert stored slip components to heave, lateral and throw."""
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    for node in graph:
        data = graph.nodes[node]
        local = dim == 3 and all(
            name in data for name in ("slip_east", "slip_north", "slip_up", "bearing")
        )
        if local and np.isfinite(data["bearing"]):
            bearing = math.radians(data["bearing"])
            east, north = data["slip_east"], data["slip_north"]
            data["heave"] = abs(east * math.cos(bearing) - north * math.sin(bearing))
            data["lateral"] = abs(east * math.sin(bearing) + north * math.cos(bearing))
            data["throw"] = abs(data["slip_up"])
            data["displacement"] = math.sqrt(east**2 + north**2 + data["slip_up"]**2)
        else:
            data["heave"] = data["slip_x"]
            if dim == 3:
                data["lateral"] = data["slip_y"]
            data["throw"] = data["slip_z"]
            data["displacement"] = data["slip"]
    return graph


__all__ = [
    "EARTH_RADIUS_KM", "EARTH_RADIUS_M", "angular_distance",
    "calculate_direction", "calculate_pickup_points", "calculate_slip_rate_sphere",
    "calculate_slip_sphere", "calculate_strike", "cartesian_to_latlon",
    "compute_similarity", "connect_across_dateline", "correlation_slow",
    "destination_point", "haversine_distance", "initial_bearing",
    "latlon_to_cartesian", "track_fault_sequence", "write_slip_to_displacement",
]
