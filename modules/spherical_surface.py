"""Prepared raster helpers for spherical Fatbox tutorials.

The spherical numerical core lives in :mod:`spherical`. This module contains
only the small amount of raster handling needed by the teaching notebooks.
The input may come from any numerical model or gridded spherical dataset.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

try:
    from . import spherical
except ImportError:  # Support the historical ``sys.path += modules`` workflow.
    import spherical


def read_surface_archive(path):
    """Read a prepared spherical tutorial archive.

    The archive must contain a two-dimensional strain-rate raster, a
    three-component surface-velocity raster, output number, model time and
    grid shape. It contains no numerical-model mesh or unrelated fields.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"strain", "velocity", "output_number", "time_years", "grid_shape"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"surface archive is missing: {sorted(missing)}")
        strain = np.asarray(archive["strain"], dtype=float)
        velocity = np.asarray(archive["velocity"], dtype=float)
        grid_shape = tuple(int(value) for value in archive["grid_shape"])
        output_number = int(archive["output_number"])
        time_years = float(archive["time_years"])
    if strain.shape != grid_shape or velocity.shape != grid_shape + (3,):
        raise ValueError("surface archive arrays do not match grid_shape")
    return {
        "strain": strain,
        "velocity": velocity,
        "output_number": output_number,
        "time_years": time_years,
        "grid_shape": grid_shape,
    }


def extract_fault_graph(strain, threshold, minimum_component_size=5):
    """Extract a labelled, periodic 8-neighbour graph from a strain raster."""
    strain = np.asarray(strain, dtype=float)
    if strain.ndim != 2:
        raise ValueError("strain must be a two-dimensional raster")
    if minimum_component_size < 1:
        raise ValueError("minimum_component_size must be positive")
    mask = strain > threshold
    skeleton = skeletonize(np.pad(mask, ((0, 0), (2, 2)), mode="wrap"))[:, 2:-2]
    nlat, nlon = skeleton.shape
    graph = nx.Graph()
    pixel_to_node = {}
    for y, x in np.argwhere(skeleton):
        node = len(pixel_to_node)
        pixel_to_node[int(y), int(x)] = node
        graph.add_node(
            node,
            pos=(-180 + (x + 0.5) * 360 / nlon, 90 - (y + 0.5) * 180 / nlat),
            pixel=(int(y), int(x)),
        )
    for (y, x), node in pixel_to_node.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbour = pixel_to_node.get((y + dy, (x + dx) % nlon))
                if (dy or dx) and neighbour is not None and neighbour > node:
                    graph.add_edge(node, neighbour)
    small = {
        node
        for component in nx.connected_components(graph)
        if len(component) < minimum_component_size
        for node in component
    }
    graph.remove_nodes_from(small)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    for label, component in enumerate(nx.connected_components(graph)):
        for node in component:
            graph.nodes[node]["fault"] = label
    for u, v in graph.edges:
        lon1, lat1 = graph.nodes[u]["pos"]
        lon2, lat2 = graph.nodes[v]["pos"]
        graph.edges[u, v]["length"] = float(
            spherical.haversine_distance(lat1, lon1, lat2, lon2)
        )
        graph.edges[u, v]["strike"] = spherical.calculate_strike(
            (lon1, lat1), (lon2, lat2)
        )
    return graph, mask, skeleton


def sample_raster(graph, raster, name, channel=None):
    """Sample a global north-up raster at graph-node lon/lat positions."""
    data = np.asarray(raster)
    if data.ndim == 3:
        if channel is None:
            raise ValueError("channel is required for a multi-channel raster")
        data = data[..., channel]
    if data.ndim != 2:
        raise ValueError("raster must be two-dimensional after channel selection")
    nlat, nlon = data.shape
    for node in graph:
        lon, lat = graph.nodes[node]["pos"]
        if not -90 <= lat <= 90:
            raise ValueError(f"latitude must be in [-90, 90], got {lat}")
        x = int(np.rint((lon + 180) / 360 * (nlon - 1)))
        y = int(np.rint((90 - lat) / 180 * (nlat - 1)))
        graph.nodes[node][name] = (
            data[y, x] if 0 <= x < nlon and 0 <= y < nlat else np.nan
        )
    return graph


def sample_velocity_difference(graph, velocity_grid, pickup_degrees, cutoff=2):
    """Sample three-component velocities across a trace and calculate the jump."""
    spherical.calculate_direction(graph, cutoff=cutoff)
    samples = spherical.calculate_pickup_points(graph, pickup_degrees)
    for channel, name in enumerate(("v_x", "v_y", "v_z")):
        sample_raster(samples, velocity_grid, name, channel=channel)
    spherical.calculate_slip_rate_sphere(graph, samples, dim=3)
    return samples


__all__ = [
    "extract_fault_graph",
    "read_surface_archive",
    "sample_raster",
    "sample_velocity_difference",
]
