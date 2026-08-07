import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
DATA = ROOT / "tutorials" / "numerical_modelling" / "data" / "spherical_example"
sys.path.insert(0, str(ROOT / "modules"))

import spherical_surface


def test_bundled_surface_sequence_is_small_complete_and_verified():
    manifest = json.loads((DATA / "manifest.json").read_text())
    entries = manifest["outputs"]
    assert [entry["number"] for entry in entries] == list(range(220, 231))
    assert np.all(np.diff([entry["time_years"] for entry in entries]) > 0)
    total_size = 0
    for entry in entries:
        path = DATA / entry["file"]
        total_size += path.stat().st_size
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        surface = spherical_surface.read_surface_archive(path)
        assert surface["output_number"] == entry["number"]
        assert surface["grid_shape"] == (180, 360)
        assert np.all(np.isfinite(surface["strain"]))
        assert np.all(np.isfinite(surface["velocity"]))
    assert total_size < 15 * 1024 * 1024
