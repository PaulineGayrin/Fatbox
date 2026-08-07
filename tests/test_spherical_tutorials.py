import json
from pathlib import Path


TUTORIALS = Path(__file__).parents[1] / "tutorials" / "numerical_modelling"


def load_notebook(name):
    notebook = json.loads((TUTORIALS / name).read_text())
    assert notebook["nbformat"] == 4
    return notebook


def code_source(notebook):
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def markdown_source(notebook):
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )


def test_spherical_extraction_tutorial_is_valid_and_stepwise():
    notebook = load_notebook("tuto_num3_spherical_extraction.ipynb")
    source = code_source(notebook)
    compile(source, "tuto_num3_spherical_extraction.ipynb", "exec")
    markdown = markdown_source(notebook)
    assert "Read one prepared spherical surface archive" in markdown
    assert "Apply the extraction to several outputs" in markdown
    assert "Southern Hemisphere" in markdown
    assert "G_XXXXX.pickle" in markdown
    assert "spherical_surface.extract_fault_graph" in source
    assert "output_numbers = available_outputs" in source
    assert "spherical_surface.read_surface_archive" in source
    assert "/Volumes/" not in source
    assert "Extraction consistency through time" in source


def test_spherical_correlation_tutorial_is_valid_and_sequential():
    notebook = load_notebook("tuto_num4_spherical_correlation_slip.ipynb")
    source = code_source(notebook)
    compile(source, "tuto_num4_spherical_correlation_slip.ipynb", "exec")
    markdown = markdown_source(notebook)
    assert "Load the independent graphs" in markdown
    assert "First understand one pair" in markdown
    assert "complete sequence" in markdown
    assert "look back to `n−2`" in markdown
    assert "Calculate slip for every interval" in markdown
    assert "G_0, G_1 =" in source and "H =" in source
    assert "metric='great_circle'" in source
    assert "spherical.track_fault_sequence" in source
    assert "for index in range(1, len(tracked_graphs))" in source
    assert "spherical_surface.read_surface_archive" in source
    assert "/Volumes/" not in source
