import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_json_and_code_cells_compile(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{path}:cell-{index}", "exec")


def test_publication_notebook_has_explicit_sources_and_no_environment_reads() -> None:
    path = Path(__file__).parents[1] / "notebooks" / "analyze_gemma4_sae.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    for setting in (
        "REPOSITORY_ROOT",
        "SOURCE_MODE",
        "HF_REPO_ID",
        "HF_REPO_REVISION",
        "HF_CACHE_DIR",
        "HF_LOCAL_FILES_ONLY",
        "LOCAL_CONFIG_PATH",
        "EXPLANATION_BASE",
        "EXPLANATION_JSON",
        "DEVICE",
    ):
        assert f"{setting} =" in source

    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "default_explanation_path" not in source
    assert "Auto-detected explanation report" not in source
    assert 'REPOSITORY_ROOT = ".."' in source
    assert 'SOURCE_MODE == "local" or EXPLANATION_BASE == "repository"' in source
