"""Guards that keep declared dependency majors honest.

`uv.lock` is gitignored here (library convention: declare ranges, let consumers resolve)
and CI installs by resolving `pyproject.toml` fresh, so this file is the *only* constraint
that survives a clone. An open `opencv-contrib-python>=4.9` is how OpenCV 5 arrived
unannounced — it changed the ChArUco point-array layout out from under
`drawDetectedCornersCharuco`, every test stayed green, and live calibration crashed the
moment a board entered frame. These two tests make that drift a red build instead.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import re
import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _declared_requirements() -> dict[str, str]:
    """Every declared requirement (runtime + extras) as ``{distribution: specifier}``."""
    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    specifiers = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        specifiers.extend(extra)
    return {re.split(r"[><=!~@\[]", s, maxsplit=1)[0].strip(): s for s in specifiers}


@pytest.mark.parametrize("distribution", sorted(_declared_requirements()))
def test_every_dependency_declares_an_upper_bound(distribution: str) -> None:
    specifier = _declared_requirements()[distribution]
    assert "<" in specifier, f"{specifier!r} has no upper bound — pin the tested major"


@pytest.mark.parametrize("distribution", sorted(_declared_requirements()))
def test_installed_version_satisfies_the_declaration(distribution: str) -> None:
    packaging_requirements = pytest.importorskip("packaging.requirements")
    specifier = _declared_requirements()[distribution]
    try:
        installed = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        pytest.skip(f"{distribution} not installed (optional extra)")
    requirement = packaging_requirements.Requirement(specifier)
    assert requirement.specifier.contains(installed, prereleases=True), (
        f"pyproject declares {specifier!r} but {distribution} {installed} is installed"
    )
