"""Every script compiles: a syntax error in a script that runs late in a batch would stop it after hours."""
import py_compile
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=[script.name for script in SCRIPTS])
def test_script_compiles(script):
    py_compile.compile(str(script), doraise=True, cfile=None)
