"""The adapters' own self-checks, run under pytest (CPU only)."""
from phase_lora.polar import demo as polar_demo
from phase_lora.stiefel import demo as polar_stiefel_demo


def test_polar():
    polar_demo()


def test_polar_stiefel():
    polar_stiefel_demo()
