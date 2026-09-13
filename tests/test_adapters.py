"""The adapters' own self-checks, run under pytest."""
from phase_lora.polar import demo as polar_demo
from phase_lora.unitary import demo as unitary_demo


def test_polar():
    polar_demo()


def test_unitary():
    unitary_demo()
