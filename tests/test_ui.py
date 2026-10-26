"""
NetSentinel SOC — Day 17: Network Intrusion Detection
UI Test Suite using Streamlit AppTest to verify component rendering and state transitions.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs("/tmp/matplotlib", exist_ok=True)

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_initial_render():
    """Verifies that the Streamlit application starts up and renders without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App threw an unhandled exception: {at.exception}"

    # Verify app rendered without exceptions and contains markdown components
    assert not at.exception
    assert len(at.markdown) > 0


def test_app_sidebar_controls():
    """Verifies that the sidebar controls and buttons execute properly."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Find and trigger 'Ingest 1 Flow' button
    if len(at.sidebar.button) > 0:
        at.sidebar.button[0].click().run()
        assert not at.exception
