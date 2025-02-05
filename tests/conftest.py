"""Configuration for pytest."""
import pytest
import os

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Add any environment variables needed for testing
    os.environ["CDDD_MODELS_DIR"] = os.path.expanduser("~/.cddd-onnx/models")
