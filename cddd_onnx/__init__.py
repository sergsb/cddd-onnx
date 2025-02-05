"""CDDD ONNX Package"""

from cddd_onnx.model_downloader import download_models, get_model_path
from cddd_onnx.main import InferenceModel

# Download models on import
download_models()

__version__ = "0.1.0"
__all__ = ["InferenceModel", "download_models", "get_model_path"]
