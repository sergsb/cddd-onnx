import os
import hashlib
import requests
from tqdm import tqdm

# Configuration for model downloads
MODELS = {
    "encoder": {
        "url": "https://zenodo.org/records/14811055/files/encoder.onnx?download=1",
        "md5": "35ef1902b85e67abfb7abed9fa06ffc2",
        "filename": "encoder.onnx"
    }
}

def get_models_dir():
    """Get the directory where models are stored."""
    home = os.path.expanduser("~")
    models_dir = os.path.join(home, ".cddd_onnx", "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return models_dir

def calculate_md5(file_path):
    """Calculate MD5 hash of a file."""
    return hashlib.md5(open(file_path, "rb").read()).hexdigest()

def download_file(url, target_path, expected_md5=None):
    """Download a file with progress bar and MD5 verification."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(target_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))
    
    if expected_md5:
        actual_md5 = calculate_md5(target_path)
        if actual_md5 != expected_md5:
            os.remove(target_path)
            raise ValueError(f"MD5 verification failed for {target_path}")

def get_model_path(model_name):
    """Get the path to a specific model."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    models_dir = get_models_dir()
    return os.path.join(models_dir, MODELS[model_name]["filename"])

def download_models():
    """Download all models if they don't exist or MD5 doesn't match."""
    models_dir = get_models_dir()
    
    for model_name, model_info in MODELS.items():
        model_path = os.path.join(models_dir, model_info["filename"])
        
        # Skip if file exists and MD5 matches
        if os.path.exists(model_path):
            if calculate_md5(model_path) == model_info["md5"]:
                print(f"Model {model_name} already exists and MD5 matches")
                continue
            else:
                print(f"MD5 mismatch for {model_name}, re-downloading...")
        
        print(f"Downloading {model_name}...")
        try:
            download_file(
                model_info["url"],
                model_path,
                expected_md5=model_info["md5"]
            )
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")
            if os.path.exists(model_path):
                os.remove(model_path)  # Remove partially downloaded file
            raise
