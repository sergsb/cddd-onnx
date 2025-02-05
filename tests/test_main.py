"""Test cases for main functionality."""
import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from cddd_onnx.main import InferenceModel, main
from cddd_onnx.preprocessing import preprocess_smiles

def test_inference_model_initialization():
    """Test initialization of inference model."""
    model = InferenceModel()
    assert model.encoder_session is not None
    assert model.hparams.batch_size == 128

def test_seq_to_emb():
    """Test SMILES to embedding conversion."""
    model = InferenceModel()
    smiles_list = ["CC(=O)O", "CCCCC"]  # Simple valid SMILES
    embeddings = model.seq_to_emb(smiles_list)
    
    # Check output shape and type
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings.shape) == 2
    assert embeddings.shape[0] == len(smiles_list)
    
    # Check that embeddings are not all zeros or identical
    assert not np.allclose(embeddings[0], 0)
    if len(smiles_list) > 1:
        assert not np.allclose(embeddings[0], embeddings[1])

def test_batch_size_parameter():
    """Test different batch sizes."""
    model = InferenceModel()
    smiles_list = ["CC(=O)O", "CCC", "CCCC", "CCCCC"]
    
    # Test with different batch sizes
    embeddings1 = model.seq_to_emb(smiles_list, batch_size=2)
    embeddings2 = model.seq_to_emb(smiles_list, batch_size=4)
    
    # Results should be the same regardless of batch size
    assert np.allclose(embeddings1, embeddings2,equal_nan=True)


def test_smiles_processing_against_reference():
    """Test SMILES processing against reference results."""
    # Load reference data
    ref_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "200_results.csv"))
    
    # Get original and processed SMILES
    orig_smiles = ref_df["smiles"].tolist()
    ref_processed = ref_df["new_smiles"].tolist()
    
    # Process SMILES with our implementation
    our_processed = [preprocess_smiles(smi) for smi in orig_smiles]
    
    # Compare results
    for i, (ref, our) in enumerate(zip(ref_processed, our_processed)):
        if pd.isna(ref) and pd.isna(our):
            continue
        assert ref == our, f"SMILES processing mismatch at index {i}"

def test_descriptors_against_reference():
    """Test descriptor generation against reference results."""
    # Load reference data
    ref_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "200_results.csv"))
    
    # Get original SMILES and reference descriptors
    smiles_list = ref_df["smiles"].tolist()
    ref_descriptors = ref_df[[col for col in ref_df.columns if col.startswith("cddd_")]].values
    # Generate descriptors with our implementation
    model = InferenceModel()
    our_descriptors = model.seq_to_emb(smiles_list)
    # Compare results
    assert our_descriptors.shape == ref_descriptors.shape
    
    # Compare non-nan values
    valid_mask = ~pd.isna(ref_descriptors).any(axis=1)
    ref_valid = ref_descriptors[valid_mask]
    our_valid = our_descriptors[valid_mask]
    
    assert np.allclose(ref_valid, our_valid, rtol=1e-5, atol=1e-5)
