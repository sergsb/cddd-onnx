import os
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Union
import onnxruntime as ort
from cddd_onnx.tokenizer import InputPipelineInferEncode
from cddd_onnx.preprocessing import preprocess_smiles

@dataclass
class HParams:
    """Hyperparameters for the model."""
    batch_size: int = 128

class InferenceModel:
    """CDDD Inference Model for encoding SMILES to embeddings and back."""
    def __init__(self):
        self.hparams = HParams()
        """Initialize the inference model."""
        # self.hparams = HParams()
        from cddd_onnx.model_downloader import get_model_path
        encoder_path = get_model_path("encoder")
        
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Model file not found at {encoder_path}. "
                "Please run the model_downloader script first."
            )
            
        self.encoder_session = ort.InferenceSession(encoder_path)

    def seq_to_emb(self, smiles_list: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Encode a list of SMILES strings into molecular descriptors.

        Args:
            smiles_list: List of SMILES strings to encode
            batch_size: Optional batch size for inference (default: 128)

        Returns:
            numpy.ndarray: Molecular descriptors for the input SMILES
        """
        if batch_size:
            self.hparams.batch_size = batch_size

        # Process SMILES and get preprocessed versions
        processed_smiles = [preprocess_smiles(smi) for smi in smiles_list]
        valid_mask = [not pd.isna(smi) for smi in processed_smiles]
        valid_smiles = [smi for smi, valid in zip(processed_smiles, valid_mask) if valid]
        
        if not valid_smiles:
            raise ValueError("No valid SMILES found after preprocessing")
            
        # Initialize input pipeline with valid SMILES
        input_pipeline = InputPipelineInferEncode(valid_smiles, self.hparams)
        input_pipeline.initialize()
        emb_list = []
        
        while True:
            try:
                # Get next batch
                input_seq, input_len = input_pipeline.get_next()
                
                # Run inference using ONNX Runtime
                outputs = self.encoder_session.run(
                    None,  # output names - passing None means return all outputs
                    {
                        'Input/Placeholder:0': input_seq.astype(np.int32),
                        'Input/Placeholder_1:0': input_len.astype(np.int32)
                    }
                )
                emb_list.append(outputs[0])
            except StopIteration:
                break
            
        if emb_list:
            embeddings = np.vstack(emb_list)
        else:
            embeddings = np.array([])
            
        # Create a mapping of original SMILES to their embeddings
        result_embeddings = []
        for smi, valid in zip(smiles_list, valid_mask):
            if valid:
                result_embeddings.append(embeddings[0])
                embeddings = embeddings[1:]
            else:
                result_embeddings.append(np.full(embeddings[0].shape, np.nan))
                
        return np.array(result_embeddings)

def main():
    """Command-line interface entry point."""
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='CDDD ONNX Model Interface')
    parser.add_argument('--input', required=True, help='Input file (.csv or .smi)')
    parser.add_argument('--output', required=True, help='Output file for descriptors (.csv)')
    parser.add_argument('--smiles_header', default='smiles', help='Header of SMILES column in CSV (default: smiles)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference (default: 128)')
    
    args = parser.parse_args()
    
    # Read input file
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
        smiles_list = df[args.smiles_header].tolist()
    elif args.input.endswith('.smi'):
        with open(args.input, 'r') as f:
            smiles_list = [line.strip() for line in f]
    else:
        raise ValueError("Input file must be .csv or .smi")
    
    # Create model and get embeddings
    model = InferenceModel()
    processed_smiles = [preprocess_smiles(smi) for smi in smiles_list]
    embeddings = model.seq_to_emb(smiles_list, args.batch_size)
    
    # Create result dataframe with SMILES and descriptors
    result_df = pd.DataFrame(embeddings)
    result_df.columns = [f'descriptor_{i}' for i in range(embeddings.shape[1])]
    result_df.insert(0, 'smiles', smiles_list)
    result_df.insert(1, 'new_smiles', processed_smiles)
    
    # Save results
    result_df.to_csv(args.output, index=False)
    print(f"Saved descriptors to {args.output}")

if __name__ == "__main__":
    main()
