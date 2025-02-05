# CDDD-ONNX

This package provides CDDD (Continuous and Data-Driven Descriptors) models in ONNX format with automatic model downloading capabilities. This is an ONNX runtime version of the original [CDDD package](https://github.com/jrwnter/cddd).

<img src="https://github.com/jrwnter/cddd/raw/master/example/model.png" width="80%"/>

## Limitations

- Currently, only encoder model is implemented. Decoder uses a TF specific opcodes that are not supported by ONNX. I am working on a solution.

## Installation

```bash
pip install cddd-onnx
```

## Usage

### Command Line Interface

Extract molecular descriptors from SMILES using the command line interface:

```bash
python -m cddd_onnx --input smiles.smi --output descriptors.csv
```

For CSV files with a custom SMILES column header:
```bash
python -m cddd_onnx --input molecules.csv --output descriptors.csv --smiles_header smiles_column
```

### Python Interface

The format is the same as in the original CDDD package. Just import `cddd_onnx` instead of `cddd`:

```python
from cddd_onnx import InferenceModel, preprocess_smiles

# Create model instance
model = InferenceModel()

# Preprocess SMILES (if needed)
smiles_list = ["CCO", "CCN", "CC1=CC=CC=C1"]
processed_smiles = [preprocess_smiles(s) for s in smiles_list]

# Get molecular descriptors
embeddings = model.seq_to_emb(processed_smiles, batch_size=128)
```

### Input Formats
Supported input formats:
- CSV files with SMILES column
- SMI files (one SMILES per line)

## Models

The models are automatically downloaded to `~/.cddd_onnx/models/` directory when first used. 

## Requirements

- onnxruntime
- numpy
- pandas
- tqdm

## License

[MIT License](LICENSE)

## Citation

If you use this software, please cite the original CDDD paper:

```bibtex
@article{Winter2019,
  title = {Learning continuous and data-driven molecular descriptors by translating equivalent chemical representations},
  volume = {10},
  ISSN = {2041-6539},
  url = {http://dx.doi.org/10.1039/C8SC04175J},
  DOI = {10.1039/c8sc04175j},
  number = {6},
  journal = {Chemical Science},
  publisher = {Royal Society of Chemistry (RSC)},
  author = {Winter,  Robin and Montanari,  Floriane and Noé,  Frank and Clevert,  Djork-Arné},
  year = {2019},
  pages = {1692–1701}
}
