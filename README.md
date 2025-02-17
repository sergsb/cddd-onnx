# CDDD-ONNX

This package provides CDDD (Continuous and Data-Driven Descriptors) models in ONNX format with automatic model downloading capabilities. This is an ONNX runtime version of the original [CDDD package](https://github.com/jrwnter/cddd).

<img src="https://github.com/jrwnter/cddd/raw/master/example/model.png" width="80%"/>

## Limitations

* Currently, only the encoder model is implemented. The decoder uses TensorFlow-specific opcodes that are not supported by ONNX. I am working on a solution.
## Installation

```bash
pip install cddd-onnx
```

## Usage

### Command Line Interface

Extract molecular descriptors from SMILES using the command line interface:

```bash
cddd-onnx --input smiles.smi --output descriptors.csv
```

For CSV files with a custom SMILES column header:
```bash
cddd-onnx --input molecules.csv --output descriptors.csv --smiles_header smiles_column
```

### Python Interface

The format is the same as in the original CDDD package. Just import `cddd_onnx` instead of `cddd`:

```python
from cddd_onnx import InferenceModel
# Create model instance
model = InferenceModel()

smiles_list = ["CCCCO", "CCCN", "CC1=CC=CC=C1"]
embeddings = model.seq_to_emb(smiles_list)
```
* The preprocessing stage is inside the seq_to_emb function, so you do not need to run it separately.
* Be aware that if the SMILES is out of AD, the preprocessor returns None, resulting in a row of NaNs for such compounds.
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
- rdkit

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
