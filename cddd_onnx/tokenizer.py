"""SMILES tokenization functionality."""
import re
import numpy as np
import pandas as pd
from cddd_onnx.preprocessing import preprocess_smiles

REGEX_SML = r'Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'
REGEX_INCHI = r'Br|Cl|[\(\)\+,-/123456789CFHINOPSchpq]'
voc = {'#': 1, '%': 2, '(': 4, ')': 3, '+': 5, '-': 6, '0': 8, '1': 7, '2': 10, '3': 9, '4': 12, '5': 11, '6': 14, '7': 13, '8': 16, '9': 15, ':': 17, '</s>': 0, '<s>': 39, '=': 18, '@': 19, 'B': 21, 'Br': 38, 'C': 20, 'Cl': 37, 'F': 22, 'H': 24, 'I': 23, 'N': 26, 'O': 25, 'P': 27, 'S': 28, '[': 29, ']': 30, 'c': 31, 'i': 32, 'n': 34, 'o': 33, 'p': 35, 's': 36}

class InputPipelineInferEncode():
    """Class that creates a python generator for list of sequnces. Used to feed
    sequnces to the encoing part during inference time.

    Atributes:
        seq_list: List with sequnces to iterate over.
        batch_size: Number of samples to output per iterator call.
        input strings.
    """

    def __init__(self, seq_list, hparams):
        """Constructor for the inference input pipeline class.

        Args:
            seq_list: List with sequnces to iterate over.
            hparams: Hyperparameters for the model, leaving it in the code just for compatibility with the original code, uses just one parameter: batch_size.
        Returns:
            None
        """
        # Preprocess SMILES and filter invalid ones
        processed_smiles = [preprocess_smiles(smi) for smi in seq_list]
        valid_mask = [not pd.isna(smi) for smi in processed_smiles]
        self.seq_list = [smi for smi, valid in zip(processed_smiles, valid_mask) if valid]
        
        if not self.seq_list:
            raise ValueError("No valid SMILES found after preprocessing")
            
        self.batch_size = hparams.batch_size
        self.encode_vocabulary = voc
        self.generator = None

    def _input_generator(self):
        """Function that defines the generator."""
        l = len(self.seq_list)
        for ndx in range(0, l, self.batch_size):
            samples = self.seq_list[ndx:min(ndx + self.batch_size, l)]
            samples = [self._seq_to_idx(seq) for seq in samples]
            seq_len_batch = np.array([len(entry) for entry in samples])
            # pad sequences to max len and concatenate to one array
            max_length = seq_len_batch.max()
            seq_batch = np.concatenate(
                [np.expand_dims(
                    np.append(
                        seq,
                        np.array([self.encode_vocabulary['</s>']]*(max_length - len(seq)))
                    ),
                    0
                )
                    for seq in samples]
            ).astype(np.int32)
            yield seq_batch, seq_len_batch

    def initialize(self):
        """Helper function to initialize the generator"""
        self.generator = self._input_generator()

    def get_next(self):
        """Helper function to get the next batch from the iterator"""
        if self.generator is None:
            self.initialize()
        return next(self.generator)

    def _char_to_idx(self, seq):
        """Helper function to tokenize a sequnce.

        Args:
            seq: Sequence to tokenize.
        Returns:
            List with ids of the tokens in the tokenized sequnce.
        """
        char_list = re.findall(REGEX_SML, seq)
        return [self.encode_vocabulary[char_list[j]] for j in range(len(char_list))]

    def _seq_to_idx(self, seq):
        """Method that tokenizes a sequnce and pads it with start and stop token.

        Args:
            seq: Sequence to tokenize.
        Returns:
            seq: List with ids of the tokens in the tokenized sequnce.
        """
        seq = np.concatenate([np.array([self.encode_vocabulary['<s>']]),
                              np.array(self._char_to_idx(seq)).astype(np.int32),
                              np.array([self.encode_vocabulary['</s>']])
                              ]).astype(np.int32)
        return seq
