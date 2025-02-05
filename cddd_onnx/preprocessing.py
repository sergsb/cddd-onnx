"""
Functions for preprocessing SMILES sequences.
Mostly inherited from https://raw.githubusercontent.com/jrwnter/cddd/refs/heads/master/cddd/preprocessing.py 
"""
import numpy as np
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import Descriptors

REMOVER = SaltRemover()
ORGANIC_ATOM_SET = set([5, 6, 7, 8, 9, 15, 16, 17, 35, 53])

def keep_largest_fragment(sml):
    """Function that returns the SMILES sequence of the largest fragment.
    
    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequence of the largest fragment.
    """
    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(sml), asMols=True)
    largest_mol = None
    largest_mol_size = 0
    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_mol_size:
            largest_mol = mol
            largest_mol_size = size
    return Chem.MolToSmiles(largest_mol)

def remove_salt_stereo(sml, remover):
    """Function that strips salts and removes stereochemistry information from a SMILES.
    
    Args:
        sml: SMILES sequence.
        remover: RDKit's SaltRemover object.
    Returns:
        canonical SMILES sequence without salts and stereochemistry information.
    """
    try:
        mol = Chem.MolFromSmiles(sml)
        if mol is None:
            return float('nan')
            
        mol = remover.StripMol(mol, dontRemoveEverything=True)
        if mol is None:
            return float('nan')
            
        sml = Chem.MolToSmiles(mol, isomericSmiles=False)
        if "." in sml:
            sml = keep_largest_fragment(sml)
    except:
        return float('nan')
    return sml

def organic_filter(sml):
    """Function that filters for organic molecules.
    
    Args:
        sml: SMILES sequence.
    Returns:
        True if sml can be interpreted by RDKit and is organic.
        False if sml cannot interpreted by RDKIT or is inorganic.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        if m is None:
            return False
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = (set(atom_num_list) <= ORGANIC_ATOM_SET)
        return is_organic
    except:
        return False

def filter_smiles(sml):
    """Filter SMILES based on molecular properties.
    
    Args:
        sml: SMILES sequence.
    Returns:
        Filtered SMILES or nan if outside applicability domain.
    """
    try:
        if isinstance(sml, float) and np.isnan(sml):
            return float('nan')
            
        m = Chem.MolFromSmiles(sml)
        if m is None:
            return float('nan')
            
        logp = Descriptors.MolLogP(m)
        mol_weight = Descriptors.MolWt(m)
        num_heavy_atoms = Descriptors.HeavyAtomCount(m)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = set(atom_num_list) <= ORGANIC_ATOM_SET
        
        if ((logp > -5) & (logp < 7) &
            (mol_weight > 12) & (mol_weight < 600) &
            (num_heavy_atoms > 3) & (num_heavy_atoms < 50) &
            is_organic):
            return Chem.MolToSmiles(m)
        else:
            return float('nan')
    except:
        return float('nan')

def preprocess_smiles(sml):
    """Preprocess a SMILES string by removing salts, stereochemistry, and filtering.
    
    Args:
        sml: SMILES sequence.
    Returns:
        Preprocessed SMILES sequence or nan if invalid/outside domain.
    """
    new_sml = remove_salt_stereo(sml, REMOVER)
    if isinstance(new_sml, float) and np.isnan(new_sml):
        return float('nan')
    new_sml = filter_smiles(new_sml)
    return new_sml
