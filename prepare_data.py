from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import train_test_split
import numpy as np
from rdkit import Chem

def smiles_to_fingerprint(smiles, fp_size=1024, fp_radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_size))

