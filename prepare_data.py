import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import train_test_split
import numpy as np
from rdkit import Chem

def spectra_to_vector(spectra, size=500):
    vector = np.zeros(size)
    for pair in spectra:
        index = int(round(float(pair[0])))
        if 0 <= index < size:
            vector[index] = float(pair[1])
    return vector
def load_data():
    data = np.load('output2.npz',allow_pickle=True)
    spectra = data['spectra_list']
    print(spectra[0])

    spectra_vectors = [spectra_to_vector(s) for s in spectra]
    print(spectra_vectors[0])
    # metadata = pd.DataFrame(data['metadata'], columns=['SMILES', 'InChI', 'InChI Key', 'Name'])
    smile = data['smile']
    return smile, spectra

def smiles_to_fingerprint(smiles, fp_size=1024, fp_radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_size))

def prepare_dataset():
    smiles, spectra = load_data()

    fingerprints = []
    valid_indices = []

    for i, s in enumerate(smiles):
        fp = smiles_to_fingerprint(s)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(i)

    fingerprints = np.vstack(fingerprints)
    spectra = spectra[valid_indices]
    spectra_vectors = [spectra_to_vector(s) for s in spectra]
    print(spectra[0])
    return train_test_split(fingerprints, spectra_vectors, test_size=0.2, random_state=42)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_dataset()
    np.savez('dataset2.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
