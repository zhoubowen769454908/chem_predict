import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from tensorflow.keras.models import load_model
from prepare_data import smiles_to_fingerprint

def predict_spectrum(smiles, model):
    fingerprint = smiles_to_fingerprint(smiles)
    if fingerprint is None:
        return None

    fingerprint = np.expand_dims(fingerprint, axis=0)
    return model.predict(fingerprint)

def draw_spectrum(spectrum):
    intensities = spectrum.flatten()
    mzs = np.arange(1, len(intensities) + 1)

    plt.stem(mzs, intensities, markerfmt=' ')
    plt.xlabel("Mass (m/z)")
    plt.ylabel("Intensity")
    plt.title("Predicted Mass Spectrum")
    plt.show()

def on_predict():
    smiles = smiles_entry.get()

    if not smiles:
        smiles = "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"  # Default SMILES input
    spectrum = predict_spectrum(smiles, model)
    print(spectrum)
    print(spectrum[0][1])
    # 找到最大的五个数值的索引
    idxs = np.argpartition(spectrum[0], -5)[-5:]
    print(idxs)

    # 创建一个全零的数组
    result1 = np.zeros_like(spectrum)

    # 将最大的五个数值设置到对应的位置
    result1[0][idxs] = spectrum[0][idxs]
    last_non_zero = np.max(np.nonzero(result1))


    # 保留最后一个非零数值以及其后的20个数值
    result = result1[0][0:last_non_zero + 21]

    print(result)
    print(result1)

    if spectrum is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            tmol=type(mol)
            print('type',tmol)
            img = MolsToGridImage([mol], molsPerRow=1, subImgSize=(300, 300))
            img.show()
        draw_spectrum(result)
    else:
        print("Invalid SMILES input")

model = load_model('spectrum_predictor2.h5')

root = tk.Tk()
root.title("Spectrum Predictor")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

smiles_label = tk.Label(frame, text="Enter SMILES:")
smiles_label.grid(row=0, column=0, sticky='w')

smiles_entry = tk.Entry(frame, width=50)

smiles_entry.grid(row=1, column=0, padx=5, pady=5)

predict_button = tk.Button(frame, text="Predict Spectrum", command=on_predict)
predict_button.grid(row=2, column=0, pady=5)

root.mainloop()