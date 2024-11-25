import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from tensorflow.keras.models import load_model
from prepare_data import smiles_to_fingerprint

# 加载模型
model = load_model('spectrum_predictor2.h5')

# 预测光谱的函数
def predict_spectrum(smiles, model):
    fingerprint = smiles_to_fingerprint(smiles)
    if fingerprint is None:
        return None

    fingerprint = np.expand_dims(fingerprint, axis=0)
    return model.predict(fingerprint)

# 绘制光谱图
def draw_spectrum(spectrum):
    intensities = spectrum.flatten()
    mzs = np.arange(1, len(intensities) + 1)

    plt.figure(figsize=(8, 4))
    plt.stem(mzs, intensities, markerfmt=' ', basefmt=' ')
    plt.xlabel("Mass (m/z)")
    plt.ylabel("Intensity")
    plt.title("Predicted Mass Spectrum")
    st.pyplot(plt)

# Streamlit App
st.title("Spectrum Predictor")

# 输入框
smiles = st.text_input("Enter SMILES:", "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1")

# 预测按钮
if st.button("Predict Spectrum"):
    spectrum = predict_spectrum(smiles, model)

    if spectrum is not None:
        # 找到最大的五个数值的索引
        idxs = np.argpartition(spectrum[0], -5)[-5:]
        # 创建一个全零的数组
        result1 = np.zeros_like(spectrum)
        # 将最大的五个数值设置到对应的位置
        result1[0][idxs] = spectrum[0][idxs]
        last_non_zero = np.max(np.nonzero(result1))
        # 保留最后一个非零数值以及其后的20个数值
        result = result1[0][0:last_non_zero + 21]

        # 绘制分子图
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            img = MolsToGridImage([mol], molsPerRow=1, subImgSize=(300, 300))
            st.image(img, caption="Molecular Structure", use_column_width=True)

        # 绘制光谱图
        draw_spectrum(result)
    else:
        st.error("Invalid SMILES input!")
