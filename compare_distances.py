import streamlit as st
import numpy as np
import json
from scipy.stats import entropy
from scipy.stats import wasserstein_distance

# Streamlitアプリのタイトル
st.title("KL Divergence and Earth Mover's Distance Calculator")

# ファイルアップロード
st.sidebar.header("Upload your JSON files")
uploaded_fileA = st.sidebar.file_uploader("Upload distancesA.json", type="json")
uploaded_fileB = st.sidebar.file_uploader("Upload distancesB.json", type="json")

# ファイルがアップロードされているか確認
if uploaded_fileA is not None and uploaded_fileB is not None:
    # データの読み込み
    distancesA = json.load(uploaded_fileA)
    distancesB = json.load(uploaded_fileB)

    # データの長さを確認
    if len(distancesA) != len(distancesB):
        st.error("Both datasets must have the same size")
    else:
        # スタージェスの公式でビン数を決定
        n = len(distancesA)
        bins = int(np.ceil(np.log2(n) + 1))

        # ヒストグラムの作成
        histA, bin_edgesA = np.histogram(distancesA, bins=bins, density=True)
        histB, bin_edgesB = np.histogram(distancesB, bins=bins, density=True)

        # ゼロ確率を避けるため小さな値を足す
        histA += 1e-10
        histB += 1e-10

        # KLダイバージェンスの計算
        kl_divergence = entropy(histA, histB)

        # ヒストグラムの平均を取る（EMD計算のため）
        bin_centers = (bin_edgesA[:-1] + bin_edgesA[1:]) / 2

        # Earth Mover距離の計算
        emd = wasserstein_distance(bin_centers, bin_centers, histA, histB)

        # 結果を表示
        st.subheader("Results")
        st.write("KL Divergence:", kl_divergence)
        st.write("Earth Mover Distance:", emd)

        # ヒストグラムの表示
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(distancesA, bins=bins, alpha=0.5, label='distancesA', density=True)
        ax.hist(distancesB, bins=bins, alpha=0.5, label='distancesB', density=True)
        ax.legend(loc='upper right')
        st.pyplot(fig)
else:
    st.info("Please upload both JSON files to proceed")
