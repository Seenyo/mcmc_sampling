import streamlit as st
import numpy as np
import json
from scipy.stats import entropy, wasserstein_distance, ks_2samp, anderson_ksamp
from scipy.spatial.distance import jensenshannon
import plotly.graph_objects as go

# Streamlitアプリのタイトル
st.title("Comprehensive Distribution Distance Calculator")

# ファイルアップロード
st.sidebar.header("Upload your JSON files")
uploaded_fileA = st.sidebar.file_uploader("Upload distancesA.json", type="json")
uploaded_fileB = st.sidebar.file_uploader("Upload distancesB.json", type="json")

# ヘリンガー距離の計算関数
def hellinger_distance(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

# バタチャリヤ距離の計算関数
def bhattacharyya_distance(p, q):
    return -np.log(np.sum(np.sqrt(p * q)))

# ファイルがアップロードされているか確認
if uploaded_fileA is not None and uploaded_fileB is not None:
    # データの読み込み
    distancesA = json.load(uploaded_fileA)
    distancesB = json.load(uploaded_fileB)

    # データの長さを確認
    if len(distancesA) != len(distancesB):
        st.error("Both datasets must have the same size")
    else:
        # rice's ruleでビン数を決定
        n = len(distancesA)
        bins = int(np.ceil(2 * n ** (1 / 3)))

        # ヒストグラムの作成
        histA, bin_edgesA = np.histogram(distancesA, bins=bins, density=True, range=(0, 1))
        histB, bin_edgesB = np.histogram(distancesB, bins=bins, density=True, range=(0, 1))

        bin_centers = (bin_edgesA[:-1] + bin_edgesA[1:]) / 2

        normalized_histA = histA / bin_centers
        normalized_histB = histB / bin_centers

        # ゼロ確率を避けるため小さな値を足す
        normalized_histA += 1e-10
        normalized_histB += 1e-10

        # 各距離の計算
        kl_divergence = entropy(normalized_histA, normalized_histB)
        emd = wasserstein_distance(bin_centers, bin_centers, normalized_histA, normalized_histB)
        js_distance = jensenshannon(normalized_histA, normalized_histB)
        hellinger_dist = hellinger_distance(normalized_histA, normalized_histB)
        bhattacharyya_dist = bhattacharyya_distance(normalized_histA, normalized_histB)

        # コルモゴロフ-スミルノフ検定
        ks_statistic, ks_pvalue = ks_2samp(distancesA, distancesB)

        # アンダーソン・ダーリング検定
        ad_result = anderson_ksamp([distancesA, distancesB])
        ad_statistic = ad_result.statistic
        ad_pvalue = ad_result.significance_level

        # 結果を表示
        st.subheader("Results")
        st.write("KL Divergence:", kl_divergence)
        st.write("Earth Mover Distance:", emd)
        st.write("Jensen-Shannon Distance:", js_distance)
        st.write("Hellinger Distance:", hellinger_dist)
        st.write("Bhattacharyya Distance:", bhattacharyya_dist)
        st.write("Kolmogorov-Smirnov Statistic:", ks_statistic)
        st.write("Kolmogorov-Smirnov p-value:", ks_pvalue)
        st.write("Anderson-Darling Statistic:", ad_statistic)
        st.write("Anderson-Darling p-value:", ad_pvalue)

        # ヒストグラムの表示
        fig = go.Figure(go.Bar(x=bin_centers, y=normalized_histA, name='distancesA'))
        fig.add_trace(go.Bar(x=bin_centers, y=normalized_histB, name='distancesB'))
        fig.update_traces(opacity=0.75)
        fig.update_layout(title='Distribution Comparison', xaxis_title='Distance', yaxis_title='Density')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload both JSON files to proceed")
