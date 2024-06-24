import streamlit as st
import os
import json
import numpy as np
import plotly.graph_objects as go
from scipy.stats import entropy, wasserstein_distance, ks_2samp, anderson_ksamp
from scipy.spatial.distance import jensenshannon

# ディレクトリ内のJSONファイルを読み込む関数
def load_json_files(directory):
    data = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "distances.json":
                file_path = os.path.join(root, file)
                parts = os.path.relpath(root, directory).split(os.path.sep)
                if len(parts) >= 1:  # parts[0]が必要な情報を含むと仮定
                    num_particles, num_chains = parts[0].split('_')
                    if num_particles not in data:
                        data[num_particles] = {}
                    with open(file_path, 'r') as f:
                        data[num_particles][num_chains] = json.load(f)
    return data

# 分布間の距離を計算する関数
def calculate_distances(distancesA, distancesB):

    if len(distancesA) != len(distancesB):
        return {
            'KL Divergence': np.nan,
            'Earth Mover Distance': np.nan,
            'Jensen-Shannon Distance': np.nan,
            'Hellinger Distance': np.nan,
            'Bhattacharyya Distance': np.nan,
            'Kolmogorov-Smirnov Statistic': np.nan,
            'Kolmogorov-Smirnov p-value': np.nan,
            'Anderson-Darling Statistic': np.nan,
            'Anderson-Darling p-value': np.nan
        }

    n = len(distancesA)

    # rice's ruleでビン数を決定
    bins = int(np.ceil(2 * n ** (1 / 3)))

    histA, bin_edgesA = np.histogram(distancesA, bins=bins, density=True, range=(0, 1))
    histB, bin_edgesB = np.histogram(distancesB, bins=bins, density=True, range=(0, 1))

    bin_centers = (bin_edgesA[:-1] + bin_edgesA[1:]) / 2

    normalized_histA = histA / bin_centers
    normalized_histB = histB / bin_centers

    # ゼロ確率を避けるため小さな値を足す
    normalized_histA += 1e-10
    normalized_histB += 1e-10

    kl_divergence = entropy(normalized_histA, normalized_histB)
    emd = wasserstein_distance(bin_centers, bin_centers, normalized_histA, normalized_histB)

    return {
        'KL Divergence': kl_divergence,
        'Earth Mover Distance': emd,
    }


# データをプロットする関数
def plot_comparative_distances(results):
    for num_particles, data in results.items():
        fig = go.Figure()
        x_values = list(data.keys())
        for metric in data[x_values[0]].keys():
            y_values = [data[x][metric] for x in x_values]
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=metric
            ))

        fig.update_layout(
            title=f'Comparative Distribution Distances for {num_particles} Particles',
            xaxis_title='Mutation Stride',
            yaxis_title='Distance',
            template='plotly_white',
            legend_title='Metrics'
        )
        st.plotly_chart(fig)


def main():
    st.title("Comprehensive Distribution Distance Visualization")
    default_dir = st.sidebar.text_input("Enter the default directory path:", "patern_results/target_distribution5/20240624_134648")

    if not os.path.exists(default_dir):
        st.error(f"Directory {default_dir} does not exist.")
        return

    data = load_json_files(default_dir)
    results = {}

    for num_particles, values in data.items():
        strides = sorted(values.keys(), key=int)
        results[num_particles] = {}
        for i in range(len(strides) - 1):
            distancesA = values[strides[i]]
            distancesB = values[strides[i + 1]]
            results[num_particles][strides[i + 1]] = calculate_distances(distancesA, distancesB)

    plot_comparative_distances(results)


if __name__ == "__main__":
    main()