import streamlit as st
import numpy as np
from scipy.stats import entropy, wasserstein_distance
import plotly.graph_objects as go

def load_and_process_npz(file):
    data = np.load(file)
    hist, bin_edges = data['hist'], data['bin_edges']
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    adjusted_hist = hist / bin_centers
    normalized_hist = adjusted_hist / np.sum(adjusted_hist)
    normalized_hist += 1e-10
    return normalized_hist, bin_centers

def calculate_distances(hist1, hist2, centers1, centers2):
    kl_divergence_AB = entropy(hist1, hist2)
    kl_divergence_BA = entropy(hist2, hist1)
    emd = wasserstein_distance(centers1, centers2, hist1, hist2)
    bhattacharyya_dist = -np.log(np.sum(np.sqrt(hist1 * hist2)))
    return kl_divergence_AB, kl_divergence_BA, emd, bhattacharyya_dist

def plot_histograms(centers1, hist1, centers2, hist2):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centers1, y=hist1, name='Histogram 1'))
    fig.add_trace(go.Bar(x=centers2, y=hist2, name='Histogram 2'))
    fig.update_traces(opacity=0.75)
    fig.update_layout(
        title='Distribution Comparison',
        xaxis_title='Distance',
        yaxis_title='Normalized Density',
        barmode='overlay'
    )
    return fig

def main():
    st.title("MCMC Histogram Comparator")

    st.sidebar.header("Upload your NPZ files")
    uploaded_file1 = st.sidebar.file_uploader("Upload first histogram file", type="npz")
    uploaded_file2 = st.sidebar.file_uploader("Upload second histogram file", type="npz")

    if uploaded_file1 is not None and uploaded_file2 is not None:
        normalized_hist1, bin_centers1 = load_and_process_npz(uploaded_file1)
        normalized_hist2, bin_centers2 = load_and_process_npz(uploaded_file2)

        kl_divergence_AB, kl_divergence_BA, emd, bhattacharyya_dist = calculate_distances(
            normalized_hist1, normalized_hist2, bin_centers1, bin_centers2
        )

        st.write("KL Divergence AB:", kl_divergence_AB)
        st.write("KL Divergence BA:", kl_divergence_BA)
        st.write("Earth Mover Distance:", emd)
        st.write("Bhattacharyya Distance:", bhattacharyya_dist)

        fig = plot_histograms(bin_centers1, normalized_hist1, bin_centers2, normalized_hist2)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload both NPZ files to proceed")

if __name__ == "__main__":
    main()