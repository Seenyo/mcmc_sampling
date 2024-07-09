import streamlit as st
import numpy as np
from scipy.stats import entropy, wasserstein_distance
import plotly.graph_objects as go
import os
import re
from tkinter import Tk, filedialog
import pandas as pd
from tqdm import tqdm

# Streamlitアプリのタイトル
st.set_page_config(layout="wide")
st.title("MCMC Histogram Comparator")


# フォルダ選択関数
def select_folder():
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path


# ファイル処理関数
def load_and_process_npz(file):
    try:
        data = np.load(file)
        hist, bin_edges = data['hist'], data['bin_edges']
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        adjusted_hist = hist / bin_centers
        normalized_hist = adjusted_hist / np.sum(adjusted_hist)
        normalized_hist += 1e-10
        return normalized_hist, bin_centers
    except Exception as e:
        st.error(f"Error processing file {file}: {str(e)}")
        return None, None


def calculate_distances(hist1, centers1, hist2, centers2):
    try:
        kl_divergence_1_2 = entropy(hist1, hist2)
        kl_divergence_2_1 = entropy(hist2, hist1)
        emd = wasserstein_distance(centers1, centers2, hist1, hist2)
        bhattacharyya_dist = -np.log(np.sum(np.sqrt(hist1 * hist2)))
        return kl_divergence_1_2, kl_divergence_2_1, emd, bhattacharyya_dist
    except Exception as e:
        st.error(f"Error calculating distances: {str(e)}")
        return None, None, None, None

def process_subdirectory(directory):
    file_pattern = re.compile(r'histogram_thread_(\d+)_mutation_(\d+)_(\d+)\.npz')
    files = [f for f in os.listdir(directory) if file_pattern.match(f)]
    files.sort(key=lambda x: int(file_pattern.match(x).group(2)))  # Sort by start mutation

    if not files:
        st.warning(f"No matching files found in directory: {directory}")
        return None

    reference_file = files[-1]
    reference_path = os.path.join(directory, reference_file)
    reference_hist, reference_centers = load_and_process_npz(reference_path)

    if reference_hist is None or reference_centers is None:
        return None

    results = []
    for file in files:
        file_path = os.path.join(directory, file)
        hist, centers = load_and_process_npz(file_path)
        if hist is None or centers is None:
            st.warning(f"Skipping file: {file}")
            continue
        kl_1_2, kl_2_1, emd, bhatt = calculate_distances(reference_hist, reference_centers, hist, centers)
        if kl_1_2 is None or kl_2_1 is None or emd is None or bhatt is None:
            continue
        match = file_pattern.match(file)
        start_mutation = int(match.group(2))
        end_mutation = int(match.group(3))
        mutation_range = f"{start_mutation:06d}_{end_mutation:06d}"
        results.append((mutation_range, kl_1_2, kl_2_1, emd, bhatt))

    return results


def process_directory(parent_directory):
    subdirectory_pattern = re.compile(r'^\d{4}$')
    subdirectories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d)) and subdirectory_pattern.match(d)]
    subdirectories.sort()

    if not subdirectories:
        st.warning(f"No valid subdirectories found in: {parent_directory}")
        return None, None

    all_results = []
    progress_bar = st.progress(0)
    for i, subdir in enumerate(tqdm(subdirectories, desc="Processing subdirectories", leave=False)):
        subdir_path = os.path.join(parent_directory, subdir)
        results = process_subdirectory(subdir_path)
        if results:
            all_results.append(results)
        progress_bar.progress((i + 1) / len(subdirectories))

    if not all_results:
        st.warning(f"No valid data found in any subdirectory of: {parent_directory}")
        return None, None

    avg_results = []
    for i in range(len(all_results[0])):
        mutation_range = all_results[0][i][0]
        avg_kl_1_2 = np.mean([r[i][1] for r in all_results])
        avg_kl_2_1 = np.mean([r[i][2] for r in all_results])
        avg_emd = np.mean([r[i][3] for r in all_results])
        avg_bhatt = np.mean([r[i][4] for r in all_results])
        avg_results.append((mutation_range, avg_kl_1_2, avg_kl_2_1, avg_emd, avg_bhatt))

    return avg_results, all_results


def plot_results(results, title):
    if not results:
        st.warning(f"No data to plot for: {title}")
        return None
    fig = go.Figure()
    mutations, kl_divs_1_2, kl_divs_2_1, emds, bhatt_dists = zip(*results)
    fig.add_trace(go.Scatter(x=mutations, y=kl_divs_1_2, mode='lines+markers', name='Avg KL Divergence (1->2)'))
    fig.add_trace(go.Scatter(x=mutations, y=kl_divs_2_1, mode='lines+markers', name='Avg KL Divergence (2->1)'))
    fig.add_trace(go.Scatter(x=mutations, y=emds, mode='lines+markers', name='Avg Earth Mover Distance'))
    fig.add_trace(go.Scatter(x=mutations, y=bhatt_dists, mode='lines+markers', name='Avg Bhattacharyya Distance'))
    fig.update_layout(
        title=f'Average Distance Metrics Over Mutations - {title}',
        xaxis_title='Mutation Range',
        yaxis_title='Average Distance',
        legend_title='Metric'
    )
    return fig

def plot_all_results(all_results, title, metric):
    if not all_results:
        st.warning(f"No data to plot for: {title}")
        return None
    fig = go.Figure()
    for i, results in enumerate(all_results):
        mutations, kl_divs_1_2, kl_divs_2_1, emds, bhatt_dists = zip(*results)
        if metric == 'KL_1_2':
            y = kl_divs_1_2
        elif metric == 'KL_2_1':
            y = kl_divs_2_1
        elif metric == 'EMD':
            y = emds
        else:  # Bhattacharyya
            y = bhatt_dists
        particle_count = particle_dirs[i]
        fig.add_trace(go.Scatter(x=mutations, y=y, mode='lines', name=f'{particle_count} particles', opacity=0.7))
    fig.update_layout(
        title=f'{metric} Distance Metrics Over Mutations - {title}',
        xaxis_title='Mutation Range',
        yaxis_title=f'{metric} Distance',
        legend_title='Particles'
    )
    return fig

# サイドバーの設定
st.sidebar.header("Select a parent folder")

if 'folder_path' not in st.session_state:
    st.session_state['folder_path'] = ''

if st.sidebar.button("Select Folder"):
    folder_path = select_folder()
    if folder_path:
        st.session_state['folder_path'] = folder_path
        st.experimental_rerun()

folder_path = st.sidebar.text_input("Selected folder path:", st.session_state['folder_path'])

# テーブル表示オプション
show_tables = st.sidebar.checkbox("Show detailed results tables", value=True)

if folder_path:
    particle_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and d.isdigit()]
    particle_dirs.sort(key=int)

    if not particle_dirs:
        st.error(f"No valid particle directories found in: {folder_path}")
    else:
        all_mh_results = []
        all_mwg_results = []

        for particle_dir in particle_dirs:
            st.write(f"## {particle_dir} particles")
            particle_path = os.path.join(folder_path, particle_dir)
            subdirs = [d for d in os.listdir(particle_path) if os.path.isdir(os.path.join(particle_path, d))]
            subdirs.sort()

            if len(subdirs) < 2:
                st.warning(f"The {particle_dir} folder should contain at least two subdirectories. Found: {subdirs}")
            else:
                mh_dir = os.path.join(particle_path, subdirs[0])
                mwg_dir = os.path.join(particle_path, subdirs[1])

                mh_results, mh_all = process_directory(mh_dir)
                mwg_results, mwg_all = process_directory(mwg_dir)

                if mh_results and mwg_results:
                    st.write(f"Processing directories: MH ({subdirs[0]}) and MWG ({subdirs[1]})")

                    col1, col2 = st.columns(2)

                    # MHのプロット
                    with col1:
                        mh_fig = plot_results(mh_results, f"MH - {particle_dir} particles")
                        if mh_fig:
                            st.plotly_chart(mh_fig, use_container_width=True)

                    # MWGのプロット
                    with col2:
                        mwg_fig = plot_results(mwg_results, f"MWG - {particle_dir} particles")
                        if mwg_fig:
                            st.plotly_chart(mwg_fig, use_container_width=True)

                    if mh_all:
                        all_mh_results.append(mh_results)
                    if mwg_all:
                        all_mwg_results.append(mwg_results)

                    if show_tables:
                        # 結果のテーブル表示
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"Detailed Results (Averages) - MH ({particle_dir} particles):")
                            mh_df = pd.DataFrame(mh_results, columns=['Mutation Range', 'Avg KL Divergence (1->2)', 'Avg KL Divergence (2->1)', 'Avg Earth Mover Distance', 'Avg Bhattacharyya Distance'])
                            st.dataframe(mh_df)

                        with col2:
                            st.write(f"Detailed Results (Averages) - MWG ({particle_dir} particles):")
                            mwg_df = pd.DataFrame(mwg_results, columns=['Mutation Range', 'Avg KL Divergence (1->2)', 'Avg KL Divergence (2->1)', 'Avg Earth Mover Distance', 'Avg Bhattacharyya Distance'])
                            st.dataframe(mwg_df)
                else:
                    st.warning(f"No valid data found in one or both of the specified directories for {particle_dir} particles.")

        st.write("## All particles")

        metrics = ['KL_1_2', 'KL_2_1', 'EMD', 'Bhattacharyya']
        for metric in metrics:
            col1, col2 = st.columns(2)

            with col1:
                all_mh_fig = plot_all_results(all_mh_results, "MH", metric)
                if all_mh_fig:
                    st.plotly_chart(all_mh_fig, use_container_width=True)

            with col2:
                all_mwg_fig = plot_all_results(all_mwg_results, "MWG", metric)
                if all_mwg_fig:
                    st.plotly_chart(all_mwg_fig, use_container_width=True)

else:
    st.info("Please select a parent folder to proceed")