import streamlit as st
import os
import json
import numpy as np
import plotly.graph_objects as go
from scipy.stats import entropy, wasserstein_distance

def get_directory_structure(directory):
    """Automatically detect PARTICLE_NUMBERS, SAMPLE_NUMBERS, and NUM_TRIALS from directory structure."""
    particle_numbers = []
    sample_numbers = set()
    num_trials = 0

    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)) and item.endswith('_particles'):
            particle_numbers.append(int(item.split('_')[0]))
            trial_path = os.path.join(directory, item, '0', 'data')  # Assuming at least one trial exists
            if os.path.exists(trial_path):
                for file in os.listdir(trial_path):
                    if file.startswith('distances_') and file.endswith('.json'):
                        sample_numbers.add(int(file.split('_')[1].split('.')[0]))
                num_trials = max(num_trials, len([d for d in os.listdir(os.path.join(directory, item)) if os.path.isdir(os.path.join(directory, item, d))]))

    return sorted(particle_numbers), sorted(list(sample_numbers)), num_trials

def load_json_file(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_distances(distancesA, distancesB):
    """Calculate KL divergence, Earth Mover Distance, and Bhattacharyya Distance between two distributions."""
    if len(distancesA) != len(distancesB):
        return {
            'KL Divergence AB': np.nan,
            'KL Divergence BA': np.nan,
            'Earth Mover Distance': np.nan,
            'Bhattacharyya Distance': np.nan,
        }

    n = len(distancesA)
    bins = int(np.ceil(2 * n ** (1 / 3)))  # Rice's rule for bin number

    histA, bin_edges = np.histogram(distancesA, bins=bins, density=True, range=(0, 0.8))
    histB, _ = np.histogram(distancesB, bins=bins, density=True, range=(0, 0.8))

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    adjusted_histA = histA / bin_centers
    adjusted_histB = histB / bin_centers

    normalized_histA = adjusted_histA / np.sum(adjusted_histA)
    normalized_histB = adjusted_histB / np.sum(adjusted_histB)

    # Add small value to avoid zero probabilities
    normalized_histA += 1e-10
    normalized_histB += 1e-10

    kl_divergence_ab = entropy(normalized_histA, normalized_histB)
    kl_divergence_ba = entropy(normalized_histB, normalized_histA)
    emd = wasserstein_distance(bin_centers, bin_centers, normalized_histA, normalized_histB)
    bhattacharyya_dist = -np.log(np.sum(np.sqrt(normalized_histA * normalized_histB)))

    return {
        'KL Divergence AB': kl_divergence_ab,
        'KL Divergence BA': kl_divergence_ba,
        'Earth Mover Distance': emd,
        'Bhattacharyya Distance': bhattacharyya_dist,
    }


def process_particle_data(directory, num_particles, sample_numbers, num_trials):
    """Process data for a specific number of particles across all trials."""
    results = []
    for trial in range(num_trials):
        trial_results = {}
        base_path = os.path.join(directory, f"{num_particles}_particles", str(trial), "data")

        # Load the base distribution (maximum sample number)
        base_dist = load_json_file(os.path.join(base_path, f"distances_{sample_numbers[-1]}.json"))

        for sample_num in sample_numbers:  # Include the maximum sample number
            current_dist = load_json_file(os.path.join(base_path, f"distances_{sample_num}.json"))
            trial_results[sample_num] = calculate_distances(base_dist, current_dist)

        results.append(trial_results)

    return results

def plot_distances(results, num_particles, trial, sample_numbers):
    """Create a plot for a specific number of particles and trial."""
    fig = go.Figure()
    metrics = ['KL Divergence AB', 'KL Divergence BA', 'Earth Mover Distance', 'Bhattacharyya Distance']

    for metric in metrics:
        y_values = [results[trial][sample][metric] for sample in sample_numbers]
        fig.add_trace(go.Scatter(
            x=sample_numbers,
            y=y_values,
            mode='lines+markers',
            name=metric
        ))

    fig.update_layout(
        title=f'Distribution Distances for {num_particles} Particles (Trial {trial})',
        xaxis_title='Number of Samples',
        yaxis_title='Distance',
        xaxis_type='log',
        template='plotly_white',
        legend_title='Metrics'
    )

    return fig

def plot_average_distances(results, num_particles, sample_numbers, num_trials):
    """Create a plot for the average distances across all trials."""
    fig = go.Figure()
    metrics = ['KL Divergence AB', 'KL Divergence BA', 'Earth Mover Distance', 'Bhattacharyya Distance']

    for metric in metrics:
        avg_values = []
        for sample in sample_numbers:
            values = [results[trial][sample][metric] for trial in range(num_trials)]
            avg_values.append(np.mean(values))

        fig.add_trace(go.Scatter(
            x=sample_numbers,
            y=avg_values,
            mode='lines+markers',
            name=metric
        ))

    fig.update_layout(
        title=f'Average Distribution Distances for {num_particles} Particles (Across All Trials)',
        xaxis_title='Number of Samples',
        yaxis_title='Average Distance',
        xaxis_type='log',
        template='plotly_white',
        legend_title='Metrics'
    )

    return fig

def main():
    st.title("Distribution Distance Comparison")

    directory = st.sidebar.text_input("Enter the base directory path:", "patern_results/target_distribution5/20240701_210440/MH_Normal")

    if st.sidebar.button("Process Data"):
        if not os.path.exists(directory):
            st.error(f"Directory {directory} does not exist.")
            return

        particle_numbers, sample_numbers, num_trials = get_directory_structure(directory)

        # Initialize session state if not already done
        if 'particle_results' not in st.session_state:
            st.session_state.particle_results = {}
            st.session_state.particle_numbers = particle_numbers
            st.session_state.sample_numbers = sample_numbers
            st.session_state.num_trials = num_trials

        # Process data for each particle number
        progress_bar = st.progress(0)
        for i, num_particles in enumerate(particle_numbers):
            if num_particles not in st.session_state.particle_results:
                st.session_state.particle_results[num_particles] = process_particle_data(directory, num_particles, sample_numbers, num_trials)
            progress_bar.progress((i + 1) / len(particle_numbers))

        progress_bar.empty()

    if 'particle_results' in st.session_state:
        tabs = st.tabs([f"{num} Particles" for num in st.session_state.particle_numbers])

        for tab, num_particles in zip(tabs, st.session_state.particle_numbers):
            with tab:
                results = st.session_state.particle_results[num_particles]

                with st.expander("Show Each Trial"):
                    for trial in range(st.session_state.num_trials):
                        st.plotly_chart(plot_distances(results, num_particles, trial, st.session_state.sample_numbers))

                # Add the average plot after all individual trial plots
                st.plotly_chart(plot_average_distances(results, num_particles, st.session_state.sample_numbers, st.session_state.num_trials))

if __name__ == "__main__":
    main()