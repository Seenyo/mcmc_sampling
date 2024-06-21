import json
import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from vispy import scene
from vispy.io import write_png
import subprocess
import datetime
import colorsys
import math

from stqdm import stqdm

# 仮想環境のアクティベーションコマンド
if os.name == 'nt':  # Windowsの場合
    venv_activate = ".\\.venv\\Scripts\\activate.bat"
else:  # Unix系の場合
    venv_activate = "source .venv/bin/activate"

def toroidal_distance(length, p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])

    if dx > length / 2:
        dx = length - dx
    if dy > length / 2:
        dy = length - dy

    return math.sqrt(dx ** 2 + dy ** 2)

def initialize_parameters():
    st.session_state.num_of_particles = st.sidebar.number_input("Number of Particles", 1, 10000, 2)
    st.session_state.target_distribution_name = st.sidebar.selectbox("Target Distribution", ["target_distribution", "target_distribution2", "target_distribution3", "target_distribution4", "target_distribution5", "target_distribution01", "target_distribution005"], index=4)
    st.session_state.a = st.sidebar.number_input("a", 0.0, 20.0, 3 * np.pi)
    st.session_state.b = st.sidebar.number_input("b", 0.0, 5.0, 0.05)
    st.session_state.c = st.sidebar.number_input("c", 0.0, 5.0, 0.1, step=0.001)
    st.session_state.s = st.sidebar.number_input("s", 0.0, 5.0, 0.1)
    st.session_state.proposal_std = st.sidebar.number_input("Proposal Standard Deviation", 0.0, 5.0, 0.5)
    st.session_state.r_threshold = st.sidebar.number_input("r Threshold", 0.0, 1.0, 0.001)
    st.session_state.num_of_chains = st.sidebar.number_input("Number of Chains", 1, 1000000, 10000)
    st.session_state.num_of_iterations_for_each_chain = st.sidebar.number_input("Number of Iterations for Each Chain", 1, 100000000, 10000)
    st.session_state.num_of_mutations = st.sidebar.number_input("Number of Mutations", 10, 10000000, 1000)
    st.session_state.scaling_factor = st.sidebar.number_input("Scaling Factor", 0.0, 100.0, 5.0, step=0.5)
    st.session_state.geta = st.sidebar.number_input("Geta", 0.0, 30.0, 5.0, step=0.5)
    st.session_state.burn_in_multiplier = st.sidebar.number_input("Burn-in Multiplier", 1.0, 10.0, 1.5, step=0.1)
    st.session_state.show_particles = st.sidebar.checkbox("Visualize Particles", False)
    st.session_state.each_particle = st.sidebar.checkbox("Visualize Each Particle Index", False)
    st.session_state.save_image = st.sidebar.checkbox("Save Image", False)
    st.session_state.plotly = st.sidebar.checkbox("Use Plotly", False)
    st.session_state.use_maximal_c = st.sidebar.checkbox("Use Maximal c", True)
    st.session_state.acceptance_ratio_calculation_with_log = st.sidebar.checkbox("Calculate Acceptance Ratio with Log", True)
    st.session_state.record_from_first_acceptance = st.sidebar.checkbox("Record from First Acceptance", True)
    st.session_state.use_metropolis_within_gibbs = st.sidebar.checkbox("Use Metropolis within Gibbs", True)

    # Hello

    variables_to_initialize = [
        'current_particles', 'result_particles', 'distances', 'min_distance_particles',
        'prev_scaling_factor', 'prev_geta', 'average_acceptance_ratio', 'calc_time'
    ]
    for var in variables_to_initialize:
        if var not in st.session_state:
            st.session_state[var] = None

    if 'min_distance' not in st.session_state:
        st.session_state.min_distance = 1.0


def calculate_maximal_c():
    Cab = -(2 * ((1 - 3 * st.session_state.a ** 2) * np.sin(st.session_state.a * st.session_state.b) + st.session_state.a * (st.session_state.a ** 2 - 3) * np.cos(st.session_state.a * st.session_state.b))) / ((st.session_state.a ** 2 + 1) ** 3)
    t = np.sin(-st.session_state.a * st.session_state.b) - (Cab / 2)

    if st.session_state.target_distribution_name in ['target_distribution', 'target_distribution4']:
        sin_ab = np.sin(st.session_state.a * st.session_state.b)
        cos_ab = np.cos(st.session_state.a * st.session_state.b)
        numerator = 2 * ((1 - 3 * st.session_state.a ** 2) * sin_ab + st.session_state.a * (st.session_state.a ** 2 - 3) * cos_ab)
        denominator = (st.session_state.a ** 2 + 1) ** 3
        Cab = -1 * numerator / denominator
        num_of_combinations = st.session_state.num_of_particles * (st.session_state.num_of_particles - 1) / 2
        st.session_state.c = -1 / ((np.sin(-st.session_state.a * st.session_state.b) - Cab) * num_of_combinations)
    elif st.session_state.target_distribution_name == 'target_distribution3':
        st.session_state.a = 5.0
        st.session_state.b = 1.6
        st.session_state.s = 1.6
        sin_ab = np.sin(st.session_state.a * st.session_state.b)
        cos_ab = np.cos(st.session_state.a * st.session_state.b)
        numerator = 2 * st.session_state.a * cos_ab - (1 - st.session_state.a**2) * sin_ab
        denominator = (1 + st.session_state.a ** 2) ** 2
        Cab = numerator / denominator
        st.info(f'Cab is {Cab}')
        st.session_state.c = -1 / (np.sin(-st.session_state.a * st.session_state.b) - Cab)
    else:
        st.session_state.c = -1 / t
    st.info(f'c is {st.session_state.c}')

def load_data():
    # ファイルの存在チェックと読み込みを繰り返し行っているコードを、ループを使用して簡略化
    files_to_load = [
        ('temp_folder/initial_particles.npy', 'initial_particles', np.load),
        ('temp_folder/result_particles.npy', 'result_particles', np.load),
        ('temp_folder/current_particles.npy', 'current_particles', np.load),
        ('temp_folder/distances.npy', 'distances', np.load),
        ('temp_folder/min_distance_particles.npy', 'min_distance_particles', np.load),
        ('temp_folder/min_distance_pair.npy', 'min_distance_pair', lambda x: tuple(np.load(x))),
    ]

    for file_path, var_name, load_func in files_to_load:
        if os.path.exists(file_path):
            st.session_state[var_name] = load_func(file_path)

    files_to_read = [
        ('temp_folder/calc_time.txt', 'calc_time', float),
        ('temp_folder/min_distance.txt', 'min_distance', float),
        ('temp_folder/average_acceptance_ratio.txt', 'average_acceptance_ratio', float),
    ]

    for file_path, var_name, cast_func in files_to_read:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                st.session_state[var_name] = cast_func(f.read())

def visualize_particles_with_plotly():
    # st.session_state.current_particlesの最後の要素を取得
    last_particles = st.session_state.current_particles[-1]
    st.info(f'Number of Particles: {len(last_particles)}')
    st.info(last_particles)

    titleStr = "Metropolis within Gibbs Sampling with Particles" if st.session_state.use_metropolis_within_gibbs else "Metropolis-Hastings Algorithm Sampling with Particles"

    # last_particlesをplotlyで描画, 正方形の範囲, スケーリングを指定
    fig = go.Figure()
    for i in range(len(last_particles)):
        fig.add_trace(go.Scatter(x=[last_particles[i][0]], y=[last_particles[i][1]], mode='markers', name=f'Particle {i}'))
    fig.update_layout(title=titleStr,
                      xaxis_title='x',
                      yaxis_title='y',
                      xaxis_range=[0, 1],
                      yaxis_range=[0, 1],
                      xaxis=dict(constrain='domain'),
                      yaxis=dict(scaleanchor='x')
                      )
    # 表示領域を正方形にする
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1], scaleanchor='x', scaleratio=1)
    st.plotly_chart(fig, theme=None)



@st.cache_data
def calculate_kappa(r_list, scaling_factor, c, s, a, b, c_ab_val, geta):
    return scaling_factor * c * np.exp(-1 * r_list / s) * (np.sin(a * (r_list / s - b)) - c_ab_val * 0.5) + geta

@st.cache_data
def calculate_kappa2(r_list, scaling_factor, c, s, a, b, c_ab_val, geta):
    return scaling_factor * c * np.exp(-1 * r_list / s) * (np.sin(a * (r_list / s - b)) - c_ab_val) + geta

@st.cache_data
def calculate_kappa3(r_list, scaling_factor, a, b, c, Cab, geta):
    kappa2 = np.where(r_list < b, -1.0, c * np.exp(-(r_list - b)) * (-np.cos(a * (r_list - b)) + Cab))
    return scaling_factor * kappa2 + scaling_factor

@st.cache_data
def calculate_kappa01(r_list, scaling_factor, geta):
    kappa_values = []
    for i in range(len(r_list)):
        if r_list[i] < 0.1:
            kappa_values.append(0.0)
        else:
            kappa_values.append(scaling_factor * (-1 * np.exp(-3*(r_list[i]-0.1)) * np.cos(10*(r_list[i]-0.1))) + geta)
    return kappa_values

@st.cache_data
def calculate_kappa005(r_list, scaling_factor, geta):
    kappa_values = []
    for i in range(len(r_list)):
        if r_list[i] < 0.05:
            kappa_values.append(0.0)
        else:
            kappa_values.append(scaling_factor * (-1 * np.exp(-5*(r_list[i]-0.05)) * np.cos(22*(r_list[i]-0.05))) + scaling_factor)
    return kappa_values

def visualize_histogram():
    if st.session_state.distances is not None:
        # Display the histogram with go
        # Display as a density plot
        # fig = go.Figure(data=[go.Histogram(x=st.session_state.distances, histnorm='density', nbinsx=50)])
        # fig.update_layout(title='Distance between Two Particles', xaxis_title='Distance', yaxis_title='Density')
        # fig.update_xaxes(range=[st.session_state.r_threshold, max(st.session_state.distances)])
        # st.plotly_chart(fig, theme=None)
        #
        # # Normalize the histogram as a density plot
        # hist, bin_edges = np.histogram(st.session_state.distances, bins=50, density=True)
        # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        #
        # normalized_hist = hist / bin_centers

        # calculate kappa value
        sin_ab = np.sin(st.session_state.a * st.session_state.b)
        cos_ab = np.cos(st.session_state.a * st.session_state.b)

        if 'kappa_values' not in st.session_state:
            st.session_state.kappa_values = None

        r_list = np.linspace(0, 5, 500)

        if (st.session_state.scaling_factor != st.session_state.prev_scaling_factor) or (st.session_state.geta != st.session_state.prev_geta):
            if st.session_state.target_distribution_name == 'target_distribution' or st.session_state.target_distribution_name == 'target_distribution2':
                numerator = 2 * ((1 - 3 * st.session_state.a ** 2) * sin_ab + st.session_state.a * (st.session_state.a ** 2 - 3) * cos_ab)
                denominator = (st.session_state.a ** 2 + 1) ** 3
                st.session_state.c_ab_val = -1 * numerator / denominator
                st.session_state.kappa_values = calculate_kappa(r_list, st.session_state.scaling_factor, st.session_state.c, st.session_state.s, st.session_state.a, st.session_state.b, st.session_state.c_ab_val, st.session_state.geta)
            elif st.session_state.target_distribution_name == 'target_distribution3':
                numerator = 2 * st.session_state.a * cos_ab - (1 - st.session_state.a**2) * sin_ab
                denominator = (1 + st.session_state.a ** 2) ** 2
                st.session_state.c_ab_val = numerator / denominator
                st.session_state.geta = st.session_state.scaling_factor
                st.session_state.kappa_values = calculate_kappa2(r_list, st.session_state.scaling_factor, st.session_state.c, st.session_state.s, st.session_state.a, st.session_state.b, st.session_state.c_ab_val, st.session_state.geta)
            elif st.session_state.target_distribution_name == 'target_distribution4':
                numerator = 2 * st.session_state.a * cos_ab - (1 - st.session_state.a ** 2) * sin_ab
                denominator = (1 + st.session_state.a ** 2) ** 2
                st.session_state.c_ab_val = numerator / denominator
                st.session_state.kappa_values = calculate_kappa2(r_list, st.session_state.scaling_factor, st.session_state.c, st.session_state.s, st.session_state.a, st.session_state.b, st.session_state.c_ab_val, st.session_state.geta)
            elif st.session_state.target_distribution_name == 'target_distribution5':
                a_squared_plus_one = st.session_state.a ** 2 + 1
                b_plus_one = st.session_state.b + 1
                b_minus_one = st.session_state.b - 1

                c_numer = (a_squared_plus_one ** 2) * (st.session_state.b ** 2 + 2 * st.session_state.b + 2)
                c_denom = 2 * ((a_squared_plus_one ** 2) * b_plus_one - a_squared_plus_one * b_minus_one - 2)
                c = c_numer / c_denom
                Cab_1 = b_minus_one / (a_squared_plus_one * b_plus_one)
                Cab_2 = 2 / (a_squared_plus_one ** 2 * b_plus_one)
                Cab_3 = st.session_state.b ** 2 / (2 * b_plus_one * c)
                Cab = Cab_1 + Cab_2 + Cab_3
                st.session_state.kappa_values = calculate_kappa3(r_list, st.session_state.scaling_factor, st.session_state.a, st.session_state.b, c, Cab, st.session_state.geta)
            elif st.session_state.target_distribution_name == 'target_distribution01':
                st.session_state.kappa_values = calculate_kappa01(r_list, st.session_state.scaling_factor, st.session_state.geta)
            elif st.session_state.target_distribution_name == 'target_distribution005':
                st.session_state.kappa_values = calculate_kappa005(r_list, st.session_state.scaling_factor, st.session_state.geta)

            st.session_state.prev_scaling_factor = st.session_state.scaling_factor
            st.session_state.prev_geta = st.session_state.geta

        # fig = go.Figure(data=[go.Bar(x=bin_centers, y=normalized_hist)])
        # fig.add_trace(go.Scatter(x=r_list, y=st.session_state.kappa_values, mode='lines', name='Kappa Values'))
        # fig.update_xaxes(range=[st.session_state.r_threshold, max(st.session_state.distances)])
        # fig.update_yaxes(range=[0, max(normalized_hist) * 1.1])
        # fig.update_layout(title='Normalized Distance between Two Particles', xaxis_title='Distance', yaxis_title='Normalized Frequency')
        # st.plotly_chart(fig, theme=None)

@st.cache_data
def calculate_min_distance(result_particles):
    distances = []
    min_distance = 100.0
    min_distance_particles = None
    min_distance_pair = None

    for i in range(len(result_particles)):
        for j in range(len(result_particles[i])):
            dist = toroidal_distance(1.0, result_particles[i][j][0], result_particles[i][j][1])
            distances.append(dist)
            if dist < min_distance:
                min_distance = dist
                min_distance_particles = result_particles[i][j]
                min_distance_pair = (0, 1)

    return distances, min_distance, min_distance_particles, min_distance_pair

def add_annotation_to_plot(fig):
    MinDistStr = f'Minimum Distance: {st.session_state.min_distance:.4e}'
    NumOfParticlesStr = f'Number of Particles: {st.session_state.num_of_particles}'
    IterationsPerTrialStr = f'Iterations per Trial: {st.session_state.num_of_iterations_for_each_chain}'
    SamplingStridesStr = f'Sampling Strides: {st.session_state.num_of_mutations}'
    AverageAcceptanceRatioStr = f'Average Acceptance Ratio: {st.session_state.average_acceptance_ratio:.2f}%'

    if st.session_state.use_metropolis_within_gibbs:
        methodStr = 'Method: Metropolis within Gibbs'
    else:
        methodStr = 'Method: Metropolis-Hastings'

    fig.add_annotation(
        x=1.05,
        y=1.0,
        showarrow=False,
        text=f'{MinDistStr}<br>{NumOfParticlesStr}<br>{IterationsPerTrialStr}<br>{SamplingStridesStr}<br>{AverageAcceptanceRatioStr}<br>{methodStr}',
        xref='paper',
        yref='paper'
    )

def visualize_min_distance_particles():
    if st.session_state.result_particles is not None:
        st.session_state.distances, st.session_state.min_distance, st.session_state.min_distance_particles, st.session_state.min_distance_pair = calculate_min_distance(st.session_state.result_particles)

        st.info(f"Number of Distances: {len(st.session_state.distances)}")
        st.info(f'Minimum distance is: {st.session_state.min_distance}')

        # plot min distance particles by plotly
        fig = go.Figure(data=[go.Scatter(x=[x[0] for x in st.session_state.min_distance_particles], y=[x[1] for x in st.session_state.min_distance_particles], mode='markers', showlegend=False)])

        # plot the minimum distance pair with red dots and a line
        # Extract coordinates for clarity
        x0, y0 = st.session_state.min_distance_particles[st.session_state.min_distance_pair[0]]
        x1, y1 = st.session_state.min_distance_particles[st.session_state.min_distance_pair[1]]

        # Add the first marker (red)
        fig.add_trace(go.Scatter(x=[x0], y=[y0], mode='markers', marker=dict(color='red'), showlegend=False))

        # Add the second marker (green)
        fig.add_trace(go.Scatter(x=[x1], y=[y1], mode='markers', marker=dict(color='green'), showlegend=False))

        # Add a line between them
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(color='blue'), showlegend=False))

        # Update the layout or add other configurations if needed
        fig.update_layout(title="Minimum Distance Pair Visualization", xaxis_title="X Axis", yaxis_title="Y Axis")
        fig.update_layout(
            title='Minimum Distance between Particles',
            xaxis_title='x',
            yaxis_title='y',
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            xaxis=dict(constrain='domain'),
            yaxis=dict(scaleanchor='x'),
            margin=dict(r=60)
        )

        MinDistStr = f'Minimum Distance: {st.session_state.min_distance:.4e}'
        NumOfParticlesStr = f'Number of Particles: {st.session_state.num_of_particles}'
        IterationsPerTrialStr = f'Iterations per Trial: {st.session_state.num_of_iterations_for_each_chain}'
        SamplingStridesStr = f'Mutations: {st.session_state.num_of_mutations}'
        AverageAcceptanceRatioStr = f'Average Acceptance Ratio: {st.session_state.average_acceptance_ratio:.2f}%'

        if st.session_state.use_metropolis_within_gibbs:
            methodStr = 'Method: Metropolis within Gibbs'
        else:
            methodStr = 'Method: Metropolis-Hastings'

        fig.add_annotation(
            x=1.05,
            y=1.0,
            showarrow=False,
            text=f'{MinDistStr}<br>{NumOfParticlesStr}<br>{IterationsPerTrialStr}<br>{SamplingStridesStr}<br>{AverageAcceptanceRatioStr}<br>{methodStr}',
            xref='paper',
            yref='paper'
        )
        add_annotation_to_plot(fig)

        st.plotly_chart(fig, theme=None)

def draw_particles(data, colors, filename, size=(1000, 1000), save=False):
    """Draw particles on a canvas and optionally save the image."""
    canvas = scene.SceneCanvas(keys='interactive', size=size, bgcolor='white', show=False)
    grid = canvas.central_widget.add_grid()
    grid.spacing = 0

    title = scene.Label(filename, color='black')
    title.height_max = 40
    grid.add_widget(title, row=0, col=0, col_span=2)

    view = canvas.central_widget.add_view()
    scatter = scene.visuals.Markers(parent=view.scene)
    scatter.set_data(data, size=10, edge_color=None, face_color=colors, edge_width=0)
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(0, 1), y=(0, 1))

    canvas.update()
    canvas.app.process_events()
    image = canvas.render()

    if save:
        path = f"./created_image/{format(datetime.datetime.now(), '%Y%m%d_%H%M%S')}"
        if not os.path.exists(path):
            os.makedirs(path)
        write_png(f'{path}/{filename}.png', image)

    return image

def visualize_particles():
    """Visualize and optionally save the results."""
    # Visualize all initial particles
    data_all = np.concatenate(st.session_state.initial_particles, axis=0)
    num_of_channels = 4
    colors_all = np.zeros((len(data_all), num_of_channels))
    for i in range(st.session_state.num_of_particles):
        hue = i / float(st.session_state.num_of_particles)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors_all[i::st.session_state.num_of_particles] = rgb + (1.0,)
    image_all = draw_particles(data_all, colors_all, 'all_initial_particles', save=st.session_state.save_image)
    st.image(image_all)

    # Visualize all result particles
    data_all = np.concatenate(st.session_state.current_particles, axis=0)
    colors_all = np.zeros((len(data_all), num_of_channels))
    for i in range(st.session_state.num_of_particles):
        hue = i / float(st.session_state.num_of_particles)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors_all[i::st.session_state.num_of_particles] = rgb + (1.0,)
    image_all = draw_particles(data_all, colors_all, 'all_result_particles', save=st.session_state.save_image)
    st.image(image_all)

    # Visualize each particle index
    if st.session_state.each_particle:
        for i in range(st.session_state.num_of_particles):
            data_index = np.array([chain[i] for chain in st.session_state.current_particles])
            colors_index = colors_all[i::st.session_state.num_of_particles]
            filename = f'particles_index_{i}_trial_count_{st.session_state.num_of_chains}_mh_steps_{st.session_state.num_of_iterations_for_each_chain}'
            image_index = draw_particles(data_index, colors_index, filename, save=st.session_state.save_image)
            st.image(image_index)

def visualize_acceptance_rate():
    if os.path.exists('temp_folder/acceptance_rate.txt'):
        global acceptance_rates
        acceptance_rates = []
        with open('temp_folder/acceptance_rate.txt', 'r') as f:
            for line in f:
                acceptance_rates.append(float(line.strip()))

        # mutations = list(range(1, len(acceptance_rates) + 1))
        # mutations = [x * st.session_state.num_of_mutations for x in mutations]

        # fig = go.Figure(data=go.Scatter(x=mutations, y=acceptance_rates))
        # fig.update_layout(title='Acceptance Rate over Mutations',
        #                   xaxis_title='Mutations',
        #                   yaxis_title='Acceptance Rate (%)')
        # st.plotly_chart(fig, theme=None)


def visualize_acceptance_rate_change():
    if os.path.exists('temp_folder/acceptance_rate_change.txt'):
        global acceptance_rate_changes
        acceptance_rate_changes = []
        with open('temp_folder/acceptance_rate_change.txt', 'r') as f:
            for line in f:
                acceptance_rate_changes.append(float(line.strip()))

        # mutations = list(range(2, len(acceptance_rate_changes) + 2))
        # mutations = [x * st.session_state.num_of_mutations for x in mutations]

        # fig = go.Figure(data=go.Scatter(x=mutations, y=acceptance_rate_changes))
        # fig.update_layout(title='Relative Change of Acceptance Rate over Mutations',
        #                   xaxis_title='Mutations',
        #                   yaxis_title='Relative Change of Acceptance Rate (%)')
        # st.plotly_chart(fig, theme=None)

def set_flags():
    log_flag = "--acceptance_ratio_calculation_with_log" if st.session_state.acceptance_ratio_calculation_with_log else ""
    record_flag = "--record_from_first_acceptance" if st.session_state.record_from_first_acceptance else ""
    use_metropolis_within_gibbs_flag = "--use_metropolis_within_gibbs" if st.session_state.use_metropolis_within_gibbs else ""
    return log_flag, record_flag, use_metropolis_within_gibbs_flag
def calculate_all_patterns():
    num_of_particles_list = [2, 5, 10, 15, 20, 30]
    num_of_mutations_list = [100, 1000, 10000, 100000]
    use_metropolis_within_gibbs_list = [True]
    use_log_calculation_list = [True, False]

    total_iterations = len(use_log_calculation_list) * len(use_metropolis_within_gibbs_list) * len(num_of_particles_list) * len(num_of_mutations_list)

    with stqdm(total=total_iterations, desc="Progress") as pbar:
        for use_log_calculation in use_log_calculation_list:
            for use_metropolis_within_gibbs in use_metropolis_within_gibbs_list:
                for num_of_particles in num_of_particles_list:
                    for num_of_mutations in num_of_mutations_list:
                        st.session_state.num_of_particles = num_of_particles
                        st.session_state.num_of_mutations = num_of_mutations
                        st.session_state.num_of_iterations_for_each_chain = num_of_mutations * 10
                        st.session_state.use_metropolis_within_gibbs = use_metropolis_within_gibbs
                        st.session_state.acceptance_ratio_calculation_with_log = use_log_calculation

                        log_flag, record_flag, use_metropolis_within_gibbs_flag = set_flags()
                        subprocess.run(
                            f"{venv_activate} && python taichi_calculator.py "
                            f"--num_of_particles {st.session_state.num_of_particles} "
                            f"--a {st.session_state.a} "
                            f"--b {st.session_state.b} "
                            f"--c {st.session_state.c} "
                            f"--s {st.session_state.s} "
                            f"--proposal_std {st.session_state.proposal_std} "
                            f"--num_of_chains {st.session_state.num_of_chains} "
                            f"--target_distribution_name {st.session_state.target_distribution_name} "
                            f"--num_of_iterations_for_each_chain {st.session_state.num_of_iterations_for_each_chain} "
                            f"--num_of_mutations {st.session_state.num_of_mutations} "
                            f"--burn_in_multiplier {st.session_state.burn_in_multiplier} "
                            f"{log_flag} "
                            f"{record_flag} "
                            f"{use_metropolis_within_gibbs_flag}",
                            shell=True
                        )
                        load_data()
                        method = "MWG" if use_metropolis_within_gibbs else "MH"
                        log_calculation = "Log" if use_log_calculation else "Normal"
                        save_dir = f"patern_results/{st.session_state.target_distribution_name}/{method}_{log_calculation}/{num_of_particles}_{num_of_mutations}"
                        visualize_acceptance_rate()
                        visualize_acceptance_rate_change()
                        visualize_histogram()
                        save_results(save_dir)
                        pbar.update(1)

def save_results(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(f"{save_dir}/img")
        os.makedirs(f"{save_dir}/data")

    # 画像を保存
    # Minimum Distance between Particlesの画像を保存
    fig_min_distance = go.Figure(data=[go.Scatter(x=[x[0] for x in st.session_state.min_distance_particles], y=[x[1] for x in st.session_state.min_distance_particles], mode='markers', marker=dict(color='black'), showlegend=False)])

    # 最小距離のペアの粒子を赤と緑でプロット
    x0, y0 = st.session_state.min_distance_particles[st.session_state.min_distance_pair[0]]
    x1, y1 = st.session_state.min_distance_particles[st.session_state.min_distance_pair[1]]

    fig_min_distance.add_trace(go.Scatter(x=[x0], y=[y0], mode='markers', marker=dict(color='red'), showlegend=False))
    fig_min_distance.add_trace(go.Scatter(x=[x1], y=[y1], mode='markers', marker=dict(color='green'), showlegend=False))

    # 最小距離のペア間に青い線を引く
    fig_min_distance.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(color='blue'), showlegend=False))

    add_annotation_to_plot(fig_min_distance)
    fig_min_distance.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    pio.write_image(fig_min_distance, f"{save_dir}/img/min_distance_particles.png")

    # Distance between Two Particlesの画像を保存
    fig_distance = go.Figure(data=[go.Histogram(x=st.session_state.distances, histnorm='density', nbinsx=50, marker=dict(color='blue'))])
    fig_distance.update_layout(title='Distance between Two Particles', xaxis_title='Distance', yaxis_title='Density', plot_bgcolor='white', paper_bgcolor='white')
    fig_distance.update_xaxes(range=[st.session_state.r_threshold, max(st.session_state.distances)])
    pio.write_image(fig_distance, f"{save_dir}/img/distance_between_two_particles.png")

    # Normalized Distance between Two Particlesの画像を保存
    hist, bin_edges = np.histogram(st.session_state.distances, bins=50, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    normalized_hist = hist / bin_centers
    r_list = np.linspace(0, 5, 500)

    fig_normalized_distance = go.Figure(data=[go.Bar(x=bin_centers, y=normalized_hist, marker=dict(color='blue'))])
    fig_normalized_distance.add_trace(go.Scatter(x=r_list, y=st.session_state.kappa_values, mode='lines', name='Kappa Values', line=dict(color='red')))
    fig_normalized_distance.update_xaxes(range=[st.session_state.r_threshold, max(st.session_state.distances)])
    fig_normalized_distance.update_yaxes(range=[0, max(normalized_hist) * 1.1])
    fig_normalized_distance.update_layout(title='Normalized Distance between Two Particles', xaxis_title='Distance', yaxis_title='Normalized Frequency', plot_bgcolor='white', paper_bgcolor='white')
    pio.write_image(fig_normalized_distance, f"{save_dir}/img/normalized_distance_between_two_particles.png")

    # acceptance_rateのプロットを保存
    fig_acceptance_rate = go.Figure(data=go.Scatter(x=list(range(0, len(acceptance_rates))), y=acceptance_rates, line=dict(color='blue')))

    mutations = list(range(0, len(acceptance_rates)))
    mutations_text = [str((x + 1) * st.session_state.num_of_mutations) for x in mutations]

    fig_acceptance_rate.update_xaxes(ticktext=mutations_text, tickvals=mutations)
    fig_acceptance_rate.update_layout(title='Acceptance Rate over Mutations', xaxis_title='Mutations', yaxis_title='Acceptance Rate (%)', plot_bgcolor='white', paper_bgcolor='white')
    pio.write_image(fig_acceptance_rate, f"{save_dir}/img/acceptance_rate.png")

    # acceptance_rate_changeのプロットを保存
    fig_acceptance_rate_change = go.Figure(data=go.Scatter(x=list(range(1, len(acceptance_rate_changes) + 1)), y=acceptance_rate_changes, line=dict(color='blue')))

    mutations = list(range(1, len(acceptance_rate_changes) + 1))
    mutations_text = [str((x + 1) * st.session_state.num_of_mutations) for x in mutations]

    fig_acceptance_rate_change.update_xaxes(ticktext=mutations_text, tickvals=mutations)
    fig_acceptance_rate_change.update_layout(title='Relative Change of Acceptance Rate over Mutations', xaxis_title='Mutations', yaxis_title='Relative Change (%)', plot_bgcolor='white', paper_bgcolor='white')
    pio.write_image(fig_acceptance_rate_change, f"{save_dir}/img/acceptance_rate_change.png")

    # パラメータと他の値を一つのJSONファイルに保存
    results = {
        "parameters": {
            "num_of_particles": st.session_state.num_of_particles,
            "target_distribution_name": st.session_state.target_distribution_name,
            "a": st.session_state.a,
            "b": st.session_state.b,
            "c": st.session_state.c,
            "s": st.session_state.s,
            "proposal_std": st.session_state.proposal_std,
            "r_threshold": st.session_state.r_threshold,
            "num_of_chains": st.session_state.num_of_chains,
            "num_of_iterations_for_each_chain": st.session_state.num_of_iterations_for_each_chain,
            "num_of_mutations": st.session_state.num_of_mutations,
            "scaling_factor": st.session_state.scaling_factor,
            "geta": st.session_state.geta,
            "burn_in_multiplier": st.session_state.burn_in_multiplier
        },
        "calc_time": st.session_state.calc_time,
        "average_acceptance_ratio": st.session_state.average_acceptance_ratio,
        "acceptance_rates": acceptance_rates,
        "acceptance_rate_changes": acceptance_rate_changes
    }
    with open(f"{save_dir}/data/results.json", "w") as outfile:
        json.dump(results, outfile, indent=4)

    st.info(f"Results saved to {save_dir}")
def main():
    st.title('Markov chain Monte Carlo Algorithm Sampling')
    initialize_parameters()

    if st.session_state.use_maximal_c:
        calculate_maximal_c()

    if st.button('Calculate All Patterns'):
        calculate_all_patterns()

    if st.button('Calculate'):
        log_flag, record_flag, use_metropolis_within_gibbs_flag = set_flags()
        # Run taichi_calculator.py using subprocess with virtual environment activation
        subprocess.run(
            f"{venv_activate} && python taichi_calculator.py "
            f"--num_of_particles {st.session_state.num_of_particles} "
            f"--a {st.session_state.a} "
            f"--b {st.session_state.b} "
            f"--c {st.session_state.c} "
            f"--s {st.session_state.s} "
            f"--proposal_std {st.session_state.proposal_std} "
            f"--num_of_chains {st.session_state.num_of_chains} "
            f"--target_distribution_name {st.session_state.target_distribution_name} "
            f"--num_of_iterations_for_each_chain {st.session_state.num_of_iterations_for_each_chain} "
            f"--num_of_mutations {st.session_state.num_of_mutations} "
            f"--burn_in_multiplier {st.session_state.burn_in_multiplier} "
            f"{log_flag} "
            f"{record_flag} "
            f"{use_metropolis_within_gibbs_flag}",
            shell=True
        )
        load_data()
        st.info(f"Calculation Time: {st.session_state.calc_time:.2f} sec")
        st.info(f"Average Acceptance Ratio: {st.session_state.average_acceptance_ratio:.2f}%")

    if st.session_state.current_particles is not None and st.session_state.plotly:
        visualize_particles_with_plotly()

    if st.session_state.result_particles is not None:
        st.info(f'Sampled a total of {len(st.session_state.result_particles)} times per chain.')
        visualize_min_distance_particles()
        visualize_acceptance_rate()
        visualize_acceptance_rate_change()

    if st.session_state.current_particles is not None and st.session_state.show_particles:
        visualize_particles()

    if st.session_state.distances is not None:
        visualize_histogram()
        if st.button('Save Results'):
            save_results(f"Results/{st.session_state.target_distribution_name}/{format(datetime.datetime.now(), '%Y%m%d_%H%M%S')}")

if __name__ == "__main__":
    main()