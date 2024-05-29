import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from vispy import scene
from vispy.io import write_png
import subprocess
import datetime
import colorsys
import math

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
    st.session_state.target_distribution_name = st.sidebar.selectbox("Target Distribution", ["target_distribution", "target_distribution2", "target_distribution3", "target_distribution4", "target_distribution5", "target_distribution01", "target_distribution005", "target_distribution_sigmoid"], index=4)
    st.session_state.a = st.sidebar.number_input("a", 0.0, 20.0, 3 * np.pi)
    st.session_state.b = st.sidebar.number_input("b", 0.0, 5.0, 0.05)
    st.session_state.c = st.sidebar.number_input("c", 0.0, 5.0, 0.1, step=0.001)
    st.session_state.s = st.sidebar.number_input("s", 0.0, 5.0, 0.1)
    st.session_state.proposal_std = st.sidebar.number_input("Proposal Standard Deviation", 0.0, 5.0, 0.5)
    st.session_state.r_threshold = st.sidebar.number_input("r Threshold", 0.0, 1.0, 0.001)
    st.session_state.num_of_independent_trials = st.sidebar.number_input("Number of Independent Trials", 1, 1000000, 10000)
    st.session_state.num_of_iterations_for_each_trial = st.sidebar.number_input("Number of Iterations for Each Trial", 1, 100000000, 10000)
    st.session_state.num_of_sampling_strides = st.sidebar.number_input("Number of Sampling Strides", 100, 10000000, 1000)
    st.session_state.scaling_factor = st.sidebar.number_input("Scaling Factor", 0.0, 100.0, 5.0, step=0.5)
    st.session_state.geta = st.sidebar.number_input("Geta", 0.0, 30.0, 5.0, step=0.5)
    st.session_state.show_particles = st.sidebar.checkbox("Visualize Particles", False)
    st.session_state.each_particle = st.sidebar.checkbox("Visualize Each Particle Index", False)
    st.session_state.save_image = st.sidebar.checkbox("Save Image", False)
    st.session_state.plotly = st.sidebar.checkbox("Use Plotly", False)
    st.session_state.use_maximal_c = st.sidebar.checkbox("Use Maximal c", True)
    st.session_state.acceptance_ratio_calculation_with_log = st.sidebar.checkbox("Calculate Acceptance Ratio with Log", False)
    st.session_state.record_from_first_acceptance = st.sidebar.checkbox("Record from First Acceptance", True)
    st.session_state.burn_in_multiplier = st.sidebar.number_input("Burn-in Multiplier", 1.0, 5.0, 1.5, step=0.1)

    if 'current_particles' not in st.session_state:
        st.session_state.current_particles = None

    if 'result_particles' not in st.session_state:
        st.session_state.result_particles = None

    if 'distances' not in st.session_state:
        st.session_state.distances = None

    if 'min_distance' not in st.session_state:
        st.session_state.min_distance = 1.0

    if 'min_distance_particles' not in st.session_state:
        st.session_state.min_distance_particles = None

    if 'prev_scaling_factor' not in st.session_state:
        st.session_state.prev_scaling_factor = None

    if 'prev_geta' not in st.session_state:
        st.session_state.prev_geta = None

def calculate_maximal_c():
    Cab = -(2 * ((1 - 3 * st.session_state.a ** 2) * np.sin(st.session_state.a * st.session_state.b) + st.session_state.a * (st.session_state.a ** 2 - 3) * np.cos(st.session_state.a * st.session_state.b))) / ((st.session_state.a ** 2 + 1) ** 3)
    t = np.sin(-st.session_state.a * st.session_state.b) - (Cab / 2)

    if st.session_state.target_distribution_name == 'target_distribution':
        num_of_combinations = st.session_state.num_of_particles * ( st.session_state.num_of_particles - 1) / 2
        st.session_state.c = -1 / (t * num_of_combinations)
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
    elif st.session_state.target_distribution_name == 'target_distribution4':
        sin_ab = np.sin(st.session_state.a * st.session_state.b)
        cos_ab = np.cos(st.session_state.a * st.session_state.b)
        numerator = 2 * st.session_state.a * cos_ab - (1 - st.session_state.a ** 2) * sin_ab
        denominator = (1 + st.session_state.a ** 2) ** 2
        Cab = numerator / denominator
        st.info(f'Cab is {Cab}')
        num_of_combinations = st.session_state.num_of_particles * (st.session_state.num_of_particles - 1) / 2
        st.session_state.c = -1 / ((np.sin(-st.session_state.a * st.session_state.b) - Cab) * num_of_combinations)
    else:
        st.session_state.c = -1 / t
    st.info(f'c is {st.session_state.c}')

def load_data():
    if os.path.exists('temp_folder/initial_particles.npy'):
        st.session_state.initial_particles = np.load('temp_folder/initial_particles.npy')
    if os.path.exists('temp_folder/result_particles.npy'):
        st.session_state.result_particles = np.load('temp_folder/result_particles.npy')
    if os.path.exists('temp_folder/current_particles.npy'):
        st.session_state.current_particles = np.load('temp_folder/current_particles.npy')
    if os.path.exists('temp_folder/calc_time.txt'):
        with open('temp_folder/calc_time.txt', 'r') as f:
            st.session_state.calc_time = float(f.read())
    if os.path.exists('temp_folder/distances.npy'):
        st.session_state.distances = np.load('temp_folder/distances.npy')
    if os.path.exists('temp_folder/min_distance_particles.npy'):
        st.session_state.min_distance_particles = np.load('temp_folder/min_distance_particles.npy')
    if os.path.exists('temp_folder/min_distance_pair.npy'):
        st.session_state.min_distance_pair = tuple(np.load('temp_folder/min_distance_pair.npy'))
    if os.path.exists('temp_folder/min_distance.txt'):
        with open('temp_folder/min_distance.txt', 'r') as f:
            st.session_state.min_distance = float(f.read())
    if os.path.exists('temp_folder/average_acceptance_ratio.txt'):
        with open('temp_folder/average_acceptance_ratio.txt', 'r') as f:
            st.session_state.average_acceptance_ratio = float(f.read())

def visualize_particles_with_plotly():
    df_list = []
    # for i, particles in enumerate(st.session_state.current_particles):
    #     df_temp = pd.DataFrame(particles, columns=['x', 'y'])
    #     df_temp['Particle Index'] = i % st.session_state.num_of_particles
    #     df_list.append(df_temp)
    #
    # df = pd.concat(df_list, ignore_index=True)
    #
    # fig = px.scatter(df, x='x', y='y', color='Particle Index',
    #                  title='Metropolis-Hastings Algorithm Sampling with Colored Particles',
    #                  range_x=[0, 1], range_y=[0, 1], width=700, height=700,
    #                  color_continuous_scale=px.colors.qualitative.G10)
    # fig.update_layout(yaxis_scaleanchor='x', xaxis_constrain='domain')
    # st.plotly_chart(fig, theme=None)

    # st.session_state.current_particlesの最後の要素を取得
    last_particles = st.session_state.current_particles[-1]
    st.info(f'Number of Particles: {len(last_particles)}')
    st.info(last_particles)

    # last_particlesをplotlyで描画, 正方形の範囲, スケーリングを指定
    fig = go.Figure()
    for i in range(len(last_particles)):
        fig.add_trace(go.Scatter(x=[last_particles[i][0]], y=[last_particles[i][1]], mode='markers', name=f'Particle {i}'))
    fig.update_layout(title='Metropolis-Hastings Algorithm Sampling with Particles',
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
def calculate_sigmoid_kappa(r_list, scaling_factor, c, s, a, b, c_ab_val, geta):
    sigmoid_formula = 1 / (1 + np.exp(-r_list ** 2))
    c = 12.4171897625123
    b = -7.20859488125615
    return scaling_factor * (c * sigmoid_formula + b) + geta

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
        fig = go.Figure(data=[go.Histogram(x=st.session_state.distances, histnorm='density', nbinsx=50)])
        fig.update_layout(title='Distance between Two Particles', xaxis_title='Distance', yaxis_title='Density')
        fig.update_xaxes(range=[st.session_state.r_threshold, max(st.session_state.distances)])
        st.plotly_chart(fig, theme=None)

        # Normalize the histogram as a density plot
        hist, bin_edges = np.histogram(st.session_state.distances, bins=50, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        normalized_hist = hist / bin_centers

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
            else:
                st.session_state.kappa_values = calculate_sigmoid_kappa(r_list, st.session_state.scaling_factor, st.session_state.c, st.session_state.s, st.session_state.a, st.session_state.b, st.session_state.c_ab_val, st.session_state.geta)

            st.session_state.prev_scaling_factor = st.session_state.scaling_factor
            st.session_state.prev_geta = st.session_state.geta

        fig = go.Figure(data=[go.Bar(x=bin_centers, y=normalized_hist)])
        fig.add_trace(go.Scatter(x=r_list, y=st.session_state.kappa_values, mode='lines', name='Kappa Values'))
        fig.update_xaxes(range=[st.session_state.r_threshold, max(st.session_state.distances)])
        fig.update_yaxes(range=[0, max(normalized_hist) * 1.1])
        fig.update_layout(title='Normalized Distance between Two Particles', xaxis_title='Distance', yaxis_title='Normalized Frequency')
        st.plotly_chart(fig, theme=None)

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

        fig.add_annotation(
            x=1.05,
            y=1.0,
            showarrow=False,
            text=f'Minimum Distance: {st.session_state.min_distance:.4e}<br>Number of Particles: {st.session_state.num_of_particles}<br>Iterations per Trial: {st.session_state.num_of_iterations_for_each_trial}<br>Sampling Strides: {st.session_state.num_of_sampling_strides}',
            xref='paper',
            yref='paper'
        )

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
            filename = f'particles_index_{i}_trial_count_{st.session_state.num_of_independent_trials}_mh_steps_{st.session_state.num_of_iterations_for_each_trial}'
            image_index = draw_particles(data_index, colors_index, filename, save=st.session_state.save_image)
            st.image(image_index)

def main():
    st.title('Metropolis-Hastings Algorithm Sampling')
    initialize_parameters()

    if st.session_state.use_maximal_c:
        calculate_maximal_c()

    if st.button('Calculate'):
        log_flag = "--acceptance_ratio_calculation_with_log" if st.session_state.acceptance_ratio_calculation_with_log else ""
        record_flag = "--record_from_first_acceptance" if st.session_state.record_from_first_acceptance else ""
        # Run taichi_calculator.py using subprocess with virtual environment activation
        subprocess.run(
            f"{venv_activate} && python taichi_calculator.py "
            f"--num_of_particles {st.session_state.num_of_particles} "
            f"--a {st.session_state.a} "
            f"--b {st.session_state.b} "
            f"--c {st.session_state.c} "
            f"--s {st.session_state.s} "
            f"--proposal_std {st.session_state.proposal_std} "
            f"--burn_in_multiplier {st.session_state.burn_in_multiplier} "
            f"--num_of_independent_trials {st.session_state.num_of_independent_trials} "
            f"--target_distribution_name {st.session_state.target_distribution_name} "
            f"--num_of_iterations_for_each_trial {st.session_state.num_of_iterations_for_each_trial} "
            f"--num_of_sampling_strides {st.session_state.num_of_sampling_strides} "
            f"{log_flag} "
            f"{record_flag}",
            shell=True
        )
        load_data()
        st.info(f"Calculation Time: {st.session_state.calc_time:.2f} sec")
        st.info(f"Average Acceptance Ratio: {st.session_state.average_acceptance_ratio:.3f}%")

    if st.session_state.current_particles is not None and st.session_state.plotly:
        visualize_particles_with_plotly()

    if st.session_state.result_particles is not None:
        st.info(f'Sampled a total of {len(st.session_state.result_particles)} times in each independent trial.')
        visualize_min_distance_particles()

    if st.session_state.current_particles is not None and st.session_state.show_particles:
        visualize_particles()

    if st.session_state.distances is not None:
        visualize_histogram()

if __name__ == "__main__":
    main()