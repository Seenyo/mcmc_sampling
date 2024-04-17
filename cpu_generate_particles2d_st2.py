import math
import os
import time
import datetime
import colorsys

import streamlit as st
import numpy as np
import taichi as ti
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from stqdm import stqdm
from vispy import scene
from vispy.io import write_png


def toroidal_distance(length, p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])

    if dx > length / 2:
        dx = length - dx
    if dy > length / 2:
        dy = length - dy

    return math.sqrt(dx ** 2 + dy ** 2)


@ti.data_oriented
class MetropolisHastings:
    def __init__(self, num_of_particles, a, b, c, s, proposal_std, num_of_independent_trials):
        self.a = a
        self.b = b
        self.c = c
        self.s = s
        self.proposal_std = proposal_std
        self.num_of_independent_trials = num_of_independent_trials
        self.num_of_particles = num_of_particles

        self.init_particles = ti.Vector.field(2, dtype=ti.f32, shape=(num_of_independent_trials, num_of_particles))
        self.proposed_particles = ti.Vector.field(2, dtype=ti.f32, shape=(num_of_independent_trials, num_of_particles))
        self.current_particles = ti.Vector.field(2, dtype=ti.f32, shape=(num_of_independent_trials, num_of_particles))

    @ti.kernel
    def initialize_particles(self):
        #Initialize the particles with the same initial values
        for trial_idx in range(self.num_of_independent_trials):
            for particle_idx in range(self.num_of_particles):
                self.init_particles[trial_idx, particle_idx] = ti.Vector(
                    [ti.random(dtype=ti.f32), ti.random(dtype=ti.f32)])
                self.current_particles[trial_idx, particle_idx] = self.init_particles[trial_idx, particle_idx]

    @ti.func
    def toroidal_distance(self, length, p1, p2):
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])

        if dx > length / 2:
            dx = length - dx
        if dy > length / 2:
            dy = length - dy

        return ti.sqrt(dx ** 2 + dy ** 2)

    @ti.func
    def target_distribution(self, trial_idx, is_proposed=False):
        sin_ab = ti.sin(self.a * self.b)
        cos_ab = ti.cos(self.a * self.b)
        numerator = 2 * ((1 - 3 * self.a ** 2) * sin_ab + self.a * (self.a ** 2 - 3) * cos_ab)
        denominator = (self.a ** 2 + 1) ** 3
        c_ab_val = -1 * numerator / denominator

        kappa2_37_sum = 0.0

        # x1 = self.proposed_particles[trial_idx, 0] if is_proposed else self.mh_particles[trial_idx, 0]
        # kappa2_37_sum = ti.sin(2 * 3.14 * x1[0]) * ti.cos(2 * 3.14 * x1[1]) + 1
        # return kappa2_37_sum

        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[trial_idx, k] if is_proposed else self.current_particles[trial_idx, k]
                x2 = self.proposed_particles[trial_idx, l] if is_proposed else self.current_particles[trial_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa2_37_sum += self.c * ti.exp(-1 * r / self.s) * (
                        ti.sin(self.a * (r / self.s - self.b)) - c_ab_val * 0.5)

        val = 1 / (1 ** self.num_of_particles) + 1 / (1 ** (self.num_of_particles - 2)) * kappa2_37_sum
        return val

    @ti.kernel
    def compute_mh(self):
        for i in range(self.num_of_independent_trials):
            for j in range(self.num_of_particles):
                val_x = (ti.random(dtype=ti.f32) - 0.5) * self.proposal_std
                val_y = (ti.random(dtype=ti.f32) - 0.5) * self.proposal_std

                self.proposed_particles[i, j][0] = (self.current_particles[i, j][0] + val_x) % 1.0
                self.proposed_particles[i, j][1] = (self.current_particles[i, j][1] + val_y) % 1.0

            acceptance_ratio = self.target_distribution(i, True) / self.target_distribution(i)

            if acceptance_ratio >= 1 or ti.random(dtype=ti.f32) < acceptance_ratio:
                for j in range(self.num_of_particles):
                    self.current_particles[i, j] = self.proposed_particles[i, j]


def initialize_parameters():
    """Initialize and return user-defined parameters from the sidebar."""
    st.session_state.num_of_particles = st.sidebar.number_input("Number of Particles", 1, 10000, 2)
    st.session_state.a = st.sidebar.number_input("a", 0.0, 10.0, np.pi)
    st.session_state.b = st.sidebar.number_input("b", 0.0, 1.0, 0.25)
    st.session_state.c = st.sidebar.number_input("c", 0.0, 5.0, 0.1, step=0.001)
    st.session_state.s = st.sidebar.number_input("s", 0.0, 1.0, 0.1)
    st.session_state.proposal_std = st.sidebar.number_input("Proposal Standard Deviation", 0.0, 5.0, 1.0)
    st.session_state.r_threshold = st.sidebar.number_input("r Threshold", 0.0, 1.0, 0.001)
    st.session_state.num_of_independent_trials = st.sidebar.number_input("Number of Independent Trials", 1, 10000000,
                                                                         10000)
    st.session_state.num_of_iterations_for_each_trial = st.sidebar.number_input("Number of Iterations for Each Trial",
                                                                                1, 1000000, 10000)
    st.session_state.num_of_sampling_strides = st.sidebar.number_input("Number of Sampling Strides", 100, 10000, 1000)
    st.session_state.scaling_factor = st.sidebar.slider("Scaling Factor", 0.0, 10000.0, 5000.0)
    st.session_state.geta = st.sidebar.slider("Geta", 0.0, 10000.0, 5000.0)
    st.session_state.show_particles = st.sidebar.checkbox("Visualize Particles", False)
    st.session_state.each_particle = st.sidebar.checkbox("Visualize Each Particle Index", False)
    st.session_state.save_image = st.sidebar.checkbox("Save Image", False)
    st.session_state.plotly = st.sidebar.checkbox("Use Plotly", False)

    if 'mh_particles' not in st.session_state:
        st.session_state.current_particles = None

    if 'result_particles' not in st.session_state:
        st.session_state.result_particles = None

    if 'distances' not in st.session_state:
        st.session_state.distances = None


def perform_calculations():
    """Perform the Metropolis-Hastings calculations and return the results."""
    Cab = -(2*((1-3*st.session_state.a**2)*np.sin(st.session_state.a*st.session_state.b)+st.session_state.a*(st.session_state.a**2-3)*np.cos(st.session_state.a*st.session_state.b))) / ((st.session_state.a**2+1)**3)
    st.session_state.c = -2 / ((np.sin(-st.session_state.a*st.session_state.b)-(Cab/2)) * st.session_state.num_of_particles*(st.session_state.num_of_particles-1))
    st.info(f'c is {st.session_state.c}')
    MH = MetropolisHastings(
        st.session_state.num_of_particles,
        st.session_state.a,
        st.session_state.b,
        st.session_state.c,
        st.session_state.s,
        st.session_state.proposal_std,
        st.session_state.num_of_independent_trials,
    )
    MH.initialize_particles()

    # Initial particles visualization setup
    st.session_state.initial_particles = MH.init_particles.to_numpy()
    st.session_state.colors = np.zeros((st.session_state.num_of_particles, 4))
    for i in range(st.session_state.num_of_particles):
        hue = i / float(st.session_state.num_of_particles)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        st.session_state.colors[i] = rgb + (1.0,)

    # Perform calculations
    taichi_start = time.time()
    st.session_state.result_particles = []
    for trial_idx in stqdm(range(st.session_state.num_of_iterations_for_each_trial)):
        MH.compute_mh()
        if trial_idx % st.session_state.num_of_sampling_strides == 0 and trial_idx != 0:
            st.session_state.result_particles.append(MH.current_particles.to_numpy())

    st.session_state.calc_time = time.time() - taichi_start

    st.session_state.current_particles = MH.current_particles.to_numpy()

    st.info(f'Sampled a total of {len(st.session_state.result_particles)} times in each independent trial.')

    st.session_state.distances = []

    for i in range(len(st.session_state.result_particles)):
        for j in range(len(st.session_state.result_particles[i])):
            dist = toroidal_distance(1.0, st.session_state.result_particles[i][j][0],
                                     st.session_state.result_particles[i][j][1])
            st.session_state.distances.append(dist)

    st.info(f"Number of Distances: {len(st.session_state.distances)}")

    if st.session_state.plotly:
        # Display the particles with plotly
        df_list = []
        for i, particles in enumerate(st.session_state.current_particles):
            df_temp = pd.DataFrame(particles, columns=['x', 'y'])
            df_temp['Particle Index'] = i % st.session_state.num_of_particles
            df_list.append(df_temp)

        df = pd.concat(df_list, ignore_index=True)

        fig = px.scatter(df, x='x', y='y', color='Particle Index',
                         title='Metropolis-Hastings Algorithm Sampling with Colored Particles',
                         range_x=[0, 1], range_y=[0, 1], width=700, height=700,
                         color_continuous_scale=px.colors.qualitative.G10)
        fig.update_layout(yaxis_scaleanchor='x', xaxis_constrain='domain')
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
        # Save the image with the given filename
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


def visualize_histogram():
    # Display the histogram with go
    fig = go.Figure(data=[go.Histogram(x=st.session_state.distances, nbinsx=1000)])
    fig.update_layout(title='Distance between Two Particles', xaxis_title='Distance', yaxis_title='Frequency')
    st.plotly_chart(fig, theme=None)

    # Normalize the histogram
    hist, bin_edges = np.histogram(st.session_state.distances, bins=1000)
    # print("hist", hist)
    # print("bin_edges", bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # print("bin_centers", bin_centers)

    # r threshold is set to avoid division by zero and ignore small distances
    valid_indices = bin_centers > st.session_state.r_threshold
    # print("valid_indices", valid_indices)
    filtered_bin_centers = bin_centers[valid_indices]
    # print("filtered_bin_centers", filtered_bin_centers)
    filtered_hist = hist[valid_indices]
    # print("filtered_hist", filtered_hist)

    normalized_hist = filtered_hist / filtered_bin_centers

    # print("normalized_hist", normalized_hist)

    # calculate kappa value
    sin_ab = np.sin(st.session_state.a * st.session_state.b)
    cos_ab = ti.cos(st.session_state.a * st.session_state.b)
    numerator = 2 * ((1 - 3 * st.session_state.a ** 2) * sin_ab + st.session_state.a * (
            st.session_state.a ** 2 - 3) * cos_ab)
    denominator = (st.session_state.a ** 2 + 1) ** 3
    c_ab_val = -1 * numerator / denominator

    def kappa(r):
        return st.session_state.scaling_factor * st.session_state.c * np.exp(-1 * r / st.session_state.s) * (np.sin(
            st.session_state.a * (
                    r / st.session_state.s - st.session_state.b)) - c_ab_val * 0.5) + st.session_state.geta

    kappa_values = [kappa(r) for r in filtered_bin_centers]

    fig = go.Figure(data=[go.Bar(x=bin_edges, y=normalized_hist)])
    fig.add_trace(go.Scatter(x=filtered_bin_centers, y=kappa_values, mode='lines', name='Kappa Value'))
    # ヒストグラムの表示範囲を設定, 最大値はdistancesの最大値
    fig.update_xaxes(range=[st.session_state.r_threshold, max(st.session_state.distances)])
    fig.update_layout(title='Normalized Distance between Two Particles', xaxis_title='Distance',
                      yaxis_title='Normalized Frequency')
    st.plotly_chart(fig, theme=None)


def main():
    st.title('Metropolis-Hastings Algorithm Sampling')
    initialize_parameters()

    if st.button('Calculate'):
        perform_calculations()
        st.info(f"Calculation Time: {st.session_state.calc_time:.2f} sec")

    if st.session_state.current_particles is not None and st.session_state.show_particles:
        visualize_particles()

    if st.session_state.distances is not None:
        visualize_histogram()


if __name__ == "__main__":
    ti.init(arch=ti.cpu, random_seed=int(time.time()))
    main()
