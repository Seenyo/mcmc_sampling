import os
import time
import datetime
import colorsys

import streamlit as st
import numpy as np
import taichi as ti
import pandas as pd
import plotly.express as px

from stqdm import stqdm
from vispy import scene
from vispy.io import write_png

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
        self.mh_particles = ti.Vector.field(2, dtype=ti.f32, shape=(num_of_independent_trials, num_of_particles))

    @ti.kernel
    def initialize_particles(self):

        # for particle_idx in range(self.num_of_particles):
        #     self.init_particles[particle_idx] = ti.Vector([ti.random(dtype=ti.f32), ti.random(dtype=ti.f32)])

        #Initialize the particles with the same initial values
        for trial_idx in range(self.num_of_independent_trials):
            for particle_idx in range(self.num_of_particles):
                self.init_particles[trial_idx, particle_idx] = ti.Vector([ti.random(dtype=ti.f32), ti.random(dtype=ti.f32)])
                self.mh_particles[trial_idx, particle_idx] = self.init_particles[trial_idx, particle_idx]

    @ti.kernel
    def compute_mh(self):
        for i, j in ti.ndrange(self.num_of_independent_trials, self.num_of_particles):
            val_x = ti.random(dtype=ti.f32) * self.proposal_std
            val_y = ti.random(dtype=ti.f32) * self.proposal_std
            self.proposed_particles[i, j][0] = (self.mh_particles[i, j][0] + val_x) % 1.0
            self.proposed_particles[i, j][1] = (self.mh_particles[i, j][1] + val_y) % 1.0

            sin_ab = ti.sin(self.a * self.b)
            cos_ab = ti.cos(self.a * self.b)
            numerator = 2 * ((1 - 3 * self.a ** 2) * sin_ab + self.a * (self.a ** 2 - 3) * cos_ab)
            denominator = (self.a ** 2 + 1) ** 3
            c_ab_val = -1 * numerator / denominator


            kappa2_37_sum = 0.0

            for k in range(self.num_of_particles):
                for l in range(k + 1, self.num_of_particles):
                    x1 = self.proposed_particles[i, k]
                    x2 = self.proposed_particles[i, l]
                    r = ti.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
                    kappa2_37_sum += self.c * ti.exp(-1 * r / self.s) * (ti.sin(self.a * (r / self.s - self.b)) - c_ab_val * 0.5)

            acceptance_ratio = 1 / (1 ** self.num_of_particles) + 1 / (1 ** (self.num_of_particles - 2)) * kappa2_37_sum


            if acceptance_ratio >= 1 or ti.random(dtype=ti.f32) < acceptance_ratio:
                self.mh_particles[i, j] = self.proposed_particles[i, j]
def initialize_parameters():
    """Initialize and return user-defined parameters from the sidebar."""
    num_of_particles = st.sidebar.number_input("Number of Particles", 1, 10000, 2)
    a = st.sidebar.number_input("a", 0.0, 10.0, np.pi)
    b = st.sidebar.number_input("b", 0.0, 1.0, 0.25)
    c = st.sidebar.number_input("c", 0.0, 1.0, 0.1)
    s = st.sidebar.number_input("s", 0.0, 1.0, 0.1)
    proposal_std = st.sidebar.number_input("Proposal Standard Deviation", 0.0, 1.0, 0.1)
    num_of_independent_trials = st.sidebar.number_input("Number of Independent Trials", 1, 1000000, 100)
    num_of_iterations_for_each_trial = st.sidebar.number_input("Number of Iterations for Each Trial", 1, 1000000, 10)
    save_image = st.sidebar.checkbox("Save Image", False)
    return num_of_particles, a, b, c, s, proposal_std, num_of_independent_trials, num_of_iterations_for_each_trial, save_image

def perform_calculations(params):
    """Perform the Metropolis-Hastings calculations and return the results."""
    num_of_particles, a, b, c, s, proposal_std, num_of_independent_trials, num_of_iterations_for_each_trial, _ = params
    MH = MetropolisHastings(num_of_particles, a, b, c, s, proposal_std, num_of_independent_trials)
    MH.initialize_particles()

    # Initial particles visualization setup
    initial_particles = MH.init_particles.to_numpy()
    colors = np.zeros((num_of_particles, 4))
    for i in range(num_of_particles):
        hue = i / float(num_of_particles)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors[i] = rgb + (1.0,)

    # Perform calculations
    taichi_start = time.time()
    for _ in stqdm(range(num_of_iterations_for_each_trial)):
        MH.compute_mh()
    taichi_end = time.time()

    mh_particles = MH.mh_particles.to_numpy()

    # Display the particles with plotly
    df_list = []
    for i, particles in enumerate(mh_particles):
        df_temp = pd.DataFrame(particles, columns=['x', 'y'])
        df_temp['Particle Index'] = i % num_of_particles  # num_of_particlesの数に応じてインデックスを割り当て
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)

    # Plotlyでプロットする
    fig = px.scatter(df, x='x', y='y', color='Particle Index',  # ここでインデックスを色情報として使用
                     title='Metropolis-Hastings Algorithm Sampling with Colored Particles',
                     range_x=[0, 1], range_y=[0, 1], width=700, height=700,
                     color_continuous_scale=px.colors.qualitative.G10)
    fig.update_layout(yaxis_scaleanchor='x', xaxis_constrain='domain')
    st.plotly_chart(fig, theme=None)

    calc_time = taichi_end - taichi_start
    return initial_particles, colors, mh_particles, calc_time


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

def visualize_results(params, initial_particles, colors, particles):
    """Visualize and optionally save the results."""
    num_of_particles, _, _, _, _, _, num_of_independent_trials, num_of_iterations_for_each_trial, save_image = params

    # Visualize all initial particles
    data_all = np.concatenate(initial_particles, axis=0)
    # rgbaの4つの要素を持つ配列を作成
    num_of_channels = 4
    colors_all = np.zeros((len(data_all), num_of_channels))
    for i in range(num_of_particles):
        hue = i / float(num_of_particles)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors_all[i::num_of_particles] = rgb + (1.0,)
    image_all = draw_particles(data_all, colors_all, 'all_initial_particles', save=save_image)
    st.image(image_all)

    # Visualize all result particles
    data_all = np.concatenate(particles, axis=0)
    colors_all = np.zeros((len(data_all), num_of_channels))
    for i in range(num_of_particles):
        hue = i / float(num_of_particles)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors_all[i::num_of_particles] = rgb + (1.0,)
    image_all = draw_particles(data_all, colors_all, 'all_result_particles', save=save_image)
    st.image(image_all)

    # Visualize each particle index
    for i in range(num_of_particles):
        data_index = np.array([chain[i] for chain in particles])
        colors_index = colors_all[i::num_of_particles]
        filename = f'particles_index_{i}_trial_count_{num_of_independent_trials}_mh_steps_{num_of_iterations_for_each_trial}'
        image_index = draw_particles(data_index, colors_index, filename, save=save_image)
        st.image(image_index)

def main():
    st.title('Metropolis-Hastings Algorithm Sampling')
    params = initialize_parameters()

    if st.button('Calculate'):
        initial_particles, colors, mh_particles, calc_time = perform_calculations(params)
        st.info(f"Calculation Time: {calc_time:.2f} sec")
        visualize_results(params, initial_particles, colors, mh_particles)

if __name__ == "__main__":
    ti.init(arch=ti.cpu, random_seed=int(time.time()))
    main()


