import argparse
import math
import os
import time

import taichi as ti
import numpy as np

from tqdm import tqdm

@ti.data_oriented
class MetropolisHastings:
    def __init__(self, num_of_particles, a, b, c, s, proposal_std, num_of_independent_trials, target_distribution_name='target_distribution'):
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

        self.target_distribution_name = target_distribution_name

    @ti.kernel
    def initialize_particles(self):
        # Initialize the particles with the same initial values
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

        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[trial_idx, k] if is_proposed else self.current_particles[trial_idx, k]
                x2 = self.proposed_particles[trial_idx, l] if is_proposed else self.current_particles[trial_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa2_37_sum += self.c * ti.exp(-1 * r / self.s) * (
                        ti.sin(self.a * (r / self.s - self.b)) - c_ab_val * 0.5)

        val = 1 / (1 ** self.num_of_particles) + 1 / (1 ** (self.num_of_particles - 2)) * kappa2_37_sum
        return val

    @ti.func
    def target_distribution2(self, trial_idx, is_proposed=False):
        sin_ab = ti.sin(self.a * self.b)
        cos_ab = ti.cos(self.a * self.b)
        numerator = 2 * ((1 - 3 * self.a ** 2) * sin_ab + self.a * (self.a ** 2 - 3) * cos_ab)
        denominator = (self.a ** 2 + 1) ** 3
        c_ab_val = -1 * numerator / denominator

        # It is clear that first_order_term should be 1.0,
        # but we dare to calculate it for the understanding of the paper.

        area = 1.0
        first_order_term = 1.0
        for i in range(self.num_of_particles):
            first_order_term *= 1 / area

        # calculate second_order_term
        second_order_term = 1.0
        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[trial_idx, k] if is_proposed else self.current_particles[trial_idx, k]
                x2 = self.proposed_particles[trial_idx, l] if is_proposed else self.current_particles[trial_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa2_37 = self.c * ti.exp(-1 * r / self.s) * (
                        ti.sin(self.a * (r / self.s - self.b)) - c_ab_val * 0.5)
                second_order_term *= (1.0 + kappa2_37)

        val = first_order_term * second_order_term

        if val < 0:
            print(f'val is a negative value: {val}')

        return val

    @ti.func
    def target_distribution_sigmoid(self, trial_idx, is_proposed=False):
        # It is clear that first_order_term should be 1.0,
        # but we dare to calculate it for the understanding of the paper.

        area = 1.0
        first_order_term = 1.0
        for i in range(self.num_of_particles):
            first_order_term *= 1 / area

        # calculate second_order_term
        second_order_term = 1.0
        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[trial_idx, k] if is_proposed else self.current_particles[trial_idx, k]
                x2 = self.proposed_particles[trial_idx, l] if is_proposed else self.current_particles[trial_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)

                sigmoid_formula = 1 / (1 + ti.exp(-r ** 2))
                c = 12.4171897625123
                b = -7.20859488125615
                second_order_term *= (1.0 + c * sigmoid_formula + b)

        val = first_order_term * second_order_term

        if val < 0:
            print(f'val is a negative value: {val}')

        return val

    @ti.func
    def target_distribution3(self, trial_idx, is_proposed=False):
        sin_ab = ti.sin(self.a * self.b)
        cos_ab = ti.cos(self.a * self.b)
        numerator = 2 * self.a * cos_ab - (1 - self.a**2) * sin_ab
        denominator = (1 + self.a ** 2) ** 2
        Cab = numerator / denominator

        # It is clear that first_order_term should be 1.0,
        # but we dare to calculate it for the understanding of the paper.

        area = 1.0
        first_order_term = 1.0
        for i in range(self.num_of_particles):
            first_order_term *= 1 / area

        # calculate second_order_term
        second_order_term = 1.0
        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[trial_idx, k] if is_proposed else self.current_particles[trial_idx, k]
                x2 = self.proposed_particles[trial_idx, l] if is_proposed else self.current_particles[trial_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa2_37 = self.c * ti.exp(-1 * r / self.s) * (
                        ti.sin(self.a * (r / self.s - self.b)) - Cab)
                second_order_term *= (1.0 + kappa2_37)

        val = first_order_term * second_order_term

        if val < 0:
            print(f'val is a negative value: {val}')

        return val

    @ti.func
    def target_distribution4(self, trial_idx, is_proposed=False):
        sin_ab = ti.sin(self.a * self.b)
        cos_ab = ti.cos(self.a * self.b)
        numerator = 2 * self.a * cos_ab - (1 - self.a**2) * sin_ab
        denominator = (1 + self.a ** 2) ** 2
        Cab = numerator / denominator

        # It is clear that first_order_term should be 1.0,
        # but we dare to calculate it for the understanding of the paper.

        area = 1.0
        first_order_term = 1.0
        for i in range(self.num_of_particles):
            first_order_term *= 1 / area

        # calculate second_order_term
        second_order_term = 0.0
        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[trial_idx, k] if is_proposed else self.current_particles[trial_idx, k]
                x2 = self.proposed_particles[trial_idx, l] if is_proposed else self.current_particles[trial_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                second_order_term += self.c * ti.exp(-1 * r / self.s) * (
                        ti.sin(self.a * (r / self.s - self.b)) - Cab)

        val = first_order_term + 1 / (1 ** (self.num_of_particles - 2)) * second_order_term

        if val < 0:
            print(f'val is a negative value: {val}')

        return val

    @ti.kernel
    def compute_mh(self):
        for i in range(self.num_of_independent_trials):
            for j in range(self.num_of_particles):
                val_x = (ti.random(dtype=ti.f32) - 0.5) * self.proposal_std
                val_y = (ti.random(dtype=ti.f32) - 0.5) * self.proposal_std

                self.proposed_particles[i, j][0] = (self.current_particles[i, j][0] + val_x) % 1.0
                self.proposed_particles[i, j][1] = (self.current_particles[i, j][1] + val_y) % 1.0

            acceptance_ratio = 0.0
            if self.target_distribution_name == 'target_distribution':
                acceptance_ratio = self.target_distribution(i, True) / self.target_distribution(i)
            elif self.target_distribution_name == 'target_distribution2':
                acceptance_ratio = self.target_distribution2(i, True) / self.target_distribution2(i)
            elif self.target_distribution_name == 'target_distribution3':
                acceptance_ratio = self.target_distribution3(i, True) / self.target_distribution3(i)
            elif self.target_distribution_name == 'target_distribution4':
                acceptance_ratio = self.target_distribution4(i, True) / self.target_distribution4(i)
            elif self.target_distribution_name == 'target_distribution_sigmoid':
                acceptance_ratio = self.target_distribution_sigmoid(i, True) / self.target_distribution_sigmoid(i)
            else:
                print('Invalid target distribution name')

            if acceptance_ratio >= 1.0 or ti.random(dtype=ti.f32) < acceptance_ratio:
                for j in range(self.num_of_particles):
                    self.current_particles[i, j] = self.proposed_particles[i, j]

def toroidal_distance(length, p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])

    if dx > length / 2:
        dx = length - dx
    if dy > length / 2:
        dy = length - dy

    return math.sqrt(dx ** 2 + dy ** 2)

def initialize_particles(mh):
    mh.initialize_particles()
    initial_particles = mh.init_particles.to_numpy()
    np.save('temp_folder/initial_particles.npy', initial_particles)

def perform_calculations(args):
    MH = MetropolisHastings(
        args.num_of_particles,
        args.a,
        args.b,
        args.c,
        args.s,
        args.proposal_std,
        args.num_of_independent_trials,
        args.target_distribution_name
    )
    initialize_particles(MH)

    # Perform calculations
    taichi_start = time.time()
    result_particles = []
    for trial_idx in tqdm(range(args.num_of_iterations_for_each_trial)):
        MH.compute_mh()
        if trial_idx % args.num_of_sampling_strides == 0 and trial_idx != 0:
            result_particles.append(MH.current_particles.to_numpy())

    calc_time = time.time() - taichi_start

    current_particles = MH.current_particles.to_numpy()

    np.save('temp_folder/result_particles.npy', result_particles)
    np.save('temp_folder/current_particles.npy', current_particles)

    # Save calculation time to a file
    with open('temp_folder/calc_time.txt', 'w') as f:
        f.write(str(calc_time))

def calculate_distances():
    result_particles = np.load('temp_folder/result_particles.npy')
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

    np.save('temp_folder/distances.npy', distances)
    np.save('temp_folder/min_distance_particles.npy', min_distance_particles)
    np.save('temp_folder/min_distance_pair.npy', min_distance_pair)

    # Save min_distance to a file
    with open('temp_folder/min_distance.txt', 'w') as f:
        f.write(str(min_distance))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_particles', type=int, default=2)
    parser.add_argument('--a', type=float, default=np.pi)
    parser.add_argument('--b', type=float, default=0.25)
    parser.add_argument('--c', type=float, default=0.1)
    parser.add_argument('--s', type=float, default=0.1)
    parser.add_argument('--proposal_std', type=float, default=1.0)
    parser.add_argument('--num_of_independent_trials', type=int, default=10000)
    parser.add_argument('--target_distribution_name', type=str, default='target_distribution')
    parser.add_argument('--num_of_iterations_for_each_trial', type=int, default=10000)
    parser.add_argument('--num_of_sampling_strides', type=int, default=1000)
    arguments = parser.parse_args()

    path = f'temp_folder'
    if not os.path.exists(path):
        os.makedirs(path)

    ti.init(arch=ti.cuda, random_seed=int(time.time()))

    perform_calculations(arguments)
    calculate_distances()

