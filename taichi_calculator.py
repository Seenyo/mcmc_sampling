import argparse
import math
import os
import time

import taichi as ti
import numpy as np

from tqdm import tqdm


@ti.data_oriented
class MetropolisHastings:
    def __init__(self, num_of_particles, a, b, c, s, proposal_std, num_of_independent_trials,
                 target_distribution_name='target_distribution', acceptance_ratio_calculation_with_log=False,
                 record_from_first_acceptance=False, use_metropolis_within_gibbs=False):
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

        # logを使ったacceptance ratioの計算を行うかどうか
        self.acceptance_ratio_calculation_with_log = acceptance_ratio_calculation_with_log

        # 最初のacceptanceが起こった時点から記録を行うかどうか
        self.record_from_first_acceptance = record_from_first_acceptance
        self.count_of_acceptance = ti.field(dtype=ti.i32, shape=num_of_independent_trials)

        # 前のステップの確率密度関数の値を保存する変数
        self.current_prob = ti.field(dtype=ti.f32, shape=num_of_independent_trials)

        # Metropolis-within-Gibbsを使うかどうか
        self.use_metropolis_within_gibbs = use_metropolis_within_gibbs

    @ti.kernel
    def initialize_all_chain_particles(self):
        # Initialize the particles with the same initial values
        for chain_idx in range(self.num_of_independent_trials):
            for particle_idx in range(self.num_of_particles):
                self.init_particles[chain_idx, particle_idx] = ti.Vector(
                    [ti.random(dtype=ti.f32), ti.random(dtype=ti.f32)])
                self.current_particles[chain_idx, particle_idx] = self.init_particles[chain_idx, particle_idx]

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
    def target_distribution(self, chain_idx, is_proposed=False):
        sin_ab = ti.sin(self.a * self.b)
        cos_ab = ti.cos(self.a * self.b)
        numerator = 2 * ((1 - 3 * self.a ** 2) * sin_ab + self.a * (self.a ** 2 - 3) * cos_ab)
        denominator = (self.a ** 2 + 1) ** 3
        c_ab_val = -1 * numerator / denominator

        kappa2_37_sum = 0.0

        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[chain_idx, k] if is_proposed else self.current_particles[chain_idx, k]
                x2 = self.proposed_particles[chain_idx, l] if is_proposed else self.current_particles[chain_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa2_37_sum += self.c * ti.exp(-1 * r / self.s) * (
                        ti.sin(self.a * (r / self.s - self.b)) - c_ab_val * 0.5)

        val = 1 / (1 ** self.num_of_particles) + 1 / (1 ** (self.num_of_particles - 2)) * kappa2_37_sum
        return val

    @ti.func
    def target_distribution2(self, chain_idx, is_proposed=False):
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
                x1 = self.proposed_particles[chain_idx, k] if is_proposed else self.current_particles[chain_idx, k]
                x2 = self.proposed_particles[chain_idx, l] if is_proposed else self.current_particles[chain_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa2_37 = self.c * ti.exp(-1 * r / self.s) * (
                        ti.sin(self.a * (r / self.s - self.b)) - c_ab_val * 0.5)
                second_order_term *= (1.0 + kappa2_37)

        val = first_order_term * second_order_term

        if val < 0:
            print(f'val is a negative value: {val}')

        return val

    @ti.func
    def target_distribution2_log(self, chain_idx, is_proposed=False):
        sin_ab = ti.sin(self.a * self.b)
        cos_ab = ti.cos(self.a * self.b)
        numerator = 2 * ((1 - 3 * self.a ** 2) * sin_ab + self.a * (self.a ** 2 - 3) * cos_ab)
        denominator = (self.a ** 2 + 1) ** 3
        c_ab_val = -1 * numerator / denominator

        # It is clear that first_order_term should be 1.0,
        # but we dare to calculate it for the understanding of the paper.

        area = 1.0
        first_order_term = 0.0
        for i in range(self.num_of_particles):
            first_order_term += ti.log(1 / area)

        # calculate second_order_term
        second_order_term = 0.0
        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[chain_idx, k] if is_proposed else self.current_particles[chain_idx, k]
                x2 = self.proposed_particles[chain_idx, l] if is_proposed else self.current_particles[chain_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa2_37 = self.c * ti.exp(-1 * r / self.s) * (
                        ti.sin(self.a * (r / self.s - self.b)) - c_ab_val * 0.5)
                second_order_term += ti.log(1.0 + kappa2_37)

        log_val = first_order_term + second_order_term

        return log_val

    @ti.func
    def target_distribution3(self, chain_idx, is_proposed=False):
        sin_ab = ti.sin(self.a * self.b)
        cos_ab = ti.cos(self.a * self.b)
        numerator = 2 * self.a * cos_ab - (1 - self.a ** 2) * sin_ab
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
                x1 = self.proposed_particles[chain_idx, k] if is_proposed else self.current_particles[chain_idx, k]
                x2 = self.proposed_particles[chain_idx, l] if is_proposed else self.current_particles[chain_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa2_37 = self.c * ti.exp(-1 * r / self.s) * (
                        ti.sin(self.a * (r / self.s - self.b)) - Cab)
                second_order_term *= (1.0 + kappa2_37)

        val = first_order_term * second_order_term

        if val < 0:
            print(f'val is a negative value: {val}')

        return val

    @ti.func
    def target_distribution3_log(self, chain_idx, is_proposed=False):
        sin_ab = ti.sin(self.a * self.b)
        cos_ab = ti.cos(self.a * self.b)
        numerator = 2 * self.a * cos_ab - (1 - self.a ** 2) * sin_ab
        denominator = (1 + self.a ** 2) ** 2
        Cab = numerator / denominator

        # It is clear that first_order_term should be 1.0,
        # but we dare to calculate it for the understanding of the paper.

        area = 1.0
        first_order_term = 0.0
        for i in range(self.num_of_particles):
            first_order_term += ti.log(1.0 / area)

        second_order_term = 0.0
        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[chain_idx, k] if is_proposed else self.current_particles[chain_idx, k]
                x2 = self.proposed_particles[chain_idx, l] if is_proposed else self.current_particles[chain_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa2_37 = self.c * ti.exp(-1 * r / self.s) * (
                        ti.sin(self.a * (r / self.s - self.b)) - Cab)
                second_order_term += ti.log(1.0 + kappa2_37)

        log_val = first_order_term + second_order_term

        return log_val

    @ti.func
    def target_distribution4(self, chain_idx, is_proposed=False):
        sin_ab = ti.sin(self.a * self.b)
        cos_ab = ti.cos(self.a * self.b)
        numerator = 2 * self.a * cos_ab - (1 - self.a ** 2) * sin_ab
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
                x1 = self.proposed_particles[chain_idx, k] if is_proposed else self.current_particles[chain_idx, k]
                x2 = self.proposed_particles[chain_idx, l] if is_proposed else self.current_particles[chain_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                second_order_term += self.c * ti.exp(-1 * r / self.s) * (
                        ti.sin(self.a * (r / self.s - self.b)) - Cab)

        val = first_order_term + 1 / (1 ** (self.num_of_particles - 2)) * second_order_term

        if val < 0:
            print(f'val is a negative value: {val}')

        return val

    @ti.func
    def target_distribution01(self, chain_idx, is_proposed=False):

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
                x1 = self.proposed_particles[chain_idx, k] if is_proposed else self.current_particles[chain_idx, k]
                x2 = self.proposed_particles[chain_idx, l] if is_proposed else self.current_particles[chain_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa_01 = -1 if r <= 0.1 else -1 * ti.exp(-3 * (r - 0.1)) * ti.cos(10 * (r - 0.1))
                second_order_term *= (1.0 + kappa_01)

        val = first_order_term * second_order_term

        if val < 0:
            print(f'val is a negative value: {val}')

        return val

    @ti.func
    def target_distribution5(self, chain_idx, is_proposed=False):
        a_squared_plus_one = self.a ** 2 + 1
        b_plus_one = self.b + 1
        b_minus_one = self.b - 1

        c_numer = (a_squared_plus_one ** 2) * (self.b ** 2 + 2 * self.b + 2)
        c_denom = 2 * ((a_squared_plus_one ** 2) * b_plus_one - a_squared_plus_one * b_minus_one - 2)
        c = c_numer / c_denom
        Cab_1 = b_minus_one / (a_squared_plus_one * b_plus_one)
        Cab_2 = 2 / (a_squared_plus_one ** 2 * b_plus_one)
        Cab_3 = self.b ** 2 / (2 * b_plus_one * c)
        Cab = Cab_1 + Cab_2 + Cab_3

        area = 1.0
        first_order_term = 1.0
        for i in range(self.num_of_particles):
            first_order_term *= 1 / area

        # calculate second_order_term
        second_order_term = 1.0
        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[chain_idx, k] if is_proposed else self.current_particles[chain_idx, k]
                x2 = self.proposed_particles[chain_idx, l] if is_proposed else self.current_particles[chain_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa2 = -1.0 if r < self.b else c * ti.exp(-(r - self.b)) * (-ti.cos(self.a * (r - self.b)) + Cab)
                second_order_term *= (1.0 + kappa2)

        val = first_order_term * second_order_term

        if val < -1e-5:
            print(f'val is a negative value: {val}')

        return val

    @ti.func
    def target_distribution5_log(self, chain_idx, is_proposed=False):
        a_squared_plus_one = self.a ** 2 + 1
        b_plus_one = self.b + 1
        b_minus_one = self.b - 1

        c_numer = (a_squared_plus_one ** 2) * (self.b ** 2 + 2 * self.b + 2)
        c_denom = 2 * ((a_squared_plus_one ** 2) * b_plus_one - a_squared_plus_one * b_minus_one - 2)
        c = c_numer / c_denom
        Cab_1 = b_minus_one / (a_squared_plus_one * b_plus_one)
        Cab_2 = 2 / (a_squared_plus_one ** 2 * b_plus_one)
        Cab_3 = self.b ** 2 / (2 * b_plus_one * c)
        Cab = Cab_1 + Cab_2 + Cab_3

        area = 1.0
        first_order_term = 0.0
        for i in range(self.num_of_particles):
            first_order_term += ti.log(1.0 / area)

        # calculate second_order_term
        second_order_term = 0.0
        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[chain_idx, k] if is_proposed else self.current_particles[chain_idx, k]
                x2 = self.proposed_particles[chain_idx, l] if is_proposed else self.current_particles[chain_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa2 = -1.0 if r < self.b else c * ti.exp(-(r - self.b)) * (-ti.cos(self.a * (r - self.b)) + Cab)
                second_order_term += ti.log(1.0 + kappa2)

        log_val = first_order_term + second_order_term

        return log_val

    @ti.func
    def target_distribution01_log(self, chain_idx, is_proposed=False):

        area = 1.0
        first_order_term = 0.0
        for i in range(self.num_of_particles):
            first_order_term += ti.log(1.0 / area)

        # calculate second_order_term
        second_order_term = 0.0
        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[chain_idx, k] if is_proposed else self.current_particles[chain_idx, k]
                x2 = self.proposed_particles[chain_idx, l] if is_proposed else self.current_particles[chain_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa_01 = -1 if r <= 0.1 else -1 * ti.exp(-3 * (r - 0.1)) * ti.cos(10 * (r - 0.1))
                second_order_term += ti.log(1.0 + kappa_01)

        log_val = first_order_term + second_order_term

        return log_val

    @ti.func
    def target_distribution005(self, chain_idx, is_proposed=False):

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
                x1 = self.proposed_particles[chain_idx, k] if is_proposed else self.current_particles[chain_idx, k]
                x2 = self.proposed_particles[chain_idx, l] if is_proposed else self.current_particles[chain_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa_005 = -1 if r <= 0.05 else -1 * ti.exp(-5 * (r - 0.05)) * ti.cos(22 * (r - 0.05))
                second_order_term *= (1.0 + kappa_005)

        val = first_order_term * second_order_term

        if val < 0:
            print(f'val is a negative value: {val}')

        return val

    @ti.func
    def target_distribution005_log(self, chain_idx, is_proposed=False):

        area = 1.0
        first_order_term = 0.0
        for i in range(self.num_of_particles):
            first_order_term += ti.log(1.0 / area)

        # calculate second_order_term
        second_order_term = 0.0
        for k in range(self.num_of_particles):
            for l in range(k + 1, self.num_of_particles):
                x1 = self.proposed_particles[chain_idx, k] if is_proposed else self.current_particles[chain_idx, k]
                x2 = self.proposed_particles[chain_idx, l] if is_proposed else self.current_particles[chain_idx, l]
                r = self.toroidal_distance(1.0, x1, x2)
                kappa_005 = -1 if r <= 0.05 else -1 * ti.exp(-5 * (r - 0.05)) * ti.cos(22 * (r - 0.05))
                second_order_term += ti.log(1.0 + kappa_005)

        log_val = first_order_term + second_order_term

        return log_val

    @ti.func
    def calculate_acceptance_direct(self, prob_current, prob_proposed):
        # Avoid division by zero
        if prob_proposed == 0.0 and prob_current == 0.0:
            acceptance_ratio = 0.0
        elif prob_proposed > 0.0 and prob_current == 0.0:
            acceptance_ratio = 1.0

        # Calculate the acceptance ratio
        acceptance_ratio = prob_proposed / prob_current
        return acceptance_ratio

    @ti.func
    def calculate_acceptance_log(self, log_prob_current, log_prob_proposed):
        # Calculate the log of the acceptance ratio
        delta = log_prob_proposed - log_prob_current

        acceptance_ratio = 0.0

        # nanが出る場合があるので、その場合は0を返す
        if delta != delta:  # Replaced ti.is_nan(delta) with delta != delta
            acceptance_ratio = 0.0
        else:
            # Convert log acceptance ratio to actual acceptance probability
            acceptance_ratio = ti.exp(delta)

        return acceptance_ratio

    @ti.func
    def sample_all_particles_from_proposal_distribution(self, independent_chain_idx):
        for i in range(self.num_of_particles):
            val_x = (ti.random(dtype=ti.f32) - 0.5) * self.proposal_std
            val_y = (ti.random(dtype=ti.f32) - 0.5) * self.proposal_std

            self.proposed_particles[independent_chain_idx, i][0] = (self.current_particles[independent_chain_idx, i][
                                                                        0] + val_x) % 1.0
            self.proposed_particles[independent_chain_idx, i][1] = (self.current_particles[independent_chain_idx, i][
                                                                        1] + val_y) % 1.0

    @ti.func
    def sample_single_particle_from_proposal_distribution(self, independent_chain_idx, sample_particle_idx):
        val_x = (ti.random(dtype=ti.f32) - 0.5) * self.proposal_std
        val_y = (ti.random(dtype=ti.f32) - 0.5) * self.proposal_std

        self.proposed_particles[independent_chain_idx, sample_particle_idx][0] = (
                (self.current_particles[independent_chain_idx, sample_particle_idx][0] + val_x) % 1.0)
        self.proposed_particles[independent_chain_idx, sample_particle_idx][1] = (
                (self.current_particles[independent_chain_idx, sample_particle_idx][1] + val_y) % 1.0)

    @ti.func
    def calculate_probability(self, chain_idx, is_proposed=False):
        prob = 0.0
        if self.target_distribution_name == 'target_distribution':
            prob = self.target_distribution(chain_idx, is_proposed)
        elif self.target_distribution_name == 'target_distribution2':
            if self.acceptance_ratio_calculation_with_log:
                # print('target_distribution2_log')
                prob = self.target_distribution2_log(chain_idx, is_proposed)
            else:
                # print('target_distribution2')
                prob = self.target_distribution2(chain_idx, is_proposed)
        elif self.target_distribution_name == 'target_distribution3':
            if self.acceptance_ratio_calculation_with_log:
                prob = self.target_distribution3_log(chain_idx, is_proposed)
            else:
                prob = self.target_distribution3(chain_idx, is_proposed)
        elif self.target_distribution_name == 'target_distribution4':
            prob = self.target_distribution4(chain_idx, is_proposed)
        elif self.target_distribution_name == 'target_distribution5':
            if self.acceptance_ratio_calculation_with_log:
                prob = self.target_distribution5_log(chain_idx, is_proposed)
            else:
                prob = self.target_distribution5(chain_idx, is_proposed)
        elif self.target_distribution_name == 'target_distribution01':
            if self.acceptance_ratio_calculation_with_log:
                prob = self.target_distribution01_log(chain_idx, is_proposed)
            else:
                prob = self.target_distribution01(chain_idx, is_proposed)
        elif self.target_distribution_name == 'target_distribution005':
            if self.acceptance_ratio_calculation_with_log:
                prob = self.target_distribution005_log(chain_idx, is_proposed)
            else:
                prob = self.target_distribution005(chain_idx, is_proposed)
        else:
            print('Invalid target distribution name')

        # check prob is nan
        if self.isnan(prob):
            # print(f'prob({prob}) is nan at chain_idx: {chain_idx}, is_proposed: {is_proposed}')
            prob = 0.0

        return prob

    @ti.func
    def calculate_acceptance_ratio(self, chain_idx, proposed_prob):
        current_prob = self.current_prob[chain_idx]

        acceptance_ratio = 0.0
        if self.target_distribution_name == 'target_distribution3' and self.acceptance_ratio_calculation_with_log:
            acceptance_ratio = self.calculate_acceptance_log(current_prob, proposed_prob)
        elif self.target_distribution_name == 'target_distribution2' and self.acceptance_ratio_calculation_with_log:
            acceptance_ratio = self.calculate_acceptance_log(current_prob, proposed_prob)
        elif self.target_distribution_name == 'target_distribution5' and self.acceptance_ratio_calculation_with_log:
            acceptance_ratio = self.calculate_acceptance_log(current_prob, proposed_prob)
        elif self.target_distribution_name == 'target_distribution01' and self.acceptance_ratio_calculation_with_log:
            acceptance_ratio = self.calculate_acceptance_log(current_prob, proposed_prob)
        elif self.target_distribution_name == 'target_distribution005' and self.acceptance_ratio_calculation_with_log:
            acceptance_ratio = self.calculate_acceptance_log(current_prob, proposed_prob)
        else:
            acceptance_ratio = self.calculate_acceptance_direct(current_prob, proposed_prob)

        return acceptance_ratio

    @ti.kernel
    def calculate_initial_probability(self):
        for chain_idx in range(self.num_of_independent_trials):
            self.current_prob[chain_idx] = self.calculate_probability(chain_idx)

    @ti.kernel
    def initialize_count_of_acceptance(self):
        for i in range(self.num_of_independent_trials):
            self.count_of_acceptance[i] = 0

    def compute_mcmc(self):
        if self.use_metropolis_within_gibbs:
            self.compute_mwg()
        else:
            self.compute_mh()

    @ti.kernel
    def compute_mh(self):
        for chain_idx in range(self.num_of_independent_trials):
            self.sample_all_particles_from_proposal_distribution(chain_idx)
            proposed_prob = self.calculate_probability(chain_idx, True)
            acceptance_ratio = self.calculate_acceptance_ratio(chain_idx, proposed_prob)

            if acceptance_ratio >= 1.0 or ti.random(dtype=ti.f32) < acceptance_ratio:
                for particle_idx in range(self.num_of_particles):
                    self.current_particles[chain_idx, particle_idx] = self.proposed_particles[chain_idx, particle_idx]
                self.current_prob[chain_idx] = proposed_prob
                self.count_of_acceptance[chain_idx] += 1
                # print(f'Accept at chain_idx: {chain_idx}, count_of_acceptance: {self.count_of_acceptance[chain_idx]}')

    # Metropolis within Gibbs
    @ti.kernel
    def compute_mwg(self):
        for chain_idx in range(self.num_of_independent_trials):
            # Gibbs sampling
            for particle_idx in range(self.num_of_particles):
                self.sample_single_particle_from_proposal_distribution(chain_idx, particle_idx)
                proposed_prob = self.calculate_probability(chain_idx, True)
                acceptance_ratio = self.calculate_acceptance_ratio(chain_idx, proposed_prob)

                if acceptance_ratio >= 1.0 or ti.random(dtype=ti.f32) < acceptance_ratio:
                    self.current_particles[chain_idx, particle_idx] = self.proposed_particles[chain_idx, particle_idx]
                    self.current_prob[chain_idx] = proposed_prob
                    self.count_of_acceptance[chain_idx] += 1
                else:
                    self.proposed_particles[chain_idx, particle_idx] = self.current_particles[chain_idx, particle_idx]

    @ti.func
    def isnan(self, x):
        return not (x < 0 or 0 < x or x == 0)


def toroidal_distance(length, p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])

    if dx > length / 2:
        dx = length - dx
    if dy > length / 2:
        dy = length - dy

    return math.sqrt(dx ** 2 + dy ** 2)


def initialize_particles(mh, verbose=True):
    print('Initializing particles')
    mh.initialize_all_chain_particles()
    initial_particles = mh.init_particles.to_numpy()
    np.save('temp_folder/initial_particles.npy', initial_particles)

    # Calculate the initial probability
    mh.calculate_initial_probability()
    if verbose: print(f'Calculating initial probability: {mh.current_prob.to_numpy()}')

    mh.initialize_count_of_acceptance()


def perform_calculations(args):
    MH = MetropolisHastings(
        args.num_of_particles,
        args.a,
        args.b,
        args.c,
        args.s,
        args.proposal_std,
        args.num_of_independent_trials,
        args.target_distribution_name,
        args.acceptance_ratio_calculation_with_log,
        args.record_from_first_acceptance,
        args.use_metropolis_within_gibbs
    )
    print(f'acceptance_ratio_calculation_with_log: {args.acceptance_ratio_calculation_with_log}')
    print(f'record_from_first_acceptance: {args.record_from_first_acceptance}')

    # burn-inの試行数を設定（num_of_independent_trialsの1.1倍）
    burn_in_trials = int(args.num_of_independent_trials * args.burn_in_multiplier)

    print(f'burn_in_trials: {burn_in_trials}')

    # burn-in用のMetropolisHastingsインスタンスを作成
    MH_burn_in = MetropolisHastings(
        args.num_of_particles,
        args.a,
        args.b,
        args.c,
        args.s,
        args.proposal_std,
        burn_in_trials,
        args.target_distribution_name,
        args.acceptance_ratio_calculation_with_log,
        args.record_from_first_acceptance,
        args.use_metropolis_within_gibbs
    )
    initialize_particles(MH_burn_in)
    MH_burn_in.initialize_count_of_acceptance()

    # burn-inを実行
    count = 0
    num_accepted = 0
    accepted_particles = []
    accepted_probs = []

    while num_accepted < args.num_of_independent_trials:
        MH_burn_in.compute_mcmc()
        count += 1

        # 10000回ごとにチェック
        if count % args.num_of_sampling_strides == 0:
            new_accepted_indices = np.where(MH_burn_in.count_of_acceptance.to_numpy() > 0)[0]

            if new_accepted_indices.size > 0:
                # 新しく受理されたサンプルを保存
                accepted_particles.extend(MH_burn_in.current_particles.to_numpy()[new_accepted_indices])
                accepted_probs.extend(MH_burn_in.current_prob.to_numpy()[new_accepted_indices])

                # 受理されたサンプルの数を更新
                num_accepted = len(accepted_particles)
                print(f'num_accepted: {num_accepted} / {args.num_of_independent_trials} at count: {count}')

                # 一度受理されたサンプルがargs.num_of_independent_trialsに達した場合
                if num_accepted >= args.num_of_independent_trials:
                    break

            # 受理されないチェーンの粒子を再初期化
            initialize_particles(MH_burn_in, False)

    print(f'Burn-in finished at count: {count}')

    # burn-inが終了したら、受理された粒子の位置をMHにコピー
    accepted_particles_array = np.array(accepted_particles[:args.num_of_independent_trials])
    accepted_probs_array = np.array(accepted_probs[:args.num_of_independent_trials])

    print(f'accepted_particles_array.shape: {accepted_particles_array.shape}')
    print(f'accepted_probs_array: {accepted_probs_array}')

    MH.current_particles.from_numpy(accepted_particles_array)
    MH.current_prob.from_numpy(accepted_probs_array)

    # Perform calculations
    taichi_start = time.time()
    result_particles = []

    for chain_idx in tqdm(range(args.num_of_iterations_for_each_trial)):
        MH.compute_mcmc()
        if chain_idx % args.num_of_sampling_strides == 0 and chain_idx != 0:
            result_particles.append(MH.current_particles.to_numpy())

    if args.use_metropolis_within_gibbs:
        average_acceptance_ratio = (
                (MH.count_of_acceptance.to_numpy().mean() / (
                        args.num_of_particles * args.num_of_iterations_for_each_trial)) * 100)
    else:
        average_acceptance_ratio = (
                                           MH.count_of_acceptance.to_numpy().mean() / args.num_of_iterations_for_each_trial) * 100

    print(f'Average acceptance ratio: {average_acceptance_ratio}%')

    calc_time = time.time() - taichi_start

    current_particles = MH.current_particles.to_numpy()

    np.save('temp_folder/result_particles.npy', result_particles)
    np.save('temp_folder/current_particles.npy', current_particles)

    with open('temp_folder/average_acceptance_ratio.txt', 'w') as f:
        f.write(str(average_acceptance_ratio))

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
    parser.add_argument('--burn_in_multiplier', type=float, default=1.5)
    parser.add_argument('--proposal_std', type=float, default=1.0)
    parser.add_argument('--num_of_independent_trials', type=int, default=10000)
    parser.add_argument('--target_distribution_name', type=str, default='target_distribution')
    parser.add_argument('--num_of_iterations_for_each_trial', type=int, default=10000)
    parser.add_argument('--num_of_sampling_strides', type=int, default=1000)
    parser.add_argument('--acceptance_ratio_calculation_with_log', action='store_true', default=False)
    parser.add_argument('--record_from_first_acceptance', action='store_true', default=False)
    parser.add_argument('--use_metropolis_within_gibbs', action='store_true', default=False)
    arguments = parser.parse_args()

    print(f'Arguments: {arguments}')

    path = f'temp_folder'
    if not os.path.exists(path):
        os.makedirs(path)

    ti.init(arch=ti.cuda, random_seed=int(time.time()))

    perform_calculations(arguments)
    calculate_distances()
