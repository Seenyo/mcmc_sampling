import itertools
import time
import streamlit as st
import numpy as np

# class Params:
#     def __init__(self, a, b, c, s, proposal_std):
#         self.a = a
#         self.b = b
#         self.c = c
#         self.s = s
#         self.proposal_std = proposal_std
#
# # CPUで実行した場合のコード
# def metropolis_hastings(initial_particles, mh_iterations, iterations, params):
#     mh_results = []
#     print("CPU calculation start")
#     for mh_iter in range(mh_iterations):
#         current_particles = initial_particles
#         samples = [current_particles]
#         for _ in range(iterations):
#             proposed_particles = []
#             for i in range(len(current_particles)):
#                 # for debug
#                 val = current_particles[i] + np.array([0.1, 0.1])
#                 # val = np.random.normal(current_particles[i], params.proposal_std) % 1.0
#                 proposed_particles.append(val)
#
#             # print('proposed_particles:', proposed_particles)
#
#             sin_ab = np.sin(params.a * params.b)
#             cos_ab = np.cos(params.a * params.b)
#             numerator = 2 * ((1 - 3 * params.a ** 2) * sin_ab + params.a * (params.a ** 2 - 3) * cos_ab)
#             denominator = (params.a ** 2 + 1) ** 3
#             c_ab_val = -1 * numerator / denominator
#
#             # print('c_ab_val:', c_ab_val)
#
#             kappa2_37_sum = 0
#             order = len(current_particles)
#             num_array = np.arange(1, order + 1)
#             comb = itertools.combinations(num_array, 2)
#             comb_array = np.array(list(comb))
#
#             for i, j in comb_array:
#                 x1 = proposed_particles[i - 1]
#                 x2 = proposed_particles[j - 1]
#                 r = np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
#                 kappa2_37_sum += params.c * np.exp(-1 * r / params.s) * (np.sin(params.a * (r / params.s - params.b)) - c_ab_val * 0.5)
#
#             # print('kappa2_37_sum:', kappa2_37_sum)
#
#             acceptance_ratio = 1 / (1 ** order) + 1 / (1 ** (order - 2)) * kappa2_37_sum
#
#             # print('acceptance_ratio:', acceptance_ratio)
#
#             if acceptance_ratio >= 1 or np.random.uniform(0, 1) < acceptance_ratio:
#                 current_particles = proposed_particles
#
#             # for debug
#             # if acceptance_ratio >= 1:
#             #     current_particles = proposed_particles
#
#             samples.append(current_particles)
#
#         mh_results.append(samples[-1])
#
#     return mh_results


class MetropolisHastingsCPU:

    def __init__(self, num_of_particles, a, b, c, s, proposal_std, num_of_independent_trials, num_of_iterations_for_each_trial):
        self.a = a
        self.b = b
        self.c = c
        self.s = s
        self.proposal_std = proposal_std
        self.num_of_independent_trials = num_of_independent_trials
        self.num_of_iterations_for_each_trial = num_of_iterations_for_each_trial
        self.num_of_particles = num_of_particles

        # for debug
        # self.init_particles = np.array([[0.5, 0.5], [0.3, 0.3]])

        self.init_particles = np.random.rand(num_of_particles, 2)
        self.mh_particles = np.zeros((num_of_independent_trials, num_of_particles, 2))

    def compute_mh(self):
        for i in range(self.num_of_independent_trials):
            self.mh_particles[i] = self.init_particles
            for j in range(self.num_of_iterations_for_each_trial):
                proposed_particles = []
                for k in range(self.num_of_particles):
                    # for debug
                    # val = self.mh_particles[i, k] + np.array([0.1, 0.1])
                    val = np.random.normal(self.mh_particles[i, k], self.proposal_std) % 1.0
                    proposed_particles.append(val)

                # print('proposed_particles:', proposed_particles)

                sin_ab = np.sin(self.a * self.b)
                cos_ab = np.cos(self.a * self.b)
                numerator = 2 * ((1 - 3 * self.a ** 2) * sin_ab + self.a * (self.a ** 2 - 3) * cos_ab)
                denominator = (self.a ** 2 + 1) ** 3
                c_ab_val = -1 * numerator / denominator

                # print('c_ab_val:', c_ab_val)

                kappa2_37_sum = 0
                order = self.num_of_particles
                num_array = np.arange(1, order + 1)
                comb = itertools.combinations(num_array, 2)
                comb_array = np.array(list(comb))

                for k, l in comb_array:
                    x1 = proposed_particles[k - 1]
                    x2 = proposed_particles[l - 1]
                    r = np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
                    kappa2_37_sum += self.c * np.exp(-1 * r / self.s) * (np.sin(self.a * (r / self.s - self.b)) - c_ab_val * 0.5)

                # print('kappa2_37_sum:', kappa2_37_sum)

                acceptance_ratio = 1 / (1 ** order) + 1 / (1 ** (order - 2)) * kappa2_37_sum

                # print('acceptance_ratio:', acceptance_ratio)

                if acceptance_ratio >= 1 or np.random.uniform(0, 1) < acceptance_ratio:
                    self.mh_particles[i] = proposed_particles

                # for debug
                # if acceptance_ratio >= 1:
                #     self.mh_particles[i] = proposed_particles


def main():
    # parameters
    num_of_particles = st.slider("Number of Particles", 1, 100, 10)
    a = st.slider("a", 0.0, 1.0, np.pi)
    b = st.slider("b", 0.0, 1.0, 0.25)
    c = st.slider("c", 0.0, 1.0, 0.1)
    s = st.slider("s", 0.0, 1.0, 0.1)
    proposal_std = st.slider("Proposal Standard Deviation", 0.0, 1.0, 0.1)
    num_of_independent_trials = st.slider("Number of Independent Trials", 1, 10000, 100)
    num_of_iterations_for_each_trial = st.slider("Number of Iterations for Each Trial", 1, 10000, 10)

    # CPU calculation start
    cpu_calc_start = time.time()

    MH_cpu = MetropolisHastingsCPU(num_of_particles, a, b, c, s, proposal_std, num_of_independent_trials, num_of_iterations_for_each_trial)
    MH_cpu.compute_mh()

    cpu_calc_end = time.time()

    print("CPU Calculation Latency [Total]:\t", format(cpu_calc_end - cpu_calc_start, '.3f'), "sec")
    print("CPU Calculation Latency [Single]:\t", format((cpu_calc_end - cpu_calc_start) / num_of_independent_trials, '.3f'), "sec")

if __name__ == "__main__":
    main()


