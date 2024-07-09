import argparse
import datetime
import math
import time
import os

import taichi as ti
import numpy as np
import multiprocessing as mp

from functools import partial
from itertools import combinations
from termcolor import colored
from tqdm import tqdm

import target_distributions

@ti.data_oriented
class MetropolisHastings:
    def __init__(self, num_of_particles, a, b, c, s, proposal_std, num_of_chains, num_of_mutations,
                 target_distribution_name='target_distribution', acceptance_ratio_calculation_with_log=False,
                 record_from_first_acceptance=False, use_metropolis_within_gibbs=False):
        self.a = a
        self.b = b
        self.c = c
        self.s = s
        self.proposal_std = proposal_std
        self.num_of_chains = num_of_chains
        self.num_of_particles = num_of_particles
        self.batch_size = num_of_mutations

        self.init_particles = ti.Vector.field(2, dtype=ti.f32, shape=(num_of_chains, num_of_particles))
        self.proposed_particles = ti.Vector.field(2, dtype=ti.f32, shape=(num_of_chains, num_of_particles))
        self.current_particles = ti.Vector.field(2, dtype=ti.f32, shape=(num_of_chains, num_of_particles))

        self.batch_buffer = ti.Vector.field(2, dtype=ti.f32, shape=(self.batch_size, self.num_of_particles))
        self.current_iteration = ti.field(dtype=ti.i32, shape=())
        self.current_iterations = ti.field(dtype=ti.i32, shape=(self.num_of_chains,))
        self.all_particles = ti.Vector.field(2, dtype=ti.f32, shape=(self.batch_size, self.num_of_chains, num_of_particles))

        self.count_of_acceptance = ti.field(dtype=ti.i32, shape=num_of_chains)
        self.current_prob = ti.field(dtype=ti.f32, shape=num_of_chains)

        self.target_distribution_name = target_distribution_name
        self.acceptance_ratio_calculation_with_log = acceptance_ratio_calculation_with_log
        self.record_from_first_acceptance = record_from_first_acceptance
        self.use_metropolis_within_gibbs = use_metropolis_within_gibbs

    @ti.kernel
    def initialize_all_chain_particles(self):
        # Initialize the particles with the same initial values
        for chain_idx in range(self.num_of_chains):
            for particle_idx in range(self.num_of_particles):
                self.init_particles[chain_idx, particle_idx] = ti.Vector(
                    [ti.random(dtype=ti.f32), ti.random(dtype=ti.f32)])
                self.current_particles[chain_idx, particle_idx] = self.init_particles[chain_idx, particle_idx]
                self.proposed_particles[chain_idx, particle_idx] = self.init_particles[chain_idx, particle_idx]

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
    def sample_all_particles_from_proposal_distribution(self, chain_idx):
        for i in range(self.num_of_particles):
            val_x = (ti.random(dtype=ti.f32) - 0.5) * self.proposal_std
            val_y = (ti.random(dtype=ti.f32) - 0.5) * self.proposal_std

            self.proposed_particles[chain_idx, i][0] = (self.current_particles[chain_idx, i][
                                                                        0] + val_x) % 1.0
            self.proposed_particles[chain_idx, i][1] = (self.current_particles[chain_idx, i][
                                                                        1] + val_y) % 1.0

    @ti.func
    def sample_single_particle_from_proposal_distribution(self, chain_idx, particle_idx):
        val_x = (ti.random(dtype=ti.f32) - 0.5) * self.proposal_std
        val_y = (ti.random(dtype=ti.f32) - 0.5) * self.proposal_std

        x = (self.current_particles[chain_idx, particle_idx][0] + val_x) % 1.0
        y = (self.current_particles[chain_idx, particle_idx][1] + val_y) % 1.0

        self.proposed_particles[chain_idx, particle_idx] = ti.Vector([x, y])

    @ti.func
    def calculate_probability(self, chain_idx, is_proposed=False):
        prob = 0.0
        if self.target_distribution_name == 'target_distribution':
            prob = target_distributions.target_distribution(self, chain_idx, is_proposed)
        elif self.target_distribution_name == 'target_distribution2':
            if self.acceptance_ratio_calculation_with_log:
                prob = target_distributions.target_distribution2_log(self, chain_idx, is_proposed)
            else:
                prob = target_distributions.target_distribution2(self, chain_idx, is_proposed)
        elif self.target_distribution_name == 'target_distribution3':
            if self.acceptance_ratio_calculation_with_log:
                prob = target_distributions.target_distribution3_log(self, chain_idx, is_proposed)
            else:
                prob = target_distributions.target_distribution3(self, chain_idx, is_proposed)
        elif self.target_distribution_name == 'target_distribution4':
            prob = target_distributions.target_distribution4(self, chain_idx, is_proposed)
        elif self.target_distribution_name == 'target_distribution5':
            if self.acceptance_ratio_calculation_with_log:
                prob = target_distributions.target_distribution5_log(self, chain_idx, is_proposed)
            else:
                prob = target_distributions.target_distribution5(self, chain_idx, is_proposed)
        elif self.target_distribution_name == 'target_distribution01':
            if self.acceptance_ratio_calculation_with_log:
                prob = target_distributions.target_distribution01_log(self, chain_idx, is_proposed)
            else:
                prob = target_distributions.target_distribution01(self, chain_idx, is_proposed)
        elif self.target_distribution_name == 'target_distribution005':
            if self.acceptance_ratio_calculation_with_log:
                prob = target_distributions.target_distribution005_log(self, chain_idx, is_proposed)
            else:
                prob = target_distributions.target_distribution005(self, chain_idx, is_proposed)
        else:
            print('Invalid target distribution name')

        # check prob is nan
        if self.isnan(prob):
            prob = 0.0

        return prob

    @ti.func
    def calculate_acceptance_ratio(self, chain_idx, proposed_prob):
        current_prob = self.current_prob[chain_idx]

        acceptance_ratio = 0.0
        if self.target_distribution_name == 'target_distribution3' and self.acceptance_ratio_calculation_with_log:
            acceptance_ratio = target_distributions.calculate_acceptance_log(self, current_prob, proposed_prob)
        elif self.target_distribution_name == 'target_distribution2' and self.acceptance_ratio_calculation_with_log:
            acceptance_ratio = target_distributions.calculate_acceptance_log(self, current_prob, proposed_prob)
        elif self.target_distribution_name == 'target_distribution5' and self.acceptance_ratio_calculation_with_log:
            acceptance_ratio = target_distributions.calculate_acceptance_log(self, current_prob, proposed_prob)
        elif self.target_distribution_name == 'target_distribution01' and self.acceptance_ratio_calculation_with_log:
            acceptance_ratio = target_distributions.calculate_acceptance_log(self, current_prob, proposed_prob)
        elif self.target_distribution_name == 'target_distribution005' and self.acceptance_ratio_calculation_with_log:
            acceptance_ratio = target_distributions.calculate_acceptance_log(self, current_prob, proposed_prob)
        else:
            acceptance_ratio = target_distributions.calculate_acceptance_direct(self, current_prob, proposed_prob)

        return acceptance_ratio

    @ti.kernel
    def calculate_initial_probability(self):
        for chain_idx in range(self.num_of_chains):
            self.current_prob[chain_idx] = self.calculate_probability(chain_idx, False )

    @ti.kernel
    def initialize_count_of_acceptance(self):
        for i in range(self.num_of_chains):
            self.count_of_acceptance[i] = 0

    def compute_mcmc(self):
        if self.use_metropolis_within_gibbs:
            self.compute_mwg()
        else:
            self.compute_mh()

    @ti.kernel
    def compute_mh(self):
        for chain_idx in range(self.num_of_chains):
            self.sample_all_particles_from_proposal_distribution(chain_idx)
            proposed_prob = self.calculate_probability(chain_idx, True)
            acceptance_ratio = self.calculate_acceptance_ratio(chain_idx, proposed_prob)

            if acceptance_ratio >= 1.0 or ti.random(dtype=ti.f32) < acceptance_ratio:
                for particle_idx in range(self.num_of_particles):
                    self.current_particles[chain_idx, particle_idx] = self.proposed_particles[chain_idx, particle_idx]
                self.current_prob[chain_idx] = proposed_prob
                self.count_of_acceptance[chain_idx] += 1

    # Metropolis within Gibbs
    @ti.kernel
    def compute_mwg(self):
        for chain_idx in range(self.num_of_chains):
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

    @ti.kernel
    def save_current_state(self):
        for chain_idx in range(self.num_of_chains):
            idx = self.current_iterations[chain_idx] % self.batch_size
            for particle_idx in range(self.num_of_particles):
                self.all_particles[idx, chain_idx, particle_idx] = self.current_particles[chain_idx, particle_idx]
            self.current_iterations[chain_idx] += 1

    def get_all_batch_numpy(self):
        return self.all_particles.to_numpy()

    def get_iteration_range(self, batch_idx, num_of_iterations_for_each_chain):
        start = batch_idx * self.batch_size
        end = min((batch_idx + 1) * self.batch_size, num_of_iterations_for_each_chain) - 1
        return start, end

    def print_debug_info(self):
        print(f"Current iterations: {self.current_iterations.to_numpy()}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of chains: {self.num_of_chains}")
        print(f"Number of particles: {self.num_of_particles}")
        print(f"All particles shape: {self.all_particles.shape}")

def toroidal_distance_cpu(length, p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])

    if dx > length / 2:
        dx = length - dx
    if dy > length / 2:
        dy = length - dy

    return math.sqrt(dx ** 2 + dy ** 2)


def initialize_particles(mh, verbose=True):
    if verbose: print('Initializing particles')
    mh.initialize_all_chain_particles()
    initial_particles = mh.init_particles.to_numpy()
    np.save('temp_folder/initial_particles.npy', initial_particles)

    # Calculate the initial probability
    mh.calculate_initial_probability()
    if verbose: print(f'Calculating initial probability (10 chains from top): {mh.current_prob.to_numpy()[:10]}')

    mh.initialize_count_of_acceptance()

def calc_distances(batch_data, num_of_particles):
    distances = []
    for iteration_data in batch_data:
        for i, j in combinations(range(num_of_particles), 2):
            particle1 = iteration_data[i]
            particle2 = iteration_data[j]
            dist = toroidal_distance_cpu(1.0, particle1, particle2)
            distances.append(dist)
    distances = np.sort(distances)
    return np.array(distances)
def create_histogram(distances, r=(0, 0.8)):
    n = len(distances)
    num_bins = int(np.ceil(2 * n ** (1 / 3)))  # Rice's rule for bin number
    hist, bin_edges = np.histogram(distances, bins=num_bins, range=r, density=True)
    return hist, bin_edges

def save_histogram_data(hist, bin_edges, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(filename, hist=hist, bin_edges=bin_edges)

def process_chain(chain_idx, batch_data, args, start_iter, end_iter, time_stamp):
    batch_distances = calc_distances(batch_data, args.num_of_particles)
    hist, bin_edges = create_histogram(batch_distances)
    if args.use_metropolis_within_gibbs:
        chain_type = 'MWG'
    else:
        chain_type = 'MH'
    save_histogram_data(hist, bin_edges, f'mutation_experiment_results/{args.target_distribution_name}/{time_stamp}/{args.num_of_particles}/{chain_type}/{chain_idx:04d}/histogram_thread_{chain_idx:04d}_mutation_{start_iter:06d}_{end_iter:06d}.npz')

def parallel_chain_processing(MH, args, start_iter, end_iter, time_stamp):
    all_batch_data = MH.get_all_batch_numpy()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        process_func = partial(process_chain, args=args, start_iter=start_iter, end_iter=end_iter, time_stamp=time_stamp)
        chain_data = [(chain_idx, all_batch_data[:, chain_idx, :]) for chain_idx in range(MH.num_of_chains)]
        list(tqdm(pool.starmap(process_func, chain_data),
                  total=MH.num_of_chains,
                  desc=colored('Processing Chains', 'magenta'),
                  bar_format="{l_bar}{bar}{r_bar}",
                  colour='magenta',
                  leave=False))
def perform_calculations(args):
    print(f'acceptance_ratio_calculation_with_log: {args.acceptance_ratio_calculation_with_log}')
    print(f'record_from_first_acceptance: {args.record_from_first_acceptance}')

    # burn-inの試行数を設定（num_of_chainsの1.1倍）
    burn_in_chains = int(args.num_of_chains * args.burn_in_multiplier)

    print(f'burn_in_chains: {burn_in_chains}')

    # burn-in用のMetropolisHastingsインスタンスを作成
    MH_burn_in = MetropolisHastings(
        args.num_of_particles,
        args.a,
        args.b,
        args.c,
        args.s,
        args.proposal_std,
        burn_in_chains,
        args.num_of_mutations,
        args.target_distribution_name,
        args.acceptance_ratio_calculation_with_log,
        args.record_from_first_acceptance,
        args.use_metropolis_within_gibbs
    )
    initialize_particles(MH_burn_in)

    # burn-inを実行
    count = 0
    num_accepted = 0
    accepted_particles = []
    accepted_probs = []

    print('Burn-in started')

    while num_accepted < args.num_of_chains:
        MH_burn_in.compute_mcmc()
        count += 1

        if count % 10000 == 0:
            new_accepted_indices = np.where(MH_burn_in.count_of_acceptance.to_numpy() > 0)[0]
            print(f'count, num_accepted: {count}, {len(new_accepted_indices)}')

            if new_accepted_indices.size > 0:
                # 新しく受理されたサンプルを保存
                accepted_particles.extend(MH_burn_in.current_particles.to_numpy()[new_accepted_indices])
                accepted_probs.extend(MH_burn_in.current_prob.to_numpy()[new_accepted_indices])

                # 受理されたサンプルの数を更新
                num_accepted = len(accepted_particles)
                print(f'num_accepted: {num_accepted} / {args.num_of_chains} at count: {count}')

                # 一度受理されたサンプルがargs.num_of_chainsに達した場合
                if num_accepted >= args.num_of_chains:
                    break

            # 受理されないチェーンの粒子を再初期化
            initialize_particles(MH_burn_in, False)

    print(f'Burn-in finished at count: {count}')

    # burn-inが終了したら、受理された粒子の位置をMHにコピー
    accepted_particles_array = np.array(accepted_particles[:args.num_of_chains])
    accepted_probs_array = np.array(accepted_probs[:args.num_of_chains])

    print(f'accepted_probs_array(From top 10): {accepted_probs_array[:10]}')

    MH = MetropolisHastings(
        args.num_of_particles,
        args.a,
        args.b,
        args.c,
        args.s,
        args.proposal_std,
        args.num_of_chains,
        args.num_of_mutations,
        args.target_distribution_name,
        args.acceptance_ratio_calculation_with_log,
        args.record_from_first_acceptance,
        args.use_metropolis_within_gibbs,
    )
    MH.current_particles.from_numpy(accepted_particles_array)
    MH.current_prob.from_numpy(accepted_probs_array)

    # Perform calculations
    start_time = time.time()

    num_batches = (args.num_of_iterations_for_each_chain + MH.batch_size - 1) // MH.batch_size
    time_stamp = args.time_stamp

    for batch_idx in tqdm(range(num_batches), desc=colored('Batch Progress', 'green'), bar_format="{l_bar}{bar}{r_bar}", colour='green', leave=False):
        for _ in tqdm(range(MH.batch_size), desc=colored('MCMC Progress', 'blue'), bar_format="{l_bar}{bar}{r_bar}", colour='blue', leave=False):
            MH.compute_mcmc()
            MH.save_current_state()

        start_iter, end_iter = MH.get_iteration_range(batch_idx, args.num_of_iterations_for_each_chain)
        parallel_chain_processing(MH, args, start_iter, end_iter, time_stamp)


    end_time = time.time()
    print("num of particles: ", args.num_of_particles)
    print("calculation time: ", end_time - start_time)

    print("Simulation completed and all data saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_particles', type=int, default=2)
    parser.add_argument('--a', type=float, default=np.pi)
    parser.add_argument('--b', type=float, default=0.25)
    parser.add_argument('--c', type=float, default=0.1)
    parser.add_argument('--s', type=float, default=0.1)
    parser.add_argument('--proposal_std', type=float, default=1.0)
    parser.add_argument('--num_of_chains', type=int, default=10000)
    parser.add_argument('--target_distribution_name', type=str, default='target_distribution')
    parser.add_argument('--num_of_iterations_for_each_chain', type=int, default=10000)
    parser.add_argument('--num_of_mutations', type=int, default=10000)
    parser.add_argument('--burn_in_multiplier', type=float, default=1.5)
    parser.add_argument('--time_stamp', type=str, default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--acceptance_ratio_calculation_with_log', action='store_true', default=False)
    parser.add_argument('--record_from_first_acceptance', action='store_true', default=False)
    parser.add_argument('--use_metropolis_within_gibbs', action='store_true', default=False)
    arguments = parser.parse_args()

    print(f'Arguments: {arguments}')
    ti.init(arch=ti.cuda, random_seed=int(time.time()))

    perform_calculations(arguments)