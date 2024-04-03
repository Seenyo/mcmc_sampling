import time
import colorsys
import numpy as np
import taichi as ti
from vispy import app, scene
from vispy.io import write_png
@ti.data_oriented
class MetropolisHastingsGPU:
    def __init__(self, num_of_particles, a, b, c, s, proposal_std, num_of_independent_trials):
        self.a = a
        self.b = b
        self.c = c
        self.s = s
        self.proposal_std = proposal_std
        self.num_of_independent_trials = num_of_independent_trials
        self.num_of_particles = num_of_particles

        self.init_particles = ti.Vector.field(2, dtype=ti.f32, shape=(num_of_particles, ))
        self.proposed_particles = ti.Vector.field(2, dtype=ti.f32, shape=(num_of_independent_trials, num_of_particles))
        self.mh_particles = ti.Vector.field(2, dtype=ti.f32, shape=(num_of_independent_trials, num_of_particles))

    @ti.kernel
    def initialize_particles(self):

        for particle in range(self.num_of_particles):
            self.init_particles[particle] = ti.Vector([ti.random(dtype=ti.f32), ti.random(dtype=ti.f32)])

        #Initialize the particles with the same initial values
        for trial in range(self.num_of_independent_trials):
            for particle in range(self.num_of_particles):
                self.mh_particles[trial, particle] = self.init_particles[particle]

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


def main():
    ti.init(arch=ti.cuda, random_seed=int(time.time()))

    ############################## Setting parameters ##############################
    num_of_particles = 4
    a = np.pi
    b = 0.25
    c = 0.1
    s = 0.1
    proposal_std = 0.1
    num_of_independent_trials = 1000
    num_of_iterations_for_each_trial = 1000

    ############################## GPU Calculation ##############################
    gpu_calc_start = time.time()

    # Initialize the Metropolis-Hastings algorithm
    MH_gpu = MetropolisHastingsGPU(num_of_particles, a, b, c, s, proposal_std, num_of_independent_trials)
    MH_gpu.initialize_particles()

    # This loop will be calculated sequentially
    for i in range(num_of_iterations_for_each_trial):
        MH_gpu.compute_mh()

    mh_particles_gpu = MH_gpu.mh_particles.to_numpy()
    gpu_calc_end = time.time()

    print('GPU Calculation Time:', format(gpu_calc_end - gpu_calc_start, '.2f'), 'seconds')

    ############################## Visualization ##############################
    data = np.concatenate(mh_particles_gpu, axis=0)
    colors = np.zeros((len(data), 4))

    for i in range(num_of_particles):
        hue = i / float(num_of_particles)  # 色相を粒子の数に基づいて均等に分割
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # HSVからRGBへ変換
        colors[i::num_of_particles] = rgb + (1.0,)  # 同じインデックスの粒子に色を割り当て

    # setting canvas with off-screen mode
    canvas = scene.SceneCanvas(keys='interactive', size=(1200, 1200), bgcolor='white', show=False)
    view = canvas.central_widget.add_view()

    scatter = scene.visuals.Markers(parent=view.scene)
    scatter.set_data(data, size=10, edge_color=None, face_color=colors, edge_width=0)
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(0, 1), y=(0, 1))

    # save results
    canvas.update()
    canvas.app.process_events()
    image = canvas.render()
    write_png('gpu_mh_last_sample.png', image)

if __name__ == "__main__":
    main()


