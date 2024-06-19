from abc import ABC, abstractmethod
import taichi as ti

# Distribution interface
class Distribution:
    @abstractmethod
    @ti.func
    def pdf(self, x):
        pass


class taregt_distribution(Distribution):
    def __init__(self, a, b, c, s):
        self.a = a
        self.b = b
        self.c = c
        self.s = s

    @ti.func
    def pdf(self, x):
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

        return acceptance_ratio