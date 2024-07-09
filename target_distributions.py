import taichi as ti

@ti.func
def toroidal_distance(length, p1, p2):
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
            r = toroidal_distance(1.0, x1, x2)
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
            r = toroidal_distance(1.0, x1, x2)
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
            r = toroidal_distance(1.0, x1, x2)
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
            r = toroidal_distance(1.0, x1, x2)
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
            r = toroidal_distance(1.0, x1, x2)
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
            r = toroidal_distance(1.0, x1, x2)
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
            r = toroidal_distance(1.0, x1, x2)
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
            r = toroidal_distance(1.0, x1, x2)
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
            r = toroidal_distance(1.0, x1, x2)
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
            r = toroidal_distance(1.0, x1, x2)
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
            r = toroidal_distance(1.0, x1, x2)
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
            r = toroidal_distance(1.0, x1, x2)
            kappa_005 = -1 if r <= 0.05 else -1 * ti.exp(-5 * (r - 0.05)) * ti.cos(22 * (r - 0.05))
            second_order_term += ti.log(1.0 + kappa_005)

    log_val = first_order_term + second_order_term

    return log_val


@ti.func
def calculate_acceptance_direct(self, prob_current, prob_proposed):
    acceptance_ratio = 0.0
    # Avoid division by zero
    if prob_proposed == 0.0 and prob_current == 0.0:
        acceptance_ratio = 0.0
    elif prob_proposed > 0.0 and prob_current == 0.0:
        acceptance_ratio = 1.0
    else:
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