import math

import numpy as np
import matplotlib.pyplot as plt
import imageio

K = 10000000  # サンプリングの最大イテレーション数

def count_digits(n):
    if n > 0:
        return int(math.floor(math.log10(n))) + 1
    elif n == 0:
        return 1  # 0の桁数は1とする
    else:
        return int(math.floor(math.log10(-n))) + 1  # 負の数の場合

def generate_intervals():
    intervals = []
    for i in range(K + 1):
        if i % 10 ** (count_digits(i) - 1) == 0 and i > 99:
            intervals.append(i)
    return intervals

save_intervals = generate_intervals()

print(f'Saving frames at iterations: {save_intervals}')

def target(x):
    return np.sin(2 * np.pi * x) + 1

data = []
x_list = np.linspace(0, 1, 1000)
y = target(x_list)
x = np.random.rand()
gif_images = []

for i in range(K + 1):
    x_new = (x + np.random.uniform(-0.5, 0.5)) % 1.0
    alpha = target(x_new) / target(x)
    if np.random.rand() < alpha:
        x = x_new

    if (i % 100 == 0) and (i > 0):
        data.append(x)

    if i in save_intervals:
        print(f'Saving frame at iteration {i}')
        fig, ax = plt.subplots()
        ax.hist(data, bins=100, range=(0, 1), density=True, alpha=0.75, label='Sampled Data')
        ax.plot(x_list, y, 'r-', label='Target Distribution')
        ax.set_xlabel('X value')
        ax.set_ylabel('Density')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2.5)
        ax.set_title(f'Sampling Distribution at Iteration {i}')
        ax.legend()
        # フレームをメモリに保存
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_images.append(image)
        plt.close()

# GIFを生成
imageio.mimsave('metropolis_sampling.gif', gif_images, duration=0.5)
