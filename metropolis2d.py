import math

import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

K = 10000000  # サンプリングのイテレーション数


def count_digits(n):
    if n > 0:
        return int(math.floor(math.log10(n))) + 1
    elif n == 0:
        return 1  # 0の桁数は1とする
    else:
        return int(math.floor(math.log10(-n))) + 1  # 負の数の場合


def generate_intervals(num_of_iterations):
    intervals = []
    for i in range(num_of_iterations + 1):
        if i % 10 ** (count_digits(i) - 1) == 0 and i > 99:
            intervals.append(i)

    return intervals


save_intervals = generate_intervals(K)


def target(x, y):
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + 1


# 初期化
x, y = np.random.rand(), np.random.rand()
data_x, data_y = [], []
gif_images = []

for i in range(K + 1):
    # 新しい提案位置
    x_new = (x + np.random.uniform(-0.5, 0.5)) % 1.0
    y_new = (y + np.random.uniform(-0.5, 0.5)) % 1.0

    # 受理確率の計算
    alpha = target(x_new, y_new) / target(x, y)
    if np.random.rand() < alpha:
        x, y = x_new, y_new

    # 定期的なデータ記録
    if i % 100 == 0 and i > 0:
        data_x.append(x)
        data_y.append(y)

    if i in save_intervals:
        print(f'Saving frame at iteration {i}')
        fig, ax = plt.subplots()
        ax.hist2d(data_x, data_y, bins=100, density=True, cmap='viridis')
        # カラーバーを追加
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Density')
        # square aspect ratio
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_title(f'Sampling Distribution at Iteration {i}')
        # フレームをメモリに保存
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_images.append(image)
        plt.close(fig)

# GIFを生成
imageio.mimsave('metropolis2d.gif', gif_images, duration=0.5)
