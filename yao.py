# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import math
def k2_example1(a, b, c, s, r):
    Cab = - 2.0 * ((1.0 - 3.0*a*a) * math.sin(a*b) + a*(a*a-3.0) * math.cos(a*b)) / ((a*a + 1.0) * (a*a + 1.0) * (a*a + 1.0))
    return c * math.exp(-r/s) * (math.sin(a*(r/s-b)) - Cab * 0.5)
    #return 0.0
def k2_example2(a, b, c, r):
    #return c * (math.exp(-r/a) / (a*a) - math.exp(-r/b) / (b*b))
    ret = c * (math.exp(-r/a) / (a*a) - math.exp(-r/b) / (b*b))
    #print("k2_ex2() a=", a, ", b=", b, ", c=", c, ", r=", r, ", ret=", ret)
    return ret
# D = [0,1]^2; k_1 = U = 1
#def joint_distribution(points, a, b ,c, s):
def joint_distribution(points, a, b ,c):
    num_points = points.shape[1]
    value = 1.0
    for j in range(num_points):
        for i in range(j + 1, num_points):
            r = np.linalg.norm(points[:, j] - points[:, i])
            value += k2_example2(a, b, c, r)
    return value
def min_dist(points, x):
    #print("----- min_dist -----")
    num_points = points.shape[1]
    min_dist = 4.0
    for i in range(num_points):
        r = np.linalg.norm(points[:, i] - x[:, 0])
        #print("p[i]: ", points[:, i].transpose(), ", x: ", x.transpose(), ", r: ", r, ", p[i]-x: ", points[:, i] - x)
        min_dist = min([min_dist, r])
    return min_dist
def sample_a_new_point(prev_points, a, b, c):
    #conditional_propability_denom = joint_distribution(prev_points, a, b, c, s)
    conditional_propability_denom = joint_distribution(prev_points, a, b, c)
    factor = 2.0
    x = np.zeros((2, 1))
    while True:
        x[0, 0] = random.random()
        x[1, 0] = random.random()
        #print("x: ", x)
        points = np.hstack((prev_points, x))
        #conditional_propability_numer = joint_distribution(points, a, b, c, s)
        conditional_propability_numer = joint_distribution(points, a, b, c)
        conditional_propability = conditional_propability_numer / conditional_propability_denom
        # if conditional_propability > factor:
        #     print("factor ", factor, " was too small, encountered cp = ", conditional_propability)
        # if conditional_propability_numer < 0.0 or conditional_propability_denom < 0.0:
        #     print("encounted negative cp: ", conditional_propability, "(", conditional_propability_numer, "/", conditional_propability_denom, ")")
        #print("cp = ", conditional_propability)
        _r = random.random() * factor
        if _r < conditional_propability:
            # _min_dist = min_dist(prev_points, x)
            # if _min_dist < 0.05:
            #     print("prev_points: ", prev_points)
            #     print("x: ", x.transpose())
            #     print("conditional_propability: ", conditional_propability)
            #     print("conditional_propability_numer: ", conditional_propability_numer)
            #     print("conditional_propability_denom: ", conditional_propability_denom)
            #     print("_r: ", _r)
            # else:
            #     print("min_dist: ", _min_dist)
            break
    return x
def main():
    size = 1.0
    st.sidebar.title('Settings')
    # theta: [deg]
    k2_a = st.sidebar.slider(label = 'k2_a (example 1)', min_value = 0, max_value = 100, value = 0)
    k2_b = st.sidebar.slider(label = 'k2_b (example 1)', min_value = 1, max_value = 100, value = 0)
    num_points = st.sidebar.slider(label = 'num_points', min_value = 0, max_value = 10, value = 1)
    plot_area = st.empty()
    fig = plt.figure()
    ax = fig.add_subplot(adjustable='box', aspect=1.0)
    ax.set_title('Generated points')
    ax.set_xlim(-0.025*size, size+0.025*size)
    ax.set_ylim(-0.025*size, size+0.025*size)
    H_color = [245/255.0, 127/255.0, 32/255.0]
    points = np.empty((2, 0))
    a = 0.2
    b = 0.199
    c = 10.0
    s = 0.1
    print("--------------------------------------")
    for j in range(10):
        x = sample_a_new_point(points, a, b, c)
        #print("j: ", j, ", x: ", x.transpose())
        points = np.hstack((points, x))
    num_points = points.shape[1]
    for k in range(num_points):
        p = points[:, k]
        ax.plot(p[0] * size, p[1] * size, color=H_color, marker='o')
    H_color = [64/255.0, 182/255.0, 245/255.0]
    # num_result_vertices = len(result_polygon)
    # for k in range(num_result_vertices):
    #     p_start = result_polygon[k]
    #     p_end = result_polygon[(k+1) % num_result_vertices]
    #     ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color=H_color)
    # num_polygons = len(result_Layout)
    # for k in range(num_polygons):
    #     num_result_vertices = len(result_Layout[k])
    #     x = np.zeros(num_result_vertices+1)
    #     y = np.zeros(num_result_vertices+1)
    #     for i in range(num_result_vertices+1):
    #         p = result_Layout[k][i % num_result_vertices]
    #         x[i] = p[0]
    #         y[i] = p[1]
    #     ax.fill(x, y, color=H_color)
    H_color = [32/255.0, 127/255.0, 245/255.0]
    # num_polygons = len(result_Layout)
    # for k in range(num_polygons):
    #     num_result_vertices = len(result_Layout[k])
    #     for i in range(num_result_vertices):
    #         p_start = result_Layout[k][i]
    #         p_end = result_Layout[k][(i+1) % num_result_vertices]
    #         ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color=H_color, lw=1)
    plot_area.pyplot(fig)
if __name__ == '__main__':
    main()