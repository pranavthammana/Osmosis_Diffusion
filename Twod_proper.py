import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import norm

import random as rand

import sys

import csv


def calculate_delta(percent):
    chance = rand.randint(1, 100)
    if 1 <= chance <= percent:
        return -1
    elif percent <= chance <= 2 * percent:
        return 1
    else:
        return 0


data = {}
diffusion_coefficients = []
for percent in range(1, 34):
    data[percent] = np.zeros((100, 100), dtype=object)
    for point in range(len(data[percent])):
        # For loop that simulates a point updating 100 times
        for second in range(len(data[percent][point])):
            delta_x = calculate_delta(percent)
            delta_y = calculate_delta(percent)
            if second == 0:
                data[percent][point][second] = (0, 0)
            else:
                data[percent][point][second] = (
                data[percent][point][second - 1][0] + delta_x, data[percent][point][second - 1][1] + delta_y)

    t_99_X = []
    t_99_Y = []
    for point in data[percent]:
        t_99_X.append(point[99][0])
        t_99_Y.append(point[99][1])
    plt.hist(t_99_X)
    (mu, sigma) = norm.fit(t_99_X)
    plt.title(r'$\mathrm{Histogram\ of\ X\ Position\ } %i\ \mu=%.3f,\ \sigma=%.3f$' % (int(percent), mu, sigma))
    plt.xlabel("Position")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(t_99_Y)
    (mu, sigma) = norm.fit(t_99_Y)
    plt.title(r'$\mathrm{Histogram\ of\ X\ Position\ } %i\ \mu=%.3f,\ \sigma=%.3f$' % (int(percent), mu, sigma))
    plt.xlabel("Position")
    plt.ylabel("Frequency")
    plt.show()
    sys.exit()


