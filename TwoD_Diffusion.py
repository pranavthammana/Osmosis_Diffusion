import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import norm

import random as rand


def calculate_delta(percent):
    chance = rand.randint(1, 100)
    if 1 <= chance <= percent:
        return -1
    elif percent <= chance <= 2 * percent:
        return 1
    else:
        return 0


data = {}
for percent in range(1, 34):
    data[percent] = np.zeros((100, 100), dtype=object)
    for point in range(len(data[percent])):

        # For loop that simulates a point updating 100 times
        for second in range(len(data[percent][point])):
            delta_x = calculate_delta(percent)
            delta_y = calculate_delta(percent)
            if second == 0:
                data[percent][point][second] = (delta_x, delta_y)
            else:
                data[percent][point][second] = (
                    data[percent][point][second - 1][0] + delta_x, data[percent][point][second - 1][1] + delta_y)

    # Creating graphs for each percent

    T_100_fo = []  # fo stands for from origin
    for i in data[percent]:
        T_100_fo.append(i[99])
    for i in range(len(T_100_fo)):
        T_100_fo[i] = np.sqrt(np.square(T_100_fo[i][0]) + np.square(T_100_fo[i][1]))
    plt.hist(T_100_fo)
    (mu, sigma) = norm.fit(T_100_fo)
    plt.title(r'$\mathrm{Histogram\ of\ Distance\ from\ Origin\ at\ } %i\ \mu=%.3f,\ \sigma=%.3f$' % (int(percent), mu, sigma))
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.show()

    percent_copy = data[percent]
    for point in range(len(percent_copy)):
        for second in range(len(percent_copy[point])):
            percent_copy[point][second] = np.sqrt(np.square(percent_copy[point][second][0]) + np.square(percent_copy[point][second][1]))
    percent_msr = np.swapaxes(percent_copy, 0, 1)
    for i in range(len(percent_msr)):
        percent_msr[i] = np.mean(percent_msr[i])
    plt.plot(percent_msr)
    plt.show()

