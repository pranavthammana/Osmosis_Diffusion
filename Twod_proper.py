import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import norm

import random as rand

import sys

from mpl_toolkits.mplot3d import Axes3D


def calculate_delta(percent):
    chance = rand.randint(1, 100)
    if 1 <= chance <= percent:
        return -1
    elif percent <= chance <= 2 * percent:
        return 1
    else:
        return 0


data = {}
diffusion_coefficients_y = []
diffusion_coefficients_x = []
for percent in range(1, 34, 2):
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
    plt.title(r'$\mathrm{Histogram\ of\ Y\ Position\ } %i\ \mu=%.3f,\ \sigma=%.3f$' % (int(percent), mu, sigma))
    plt.xlabel("Position")
    plt.ylabel("Frequency")
    plt.show()

    X_Array = np.zeros((100, 100))
    for point in range(len(X_Array)):
        for time in range(len(X_Array)):
            X_Array[point][time] = data[percent][point][time][0]

    Y_Array = np.zeros((100, 100))
    for point in range(len(Y_Array)):
        for time in range(len(Y_Array)):
            Y_Array[point][time] = data[percent][point][time][1]

    X_Array2 = np.transpose(X_Array)
    Y_Array2 = np.transpose(Y_Array)

    X_Mean = []
    Y_Mean = []
    T_Vals = []

    for time in range(len(X_Array2)):
        X_Mean.append(np.mean(X_Array2[time]))

    for time in range(len(Y_Array2)):
        Y_Mean.append(np.mean(Y_Array2[time]))

    for time in range(len(X_Array2)):
        T_Vals.append(time)

    plt.plot(T_Vals, X_Mean, '.')
    plt.xlabel("Time")
    plt.ylabel("X Position")
    plt.title("Mean - Percent = " + str(percent))
    plt.show()

    plt.plot(T_Vals, Y_Mean, '.')
    plt.xlabel("Time")
    plt.ylabel("Y Position")
    plt.title("Mean - Percent = " + str(percent))
    plt.show()

    X_mean_square = []
    Y_mean_square = []

    for time in range(len(X_Array2)):
        for position in range(len(X_Array2[time])):
            X_Array2[time][position] = X_Array2[time][position] ** 2
        X_mean_square.append(np.mean(X_Array2[time]))
    for time in range(len(Y_Array2)):
        for position in range(len(Y_Array2[time])):
            Y_Array2[time][position] = Y_Array2[time][position] ** 2
        Y_mean_square.append(np.mean(Y_Array2[time]))

    plt.plot(T_Vals, X_mean_square)
    plt.xlabel("Time")
    plt.ylabel("X Position")
    plt.title("Mean Square - Percent = " + str(percent))
    plt.show()
    plt.plot(T_Vals, Y_mean_square)
    plt.xlabel("Time")
    plt.ylabel("Y Position")
    plt.title("Mean Square - Percent = " + str(percent))
    plt.show()

    
    m_x, b_x = np.polyfit(T_Vals, X_mean_square, 1)
    m_y, b_y = np.polyfit(T_Vals, Y_mean_square, 1)
    diffusion_coefficients_x.append(m_x)
    diffusion_coefficients_y.append(m_y)
    
    plt.scatter(X_Mean, Y_Mean)
    plt.show()

percents = []
for i in range(1, 34):
    percents.append(i)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(diffusion_coefficients_x, diffusion_coefficients_y, percents, zdir='z', s=20, c=None, depthshade=True)

