# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:11:00 2020

@author: Pranav
"""

import random as rand

import csv

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import norm

import pandas as pd

# Point class setup - basically allows me to get one simulation done
class Point:
    def __init__(self, pos, percent):
        self.pos = pos
        self.pos_track = [pos]
        self.percent = percent

    def pos_update(self):
        num = rand.randint(1, 100)
        if 1 <= num <= self.percent:
            self.pos = self.pos + 1
            self.pos_track.append(self.pos)
            return 1
        elif self.percent + 1 <= num <= self.percent * 2:
            self.pos = self.pos - 1
            self.pos_track.append(self.pos)
            return -1
        self.pos_track.append(self.pos)
        return 0


d_coeffs = []
big_comp = []
for percent in range(1, 34):
    trajectory_compilation = dict()  # Dictionary with key being trial number
    for point in range(100):  # Spawing in 100 points
        p = Point(0, percent)
        for update in range(100):
            p.pos_update()
        trajectory_compilation[point] = p.pos_track
    '''
    d = trajectory_compilation
    keys = sorted(d.keys())
    with open("test.csv", "wt") as outfile:
       writer = csv.writer(outfile, delimiter = ",")
       writer.writerow(keys)
       writer.writerows([d[key] for key in keys])
    
    '''
    df = pd.DataFrame(trajectory_compilation)  # Turning trajectory_compilation into dataframe
    x_vals = df.loc[99]  # x_vals = dataframe with only last row
    big_comp.append(list(df.loc[99]))
    '''
    df.to_csv('test2.csv')
    '''
    (mu, sigma) = norm.fit(
        x_vals)  # Specifically doing stats stuff to find mu and sigma - sigma is standard deviation and mu is mean

    y = norm.pdf(mu, sigma)

    plt.hist(x_vals, bins=percent)
    plt.title(r'$\mathrm{Histogram\ of\ X\ Position\ at\ } %i\ \mu=%.3f,\ \sigma=%.3f$' % (int(percent), mu, sigma))
    plt.show()

    dff = df.mean(1)  # Finding mean over all columns (so collapsing columns)

    plt.plot(dff)  # Line plot of dff

    plt.title(f"Mean of X position vs. Time for percent = {percent}")
    plt.xlabel("time")
    plt.ylabel("position")
    plt.show()

    squared = np.square(df).mean(1)  # Squares and finds the mean over all columns
    '''    
    squared.to_csv('test3.csv')
    '''
    plt.plot(
        squared.values.tolist())  # plots all the values in squared which are in an organized list in chronological
    # order
    plt.title(f"Mean squared of X position vs. Time for percent = {percent}")
    plt.xlabel("time")
    plt.ylabel("position")
    plt.show()
    list_of_100 = []
    for i in range(101):
        list_of_100.append(i)

    m, b = np.polyfit(list_of_100, np.square(df).mean(1), 1)
    d_coeffs.append(m)

plt.plot(d_coeffs)
plt.xlabel("Percent")
plt.ylabel("Diffusion coefficient")
plt.title("Coefficient of Diffusion versus percent chance of moving")
plt.show()

theoretical_d = []
for i in range(1, 34):
    theoretical_d.append((i*2/100)**2)

plt.plot(theoretical_d, d_coeffs)
plt.xlabel("Theoretical Diffusion Coefficient")
plt.ylabel("Experimental Diffusion coefficient")
plt.title("Theoretical versus Experimental Diffusion Coefficient")
plt.show()

big_comp2 = []
for i in big_comp:
    for j in i:
        big_comp2.append(j)
(mu, sigma) = norm.fit(big_comp2)
y = norm.pdf(mu, sigma)
plt.hist(big_comp2, bins=34)
plt.title(f"Histogram for total X position for all percents mu={mu} sigma={sigma}")
