# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:11:00 2020

@author: Pranav
"""

import random as rand

import pandas as pd

import csv as csv

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.stats import norm


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
plt.xlablel("Percent")
plt.ylabel("Diffusion coefficient")
plt.title("Coefficient of Diffusion versus percent chance of moving")
plt.show()

plt.hist(big_comp, bins=34)
plt.title("Total")

'''#This is the looping mechanism per se - does 100 trajectories over 100 time intervals
d_coeff = {}
big_compilation = dict()
for i in range(100):
    big_compilation[i] = []

for percent in range(1,34):
    trajectory_compilation = dict() #Will be a dictionary with key being trajectory number and entry being a list with x values
    for i in range(100):
        point = Point(0, percent)
        for j in range(100):
           point.pos_update()
        trajectory_compilation['t' + str(i)] = point.pos_track
    
    df = pd.DataFrame(data=trajectory_compilation) #data frame from which work will be done from for now
    
    plt.plot(df.mean(1))
    
    plt.xlabel('time')
    plt.ylabel('x position')
    plt.title(f'Mean for {percent}')
    

    x = []
    for i in range(101):
        x.append(i)
    
    meansquare = np.square(df).mean(1)
    
    m, b = np.polyfit(x, list(meansquare), 1)
    
    d_coeff[percent] = m
    
    #print(m, percent)
    
    
    plt.show()
    
    plt.plot(np.square(df).mean(1))
    
    plt.xlabel('time')
    plt.ylabel('x position')
    plt.title(f'Mean square for {percent}')
    
    plt.show()
    

    bins = max(trajectory_compilation['t99']) - min(trajectory_compilation['t99'])
    
    (mu, sigma) = norm.fit(trajectory_compilation['t99'])
    
    y = norm.pdf( bins, mu, sigma)
    
    l = plt.plot(bins, y, 'r--', linewidth=2)

    plt.hist(trajectory_compilation['t99'], bins = bins)
    
    plt.xlabel('x position at ' + str(percent))
    plt.ylabel('frequency')
    plt.title(r'$\mathrm{Histogram\ of\ X\ Position\ at\ } %i\ \mu=%.3f,\ \sigma=%.3f$' %(int(percent), mu, sigma))
    
    plt.show()
    
    for i in trajectory_compilation.keys():
        for index in range(len(trajectory_compilation[i])-1):
            big_compilation[index].append(trajectory_compilation[i][index])
        

y_vals = []
for i in d_coeff.keys():
    y_vals.append(d_coeff[i])

plt.scatter(list(d_coeff.keys()), y_vals)
plt.xlabel('percentage')
plt.ylabel('diffusion coefficient')
plt.title('diffusion coefficient vs percentage')
plt.show()


df = pd.DataFrame(data=big_compilation) #data frame from which work will be done from for now

plt.plot(df.mean(0))

plt.xlabel('time')
plt.ylabel('x position')
plt.title('Mean for total')

plt.show()

plt.plot(np.square(df).mean(0))

plt.xlabel('time')
plt.ylabel('x position')
plt.title('Mean square for total')

plt.show()

list_of_x = []
for i in big_compilation.keys():
    for j in big_compilation[i]:
        list_of_x.append(j)
bins = max(list_of_x) - min(list_of_x)

(mu, sigma) = norm.fit(list_of_x)

y = norm.pdf( bins, mu, sigma)

l = plt.plot(bins, y, 'r--', linewidth=2)

plt.hist(list_of_x, bins = bins)

plt.xlabel('x position for all probabilities')
plt.ylabel('frequency')
plt.title(r'$\mathrm{Histogram\ of\ X\ Position\ at\ total\ } \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))

plt.show()'''
