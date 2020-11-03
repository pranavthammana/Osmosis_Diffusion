# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:11:00 2020

@author: Pranav
"""

import random as rand

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import norm

import matplotlib.mlab as mlab

#Point class setup - basically allows me to get one simulation done
class Point:
    def __init__(self, pos):
        self.pos = pos
        self.pos_track = [pos]
        
    def pos_update(self):
        num = rand.randint(1, 100)
        if 1 <= num <= 25:
            self.pos = self.pos + 1
            self.pos_track.append(self.pos)
            return 1
        elif 26 <= num <= 50:
            self.pos = self.pos - 1
            self.pos_track.append(self.pos)
            return -1
        self.pos_track.append(self.pos)
        return 0


#This is the looping mechanism per se - does 100 trajectories over 100 time intervals
trajectory_compilation = dict() #Will be a dictionary with key being trajectory number and entry being a list with x values

for i in range(100):
    point = Point(0)
    for j in range(100):
       point.pos_update()
    trajectory_compilation['t' + str(i)] = point.pos_track

df = pd.DataFrame(data=trajectory_compilation) #data frame from which work will be done from for now

plt.plot(df.mean(1))

plt.xlabel('time')
plt.ylabel('x position')
plt.title('Mean')

x = []
for i in range(101):
    x.append(i)

meansquare = np.square(df).mean(1)

m, b = np.polyfit(x, list(meansquare), 1)

print(m)


plt.show()

plt.plot(np.square(df).mean(1))

plt.xlabel('time')
plt.ylabel('x position')
plt.title('Mean square')

plt.show()

list_of_x = []
for i in trajectory_compilation.keys():
    for j in trajectory_compilation[i]:
        list_of_x.append(j)
bins = max(list_of_x) - min(list_of_x)

(mu, sigma) = norm.fit(list_of_x)

y = norm.pdf( bins, mu, sigma)

l = plt.plot(bins, y, 'r--', linewidth=2)

plt.hist(list_of_x, bins = bins)

plt.xlabel('x position')
plt.ylabel('frequency')
plt.title(r'$\mathrm{Histogram\ of\ X Position:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))

plt.show()
#df.to_excel(r'/Users/home/Dropbox/Internship App stuff/pain.xlsx', index = False)
