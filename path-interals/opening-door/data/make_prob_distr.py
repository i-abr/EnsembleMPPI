#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

data = np.load('door-hinge-distribution.npy')
time = [4*i*0.002 for i in range(len(data[:,1]))]

plt.plot(time, data[:,0], label='Horizontal Hinge')
plt.plot(time, data[:,1], label='Vertical Hinge')
plt.plot(time, data[:,2], label='Prismatic Joint')

plt.ylabel('Probability')
plt.xlabel('sampling time (s)')


plt.legend()
plt.show()
