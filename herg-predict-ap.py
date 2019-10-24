#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import herg
import pints

"""
Predict.
"""

savedir = './herg-fig/'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# True parameter
p_true = np.array([27, 7, 450, 45, 10, 30, 11.5, 88, 24, 3, 60, 20])

# AP
ap_protocol = np.loadtxt('ap-protocol.csv', delimiter=',')
ap_herg = herg.Model(ap_protocol, set_x0=None)
ap_times = np.arange(0, 400, 0.2)
np.random.seed(101)  # fix data noise seed
ap_data = ap_herg.simulate(np.log(p_true), ap_times)
ap_data += np.random.normal(0, 0.15, size=ap_data.shape)

# Staircase
stair_protocol = np.loadtxt('staircase-protocol.csv', delimiter=',')
stair_herg = herg.Model(stair_protocol, set_x0=None)
stair_times = np.arange(0, 15.3 * 1e3, 0.2)
np.random.seed(101)  # fix data noise seed
stair_data = stair_herg.simulate(np.log(p_true), stair_times)
stair_data += np.random.normal(0, 0.15, size=stair_data.shape)

# Score
problem = pints.SingleOutputProblem(stair_herg, stair_times, stair_data)
score = pints.RootMeanSquaredError(problem)

# Load parameters
files = glob.glob('./herg-out/stair-fit-*.txt')
p_all = []
s_all = []
for f in files:
    try:
        p = np.log(np.loadtxt(f))
        s = score(p)
    except:
        continue
    p_all.append(p)
    s_all.append(s)

# Sort
order = np.argsort(s_all)  # (use [::-1] for LL)
s_all = np.asarray(s_all)[order]
p_all = np.asarray(p_all)[order]
print(s_all)

# Best N
N = 50
s_best = s_all[:N]
p_best = p_all[:N]
print(s_best)

# Inspect plot
fig, axes = plt.subplots(2, 2, figsize=(14, 6))

axes[0, 0].plot(stair_times, stair_herg.v_func(stair_times), c='#7f7f7f')
axes[0, 0].set_ylabel('Voltage (mV)')
axes[0, 0].set_xticks([])
axes[1, 0].plot(stair_times, stair_data, alpha=0.5, c='#1f77b4', label='data')
for i, p in enumerate(p_best):
    predict = stair_herg.simulate(p, stair_times)
    axes[1, 0].plot(stair_times, predict, c='#ff7f0e',
            label='__nolegend__' if i else 'Prediction')
axes[1, 0].legend(loc=4)
axes[1, 0].set_ylim([-2, 8])
axes[1, 0].set_ylabel('Current (pA)')
axes[1, 0].set_xlabel('Time (ms)')

axes[0, 1].plot(ap_times, ap_herg.v_func(ap_times), c='#7f7f7f')
axes[0, 1].set_ylabel('Voltage (mV)')
axes[0, 1].set_xticks([])
axes[1, 1].plot(ap_times, ap_data, alpha=0.5, c='#1f77b4', label='data')
for i, p in enumerate(p_best):
    fitted = ap_herg.simulate(p, ap_times)
    axes[1, 1].plot(ap_times, fitted, c='#ff7f0e',
            label='__nolegend__' if i else 'Fitted')
axes[1, 1].legend()
axes[1, 1].set_ylabel('Current (pA)')
axes[1, 1].set_xlabel('Time (ms)')

plt.subplots_adjust(hspace=0)
plt.savefig('%s/staircase-predict-ap.png' % (savedir), dpi=200,
        bbox_inches='tight')
plt.close()
