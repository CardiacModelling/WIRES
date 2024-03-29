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
bestn = 45

# AP
ap_protocol = np.loadtxt('ap-protocol.csv', delimiter=',')
ap_protocol[:, 1] = np.roll(ap_protocol[:, 1], 50)  # shift the AP for plotting
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

# Load AP parameters
with open('./herg-out/ap-fit-rank.txt', 'r') as f:
    files = f.read().splitlines()
ap_p_all = []
for f in files:
    p = np.log(np.loadtxt('./herg-out/' + f))
    ap_p_all.append(p)
# Best N
ap_p_best = ap_p_all[:bestn]

# Load Staircase parameters
with open('./herg-out/stair-fit-rank.txt', 'r') as f:
    files = f.read().splitlines()
stair_p_all = []
for f in files:
    p = np.log(np.loadtxt('./herg-out/' + f))
    stair_p_all.append(p)
# Best N
stair_p_best = stair_p_all[:bestn]

# Plot
fig, axes = plt.subplots(3, 2, figsize=(14, 7),
        gridspec_kw={'height_ratios': [1, 2, 2]})

axes[0, 0].plot(ap_times, ap_herg.v_func(ap_times), c='#7f7f7f')
axes[0, 0].set_ylabel('Voltage (mV)', fontsize=12)
axes[0, 0].set_xticks([])
axes[1, 0].plot(ap_times, ap_data, alpha=0.8, c='#7f7f7f', label='Data')
for i, p in enumerate(ap_p_best):
    fitted = ap_herg.simulate(p, ap_times)
    alpha = 0.35 if i else 0.6
    axes[1, 0].plot(ap_times, fitted, c='#1f77b4', alpha=alpha,
            label='__nolegend__' if i else 'Fit')
axes[1, 0].legend(loc=1, ncol=2)
axes[1, 0].set_ylim([-0.5, 8.5])
axes[1, 0].set_ylabel('Current\n(normalised)', fontsize=13)
axes[1, 0].set_xticks([])
axes[1, 0].text(.025, .875, 'Training', ha='left', fontsize=14,
        color='#1f77b4', transform=axes[1, 0].transAxes)
axes[2, 0].plot(ap_times, ap_data, alpha=0.8, c='#7f7f7f', label='Data')
for i, p in enumerate(stair_p_best):
    fitted = ap_herg.simulate(p, ap_times)
    alpha = 0.35 if i else 0.6
    axes[2, 0].plot(ap_times, fitted, c='#ff7f0e', alpha=alpha,
            label='__nolegend__' if i else 'Prediction')
axes[2, 0].legend(loc=1, ncol=2)
axes[2, 0].set_ylim([-0.5, 8.5])
axes[2, 0].set_ylabel('Current\n(normalised)', fontsize=13)
axes[2, 0].set_xlabel('Time (ms)', fontsize=16)
axes[2, 0].text(.025, .875, 'Validation', ha='left', fontsize=14,
        color='#ff7f0e', transform=axes[2, 0].transAxes)

axes[1, 0].arrow(1.025, 0.5, 0.14, 0, clip_on=False, width=0.02,
        length_includes_head=True, head_width=0.06, head_length=0.04,
        transform=axes[1, 0].transAxes)
axes[2, 0].arrow(1.025 + 0.14, 0.5, -0.14, 0, clip_on=False, width=0.02,
        length_includes_head=True, head_width=0.06, head_length=0.04,
        transform=axes[2, 0].transAxes)

axes[0, 1].plot(stair_times, stair_herg.v_func(stair_times), c='#7f7f7f')
axes[0, 1].set_ylabel('Voltage (mV)', fontsize=12)
axes[0, 1].set_xticks([])
axes[1, 1].plot(stair_times, stair_data, alpha=0.8, c='#7f7f7f', label='Data')
for i, p in enumerate(ap_p_best):
    predict = stair_herg.simulate(p, stair_times)
    alpha = 0.35 if i else 0.6
    axes[1, 1].plot(stair_times, predict, c='#ff7f0e', alpha=alpha,
            label='__nolegend__' if i else 'Prediction')
axes[1, 1].legend(loc=1, ncol=2)
axes[1, 1].set_ylim([-1.5, 9.5])
axes[1, 1].set_ylabel('Current\n(normalised)', fontsize=13)
axes[1, 1].set_xticks([])
axes[1, 1].text(.025, .875, 'Validation', ha='left', fontsize=14,
        color='#ff7f0e', transform=axes[1, 1].transAxes)
axes[2, 1].plot(stair_times, stair_data, alpha=0.8, c='#7f7f7f', label='Data')
for i, p in enumerate(stair_p_best):
    predict = stair_herg.simulate(p, stair_times)
    alpha = 0.35 if i else 0.6
    axes[2, 1].plot(stair_times, predict, c='#1f77b4', alpha=alpha,
            label='__nolegend__' if i else 'Fit')
axes[2, 1].legend(loc=1, ncol=2)
axes[2, 1].set_ylim([-1.5, 9.5])
axes[2, 1].set_ylabel('Current\n(normalised)', fontsize=13)
axes[2, 1].set_xlabel('Time (ms)', fontsize=16)
axes[2, 1].text(.025, .875, 'Training', ha='left', fontsize=14,
        color='#1f77b4', transform=axes[2, 1].transAxes)

plt.subplots_adjust(hspace=0.075, wspace=0.3)
plt.savefig('%s/herg-fits-and-predictions.png' % (savedir), dpi=200,
        bbox_inches='tight')
plt.savefig('%s/herg-fits-and-predictions.pdf' % (savedir), format='pdf',
        bbox_inches='tight')
plt.close()
