#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import herg
import pints

"""
Run fit.
"""

try:
    fit_id = int(sys.argv[1])
except:
    print('Usage: python %s [int:fit_id]' % os.path.basename(__file__))
    sys.exit()

savedir = './herg-out/'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Load fitting protocol
ap_protocol = np.loadtxt('ap-protocol.csv', delimiter=',')

# True parameter
p_true = np.array([27, 7, 450, 45, 10, 30, 11.5, 88, 24, 3, 60, 20])
times = np.arange(0, 400, 0.2)

# Set up model
ap_herg = herg.Model(ap_protocol, set_x0=None)

# Generate data
np.random.seed(101)  # fix data noise seed
data = ap_herg.simulate(np.log(p_true), times)
data += np.random.normal(0, 0.15, size=data.shape)

# Set fitting seed
np.random.seed(fit_id)
fit_seed = np.random.randint(0, 2**30)
np.random.seed(fit_seed)

# Score
problem = pints.SingleOutputProblem(ap_herg, times, data)
score = pints.RootMeanSquaredError(problem)
print('Score at true value: ', score(np.log(p_true)))

# Fit
for _ in range(100):
    # Randomly pick a starting point
    x0 = p_true * np.random.uniform(0.66, 1.5, size=len(p_true))
    x0 = np.log(x0)
    # Make sure it can be simulated
    if not np.isfinite(score(x0)):
        continue
    else:
        break
print('Starting point: ', x0)

# Create optimiser
print('Starting error: ', score(x0))
opt = pints.OptimisationController(score, x0, method=pints.CMAES)
opt.set_max_iterations(None)
opt.set_max_unchanged_iterations(iterations=100, threshold=1e-5)
opt.set_parallel(False)

# Run optimisation
try:
    with np.errstate(all='ignore'):
        # Tell numpy not to issue warnings
        p, s = opt.run()
        p = np.exp(p)
        print('Found solution:          True parameters:' )
        for k, x in enumerate(p):
            print(pints.strfloat(x) + '    ' + pints.strfloat(p_true[k]))
except ValueError:
    import traceback
    traceback.print_exc()

# Store output
with open('%s/ap-fit-%s.txt' % (savedir, fit_id), 'w') as f:
    for k, x in enumerate(p):
        f.write(pints.strfloat(x) + '\n')

# Inspect plot
fig, axes = plt.subplots(2, 2, figsize=(14, 6))
fitted = problem.evaluate(np.log(p))
axes[0, 0].plot(times, ap_herg.v_func(times), c='#7f7f7f')
axes[0, 0].set_ylabel('Voltage (mV)')
axes[1, 0].plot(times, data, alpha=0.5, label='data')
axes[1, 0].plot(times, fitted, label='Fitted')
axes[1, 0].legend()
axes[1, 0].set_ylabel('Current (pA)')
axes[1, 0].set_xlabel('Time (ms)')
stair_protocol = np.loadtxt('staircase-protocol.csv', delimiter=',')
stair_herg = herg.Model(stair_protocol, set_x0=None)
times = np.arange(0, 15.3 * 1e3, 0.2)
data = stair_herg.simulate(np.log(p_true), times)
predict = stair_herg.simulate(p, times)
axes[0, 1].plot(times, stair_herg.v_func(times), c='#7f7f7f')
axes[0, 1].set_ylabel('Voltage (mV)')
axes[1, 1].plot(times, data, alpha=0.5, label='data')
axes[1, 1].plot(times, predict, label='Predict')
axes[1, 1].legend()
axes[1, 1].set_ylabel('Current (pA)')
axes[1, 1].set_xlabel('Time (ms)')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/ap-fit-%s.png' % (savedir, fit_id), dpi=200,
        bbox_inches='tight')
plt.close()
