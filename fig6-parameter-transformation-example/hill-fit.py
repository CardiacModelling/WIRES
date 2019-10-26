#!/usr/bin/env python3
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

savedir = './hill-fig/'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

np.random.seed(101)

def hill(x, ic50, n):
    x = np.asarray(x)
    ic50 = np.float(ic50)
    n = np.float(n)
    return 1. / (1. + (ic50 / x) ** n)

def transformed_func(x, tt1, tt2):
    ut1 = np.exp(tt1)
    ut2 = np.copy(tt2)
    return hill(x, ut1, ut2)

# Generate
xdata = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
xfunc = np.logspace(-5., 5., 200)
ic50_true = 1.2
n_true = 0.85
ydata = hill(xdata, ic50_true, n_true)
ydata += np.random.normal(0, 0.05, size=xdata.shape)
plt.plot(xdata, ydata, 'x')
plt.ylabel('Fraction block')
plt.xlabel('Concentration')
plt.xscale('log')
plt.close()

# Bounds
ic50u = 1000.
ic50l = 0.01
nu = 10.
nl = 0.0

# Simple fit
p0_all = []
popt_all = []
for _ in range(18):
    p0 = np.random.uniform([ic50l, nl], [ic50u, nu])
    popt, pcov = curve_fit(hill, xdata, ydata, p0=p0,
            bounds=([ic50l, nl], [ic50u, nu]))
    p0_all.append(p0)
    popt_all.append(popt)
# Manual init
p0 = [1., 0.3]
popt, pcov = curve_fit(hill, xdata, ydata, p0=p0,
        bounds=([ic50l, nl], [ic50u, nu]))
p0_all.append(p0)
p0 = [100., 1.]
popt, pcov = curve_fit(hill, xdata, ydata, p0=p0,
        bounds=([ic50l, nl], [ic50u, nu]))
p0_all.append(p0)
popt_all.append(popt)
plt.plot(xdata, ydata, 'x')
plt.plot(xfunc, hill(xfunc, *popt))
plt.plot(xfunc, hill(xfunc, *p0_all[-1]), c='#fdbf6f')
plt.plot(xfunc, hill(xfunc, *p0_all[-2]), c='#e41a1c')
plt.ylabel('Fraction block', fontsize=17)
plt.xlabel('Concentration', fontsize=17)
plt.xscale('log')
plt.savefig('hill-fig/dose-response-example', bbox_inches='tight')
plt.savefig('hill-fig/dose-response-example.pdf', format='pdf',
        bbox_inches='tight')
plt.close()

# Inspect contour
ic50_sweep = np.linspace(ic50l, ic50u, 250)
n_sweep = np.linspace(nl, nu, 250)
class Error(object):
    def __init__(self, data, x):
        self.d = data
        self.x = x
    def __call__(self, ic50, n):
        return np.sqrt(np.mean((self.d - hill(self.x, ic50, n))**2))
error = Error(ydata, xdata)
IC50, N = np.meshgrid(ic50_sweep, n_sweep)
E = np.zeros(IC50.shape)
for i in range(IC50.shape[0]):
    for j in range(IC50.shape[1]):
        E[i, j] = error(IC50[i, j], N[i, j])
x_min, x_max = np.min(ic50_sweep), np.max(ic50_sweep)
y_min, y_max = np.min(n_sweep), np.max(n_sweep)
z_min, z_max = np.min(E), np.max(E)
fig, ax = plt.subplots()
c = ax.pcolormesh(IC50, N, E, cmap='viridis_r', vmin=z_min, vmax=z_max)
# cmap: 'RdBu', 'YlGnBu'
#ax.plot(ic50_true, n_true, marker='x', c='w', ls='')
for i, (p0, popt) in enumerate(zip(p0_all, popt_all)):
    if i == len(p0_all) - 1:
        colour = '#fdbf6f'
    elif i == len(p0_all) - 2:
        colour = '#e41a1c'
    else:
        colour = '#cccccc'
    ax.plot([p0[0], popt[0]], [p0[1], popt[1]], marker='x', c=colour, ls='--',
            alpha=1)
    ax.plot(popt[0], popt[1], marker='x', c='C1', ls='')
ax.axis([x_min, x_max, y_min, y_max])
ax.set_xlabel(r'IC$_{50}$', fontsize=16)
ax.set_ylabel('Hill coefficient', fontsize=16)
cbar = fig.colorbar(c, ax=ax)
cbar.ax.set_ylabel('RMSE', fontsize=16)
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('hill-fig/dose-response-simple-fit', dpi=200, bbox_inches='tight')
plt.savefig('hill-fig/dose-response-simple-fit.pdf', format='pdf',
        bbox_inches='tight')
plt.close()

'''
# Simple transformed fit
tpopt_all = []
for p0 in p0_all:
    tp0 = [np.log(p0[0]), p0[1]]
    tpopt, pcov = curve_fit(transformed_func, xdata, ydata, p0=tp0,
            bounds=([np.log(ic50l), nl], [np.log(ic50u), nu]))
    tpopt_all.append(tpopt)

# Inspect transformed contour
tic50_sweep = np.log(np.logspace(np.log10(ic50l), np.log10(ic50u), 250))
tn_sweep = np.copy(n_sweep)
error = Error(ydata, xdata)
TIC50, TN = np.meshgrid(tic50_sweep, tn_sweep)
TE = np.zeros(TIC50.shape)
for i in range(TIC50.shape[0]):
    for j in range(TIC50.shape[1]):
        TE[i, j] = error(np.exp(TIC50[i, j]), TN[i, j])
x_min, x_max = np.min(tic50_sweep), np.max(tic50_sweep)
y_min, y_max = np.min(tn_sweep), np.max(tn_sweep)
z_min, z_max = np.min(TE), np.max(TE)
fig, ax = plt.subplots()
c = ax.pcolormesh(TIC50, TN, TE, cmap='viridis_r', vmin=z_min, vmax=z_max)
# cmap: 'RdBu', 'YlGnBu'
#ax.plot(np.log(ic50_true), n_true, marker='x', c='w', ls='')
for p0, tpopt in zip(p0_all, tpopt_all):
    ax.plot([np.log(p0[0]), tpopt[0]], [p0[1], tpopt[1]], marker='x',
            c='#cccccc', ls='--', alpha=1)
    ax.plot(tpopt[0], tpopt[1], marker='x', c='C1', ls='')
ax.axis([x_min, x_max, y_min, y_max])
ax.set_xlabel(r'$\ln$(IC$_{50})$')
ax.set_ylabel('Hill coefficient')
cbar = fig.colorbar(c, ax=ax)
cbar.ax.set_ylabel('RMSE')
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.close()
'''

# Simple transformed fit (resampled p0)
#def transformed2_func(x, tt1, tt2):
#    ut1 = np.exp(tt1)
#    ut2 = np.exp(tt2)
#    return hill(x, ut1, ut2)
t2p0_all = []
t2popt_all = []
for _ in range(18):
    t2p0 = np.random.uniform([np.log(ic50l), nl], [np.log(ic50u), nu])
    t2popt, pcov = curve_fit(transformed_func, xdata, ydata, p0=t2p0,
            bounds=([np.log(ic50l), nl], [np.log(ic50u), nu]))
    t2p0_all.append(t2p0)
    t2popt_all.append(t2popt)
# Manual init
t2p0 = [np.log(1.), 0.3]
t2popt, pcov = curve_fit(transformed_func, xdata, ydata, p0=t2p0,
        bounds=([np.log(ic50l), nl], [np.log(ic50u), nu]))
t2p0_all.append(t2p0)
t2popt_all.append(t2popt)
t2p0 = [np.log(100.), 1.]
t2popt, pcov = curve_fit(transformed_func, xdata, ydata, p0=t2p0,
        bounds=([np.log(ic50l), nl], [np.log(ic50u), nu]))
t2p0_all.append(t2p0)
t2popt_all.append(t2popt)

# Inspect transformed contour (resampled p0)
t2ic50_sweep = np.log(np.logspace(np.log10(ic50l), np.log10(ic50u), 250))
#t2n_sweep = np.log(np.logspace(np.log10(nl), np.log10(nu), 250))
t2n_sweep = np.copy(n_sweep)
error = Error(ydata, xdata)
T2IC50, T2N = np.meshgrid(t2ic50_sweep, t2n_sweep)
T2E = np.zeros(T2IC50.shape)
for i in range(T2IC50.shape[0]):
    for j in range(T2IC50.shape[1]):
        T2E[i, j] = error(np.exp(T2IC50[i, j]), T2N[i, j])
x_min, x_max = np.min(t2ic50_sweep), np.max(t2ic50_sweep)
y_min, y_max = np.min(t2n_sweep), np.max(t2n_sweep)
z_min, z_max = np.min(T2E), np.max(T2E)
fig, ax = plt.subplots()
c = ax.pcolormesh(T2IC50, T2N, T2E, cmap='viridis_r', vmin=z_min, vmax=z_max)
# cmap: 'RdBu', 'YlGnBu'
#ax.plot(np.log(ic50_true), n_true, marker='x', c='w', ls='')
for i, (tp0, tpopt) in enumerate(zip(t2p0_all, t2popt_all)):
    if i == len(t2p0_all) - 1:
        colour = '#fdbf6f'
    elif i == len(t2p0_all) - 2:
        colour = '#e41a1c'
    else:
        colour = '#cccccc'
    ax.plot([tp0[0], tpopt[0]], [tp0[1], tpopt[1]], marker='x', c=colour,
            ls='--', alpha=1)
    ax.plot(tpopt[0], tpopt[1], marker='x', c='C1', ls='')
ax.axis([x_min, x_max, y_min, y_max])
ax.set_xlabel(r'$\ln$(IC$_{50})$', fontsize=16)
ax.set_ylabel('Hill coefficient', fontsize=16)
cbar = fig.colorbar(c, ax=ax)
cbar.ax.set_ylabel('RMSE', fontsize=16)
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('hill-fig/dose-response-simple-transformed-fit', dpi=200,
        bbox_inches='tight')
plt.savefig('hill-fig/dose-response-simple-transformed-fit.pdf', format='pdf',
        bbox_inches='tight')
plt.close()

