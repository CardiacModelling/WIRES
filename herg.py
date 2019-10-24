import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import pints

ap_protocol = np.loadtxt('ap-protocol.csv', delimiter=',')

class Model(pints.ForwardModel):

    def __init__(self, protocol, set_x0=None):
        # protocol: 2-D array-like [[t_0, t_1, ...], [V_0, V_1, ...]]
        # set_x0: if None, use simulated x0
        self.v_func = interp1d(protocol[:, 0], protocol[:, 1], kind='linear')
        self.set_x0 = set_x0

    def n_parameters(self):
        return 12

    def v_func(self, times):
        return v_func(times)

    def _rhs(self, x, t, p):
        p = np.exp(p)
        V = self.v_func(t)

        inf1 = 1.0 / (1.0 + np.exp((-p[0] - V) / p[1]))
        a1 = p[2] / (1.0 + np.exp((-p[3] - V) / p[4]))
        b1 = 6.0 / (1.0 + np.exp((p[5] + V) / p[6]))
        # 6 is a-priori unidentifiable with p(3) so we remove it from
        # optimisation.

        inf2 = 1.0 / (1.0 + np.exp((V + p[7]) / p[8]))
        a2 = p[9] / (1.0 + np.exp((-p[10] - V) / p[11]))
        b2 = 1.12 / (1.0 + np.exp((V - p[10]) / p[11]))
        # 1.12 is a-priori unidentifiable with p(10) so we remove it.

        t1 = a1 * b1
        t2 = a2 * b2

        return np.array([(inf1 - x[0]) / t1, (inf2 - x[1]) / t2])

    def _int(self, p, times):
        if self.set_x0 is not None:
            x0 = self.set_x0
        else:
            x0 = self._ic(p)
        try:
	    return odeint(self._rhs, x0, times, args=(p,))
        except:
            return np.inf * np.ones((len(times), 2))

    def _ic(self, p, x0=[0, 1]):
        dt = 0.2
        times = np.arange(0, 500 + dt, dt)
        return odeint(self._rhs, x0, times, args=(p,))[-1, :]

    def simulate(self, p, times):
        x = self._int(p, times)
        return x[:, 0] * x[:, 1] * (self.v_func(times) - (-88))

