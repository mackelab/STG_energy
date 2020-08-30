from sbi_summstats import PrinzStats
import numpy

import pyximport

# setup_args needed on my MacOS
pyximport.install(setup_args={"include_dirs": numpy.get_include()}, reload_support=True)

from sbi_simulator import sim_time
from sbi_simulator_energyScape import sim_time_energyscape
import numpy as np


t_burnin = 1000
t_window = 10000
noise_fact = 0.001
tmax = t_burnin + t_window
dt = 0.025
seed = 0


def get_time():
    return np.arange(0, tmax, dt)


def wrapper(params):

    full_data = simulate(params)
    ss = stats(full_data)

    return ss


def simulate(params, seed=None):
    # note: make sure to generate all randomness through self.rng (!)
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    t = np.arange(0, tmax, dt)

    membrane_params = params[0:-7]
    membrane_params = np.float64(np.reshape(membrane_params, (3, 8)))
    synaptic_params = np.exp(params[-7:])
    conns = build_conns(-synaptic_params)

    I = rng.normal(scale=noise_fact, size=(3, len(t)))

    # calling the solver --> HH.HH()
    data = sim_time(
        dt,
        t,
        I,
        membrane_params,  # membrane conductances
        conns,  # synaptic conductances (always variable)
        temp=284,
        init=None,
        start_val_input=0.0,
        verbose=False,
    )

    full_data = {
        'data': data['Vs'],
        'tmax': tmax,
        'dt': dt,
        'I': I,
        'energy': data['energy'],
    }

    return full_data



def simulate_energyscape(params, seed=None):
    # note: make sure to generate all randomness through self.rng (!)
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    t = np.arange(0, tmax, dt)

    membrane_params = params[0:-7]
    membrane_params = np.float64(np.reshape(membrane_params, (3, 8)))
    synaptic_params = np.exp(params[-7:])
    conns = build_conns(-synaptic_params)

    I = rng.normal(scale=noise_fact, size=(3, len(t)))

    # calling the solver --> HH.HH()
    data = sim_time_energyscape(
        dt,
        t,
        I,
        membrane_params,  # membrane conductances
        conns,  # synaptic conductances (always variable)
        temp=284,
        init=None,
        start_val_input=0.0,
        verbose=False,
    )

    full_data = {
        'data': data['Vs'],
        'tmax': tmax,
        'dt': dt,
        'I': I,
        'energy': data['energy'],
        'all_energies': data['all_energies']
    }

    return full_data


def stats(full_data):
    stats_object = PrinzStats(
        t_on=t_burnin,
        t_off=t_burnin + t_window,
        include_pyloric_ness=True,
        include_plateaus=True,
        seed=seed,
        energy=True
    )

    ss = stats_object.calc([full_data])[0]
    return ss


def build_conns(params):

    # Reversal voltages and dissipation time constants for the synapses, taken from
    # Prinz 2004, p. 1351
    Esglut = -70            # mV
    kminusglut = 40         # ms

    Eschol = -80            # mV
    kminuschol = 100        # ms

    return np.asarray([
        [1, 0, params[0], Esglut, kminusglut],
        [1, 0, params[1], Eschol, kminuschol],
        [2, 0, params[2], Esglut, kminusglut],
        [2, 0, params[3], Eschol, kminuschol],
        [0, 1, params[4], Esglut, kminusglut],
        [2, 1, params[5], Esglut, kminusglut],
        [1, 2, params[6], Esglut, kminusglut]
    ])
