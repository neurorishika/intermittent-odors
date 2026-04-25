import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.backend import get_backend_name, integrate_trajectory

metadata = np.load('__simcache__/metadata.npy',allow_pickle=True).item()

sim_res = metadata['sim_res']

n_n = metadata['n_n']             # number of neurons

p_n = metadata['p_n']                  # number of PNs
l_n = metadata['l_n']                  # number of LNs

C_m  = [1.0]*n_n                     # Capacitance

# Defining Common Current Parameters #

g_K  = [3.6]*p_n+[36]*l_n          # K conductance
g_L  = [0.3]*n_n                    # Leak conductance

E_K  = [-95.0]*p_n + [-95.0]*l_n     # K Potential
E_L  = [-64.0]*p_n + [-50.0]*l_n     # Leak Potential (first 90 for PNs and next 30 for LNs)

# Defining Cell Type Specific Current Parameters #

## PNs

g_Na = [7.15]*p_n                   # Na conductance
g_A  = [1.43]*p_n                    # Transient K conductance

E_Na = [50.0]*p_n                    # Na Potential
E_A  = [-95.0]*p_n                   # Transient K Potential

## LNs

g_Ca = [5.0]*l_n                     # Ca conductance
g_KCa = [0.045]*l_n                    # Ca dependent K conductance

E_Ca = [140.0]*l_n                   # Ca Potential
E_KCa = [-95]*l_n                    # Ca dependent K Potential

A_Ca = 2*(10**(-4))                  # Ca outflow rate
Ca0 = 2.4*(10**(-4))                 # Equilibrium Calcium Concentration
t_Ca = 150                           # Ca recovery time constant

## Defining Firing Thresholds ##

F_b = [0.0]*p_n+[-20.0]*l_n                      # Fire threshold

## Defining Acetylcholine Synapse Connectivity ##

ach_mat = metadata['ach_mat']

## Defining Acetylcholine Synapse Parameters ##

n_syn_ach = int(np.sum(ach_mat))     # Number of Acetylcholine (Ach) Synapses
alp_ach = [10.0]*n_syn_ach           # Alpha for Ach Synapse
bet_ach = [0.2]*n_syn_ach            # Beta for Ach Synapse
t_max = 0.3                          # Maximum Time for Synapse
t_delay = 0                          # Axonal Transmission Delay
A = [0.5]*n_n                        # Synaptic Response Strength
# g_ach = [0.09]*p_n+[0.45]*l_n         # Ach Conductance
g_ach = [0.0]*p_n+[0.225]*l_n         # Ach Conductance
E_ach = [0.0]*n_n                    # Ach Potential

## Defining GABAa Synapse Connectivity ##

fgaba_mat = metadata['fgaba_mat']

## Defining GABAa Synapse Parameters ##

n_syn_fgaba = int(np.sum(fgaba_mat)) # Number of GABAa (fGABA) Synapses
alp_fgaba = [10.0]*n_syn_fgaba       # Alpha for fGABA Synapse
bet_fgaba = [0.16]*n_syn_fgaba       # Beta for fGABA Synapse
V0 = [-20.0]*n_n                     # Decay Potential
sigma = [1.5]*n_n                    # Decay Time Constant
g_fgaba = [2.16]*p_n+[3.6]*l_n #0.4       # fGABA Conductance
E_fgaba = [-70.0]*n_n                # fGABA Potential

## Defining GABAslow Synapse Connectivity ##

sgaba_mat = metadata['sgaba_mat']

## Defining GABAslow Synapse Parameters ##

n_syn_sgaba = int(np.sum(sgaba_mat)) # Number of GABAslow (sGABA) Synapses
K_sgaba = [100e-12]*n_syn_sgaba          # K for sGABA Synapse
r1_sgaba = [1]*n_syn_sgaba         # r1 for sGABA Synapse
r2_sgaba = [0.025]*n_syn_sgaba      # r2 for sGABA Synapse
r3_sgaba = [0.1]*n_syn_sgaba         # r3 for sGABA Synapse
r4_sgaba = [0.06]*n_syn_sgaba       # r4 for sGABA Synapse
V0_sgaba = [-20.0]*n_n               # Decay Potential
sigma_sgaba = [1.5]*n_n              # Decay Time Constant
G_sgaba = [0.054]*p_n+[0.0]*l_n      # sGABA Conductance
E_sgaba = [-95.0]*n_n                # sGABA Potential


g_ach = np.divide(np.array(g_ach),np.sum(ach_mat,axis=1),where=np.sum(ach_mat,axis=1)!=0)
G_sgaba = np.divide(np.array(G_sgaba),np.sum(sgaba_mat,axis=1),where=np.sum(sgaba_mat,axis=1)!=0)
g_fgaba = np.divide(np.array(g_fgaba),np.sum(fgaba_mat,axis=1),where=np.sum(fgaba_mat,axis=1)!=0)

t = np.load("__simcache__/time.npy")[int(sys.argv[1])]
current_input = np.load("__simcache__/current_input.npy")

config = {
    'n_n': n_n,
    'p_n': p_n,
    'l_n': l_n,
    'C_m': C_m,
    'g_K': g_K,
    'g_L': g_L,
    'E_K': E_K,
    'E_L': E_L,
    'g_Na': g_Na,
    'g_A': g_A,
    'E_Na': E_Na,
    'E_A': E_A,
    'g_Ca': g_Ca,
    'g_KCa': g_KCa,
    'E_Ca': E_Ca,
    'E_KCa': E_KCa,
    'A_Ca': A_Ca,
    'Ca0': Ca0,
    't_Ca': t_Ca,
    'ach_mat': ach_mat,
    'alp_ach': alp_ach,
    'bet_ach': bet_ach,
    't_max': t_max,
    't_delay': t_delay,
    'A': A,
    'g_ach': g_ach,
    'E_ach': E_ach,
    'fgaba_mat': fgaba_mat,
    'alp_fgaba': alp_fgaba,
    'bet_fgaba': bet_fgaba,
    'V0': V0,
    'sigma': sigma,
    'g_fgaba': g_fgaba,
    'E_fgaba': E_fgaba,
    'sgaba_mat': sgaba_mat,
    'K_sgaba': K_sgaba,
    'r1_sgaba': r1_sgaba,
    'r2_sgaba': r2_sgaba,
    'r3_sgaba': r3_sgaba,
    'r4_sgaba': r4_sgaba,
    'G_sgaba': G_sgaba,
    'E_sgaba': E_sgaba,
}

backend = get_backend_name()
print(f"Using {backend} backend...")

state_vector = np.load("__simcache__/state_vector.npy")

n_batch = 1
t_batch = np.array_split(t,n_batch)

t_ = time.time()

states = []

for n,i in tqdm(enumerate(t_batch)):

    if n>0:
        i = np.append(i[0]-sim_res,i)

    state = integrate_trajectory(config, current_input, state_vector, i, F_b, backend=backend)

    state_vector = state[-1,:]
    states.append(state[::int(1/sim_res),:][:-1,:])
state = np.concatenate(states)
print("Completed. Total Execution Time:",np.round(time.time()-t_,3),"secs")

np.save("__simcache__/state_vector.npy",state_vector)
np.save(f"__simoutput__/state_{sys.argv[1]}.npy",state)
np.save(f"__simoutput__/state_{sys.argv[1]}.npy",state)
