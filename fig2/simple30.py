import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import tf_integrator as tf_int
import time
import sys
import pickle
import os
from tqdm import tqdm

tf.disable_v2_behavior()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

graphno,pertseed = int(sys.argv[2]),int(sys.argv[3])

def K_prop(V):

    V = V-(-50)

    T = 23

    phi = 3.0**((T-36)/10)

    alpha_n = 0.02*(15-V)/(tf.exp((15-V)*0.2) - 1.0)
    beta_n = 0.5*tf.exp((10.0-V)/40.0)

    t_n = 1.0/(alpha_n+beta_n)/phi
    n_inf = alpha_n/(alpha_n+beta_n)

    return n_inf, t_n


def Na_prop(V):

    V = V-(-50)

    T = 23

    phi = 3.0**((T-36)/10)

    alpha_m = 0.32*(13-V)/(tf.exp((13-V)*0.25)-1)
    beta_m = 0.28*(V-40)/(tf.exp((V-40)*0.2)-1)

    alpha_h = 0.128*tf.exp((17-V)/18.0)
    beta_h = 4.0/(tf.exp((40-V)*0.2) + 1.0)

    t_m = 1.0/(alpha_m+beta_m)/phi
    t_h = 1.0/(alpha_h+beta_h)/phi

    m_inf = alpha_m/(alpha_m+beta_m)
    h_inf = alpha_h/(alpha_h+beta_h)

    return m_inf, t_m, h_inf, t_h

def A_prop(V):

    T = 23

    phi = 3.0**((T-23.5)/10)

    m_inf = 1/(1+tf.exp(-(V+60.0)/8.5))
    h_inf = 1/(1+tf.exp((V+78.0)/6.0))

    tau_m = 0.27/(tf.exp((V + 35.8)/19.7) + tf.exp(-(V + 79.7)/12.7)) + 0.1

    t1 = 0.27*1/(tf.exp((V+46.05)/5.0) + tf.exp(-(V+238.4)/37.45)) #/ phi
    t2 = tf.ones(tf.shape(V),dtype=V.dtype)* 5.1 #*(19.0/phi) #*5.1
    tau_h = tf.where(tf.less(V,-63.0),t1,t2)

    return m_inf, tau_m, h_inf, tau_h

def Ca_prop(V):

    m_0 = 1/(1+tf.exp(-(V+20.0)/6.5))
    h_0 = 1/(1+tf.exp((V+25.0)/12))

    tau_m = 1.5
    tau_h = 0.3*tf.exp((V-40.0)/13.0) + 0.002*tf.exp((60.0-V)/29)

    return m_0, tau_m, h_0, tau_h

def KCa_prop(Ca):
    return Ca/(Ca+2), 100/(Ca+2)


# NEURONAL CURRENTS

# Common Currents #

def I_K(V, n):
    return g_K  * n**4 * (V - E_K)

def I_L(V):
    return g_L * (V - E_L)

def I_KL(V):
    return g_KL * (V - E_KL)

# PN Currents #

def I_Na(V, m, h):
    return g_Na * m**3 * h * (V - E_Na)

def I_A(V, m, h):
    return g_A * m**4 * h * (V - E_A)

# LN Currents #

def I_Ca(V, m, h):
    return g_Ca * m**2 * h * (V - E_Ca)

def I_KCa(V, m):
    return g_KCa * m * (V - E_KCa)

# SYNAPTIC CURRENTS

def I_ach(o,V):
    o_ = tf.Variable([0.0]*n_n**2,dtype=tf.float64)
    ind = tf.boolean_mask(tf.range(n_n**2),ach_mat.reshape(-1) == 1)
    o_ = tf.scatter_update(o_,ind,o)
    o_ = tf.transpose(tf.reshape(o_,(n_n,n_n)))
    return tf.reduce_sum(tf.transpose((o_*(V-E_ach))*g_ach),1)

def I_fgaba(o,V):
    o_ = tf.Variable([0.0]*n_n**2,dtype=tf.float64)
    ind = tf.boolean_mask(tf.range(n_n**2),fgaba_mat.reshape(-1) == 1)
    o_ = tf.scatter_update(o_,ind,o)
    o_ = tf.transpose(tf.reshape(o_,(n_n,n_n)))
    return tf.reduce_sum(tf.transpose((o_*(V-E_fgaba))*g_fgaba),1)

def I_sgaba(G,V):
    G4 = tf.pow(G,4)/(tf.pow(G,4)+K_sgaba)
    G_ = tf.Variable([0.0]*n_n**2,dtype=tf.float64)
    ind = tf.boolean_mask(tf.range(n_n**2),sgaba_mat.reshape(-1) == 1)
    G_ = tf.scatter_update(G_,ind,G4)
    G_ = tf.transpose(tf.reshape(G_,(n_n,n_n)))
    return tf.reduce_sum(tf.transpose((G_*(V-E_sgaba))*G_sgaba),1)

# INPUT CURRENTS

def I_inj_t(t,V):
    return tf.constant(current_input.T,dtype=tf.float64)[tf.to_int32(t*100)]*(V-E_ach)


# DIFFERENTIAL EQUATION FORM

def dXdt(X, t): # X is the state vector

    V_p   = X[0   : p_n] # Voltage(PN)
    V_l   = X[p_n : n_n] # Voltage(LN)

    n_K   = X[n_n : 2*n_n] # K-gating(ALL)

    m_Na  = X[2*n_n : 2*n_n + p_n] # Na-activation-gating(PN)
    h_Na  = X[2*n_n + p_n : 2*n_n + 2*p_n] # Na-inactivation-gating(PN)

    m_A   = X[2*n_n + 2*p_n : 2*n_n + 3*p_n] # Transient-K-activation-gating(PN)
    h_A   = X[2*n_n + 3*p_n : 2*n_n + 4*p_n] # Transient-K-inactivation-gating(PN)

    m_Ca  = X[2*n_n + 4*p_n : 2*n_n + 4*p_n + l_n] # Ca-activation-gating(LN)
    h_Ca  = X[2*n_n + 4*p_n + l_n: 2*n_n + 4*p_n + 2*l_n] # Ca-inactivation-gating(LN)

    m_KCa = X[2*n_n + 4*p_n + 2*l_n : 2*n_n + 4*p_n + 3*l_n] # K(Ca)-gating(LN)
    Ca    = X[2*n_n + 4*p_n + 3*l_n: 2*n_n + 4*p_n + 4*l_n] # Ca-concentration(LN)

    o_ach = X[6*n_n : 6*n_n + n_syn_ach] # Acetylcholine Open Fraction
    o_fgaba = X[6*n_n + n_syn_ach : 6*n_n + n_syn_ach + n_syn_fgaba] # GABAa Open Fraction
    r_sgaba = X[6*n_n + n_syn_ach + n_syn_fgaba : 6*n_n + n_syn_ach + n_syn_fgaba + n_syn_sgaba] # GABAa Open Fraction
    g_sgaba = X[6*n_n + n_syn_ach + n_syn_fgaba + n_syn_sgaba : 6*n_n + n_syn_ach + n_syn_fgaba + 2*n_syn_sgaba] # GABAa Open Fraction
    fire_t = X[-n_n:] # Fire-times

    V = X[:n_n] # Overall Voltage (PN + LN)


    # Evaluate Differentials for Gating variables and Ca concentration

    n0,tn = K_prop(V)

    dn_k = - (1.0/tn)*(n_K-n0)

    m0,tm,h0,th = Na_prop(V_p)

    dm_Na = - (1.0/tm)*(m_Na-m0)
    dh_Na = - (1.0/th)*(h_Na-h0)

    m0,tm,h0,th = A_prop(V_p)

    dm_A = - (1.0/tm)*(m_A-m0)
    dh_A = - (1.0/th)*(h_A-h0)

    m0,tm,h0,th = Ca_prop(V_l)

    dm_Ca = - (1.0/tm)*(m_Ca-m0)
    dh_Ca = - (1.0/th)*(h_Ca-h0)

    m0,tm = KCa_prop(Ca)

    dm_KCa = - (1.0/tm)*(m_KCa-m0)

    dCa = - A_Ca*I_Ca(V_l,m_Ca,h_Ca) - (Ca - Ca0)/t_Ca

    # Evaluate differential for Voltage

    CmdV_p = - I_Na(V_p, m_Na, h_Na) - I_A(V_p, m_A, h_A)
    CmdV_l = - I_Ca(V_l, m_Ca, h_Ca) - I_KCa(V_l, m_KCa)

    # Once we have that, we merge the two into a single 120-vector.

    CmdV = tf.concat([CmdV_p,CmdV_l],0)

    # Finally we add the common currents and divide by Cm to get dV/dt.

    dV = (-I_inj_t(t,V) + CmdV - I_K(V, n_K) - I_L(V) - I_ach(o_ach,V) - I_fgaba(o_fgaba,V) - I_sgaba(g_sgaba,V)) / C_m


    # Evaluate dynamics in synapses

    A_ = tf.constant(A,dtype=tf.float64)
    T_ach = tf.where(tf.logical_and(tf.greater(t,fire_t+t_delay),tf.less(t,fire_t+t_max+t_delay)),A_,tf.zeros(tf.shape(A_),dtype=A_.dtype))
    T_ach = tf.multiply(tf.constant(ach_mat,dtype=tf.float64),T_ach)
    T_ach = tf.boolean_mask(tf.reshape(T_ach,(-1,)),ach_mat.reshape(-1) == 1)
    do_achdt = alp_ach*(1.0-o_ach)*T_ach - bet_ach*o_ach

    T_fgaba = 1.0/(1.0+tf.exp(-(V-V0)/sigma))
    T_fgaba = tf.multiply(tf.constant(fgaba_mat,dtype=tf.float64),T_fgaba)
    T_fgaba = tf.boolean_mask(tf.reshape(T_fgaba,(-1,)),fgaba_mat.reshape(-1) == 1)
    do_fgabadt = alp_fgaba*(1.0-o_fgaba)*T_fgaba - bet_fgaba*o_fgaba

    dg_sgabadt = - np.array(r4_sgaba)*g_sgaba + np.array(r3_sgaba)*r_sgaba

    A_ = tf.constant(A,dtype=tf.float64)
    T_sgaba = tf.where(tf.logical_and(tf.greater(t,fire_t+t_delay),tf.less(t,fire_t+t_max+t_delay)),A_,tf.zeros(tf.shape(A_),dtype=A_.dtype))
    T_sgaba = tf.multiply(tf.constant(sgaba_mat,dtype=tf.float64),T_sgaba)
    T_sgaba = tf.boolean_mask(tf.reshape(T_sgaba,(-1,)),sgaba_mat.reshape(-1) == 1)
    dr_sgabadt = r1_sgaba*(1.0-r_sgaba)*T_sgaba - r2_sgaba*r_sgaba

    # Set change in fire-times as zero

    dfdt = tf.zeros(tf.shape(fire_t),dtype=fire_t.dtype)

    # Combine to a single vector

    out = tf.concat([dV,         dn_k,
                     dm_Na,      dh_Na,
                     dm_A,       dh_A,
                     dm_Ca,      dh_Ca,
                     dm_KCa,
                     dCa,        do_achdt,
                     do_fgabadt, dr_sgabadt,
                     dg_sgabadt, dfdt   ],0)
    return out


metadata = np.load(f'__simcache__/metadata_{graphno}_{pertseed}.npy',allow_pickle=True).item()

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

ach_mat = np.zeros((n_n,n_n))

## Defining Acetylcholine Synapse Parameters ##

n_syn_ach = int(np.sum(ach_mat))     # Number of Acetylcholine (Ach) Synapses
alp_ach = [10.0]*n_syn_ach           # Alpha for Ach Synapse
bet_ach = [0.2]*n_syn_ach            # Beta for Ach Synapse
t_max = 0.3                          # Maximum Time for Synapse
t_delay = 0                          # Axonal Transmission Delay
A = [0.5]*n_n                        # Synaptic Response Strength
# g_ach = [0.09]*p_n+[0.45]*l_n         # Ach Conductance
g_ach = [0.0]*p_n+[0.0]*l_n         # Ach Conductance
E_ach = [0.0]*n_n                    # Ach Potential

## Defining GABAa Synapse Connectivity ##

fgaba_mat = metadata['fgaba_mat']

## Defining GABAa Synapse Parameters ##

n_syn_fgaba = int(np.sum(fgaba_mat)) # Number of GABAa (fGABA) Synapses
alp_fgaba = [10.0]*n_syn_fgaba       # Alpha for fGABA Synapse
bet_fgaba = [0.16]*n_syn_fgaba       # Beta for fGABA Synapse
V0 = [-20.0]*n_n                     # Decay Potential
sigma = [1.5]*n_n                    # Decay Time Constant
g_fgaba = [0.0]*p_n+[30*0.2/2*metadata['g_gaba']]*l_n #0.4       # fGABA Conductance
E_fgaba = [-70.0]*n_n                # fGABA Potential

## Defining GABAslow Synapse Connectivity ##

sgaba_mat = np.zeros((n_n,n_n))

## Defining GABAslow Synapse Parameters ##

n_syn_sgaba = int(np.sum(sgaba_mat)) # Number of GABAslow (sGABA) Synapses
K_sgaba = [100e-12]*n_syn_sgaba          # K for sGABA Synapse
r1_sgaba = [1]*n_syn_sgaba         # r1 for sGABA Synapse
r2_sgaba = [0.025]*n_syn_sgaba      # r2 for sGABA Synapse
r3_sgaba = [0.1]*n_syn_sgaba         # r3 for sGABA Synapse
r4_sgaba = [0.06]*n_syn_sgaba       # r4 for sGABA Synapse
V0_sgaba = [-20.0]*n_n               # Decay Potential
sigma_sgaba = [1.5]*n_n              # Decay Time Constant
G_sgaba = [0.0]*p_n+[0.0]*l_n      # sGABA Conductance
E_sgaba = [-95.0]*n_n                # sGABA Potential


g_ach = np.divide(np.array(g_ach),np.sum(ach_mat,axis=1),where=np.sum(ach_mat,axis=1)!=0)
G_sgaba = np.divide(np.array(G_sgaba),np.sum(sgaba_mat,axis=1),where=np.sum(sgaba_mat,axis=1)!=0)
g_fgaba = np.divide(np.array(g_fgaba),np.sum(fgaba_mat,axis=1),where=np.sum(fgaba_mat,axis=1)!=0)

t = np.load(f"__simcache__/time_{graphno}_{pertseed}.npy")[int(sys.argv[1])]
current_input = np.load(f"__simcache__/current_input_{graphno}_{pertseed}.npy")

state_vector = np.load(f"__simcache__/state_vector_{graphno}_{pertseed}.npy")

n_batch = 1
t_batch = np.array_split(t,n_batch)

t_ = time.time()

states = []

for n,i in tqdm(enumerate(t_batch)):

    if n>0:
        i = np.append(i[0]-sim_res,i)

    init_state = tf.constant(state_vector, dtype=tf.float64)
    tensor_state = tf_int.odeint(dXdt, init_state, i, n_n, F_b)

    with tf.Session() as sess:
        print("Session started...",end="")
        tf.global_variables_initializer().run()
        state = sess.run(tensor_state)
        sess.close()

    state_vector = state[-1,:]
    states.append(state[::int(1/sim_res),:][:-1,:])
state = np.concatenate(states)
print("Completed. Total Execution Time:",np.round(time.time()-t_,3),"secs")

np.save(f"__simcache__/state_vector_{graphno}_{pertseed}.npy",state_vector)
np.save(f"__simoutput__/state_{sys.argv[1]}_{graphno}_{pertseed}.npy",state)
