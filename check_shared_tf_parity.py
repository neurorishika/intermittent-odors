import sys
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.tf_integrator import odeint
from shared.tf_network import build_dynamics


def normalize_by_indegree(values, matrix):
    indegree = np.sum(matrix, axis=1)
    out = np.zeros_like(values, dtype=np.float64)
    np.divide(values, indegree, out=out, where=indegree != 0)
    return out


def build_stable_current_input(rng, n_n, p_n, steps):
    current_input = np.zeros((n_n, steps), dtype=np.float64)
    activation = rng.choice([0.0, 1.0], size=steps, p=[0.375, 0.625])
    if not np.any(activation):
        activation[rng.integers(steps)] = 1.0

    pn_drive = np.zeros(p_n, dtype=np.float64)
    active_pn = max(1, int(np.ceil(p_n * 0.3)))
    pn_drive[rng.choice(p_n, size=active_pn, replace=False)] = 1.0

    current_input[:p_n, :] = 0.18 * pn_drive[:, None] * activation[None, :]
    if n_n > p_n:
        ln_activation = np.roll(activation, 1)
        current_input[p_n:, :] = 0.055 * ln_activation[None, :]

    current_input += 0.03 * current_input * rng.normal(size=current_input.shape)
    current_input += 1e-4 * rng.normal(size=current_input.shape)
    return np.clip(current_input, 0.0, None)


def build_stable_state_vector(rng, n_n, p_n, l_n, n_syn_ach, n_syn_fgaba, n_syn_sgaba, sim_time):
    state_vector = np.array(
        [-45.0] * p_n
        + [-45.0] * l_n
        + [0.5] * (n_n + 4 * p_n + 3 * l_n)
        + [2.4e-4] * l_n
        + [0.0] * (n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba)
        + [-(sim_time + 1.0)] * n_n,
        dtype=np.float64,
    )

    state_vector[:n_n] += rng.normal(scale=0.75, size=n_n)

    gate_start = n_n
    gate_stop = 2 * n_n + 4 * p_n + 3 * l_n
    state_vector[gate_start:gate_stop] = np.clip(
        state_vector[gate_start:gate_stop] + 0.04 * rng.normal(size=gate_stop - gate_start),
        0.0,
        1.0,
    )

    ca_start = 2 * n_n + 4 * p_n + 3 * l_n
    ca_stop = ca_start + l_n
    if l_n:
        state_vector[ca_start:ca_stop] = np.clip(
            state_vector[ca_start:ca_stop] * (1.0 + 0.03 * rng.normal(size=l_n)),
            1e-6,
            None,
        )

    syn_start = 6 * n_n
    syn_stop = syn_start + n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba
    if syn_stop > syn_start:
        state_vector[syn_start:syn_stop] = np.clip(
            0.01 * rng.normal(size=syn_stop - syn_start),
            0.0,
            0.05,
        )

    fire_start = syn_stop
    state_vector[fire_start:] += 0.1 * rng.normal(size=n_n)
    return state_vector


def legacy_build_dynamics(config, current_input):
    n_n = int(config['n_n'])
    p_n = int(config['p_n'])
    l_n = int(config['l_n'])

    C_m = np.asarray(config['C_m'], dtype=np.float64)
    g_K = np.asarray(config['g_K'], dtype=np.float64)
    g_L = np.asarray(config['g_L'], dtype=np.float64)
    E_K = np.asarray(config['E_K'], dtype=np.float64)
    E_L = np.asarray(config['E_L'], dtype=np.float64)
    g_Na = np.asarray(config['g_Na'], dtype=np.float64)
    g_A = np.asarray(config['g_A'], dtype=np.float64)
    E_Na = np.asarray(config['E_Na'], dtype=np.float64)
    E_A = np.asarray(config['E_A'], dtype=np.float64)
    g_Ca = np.asarray(config['g_Ca'], dtype=np.float64)
    g_KCa = np.asarray(config['g_KCa'], dtype=np.float64)
    E_Ca = np.asarray(config['E_Ca'], dtype=np.float64)
    E_KCa = np.asarray(config['E_KCa'], dtype=np.float64)
    A_Ca = float(config['A_Ca'])
    Ca0 = float(config['Ca0'])
    t_Ca = float(config['t_Ca'])

    ach_mat = np.asarray(config['ach_mat'], dtype=np.float64)
    n_syn_ach = int(np.sum(ach_mat))
    alp_ach = np.asarray(config['alp_ach'], dtype=np.float64)
    bet_ach = np.asarray(config['bet_ach'], dtype=np.float64)
    t_max = float(config['t_max'])
    t_delay = float(config['t_delay'])
    A = np.asarray(config['A'], dtype=np.float64)
    g_ach = np.asarray(config['g_ach'], dtype=np.float64)
    E_ach = np.asarray(config['E_ach'], dtype=np.float64)

    fgaba_mat = np.asarray(config['fgaba_mat'], dtype=np.float64)
    n_syn_fgaba = int(np.sum(fgaba_mat))
    alp_fgaba = np.asarray(config['alp_fgaba'], dtype=np.float64)
    bet_fgaba = np.asarray(config['bet_fgaba'], dtype=np.float64)
    V0 = np.asarray(config['V0'], dtype=np.float64)
    sigma = np.asarray(config['sigma'], dtype=np.float64)
    g_fgaba = np.asarray(config['g_fgaba'], dtype=np.float64)
    E_fgaba = np.asarray(config['E_fgaba'], dtype=np.float64)

    sgaba_mat = np.asarray(config['sgaba_mat'], dtype=np.float64)
    n_syn_sgaba = int(np.sum(sgaba_mat))
    K_sgaba = np.asarray(config['K_sgaba'], dtype=np.float64)
    r1_sgaba = np.asarray(config['r1_sgaba'], dtype=np.float64)
    r2_sgaba = np.asarray(config['r2_sgaba'], dtype=np.float64)
    r3_sgaba = np.asarray(config['r3_sgaba'], dtype=np.float64)
    r4_sgaba = np.asarray(config['r4_sgaba'], dtype=np.float64)
    G_sgaba = np.asarray(config['G_sgaba'], dtype=np.float64)
    E_sgaba = np.asarray(config['E_sgaba'], dtype=np.float64)

    def K_prop(V):
        V = V - (-50)
        T = 23
        phi = 3.0 ** ((T - 36) / 10)
        alpha_n = 0.02 * (15 - V) / (tf.exp((15 - V) * 0.2) - 1.0)
        beta_n = 0.5 * tf.exp((10.0 - V) / 40.0)
        t_n = 1.0 / (alpha_n + beta_n) / phi
        n_inf = alpha_n / (alpha_n + beta_n)
        return n_inf, t_n

    def Na_prop(V):
        V = V - (-50)
        T = 23
        phi = 3.0 ** ((T - 36) / 10)
        alpha_m = 0.32 * (13 - V) / (tf.exp((13 - V) * 0.25) - 1)
        beta_m = 0.28 * (V - 40) / (tf.exp((V - 40) * 0.2) - 1)
        alpha_h = 0.128 * tf.exp((17 - V) / 18.0)
        beta_h = 4.0 / (tf.exp((40 - V) * 0.2) + 1.0)
        t_m = 1.0 / (alpha_m + beta_m) / phi
        t_h = 1.0 / (alpha_h + beta_h) / phi
        m_inf = alpha_m / (alpha_m + beta_m)
        h_inf = alpha_h / (alpha_h + beta_h)
        return m_inf, t_m, h_inf, t_h

    def A_prop(V):
        T = 23
        phi = 3.0 ** ((T - 23.5) / 10)
        m_inf = 1 / (1 + tf.exp(-(V + 60.0) / 8.5))
        h_inf = 1 / (1 + tf.exp((V + 78.0) / 6.0))
        tau_m = 0.27 / (tf.exp((V + 35.8) / 19.7) + tf.exp(-(V + 79.7) / 12.7)) + 0.1
        t1 = 0.27 / (tf.exp((V + 46.05) / 5.0) + tf.exp(-(V + 238.4) / 37.45))
        t2 = tf.ones(tf.shape(V), dtype=V.dtype) * 5.1
        tau_h = tf.where(tf.less(V, -63.0), t1, t2)
        return m_inf, tau_m, h_inf, tau_h

    def Ca_prop(V):
        m_0 = 1 / (1 + tf.exp(-(V + 20.0) / 6.5))
        h_0 = 1 / (1 + tf.exp((V + 25.0) / 12))
        tau_m = 1.5
        tau_h = 0.3 * tf.exp((V - 40.0) / 13.0) + 0.002 * tf.exp((60.0 - V) / 29)
        return m_0, tau_m, h_0, tau_h

    def KCa_prop(Ca):
        return Ca / (Ca + 2), 100 / (Ca + 2)

    def I_K(V, n):
        return g_K * n ** 4 * (V - E_K)

    def I_L(V):
        return g_L * (V - E_L)

    def I_Na(V, m, h):
        return g_Na * m ** 3 * h * (V - E_Na)

    def I_A(V, m, h):
        return g_A * m ** 4 * h * (V - E_A)

    def I_Ca(V, m, h):
        return g_Ca * m ** 2 * h * (V - E_Ca)

    def I_KCa(V, m):
        return g_KCa * m * (V - E_KCa)

    def I_ach(o, V):
        o_ = tf.Variable([0.0] * n_n ** 2, dtype=tf.float64)
        ind = tf.boolean_mask(tf.range(n_n ** 2), ach_mat.reshape(-1) == 1)
        o_ = tf.scatter_update(o_, ind, o)
        o_ = tf.transpose(tf.reshape(o_, (n_n, n_n)))
        return tf.reduce_sum(tf.transpose((o_ * (V - E_ach)) * g_ach), 1)

    def I_fgaba(o, V):
        o_ = tf.Variable([0.0] * n_n ** 2, dtype=tf.float64)
        ind = tf.boolean_mask(tf.range(n_n ** 2), fgaba_mat.reshape(-1) == 1)
        o_ = tf.scatter_update(o_, ind, o)
        o_ = tf.transpose(tf.reshape(o_, (n_n, n_n)))
        return tf.reduce_sum(tf.transpose((o_ * (V - E_fgaba)) * g_fgaba), 1)

    def I_sgaba(G, V):
        G4 = tf.pow(G, 4) / (tf.pow(G, 4) + K_sgaba)
        G_ = tf.Variable([0.0] * n_n ** 2, dtype=tf.float64)
        ind = tf.boolean_mask(tf.range(n_n ** 2), sgaba_mat.reshape(-1) == 1)
        G_ = tf.scatter_update(G_, ind, G4)
        G_ = tf.transpose(tf.reshape(G_, (n_n, n_n)))
        return tf.reduce_sum(tf.transpose((G_ * (V - E_sgaba)) * G_sgaba), 1)

    def I_inj_t(t, V):
        return tf.constant(np.asarray(current_input, dtype=np.float64).T, dtype=tf.float64)[tf.to_int32(t * 100)] * (V - E_ach)

    def dXdt(X, t):
        V_p = X[0:p_n]
        V_l = X[p_n:n_n]
        n_K = X[n_n:2 * n_n]
        m_Na = X[2 * n_n:2 * n_n + p_n]
        h_Na = X[2 * n_n + p_n:2 * n_n + 2 * p_n]
        m_A = X[2 * n_n + 2 * p_n:2 * n_n + 3 * p_n]
        h_A = X[2 * n_n + 3 * p_n:2 * n_n + 4 * p_n]
        m_Ca = X[2 * n_n + 4 * p_n:2 * n_n + 4 * p_n + l_n]
        h_Ca = X[2 * n_n + 4 * p_n + l_n:2 * n_n + 4 * p_n + 2 * l_n]
        m_KCa = X[2 * n_n + 4 * p_n + 2 * l_n:2 * n_n + 4 * p_n + 3 * l_n]
        Ca = X[2 * n_n + 4 * p_n + 3 * l_n:2 * n_n + 4 * p_n + 4 * l_n]
        o_ach = X[6 * n_n:6 * n_n + n_syn_ach]
        o_fgaba = X[6 * n_n + n_syn_ach:6 * n_n + n_syn_ach + n_syn_fgaba]
        r_sgaba = X[6 * n_n + n_syn_ach + n_syn_fgaba:6 * n_n + n_syn_ach + n_syn_fgaba + n_syn_sgaba]
        g_sgaba = X[6 * n_n + n_syn_ach + n_syn_fgaba + n_syn_sgaba:6 * n_n + n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba]
        fire_t = X[-n_n:]
        V = X[:n_n]

        n0, tn = K_prop(V)
        dn_k = -(1.0 / tn) * (n_K - n0)

        m0, tm, h0, th = Na_prop(V_p)
        dm_Na = -(1.0 / tm) * (m_Na - m0)
        dh_Na = -(1.0 / th) * (h_Na - h0)

        m0, tm, h0, th = A_prop(V_p)
        dm_A = -(1.0 / tm) * (m_A - m0)
        dh_A = -(1.0 / th) * (h_A - h0)

        m0, tm, h0, th = Ca_prop(V_l)
        dm_Ca = -(1.0 / tm) * (m_Ca - m0)
        dh_Ca = -(1.0 / th) * (h_Ca - h0)

        m0, tm = KCa_prop(Ca)
        dm_KCa = -(1.0 / tm) * (m_KCa - m0)

        dCa = -A_Ca * I_Ca(V_l, m_Ca, h_Ca) - (Ca - Ca0) / t_Ca

        CmdV_p = -I_Na(V_p, m_Na, h_Na) - I_A(V_p, m_A, h_A)
        CmdV_l = -I_Ca(V_l, m_Ca, h_Ca) - I_KCa(V_l, m_KCa)
        CmdV = tf.concat([CmdV_p, CmdV_l], 0)

        dV = (-I_inj_t(t, V) + CmdV - I_K(V, n_K) - I_L(V) - I_ach(o_ach, V) - I_fgaba(o_fgaba, V) - I_sgaba(g_sgaba, V)) / C_m

        A_ = tf.constant(A, dtype=tf.float64)
        T_ach = tf.where(tf.logical_and(tf.greater(t, fire_t + t_delay), tf.less(t, fire_t + t_max + t_delay)), A_, tf.zeros(tf.shape(A_), dtype=A_.dtype))
        T_ach = tf.multiply(tf.constant(ach_mat, dtype=tf.float64), T_ach)
        T_ach = tf.boolean_mask(tf.reshape(T_ach, (-1,)), ach_mat.reshape(-1) == 1)
        do_achdt = alp_ach * (1.0 - o_ach) * T_ach - bet_ach * o_ach

        T_fgaba = 1.0 / (1.0 + tf.exp(-(V - V0) / sigma))
        T_fgaba = tf.multiply(tf.constant(fgaba_mat, dtype=tf.float64), T_fgaba)
        T_fgaba = tf.boolean_mask(tf.reshape(T_fgaba, (-1,)), fgaba_mat.reshape(-1) == 1)
        do_fgabadt = alp_fgaba * (1.0 - o_fgaba) * T_fgaba - bet_fgaba * o_fgaba

        dg_sgabadt = -np.array(r4_sgaba) * g_sgaba + np.array(r3_sgaba) * r_sgaba

        A_ = tf.constant(A, dtype=tf.float64)
        T_sgaba = tf.where(tf.logical_and(tf.greater(t, fire_t + t_delay), tf.less(t, fire_t + t_max + t_delay)), A_, tf.zeros(tf.shape(A_), dtype=A_.dtype))
        T_sgaba = tf.multiply(tf.constant(sgaba_mat, dtype=tf.float64), T_sgaba)
        T_sgaba = tf.boolean_mask(tf.reshape(T_sgaba, (-1,)), sgaba_mat.reshape(-1) == 1)
        dr_sgabadt = r1_sgaba * (1.0 - r_sgaba) * T_sgaba - r2_sgaba * r_sgaba

        dfdt = tf.zeros(tf.shape(fire_t), dtype=fire_t.dtype)

        return tf.concat([dV, dn_k, dm_Na, dh_Na, dm_A, dh_A, dm_Ca, dh_Ca, dm_KCa, dCa, do_achdt, do_fgabadt, dr_sgabadt, dg_sgabadt, dfdt], 0)

    return dXdt


def build_case(seed, n_n, p_n, ach_density, fgaba_density, sgaba_density):
    rng = np.random.default_rng(seed)
    l_n = n_n - p_n
    ach_mat = (rng.random((n_n, n_n)) < ach_density).astype(np.float64)
    fgaba_mat = (rng.random((n_n, n_n)) < fgaba_density).astype(np.float64)
    sgaba_mat = (rng.random((n_n, n_n)) < sgaba_density).astype(np.float64)
    np.fill_diagonal(ach_mat, 0.0)
    np.fill_diagonal(fgaba_mat, 0.0)
    np.fill_diagonal(sgaba_mat, 0.0)

    g_ach = np.concatenate([np.zeros(p_n), np.full(l_n, 0.225)])
    g_fgaba = np.concatenate([np.full(p_n, 2.16), np.full(l_n, 3.6)])
    G_sgaba = np.concatenate([np.full(p_n, 0.054), np.zeros(l_n)])

    g_ach = normalize_by_indegree(g_ach, ach_mat)
    g_fgaba = normalize_by_indegree(g_fgaba, fgaba_mat)
    G_sgaba = normalize_by_indegree(G_sgaba, sgaba_mat)

    n_syn_ach = int(np.sum(ach_mat))
    n_syn_fgaba = int(np.sum(fgaba_mat))
    n_syn_sgaba = int(np.sum(sgaba_mat))

    config = {
        'n_n': n_n,
        'p_n': p_n,
        'l_n': l_n,
        'C_m': [1.0] * n_n,
        'g_K': [3.6] * p_n + [36.0] * l_n,
        'g_L': [0.3] * n_n,
        'E_K': [-95.0] * n_n,
        'E_L': [-64.0] * p_n + [-50.0] * l_n,
        'g_Na': [7.15] * p_n,
        'g_A': [1.43] * p_n,
        'E_Na': [50.0] * p_n,
        'E_A': [-95.0] * p_n,
        'g_Ca': [5.0] * l_n,
        'g_KCa': [0.045] * l_n,
        'E_Ca': [140.0] * l_n,
        'E_KCa': [-95.0] * l_n,
        'A_Ca': 2e-4,
        'Ca0': 2.4e-4,
        't_Ca': 150.0,
        'ach_mat': ach_mat,
        'alp_ach': [10.0] * n_syn_ach,
        'bet_ach': [0.2] * n_syn_ach,
        't_max': 0.3,
        't_delay': 0.0,
        'A': [0.5] * n_n,
        'g_ach': g_ach,
        'E_ach': [0.0] * n_n,
        'fgaba_mat': fgaba_mat,
        'alp_fgaba': [10.0] * n_syn_fgaba,
        'bet_fgaba': [0.16] * n_syn_fgaba,
        'V0': [-20.0] * n_n,
        'sigma': [1.5] * n_n,
        'g_fgaba': g_fgaba,
        'E_fgaba': [-70.0] * n_n,
        'sgaba_mat': sgaba_mat,
        'K_sgaba': [100e-12] * n_syn_sgaba,
        'r1_sgaba': [1.0] * n_syn_sgaba,
        'r2_sgaba': [0.025] * n_syn_sgaba,
        'r3_sgaba': [0.1] * n_syn_sgaba,
        'r4_sgaba': [0.06] * n_syn_sgaba,
        'G_sgaba': G_sgaba,
        'E_sgaba': [-95.0] * n_n,
    }

    times = np.arange(0.0, 0.08, 0.01, dtype=np.float64)
    sim_time = float(times[-1] + (times[1] - times[0]))
    state = build_stable_state_vector(rng, n_n, p_n, l_n, n_syn_ach, n_syn_fgaba, n_syn_sgaba, sim_time)
    current_input = build_stable_current_input(rng, n_n, p_n, times.shape[0])
    thresholds = [0.0] * p_n + [-20.0] * l_n
    return config, current_input, state, times, thresholds


def build_repo_production_case(graph_no=1, odor_seed=59428, trial_seed=1):
    n_n = 120
    p_n = 90
    l_n = 30

    pPNPN = 0.0
    pPNLN = 0.1
    pLNPN = 0.2

    ach_rng = np.random.default_rng(64163 + graph_no)
    ach_mat = np.zeros((n_n, n_n), dtype=np.float64)
    ach_mat[p_n:, :p_n] = ach_rng.choice([0.0, 1.0], size=(l_n, p_n), p=(1 - pPNLN, pPNLN))
    ach_mat[:p_n, :p_n] = ach_rng.choice([0.0, 1.0], size=(p_n, p_n), p=(1 - pPNPN, pPNPN))

    LNPN = np.zeros((p_n, l_n), dtype=np.float64)
    stride = int(p_n / l_n)
    spread = (round(pLNPN * p_n) // 2) * 2 + 1
    center = 0
    index = np.arange(p_n)
    for i in range(l_n):
        idx = index[np.arange(center - spread // 2, 1 + center + spread // 2) % p_n]
        LNPN[idx, i] = 1.0
        center += stride

    lnln = np.loadtxt(ROOT / 'modules' / 'networks' / f'matrix_{graph_no}.csv', delimiter=',')

    fgaba_mat = np.zeros((n_n, n_n), dtype=np.float64)
    fgaba_mat[:p_n, p_n:] = LNPN
    fgaba_mat[p_n:, p_n:] = lnln
    np.fill_diagonal(fgaba_mat, 0.0)

    sgaba_mat = np.zeros((n_n, n_n), dtype=np.float64)
    sgaba_mat[:p_n, p_n:] = LNPN
    np.fill_diagonal(sgaba_mat, 0.0)

    g_ach = np.concatenate([np.zeros(p_n), np.full(l_n, 2 * 90 * 0.5 * 0.1 * 0.05)])
    g_fgaba = np.concatenate([np.full(p_n, 0.3 * 6 * 1.2), np.full(l_n, 30 * 0.2 / 2 * 1.2)])
    G_sgaba = np.concatenate([np.full(p_n, 0.3 * 6 * 0.03), np.zeros(l_n)])

    g_ach = normalize_by_indegree(g_ach, ach_mat)
    g_fgaba = normalize_by_indegree(g_fgaba, fgaba_mat)
    G_sgaba = normalize_by_indegree(G_sgaba, sgaba_mat)

    n_syn_ach = int(np.sum(ach_mat))
    n_syn_fgaba = int(np.sum(fgaba_mat))
    n_syn_sgaba = int(np.sum(sgaba_mat))

    current_rng = np.random.default_rng(graph_no + odor_seed + trial_seed)
    current_input = np.zeros((n_n, 8), dtype=np.float64)
    set_pn = np.concatenate([np.ones(9), np.zeros(81)])
    current_rng.shuffle(set_pn)
    ts = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=np.float64)
    current_input[:p_n, :] = 0.24 * set_pn[:, None] * ts[None, :]
    current_input[p_n:, :] = 0.0735 * ts[None, :]
    current_input += 0.05 * current_input * current_rng.normal(size=current_input.shape) + 0.001 * current_rng.normal(size=current_input.shape)

    sim_time = 0.08
    state_vector = np.array(
        [-45] * p_n + [-45] * l_n
        + [0.5] * (n_n + 4 * p_n + 3 * l_n)
        + [2.4 * (10 ** (-4))] * l_n
        + [0] * (n_syn_ach + n_syn_fgaba + 2 * n_syn_sgaba)
        + [-(sim_time + 1)] * n_n,
        dtype=np.float64,
    )
    state_rng = np.random.default_rng(odor_seed + trial_seed)
    state_vector = state_vector + 0.005 * state_vector * state_rng.normal(size=state_vector.shape)

    config = {
        'n_n': n_n,
        'p_n': p_n,
        'l_n': l_n,
        'C_m': [1.0] * n_n,
        'g_K': [3.6] * p_n + [36.0] * l_n,
        'g_L': [0.3] * n_n,
        'E_K': [-95.0] * n_n,
        'E_L': [-64.0] * p_n + [-50.0] * l_n,
        'g_Na': [7.15] * p_n,
        'g_A': [1.43] * p_n,
        'E_Na': [50.0] * p_n,
        'E_A': [-95.0] * p_n,
        'g_Ca': [5.0] * l_n,
        'g_KCa': [0.045] * l_n,
        'E_Ca': [140.0] * l_n,
        'E_KCa': [-95.0] * l_n,
        'A_Ca': 2e-4,
        'Ca0': 2.4e-4,
        't_Ca': 150.0,
        'ach_mat': ach_mat,
        'alp_ach': [10.0] * n_syn_ach,
        'bet_ach': [0.2] * n_syn_ach,
        't_max': 0.3,
        't_delay': 0.0,
        'A': [0.5] * n_n,
        'g_ach': g_ach,
        'E_ach': [0.0] * n_n,
        'fgaba_mat': fgaba_mat,
        'alp_fgaba': [10.0] * n_syn_fgaba,
        'bet_fgaba': [0.16] * n_syn_fgaba,
        'V0': [-20.0] * n_n,
        'sigma': [1.5] * n_n,
        'g_fgaba': g_fgaba,
        'E_fgaba': [-70.0] * n_n,
        'sgaba_mat': sgaba_mat,
        'K_sgaba': [100e-12] * n_syn_sgaba,
        'r1_sgaba': [1.0] * n_syn_sgaba,
        'r2_sgaba': [0.025] * n_syn_sgaba,
        'r3_sgaba': [0.1] * n_syn_sgaba,
        'r4_sgaba': [0.06] * n_syn_sgaba,
        'G_sgaba': G_sgaba,
        'E_sgaba': [-95.0] * n_n,
    }
    times = np.arange(0.0, sim_time, 0.01, dtype=np.float64)
    thresholds = [0.0] * p_n + [-20.0] * l_n
    return config, current_input, state_vector, times, thresholds


def evaluate(builder, config, current_input, state, times, thresholds):
    with tf.Graph().as_default():
        tf.disable_v2_behavior()
        dXdt = builder(config, current_input)
        state_tensor = tf.constant(state, dtype=tf.float64)
        t_value = tf.constant(times[1], dtype=tf.float64)
        derivative = dXdt(state_tensor, t_value)
        rollout = odeint(dXdt, state_tensor, times, config['n_n'], thresholds)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run([derivative, rollout])


def max_abs_diff(left, right):
    matching_special = (
        (np.isnan(left) & np.isnan(right))
        | (np.isposinf(left) & np.isposinf(right))
        | (np.isneginf(left) & np.isneginf(right))
    )
    valid = ~matching_special
    if not np.any(valid):
        return 0.0
    diff = np.abs(np.nan_to_num(left[valid] - right[valid], nan=np.inf, posinf=np.inf, neginf=np.inf))
    return float(np.max(diff))


def run_case(name, config, current_input, state, times, thresholds):
    legacy_derivative, legacy_rollout = evaluate(legacy_build_dynamics, config, current_input, state, times, thresholds)
    shared_derivative, shared_rollout = evaluate(build_dynamics, config, current_input, state, times, thresholds)

    derivative_close = np.allclose(legacy_derivative, shared_derivative, atol=1e-12, rtol=1e-12, equal_nan=True)
    rollout_close = np.allclose(legacy_rollout, shared_rollout, atol=1e-12, rtol=1e-12, equal_nan=True)

    print(f'{name}: derivative max abs diff = {max_abs_diff(legacy_derivative, shared_derivative):.3e}')
    print(f'{name}: rollout max abs diff = {max_abs_diff(legacy_rollout, shared_rollout):.3e}')

    if not derivative_close or not rollout_close:
        raise SystemExit(f'Parity check failed for {name}')


def main():
    run_case('production_like', *build_case(seed=1, n_n=6, p_n=4, ach_density=0.35, fgaba_density=0.45, sgaba_density=0.25))
    run_case('ln_only_like', *build_case(seed=2, n_n=5, p_n=1, ach_density=0.0, fgaba_density=0.5, sgaba_density=0.0))
    run_case('repo_production_topology', *build_repo_production_case(graph_no=1, odor_seed=59428, trial_seed=1))
    print('Shared TensorFlow core parity checks passed.')


if __name__ == '__main__':
    main()
