import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


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


def _to_float64(value):
    return tf.constant(np.asarray(value, dtype=np.float64), dtype=tf.float64)


def _to_int32(value):
    return tf.constant(np.asarray(value, dtype=np.int32), dtype=tf.int32)


def _sum_synaptic_current(values, row_ids, V, reversal, conductance, n_n):
    incoming = tf.math.unsorted_segment_sum(values, row_ids, n_n)
    return incoming * (V - reversal) * conductance


def _input_index_at_time(t, input_scale, input_steps):
    index = tf.cast(t * input_scale, tf.int32)
    return tf.clip_by_value(index, 0, input_steps - 1)


def _get_synapse_layout(config, prefix, n_n):
    flat_indices = config.get(f'{prefix}_indices')
    if flat_indices is None:
        matrix = np.asarray(config[f'{prefix}_mat'], dtype=np.float64)
        flat_indices = np.flatnonzero(matrix.reshape(-1) != 0.0).astype(np.int32, copy=False)
    else:
        flat_indices = np.asarray(flat_indices, dtype=np.int32)
    row_ids = np.asarray(config.get(f'{prefix}_row_ids', flat_indices // n_n), dtype=np.int32)
    col_ids = np.asarray(config.get(f'{prefix}_col_ids', flat_indices % n_n), dtype=np.int32)
    return int(flat_indices.size), _to_int32(row_ids), _to_int32(col_ids)


def build_dynamics(config, current_input):
    n_n = int(config['n_n'])
    p_n = int(config['p_n'])
    l_n = int(config['l_n'])

    C_m = _to_float64(config['C_m'])
    g_K = _to_float64(config['g_K'])
    g_L = _to_float64(config['g_L'])
    E_K = _to_float64(config['E_K'])
    E_L = _to_float64(config['E_L'])
    g_Na = _to_float64(config['g_Na'])
    g_A = _to_float64(config['g_A'])
    E_Na = _to_float64(config['E_Na'])
    E_A = _to_float64(config['E_A'])
    g_Ca = _to_float64(config['g_Ca'])
    g_KCa = _to_float64(config['g_KCa'])
    E_Ca = _to_float64(config['E_Ca'])
    E_KCa = _to_float64(config['E_KCa'])
    A_Ca = float(config['A_Ca'])
    Ca0 = float(config['Ca0'])
    t_Ca = float(config['t_Ca'])
    input_scale = float(config.get('input_scale', 100.0))

    n_syn_ach, ach_row_ids, ach_col_ids = _get_synapse_layout(config, 'ach', n_n)
    alp_ach = _to_float64(config['alp_ach'])
    bet_ach = _to_float64(config['bet_ach'])
    t_max = float(config['t_max'])
    t_delay = float(config['t_delay'])
    A = _to_float64(config['A'])
    g_ach = _to_float64(config['g_ach'])
    E_ach = _to_float64(config['E_ach'])

    n_syn_fgaba, fgaba_row_ids, fgaba_col_ids = _get_synapse_layout(config, 'fgaba', n_n)
    alp_fgaba = _to_float64(config['alp_fgaba'])
    bet_fgaba = _to_float64(config['bet_fgaba'])
    V0 = _to_float64(config['V0'])
    sigma = _to_float64(config['sigma'])
    g_fgaba = _to_float64(config['g_fgaba'])
    E_fgaba = _to_float64(config['E_fgaba'])

    n_syn_sgaba, sgaba_row_ids, sgaba_col_ids = _get_synapse_layout(config, 'sgaba', n_n)
    K_sgaba = _to_float64(config['K_sgaba'])
    r1_sgaba = _to_float64(config['r1_sgaba'])
    r2_sgaba = _to_float64(config['r2_sgaba'])
    r3_sgaba = _to_float64(config['r3_sgaba'])
    r4_sgaba = _to_float64(config['r4_sgaba'])
    G_sgaba = _to_float64(config['G_sgaba'])
    E_sgaba = _to_float64(config['E_sgaba'])

    current_input_tensor = tf.constant(np.asarray(current_input, dtype=np.float64).T, dtype=tf.float64)

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
        return _sum_synaptic_current(o, ach_row_ids, V, E_ach, g_ach, n_n)

    def I_fgaba(o, V):
        return _sum_synaptic_current(o, fgaba_row_ids, V, E_fgaba, g_fgaba, n_n)

    def I_sgaba(G, V):
        G4 = tf.pow(G, 4) / (tf.pow(G, 4) + K_sgaba)
        return _sum_synaptic_current(G4, sgaba_row_ids, V, E_sgaba, G_sgaba, n_n)

    def I_inj_t(t, V):
        index = _input_index_at_time(t, input_scale, tf.shape(current_input_tensor)[0])
        return current_input_tensor[index] * (V - E_ach)

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

        T_ach = tf.where(
            tf.logical_and(tf.greater(t, fire_t + t_delay), tf.less(t, fire_t + t_max + t_delay)),
            A,
            tf.zeros(tf.shape(A), dtype=A.dtype),
        )
        T_ach = tf.gather(T_ach, ach_col_ids)
        do_achdt = alp_ach * (1.0 - o_ach) * T_ach - bet_ach * o_ach

        T_fgaba = 1.0 / (1.0 + tf.exp(-(V - V0) / sigma))
        T_fgaba = tf.gather(T_fgaba, fgaba_col_ids)
        do_fgabadt = alp_fgaba * (1.0 - o_fgaba) * T_fgaba - bet_fgaba * o_fgaba

        dg_sgabadt = -r4_sgaba * g_sgaba + r3_sgaba * r_sgaba

        T_sgaba = tf.where(
            tf.logical_and(tf.greater(t, fire_t + t_delay), tf.less(t, fire_t + t_max + t_delay)),
            A,
            tf.zeros(tf.shape(A), dtype=A.dtype),
        )
        T_sgaba = tf.gather(T_sgaba, sgaba_col_ids)
        dr_sgabadt = r1_sgaba * (1.0 - r_sgaba) * T_sgaba - r2_sgaba * r_sgaba

        dfdt = tf.zeros(tf.shape(fire_t), dtype=fire_t.dtype)

        return tf.concat([
            dV,
            dn_k,
            dm_Na,
            dh_Na,
            dm_A,
            dh_A,
            dm_Ca,
            dh_Ca,
            dm_KCa,
            dCa,
            do_achdt,
            do_fgabadt,
            dr_sgabadt,
            dg_sgabadt,
            dfdt,
        ], 0)

    return dXdt