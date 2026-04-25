import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)


def K_prop(V):
    V = V - (-50)
    T = 23
    phi = 3.0 ** ((T - 36) / 10)

    alpha_n = 0.02 * (15 - V) / (jnp.exp((15 - V) * 0.2) - 1.0)
    beta_n = 0.5 * jnp.exp((10.0 - V) / 40.0)

    t_n = 1.0 / (alpha_n + beta_n) / phi
    n_inf = alpha_n / (alpha_n + beta_n)

    return n_inf, t_n


def Na_prop(V):
    V = V - (-50)
    T = 23
    phi = 3.0 ** ((T - 36) / 10)

    alpha_m = 0.32 * (13 - V) / (jnp.exp((13 - V) * 0.25) - 1)
    beta_m = 0.28 * (V - 40) / (jnp.exp((V - 40) * 0.2) - 1)

    alpha_h = 0.128 * jnp.exp((17 - V) / 18.0)
    beta_h = 4.0 / (jnp.exp((40 - V) * 0.2) + 1.0)

    t_m = 1.0 / (alpha_m + beta_m) / phi
    t_h = 1.0 / (alpha_h + beta_h) / phi

    m_inf = alpha_m / (alpha_m + beta_m)
    h_inf = alpha_h / (alpha_h + beta_h)

    return m_inf, t_m, h_inf, t_h


def A_prop(V):
    T = 23
    phi = 3.0 ** ((T - 23.5) / 10)

    m_inf = 1 / (1 + jnp.exp(-(V + 60.0) / 8.5))
    h_inf = 1 / (1 + jnp.exp((V + 78.0) / 6.0))

    tau_m = 0.27 / (jnp.exp((V + 35.8) / 19.7) + jnp.exp(-(V + 79.7) / 12.7)) + 0.1

    t1 = 0.27 / (jnp.exp((V + 46.05) / 5.0) + jnp.exp(-(V + 238.4) / 37.45))
    t2 = jnp.ones_like(V) * 5.1
    tau_h = jnp.where(V < -63.0, t1, t2)

    return m_inf, tau_m, h_inf, tau_h


def Ca_prop(V):
    m_0 = 1 / (1 + jnp.exp(-(V + 20.0) / 6.5))
    h_0 = 1 / (1 + jnp.exp((V + 25.0) / 12))

    tau_m = 1.5
    tau_h = 0.3 * jnp.exp((V - 40.0) / 13.0) + 0.002 * jnp.exp((60.0 - V) / 29)

    return m_0, tau_m, h_0, tau_h


def KCa_prop(Ca):
    return Ca / (Ca + 2), 100 / (Ca + 2)


def _dense_from_sparse(values, indices, n_n):
    dense = jnp.zeros((n_n ** 2,), dtype=jnp.float64)
    dense = dense.at[indices].set(values)
    return jnp.transpose(dense.reshape((n_n, n_n)))


def _to_float64(value):
    return jnp.asarray(np.asarray(value, dtype=np.float64), dtype=jnp.float64)


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

    ach_mat = np.asarray(config['ach_mat'], dtype=np.float64)
    ach_mask = ach_mat.reshape(-1) == 1
    ach_indices = jnp.asarray(np.flatnonzero(ach_mask).astype(np.int32), dtype=jnp.int32)
    ach_mat_tensor = _to_float64(ach_mat)
    n_syn_ach = int(np.sum(ach_mat))
    alp_ach = _to_float64(config['alp_ach'])
    bet_ach = _to_float64(config['bet_ach'])
    t_max = float(config['t_max'])
    t_delay = float(config['t_delay'])
    A = _to_float64(config['A'])
    g_ach = _to_float64(config['g_ach'])
    E_ach = _to_float64(config['E_ach'])

    fgaba_mat = np.asarray(config['fgaba_mat'], dtype=np.float64)
    fgaba_mask = fgaba_mat.reshape(-1) == 1
    fgaba_indices = jnp.asarray(np.flatnonzero(fgaba_mask).astype(np.int32), dtype=jnp.int32)
    fgaba_mat_tensor = _to_float64(fgaba_mat)
    n_syn_fgaba = int(np.sum(fgaba_mat))
    alp_fgaba = _to_float64(config['alp_fgaba'])
    bet_fgaba = _to_float64(config['bet_fgaba'])
    V0 = _to_float64(config['V0'])
    sigma = _to_float64(config['sigma'])
    g_fgaba = _to_float64(config['g_fgaba'])
    E_fgaba = _to_float64(config['E_fgaba'])

    sgaba_mat = np.asarray(config['sgaba_mat'], dtype=np.float64)
    sgaba_mask = sgaba_mat.reshape(-1) == 1
    sgaba_indices = jnp.asarray(np.flatnonzero(sgaba_mask).astype(np.int32), dtype=jnp.int32)
    sgaba_mat_tensor = _to_float64(sgaba_mat)
    n_syn_sgaba = int(np.sum(sgaba_mat))
    K_sgaba = _to_float64(config['K_sgaba'])
    r1_sgaba = _to_float64(config['r1_sgaba'])
    r2_sgaba = _to_float64(config['r2_sgaba'])
    r3_sgaba = _to_float64(config['r3_sgaba'])
    r4_sgaba = _to_float64(config['r4_sgaba'])
    G_sgaba = _to_float64(config['G_sgaba'])
    E_sgaba = _to_float64(config['E_sgaba'])

    current_input_tensor = jnp.asarray(np.asarray(current_input, dtype=np.float64).T, dtype=jnp.float64)

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
        o_ = _dense_from_sparse(o, ach_indices, n_n)
        return jnp.sum(jnp.transpose((o_ * (V - E_ach)) * g_ach), axis=1)

    def I_fgaba(o, V):
        o_ = _dense_from_sparse(o, fgaba_indices, n_n)
        return jnp.sum(jnp.transpose((o_ * (V - E_fgaba)) * g_fgaba), axis=1)

    def I_sgaba(G, V):
        G4 = jnp.power(G, 4) / (jnp.power(G, 4) + K_sgaba)
        G_ = _dense_from_sparse(G4, sgaba_indices, n_n)
        return jnp.sum(jnp.transpose((G_ * (V - E_sgaba)) * G_sgaba), axis=1)

    def I_inj_t(t, V):
        return current_input_tensor[jnp.asarray(t * 100, dtype=jnp.int32)] * (V - E_ach)

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
        CmdV = jnp.concatenate([CmdV_p, CmdV_l], axis=0)

        dV = (-I_inj_t(t, V) + CmdV - I_K(V, n_K) - I_L(V) - I_ach(o_ach, V) - I_fgaba(o_fgaba, V) - I_sgaba(g_sgaba, V)) / C_m

        T_ach = jnp.where(
            jnp.logical_and(t > fire_t + t_delay, t < fire_t + t_max + t_delay),
            A,
            jnp.zeros_like(A),
        )
        T_ach = (ach_mat_tensor * T_ach).reshape(-1)[ach_indices]
        do_achdt = alp_ach * (1.0 - o_ach) * T_ach - bet_ach * o_ach

        T_fgaba = 1.0 / (1.0 + jnp.exp(-(V - V0) / sigma))
        T_fgaba = (fgaba_mat_tensor * T_fgaba).reshape(-1)[fgaba_indices]
        do_fgabadt = alp_fgaba * (1.0 - o_fgaba) * T_fgaba - bet_fgaba * o_fgaba

        dg_sgabadt = -r4_sgaba * g_sgaba + r3_sgaba * r_sgaba

        T_sgaba = jnp.where(
            jnp.logical_and(t > fire_t + t_delay, t < fire_t + t_max + t_delay),
            A,
            jnp.zeros_like(A),
        )
        T_sgaba = (sgaba_mat_tensor * T_sgaba).reshape(-1)[sgaba_indices]
        dr_sgabadt = r1_sgaba * (1.0 - r_sgaba) * T_sgaba - r2_sgaba * r_sgaba

        dfdt = jnp.zeros_like(fire_t)

        return jnp.concatenate([
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
        ], axis=0)

    return dXdt