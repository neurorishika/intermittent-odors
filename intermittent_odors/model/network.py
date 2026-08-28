"""NetworkModel, custom dynamics builder, and pnln_network factory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from intermittent_odors.backends.jax_network import (
    A_prop, Ca_prop, K_prop, KCa_prop, Na_prop,
    _get_synapse_layout, _input_index_at_time, _sum_synaptic_current,
    _to_jax_array, build_dynamics_core,
)

from .channels import (
    ChannelSpec, _GATE_DEFAULTS,
    K_channel, Na_channel, A_channel, Ca_channel, KCa_channel,
)
from .synapses import SynapseSpec, ach_synapse, fgaba_synapse, sgaba_synapse
from .topology import Population, Projection, StateVectorLayout

# ---------------------------------------------------------------------------
# NetworkModel
# ---------------------------------------------------------------------------

@dataclass
class NetworkModel:
    """Composable HH network model.

    Build the standard PN/LN paper model via ``pnln_network()``, which ensures
    bit-for-bit numerical parity with previously published results.  For custom
    network designs, construct this class directly and call ``compile()``.

    Parameters
    ----------
    populations : list of Population
        Defined in order; index 0 is first block in the voltage section.
    projections : list of Projection
        Synaptic connections; order determines synapse-state block layout.
    global_params : dict
        Network-wide parameters (e.g., ``input_scale``, ``t_max``).
    """

    populations: list
    projections: list
    global_params: dict = field(default_factory=dict)

    # ---- Internal flags for the standard PN/LN path -------------------------
    # When True, build_dynamics_fn() delegates to the unchanged build_dynamics_core.
    _is_standard_pnln: bool = field(default=False, repr=False, compare=False)
    # If set, to_config_dict() calls this rather than building from populations.
    _config_builder: Callable | None = field(default=None, repr=False, compare=False)

    # ---- Basic properties ---------------------------------------------------

    @property
    def n_n(self) -> int:
        return sum(p.n for p in self.populations)

    def population(self, name: str) -> Population:
        for p in self.populations:
            if p.name == name:
                return p
        raise KeyError(f"No population named {name!r}")

    def population_offset(self, name: str) -> int:
        offset = 0
        for p in self.populations:
            if p.name == name:
                return offset
            offset += p.n
        raise KeyError(f"No population named {name!r}")

    def fire_thresholds(self) -> np.ndarray:
        """Return the per-neuron spike-detection threshold array."""
        return np.concatenate([
            np.full(p.n, p.threshold, dtype=np.float64)
            for p in self.populations
        ])

    # ---- State-vector layout ------------------------------------------------

    @property
    def layout(self) -> StateVectorLayout:
        """Compute and return the state-vector layout for this network.

        For the standard PN/LN model the returned layout matches the existing
        ``build_dynamics_core`` layout exactly so that ``default_state_vector``
        initialises the vector in the right slots.
        """
        return self._compute_layout()

    def _compute_layout(self) -> StateVectorLayout:
        n_n = self.n_n
        sections: dict[str, slice] = {}
        offset = 0

        # ---- Section 1: voltages (all populations concatenated) -------------
        sections["V"] = slice(offset, offset + n_n)
        offset += n_n

        # ---- Section 2: channel gates and extra states -----------------------
        # Iterate populations in declaration order.
        for pop in self.populations:
            for ch in pop.channels:
                for gate in ch.gates():
                    key = f"{pop.name}.{ch.kind}.{gate}"
                    sections[key] = slice(offset, offset + pop.n)
                    offset += pop.n
                if ch.has_extra_state():
                    key = f"{pop.name}.{ch.kind}.extra"
                    sections[key] = slice(offset, offset + pop.n)
                    offset += pop.n

        # ---- Section 3: synapse states (per projection) ---------------------
        for proj in self.projections:
            mat = np.asarray(proj.matrix, dtype=np.float64)
            n_syn = int(np.sum(mat != 0))
            kin = proj.synapse.kinetics
            tag = f"{proj.src}->{proj.dst}.{kin}"
            if kin == "sgaba":
                sections[f"syn.{tag}.r"] = slice(offset, offset + n_syn)
                offset += n_syn
                sections[f"syn.{tag}.g"] = slice(offset, offset + n_syn)
                offset += n_syn
            else:
                sections[f"syn.{tag}"] = slice(offset, offset + n_syn)
                offset += n_syn

        # ---- Section 4: fire times ------------------------------------------
        sections["fire_t"] = slice(offset, offset + n_n)
        offset += n_n

        return StateVectorLayout(n_n=n_n, sections=sections, state_size=offset)

    # ---- State-vector initialisation ----------------------------------------

    def default_state_vector(
        self,
        rng_or_seed: Any,
        sim_time: float,
        *,
        noise_scale: float = 0.0,
    ) -> np.ndarray:
        """Build a default initial state vector.

        Parameters
        ----------
        rng_or_seed : int or numpy.random.Generator
            Seed or RNG used for optional noise perturbation.
        sim_time : float
            Total simulation time (ms).  Sets fire_t to ``-(sim_time + 1)``.
        noise_scale : float
            Fractional noise added to gate variables (0 = no noise).
        """
        rng = (
            rng_or_seed
            if isinstance(rng_or_seed, np.random.Generator)
            else np.random.default_rng(int(rng_or_seed))
        )

        layout = self.layout
        sv = np.zeros(layout.state_size, dtype=np.float64)

        # Voltages
        v_offset = 0
        for pop in self.populations:
            sv[v_offset:v_offset + pop.n] = -45.0
            v_offset += pop.n

        # Gate variables and extra states
        for pop in self.populations:
            for ch in pop.channels:
                gate_defaults = _GATE_DEFAULTS.get(ch.kind, {})
                for gate in ch.gates():
                    key = f"{pop.name}.{ch.kind}.{gate}"
                    sl = layout.get(key)
                    default = gate_defaults.get(gate, 0.5)
                    sv[sl] = default
                if ch.has_extra_state():
                    key = f"{pop.name}.{ch.kind}.extra"
                    sl = layout.get(key)
                    sv[sl] = ch.extra_state_default()

        # Synapse states: 0
        for proj in self.projections:
            kin = proj.synapse.kinetics
            tag = f"{proj.src}->{proj.dst}.{kin}"
            if kin == "sgaba":
                sv[layout.get(f"syn.{tag}.r")] = 0.0
                sv[layout.get(f"syn.{tag}.g")] = 0.0
            else:
                sv[layout.get(f"syn.{tag}")] = 0.0

        # Fire times
        sv[layout.get("fire_t")] = -(sim_time + 1.0)

        # Optional noise
        if noise_scale > 0.0:
            sv += noise_scale * sv * rng.normal(size=sv.shape)

        return sv

    # ---- Config dict (standard PN/LN path) ----------------------------------

    def to_config_dict(self) -> dict:
        """Return the flat config dict consumed by ``build_dynamics_core``.

        For standard PN/LN networks created via ``pnln_network()`` this dict
        is identical to what ``benchmark_cases.build_case`` produces.  For
        custom networks an equivalent dict is built from the population and
        projection descriptors.
        """
        if self._config_builder is not None:
            return self._config_builder()
        return self._build_config_dict_from_descriptors()

    def _build_config_dict_from_descriptors(self) -> dict:
        """Build a config dict from channel/synapse descriptors.

        This produces the same dict format as the existing builders for the
        standard PN/LN model.  For truly custom models the dict is a canonical
        representation used only to compute the ``model_digest``; the actual
        dynamics come from ``build_dynamics_fn``.
        """
        n_n = self.n_n

        # ----- Intrinsic channel params per neuron --------------------------
        C_m = np.concatenate([np.full(p.n, p.C_m) for p in self.populations])
        g_L = np.concatenate([np.full(p.n, p.g_L) for p in self.populations])
        E_L = np.concatenate([np.full(p.n, p.E_L) for p in self.populations])

        def _pop_param(kind, key, default=0.0):
            parts = []
            for pop in self.populations:
                if pop.has_channel(kind):
                    ch = pop.channel(kind)
                    val = ch.extra_params.get(key, getattr(ch, key, default))
                    parts.append(np.full(pop.n, float(val)))
                else:
                    parts.append(np.zeros(pop.n))
            return np.concatenate(parts) if parts else np.array([], dtype=np.float64)

        def _pop_conductance(kind):
            parts = []
            for pop in self.populations:
                if pop.has_channel(kind):
                    g = pop.channel(kind).conductance
                    parts.append(np.broadcast_to(np.atleast_1d(np.asarray(g, np.float64)), (pop.n,)).copy())
                else:
                    parts.append(np.zeros(pop.n))
            return np.concatenate(parts)

        def _pop_reversal(kind):
            parts = []
            for pop in self.populations:
                if pop.has_channel(kind):
                    E = pop.channel(kind).reversal
                    parts.append(np.broadcast_to(np.atleast_1d(np.asarray(E, np.float64)), (pop.n,)).copy())
                else:
                    parts.append(np.zeros(pop.n))
            return np.concatenate(parts)

        # Determine which populations have which channels
        # (only include non-zero entries in the config arrays)
        g_K = _pop_conductance("K")
        E_K = _pop_reversal("K")
        g_Na = np.concatenate([
            np.full(p.n, float(p.channel("Na").conductance))
            if p.has_channel("Na") else np.zeros(0)
            for p in self.populations
        ])
        E_Na = np.concatenate([
            np.full(p.n, float(p.channel("Na").reversal))
            if p.has_channel("Na") else np.zeros(0)
            for p in self.populations
        ])
        g_A = np.concatenate([
            np.full(p.n, float(p.channel("A").conductance))
            if p.has_channel("A") else np.zeros(0)
            for p in self.populations
        ])
        E_A = np.concatenate([
            np.full(p.n, float(p.channel("A").reversal))
            if p.has_channel("A") else np.zeros(0)
            for p in self.populations
        ])
        g_Ca = np.concatenate([
            np.full(p.n, float(p.channel("Ca").conductance))
            if p.has_channel("Ca") else np.zeros(0)
            for p in self.populations
        ])
        E_Ca = np.concatenate([
            np.full(p.n, float(p.channel("Ca").reversal))
            if p.has_channel("Ca") else np.zeros(0)
            for p in self.populations
        ])
        g_KCa = np.concatenate([
            np.full(p.n, float(p.channel("KCa").conductance))
            if p.has_channel("KCa") else np.zeros(0)
            for p in self.populations
        ])
        E_KCa = np.concatenate([
            np.full(p.n, float(p.channel("KCa").reversal))
            if p.has_channel("KCa") else np.zeros(0)
            for p in self.populations
        ])

        # Ca channel scalars (take from first population with Ca)
        A_Ca = Ca0 = t_Ca = None
        for pop in self.populations:
            if pop.has_channel("Ca"):
                ch = pop.channel("Ca")
                A_Ca = ch.extra_params.get("A_Ca", 2e-4)
                Ca0 = ch.extra_params.get("Ca0", 2.4e-4)
                t_Ca = ch.extra_params.get("t_Ca", 150.0)
                break
        if A_Ca is None:
            A_Ca, Ca0, t_Ca = 2e-4, 2.4e-4, 150.0

        # ----- Synapse params -----------------------------------------------
        def _proj_by_kin(kin: str) -> list:
            return [p for p in self.projections if p.synapse.kinetics == kin]

        def _concat_syn_param(projlist, key, default):
            parts = []
            for proj in projlist:
                mat = np.asarray(proj.matrix, dtype=np.float64)
                n_syn = int(np.sum(mat != 0))
                val = proj.synapse.params.get(key, default)
                parts.append(np.full(n_syn, float(val)))
            return np.concatenate(parts) if parts else np.zeros(0)

        def _concat_mat(projlist) -> np.ndarray:
            # Sum matrices (assumes non-overlapping; each projection has its own matrix)
            if not projlist:
                return np.zeros((n_n, n_n), dtype=np.float64)
            result = np.zeros((n_n, n_n), dtype=np.float64)
            for proj in projlist:
                result += np.asarray(proj.matrix, dtype=np.float64)
            return result

        def _concat_conductance_per_dst(projlist) -> np.ndarray:
            """Conductance per destination neuron (max across projections for same kin)."""
            if not projlist:
                return np.zeros(n_n, dtype=np.float64)
            result = np.zeros(n_n, dtype=np.float64)
            for proj in projlist:
                g = np.asarray(proj.synapse.conductance, dtype=np.float64)
                if g.ndim == 0:
                    result += float(g)
                else:
                    result += np.broadcast_to(g, (n_n,))
            return result

        ach_projs   = _proj_by_kin("ach")
        fgaba_projs = _proj_by_kin("fgaba")
        sgaba_projs = _proj_by_kin("sgaba")

        ach_mat   = _concat_mat(ach_projs)
        fgaba_mat = _concat_mat(fgaba_projs)
        sgaba_mat = _concat_mat(sgaba_projs)

        n_syn_ach   = int(np.sum(ach_mat   != 0))
        n_syn_fgaba = int(np.sum(fgaba_mat != 0))
        n_syn_sgaba = int(np.sum(sgaba_mat != 0))

        # Global t_max / t_delay come from ach params or global_params
        t_max   = self.global_params.get("t_max",   0.3)
        t_delay = self.global_params.get("t_delay", 0.0)
        A_val   = self.global_params.get("A",       0.5)
        input_scale = self.global_params.get("input_scale", 100.0)

        if ach_projs:
            t_max   = ach_projs[0].synapse.params.get("t_max",   t_max)
            t_delay = ach_projs[0].synapse.params.get("t_delay", t_delay)
            A_val   = ach_projs[0].synapse.params.get("A",       A_val)

        E_ach_arr = np.concatenate([
            np.full(p.n, float(
                next((proj.synapse.reversal for proj in ach_projs), 0.0)
                if isinstance(next((proj.synapse.reversal for proj in ach_projs), 0.0), (int, float))
                else 0.0
            ))
            for p in self.populations
        ])

        E_fgaba_arr = np.concatenate([
            np.full(p.n, float(
                next((proj.synapse.reversal for proj in fgaba_projs), -70.0)
                if isinstance(next((proj.synapse.reversal for proj in fgaba_projs), -70.0), (int, float))
                else -70.0
            ))
            for p in self.populations
        ])

        E_sgaba_arr = np.concatenate([
            np.full(p.n, float(
                next((proj.synapse.reversal for proj in sgaba_projs), -95.0)
                if isinstance(next((proj.synapse.reversal for proj in sgaba_projs), -95.0), (int, float))
                else -95.0
            ))
            for p in self.populations
        ])

        V0_arr    = np.full(n_n, float(fgaba_projs[0].synapse.params.get("V0",    -20.0)) if fgaba_projs else -20.0)
        sigma_arr = np.full(n_n, float(fgaba_projs[0].synapse.params.get("sigma",  1.5)) if fgaba_projs else 1.5)

        # Conductances (already per dst neuron from to_config_dict callers)
        g_ach_arr   = _concat_conductance_per_dst(ach_projs)
        g_fgaba_arr = _concat_conductance_per_dst(fgaba_projs)
        G_sgaba_arr = _concat_conductance_per_dst(sgaba_projs)

        # p_n and l_n: counts of neurons with Na channel vs Ca channel
        p_n = sum(p.n for p in self.populations if p.has_channel("Na"))
        l_n = sum(p.n for p in self.populations if p.has_channel("Ca"))

        return {
            "n_n": n_n,
            "p_n": p_n,
            "l_n": l_n,
            "C_m": C_m.tolist(),
            "g_K": g_K.tolist(),
            "g_L": g_L.tolist(),
            "E_K": E_K.tolist(),
            "E_L": E_L.tolist(),
            "g_Na": g_Na.tolist(),
            "g_A":  g_A.tolist(),
            "E_Na": E_Na.tolist(),
            "E_A":  E_A.tolist(),
            "g_Ca": g_Ca.tolist(),
            "g_KCa": g_KCa.tolist(),
            "E_Ca": E_Ca.tolist(),
            "E_KCa": E_KCa.tolist(),
            "A_Ca": A_Ca,
            "Ca0": Ca0,
            "t_Ca": t_Ca,
            "input_scale": input_scale,
            "ach_mat": ach_mat,
            "alp_ach": _concat_syn_param(ach_projs, "alp", 10.0).tolist(),
            "bet_ach": _concat_syn_param(ach_projs, "bet", 0.2).tolist(),
            "t_max": t_max,
            "t_delay": t_delay,
            "A": np.full(n_n, float(A_val)).tolist(),
            "g_ach": g_ach_arr.tolist(),
            "E_ach": E_ach_arr.tolist(),
            "fgaba_mat": fgaba_mat,
            "alp_fgaba": _concat_syn_param(fgaba_projs, "alp", 10.0).tolist(),
            "bet_fgaba": _concat_syn_param(fgaba_projs, "bet", 0.16).tolist(),
            "V0": V0_arr.tolist(),
            "sigma": sigma_arr.tolist(),
            "g_fgaba": g_fgaba_arr.tolist(),
            "E_fgaba": E_fgaba_arr.tolist(),
            "sgaba_mat": sgaba_mat,
            "K_sgaba": _concat_syn_param(sgaba_projs, "K", 100e-12).tolist(),
            "r1_sgaba": _concat_syn_param(sgaba_projs, "r1", 1.0).tolist(),
            "r2_sgaba": _concat_syn_param(sgaba_projs, "r2", 0.025).tolist(),
            "r3_sgaba": _concat_syn_param(sgaba_projs, "r3", 0.1).tolist(),
            "r4_sgaba": _concat_syn_param(sgaba_projs, "r4", 0.06).tolist(),
            "G_sgaba": G_sgaba_arr.tolist(),
            "E_sgaba": E_sgaba_arr.tolist(),
        }

    # ---- Dynamics builder ---------------------------------------------------

    def build_dynamics_fn(self, config: dict) -> Callable:
        """Return ``dXdt(X, t, current_input_tensor)`` for this network.

        For standard PN/LN networks (created via ``pnln_network()``) this
        delegates to the unchanged ``build_dynamics_core`` so that outputs are
        numerically identical to the original model.  For custom networks a
        new ODE function is synthesised from the channel descriptors.
        """
        if self._is_standard_pnln:
            return build_dynamics_core(config)
        return _build_custom_dynamics(self, config)

    # ---- Convenience: ExperimentSpec assembly -------------------------------

    def to_experiment_spec(
        self,
        current_input: Any,
        state_vector: Any,
        times: Any,
        thresholds: Any | None = None,
        *,
        config: dict | None = None,
        input_dt: float | None = None,
        sample_stride: int = 1,
        sample_neurons: int | None = None,
        time_batches: Any = None,
        metadata: dict | None = None,
    ):
        """Assemble an ``ExperimentSpec`` from this network model.

        For standard PN/LN networks the spec is identical (same config dict,
        same dynamics path) to specs produced by ``build_fig2_experiment_spec``
        or ``trial_case_to_experiment_spec``.

        Parameters
        ----------
        current_input : array-like, shape (n_n, T)
        state_vector  : array-like, flat initial state
        times         : array-like, time points
        thresholds    : array-like, optional – defaults to ``fire_thresholds()``
        config        : dict, optional – use a prebuilt config dict instead of
                        calling ``to_config_dict()``.
        """
        from intermittent_odors.experiment import build_experiment_spec

        cfg = config if config is not None else self.to_config_dict()
        thr = thresholds if thresholds is not None else self.fire_thresholds()

        # For custom models we pass the dynamics builder so the runtime uses it
        dynamics_builder = None if self._is_standard_pnln else self.build_dynamics_fn

        n_n = self.n_n
        s_neurons = sample_neurons if sample_neurons is not None else n_n

        spec = build_experiment_spec(
            cfg,
            np.asarray(current_input, dtype=np.float64),
            np.asarray(state_vector, dtype=np.float64),
            np.asarray(times, dtype=np.float64),
            thr,
            input_dt=input_dt,
            sample_stride=sample_stride,
            sample_neurons=s_neurons,
            time_batches=time_batches,
            metadata={} if metadata is None else metadata,
        )

        if dynamics_builder is not None:
            # Return a spec with the dynamics_builder attached
            from dataclasses import replace

            prepared = spec.prepare()
            # We can't use replace() on a frozen dataclass with a computed field like model_digest
            # Instead we carry dynamics_builder in the ExperimentSpec metadata
            # and pass it through to PreparedExperiment via a wrapper.
            return _CustomDynamicsExperimentSpec(spec, dynamics_builder)

        return spec


# ---------------------------------------------------------------------------
# Custom dynamics ODE generator
# ---------------------------------------------------------------------------

def _build_custom_dynamics(model: NetworkModel, config: dict) -> Callable:
    """Generate a ``dXdt(X, t, current_input_tensor)`` for an arbitrary network.

    The state-vector layout is the one returned by ``model.layout``; callers
    must ensure the initial state vector was built using the same model.
    """
    layout = model.layout
    n_n = model.n_n
    input_scale = _to_jax_array(config.get("input_scale", 100.0))

    # --- Channel parameters (captured as JAX arrays) -------------------------
    pop_params = {}
    for pop in model.populations:
        p_offset = model.population_offset(pop.name)
        p_slice = slice(p_offset, p_offset + pop.n)
        C_m = _to_jax_array(np.full(pop.n, pop.C_m))
        g_L = _to_jax_array(np.full(pop.n, pop.g_L))
        E_L = _to_jax_array(np.full(pop.n, pop.E_L))
        channels = {}
        for ch in pop.channels:
            g = _to_jax_array(np.broadcast_to(np.atleast_1d(np.asarray(ch.conductance, np.float64)), (pop.n,)))
            E = _to_jax_array(np.broadcast_to(np.atleast_1d(np.asarray(ch.reversal,    np.float64)), (pop.n,)))
            extra = {k: float(v) for k, v in ch.extra_params.items()}
            channels[ch.kind] = {"g": g, "E": E, **extra, "props_fn": ch.get_props_fn()}
        pop_params[pop.name] = {"V_slice": p_slice, "n": pop.n, "C_m": C_m, "g_L": g_L, "E_L": E_L, "channels": channels}

    # --- Synapse parameters --------------------------------------------------
    t_max   = float(config.get("t_max",   0.3))
    t_delay = float(config.get("t_delay", 0.0))
    A_vec   = _to_jax_array(config.get("A", [0.5] * n_n))

    # E_input for injected current (use ach reversal if present, else 0)
    E_input = _to_jax_array(np.zeros(n_n))
    ach_projs   = [p for p in model.projections if p.synapse.kinetics == "ach"]
    fgaba_projs = [p for p in model.projections if p.synapse.kinetics == "fgaba"]
    sgaba_projs = [p for p in model.projections if p.synapse.kinetics == "sgaba"]

    # Pre-compute row/col ids for each projection
    def _proj_layout(proj):
        mat = np.asarray(proj.matrix, dtype=np.float64)
        flat_idx = np.flatnonzero(mat.reshape(-1) != 0).astype(np.int32)
        row_ids = jnp.asarray((flat_idx // n_n).astype(np.int32))
        col_ids = jnp.asarray((flat_idx %  n_n).astype(np.int32))
        n_syn = int(flat_idx.size)
        return n_syn, row_ids, col_ids

    ach_layouts   = [_proj_layout(p) for p in ach_projs]
    fgaba_layouts = [_proj_layout(p) for p in fgaba_projs]
    sgaba_layouts = [_proj_layout(p) for p in sgaba_projs]

    def _ach_params(proj):
        n_syn = int(np.sum(np.asarray(proj.matrix) != 0))
        alp = _to_jax_array(np.full(n_syn, proj.synapse.params.get("alp", 10.0)))
        bet = _to_jax_array(np.full(n_syn, proj.synapse.params.get("bet", 0.2)))
        g   = _to_jax_array(np.asarray(proj.synapse.conductance, np.float64))
        E   = _to_jax_array(np.full(n_n, float(proj.synapse.reversal) if not hasattr(proj.synapse.reversal, "__len__") else proj.synapse.reversal[0]))
        return alp, bet, g, E

    def _fgaba_params(proj):
        n_syn = int(np.sum(np.asarray(proj.matrix) != 0))
        alp   = _to_jax_array(np.full(n_syn, proj.synapse.params.get("alp",   10.0)))
        bet   = _to_jax_array(np.full(n_syn, proj.synapse.params.get("bet",   0.16)))
        V0    = _to_jax_array(np.full(n_n,  proj.synapse.params.get("V0",   -20.0)))
        sigma = _to_jax_array(np.full(n_n,  proj.synapse.params.get("sigma",  1.5)))
        g     = _to_jax_array(np.asarray(proj.synapse.conductance, np.float64))
        E     = _to_jax_array(np.full(n_n, float(proj.synapse.reversal) if not hasattr(proj.synapse.reversal, "__len__") else proj.synapse.reversal[0]))
        return alp, bet, V0, sigma, g, E

    def _sgaba_params(proj):
        n_syn = int(np.sum(np.asarray(proj.matrix) != 0))
        K  = _to_jax_array(np.full(n_syn, proj.synapse.params.get("K",  100e-12)))
        r1 = _to_jax_array(np.full(n_syn, proj.synapse.params.get("r1", 1.0)))
        r2 = _to_jax_array(np.full(n_syn, proj.synapse.params.get("r2", 0.025)))
        r3 = _to_jax_array(np.full(n_syn, proj.synapse.params.get("r3", 0.1)))
        r4 = _to_jax_array(np.full(n_syn, proj.synapse.params.get("r4", 0.06)))
        G  = _to_jax_array(np.asarray(proj.synapse.conductance, np.float64))
        E  = _to_jax_array(np.full(n_n, float(proj.synapse.reversal) if not hasattr(proj.synapse.reversal, "__len__") else proj.synapse.reversal[0]))
        return K, r1, r2, r3, r4, G, E

    ach_syn_params   = [_ach_params(p)   for p in ach_projs]
    fgaba_syn_params = [_fgaba_params(p) for p in fgaba_projs]
    sgaba_syn_params = [_sgaba_params(p) for p in sgaba_projs]

    def dXdt(X, t, current_input_tensor):  # noqa: N802
        V = X[layout.get("V")]
        fire_t = X[layout.get("fire_t")]

        # --- Channel gate derivatives and intrinsic currents per population --
        dgate_parts = []   # (derivative, slice_in_state)
        I_channel = jnp.zeros(n_n)

        for pop in model.populations:
            pp = pop_params[pop.name]
            V_p = V[pp["V_slice"]]

            # Leak
            I_L_pop = pp["g_L"] * (V_p - pp["E_L"])

            dV_channel_contribution = -I_L_pop

            for ch in pop.channels:
                cp = pp["channels"][ch.kind]

                if ch.kind == "K":
                    n_gate_key = f"{pop.name}.K.n"
                    n_gate = X[layout.get(n_gate_key)]
                    n0, tn = K_prop(V_p)
                    dn = -(1.0 / tn) * (n_gate - n0)
                    I_K_pop = cp["g"] * n_gate ** 4 * (V_p - cp["E"])
                    dV_channel_contribution = dV_channel_contribution - I_K_pop
                    dgate_parts.append((dn, layout.get(n_gate_key)))

                elif ch.kind == "Na":
                    m_key = f"{pop.name}.Na.m"
                    h_key = f"{pop.name}.Na.h"
                    m = X[layout.get(m_key)]
                    h = X[layout.get(h_key)]
                    m0, tm, h0, th = Na_prop(V_p)
                    dm = -(1.0 / tm) * (m - m0)
                    dh = -(1.0 / th) * (h - h0)
                    I_Na_pop = cp["g"] * m ** 3 * h * (V_p - cp["E"])
                    dV_channel_contribution = dV_channel_contribution - I_Na_pop
                    dgate_parts.append((dm, layout.get(m_key)))
                    dgate_parts.append((dh, layout.get(h_key)))

                elif ch.kind == "A":
                    m_key = f"{pop.name}.A.m"
                    h_key = f"{pop.name}.A.h"
                    m = X[layout.get(m_key)]
                    h = X[layout.get(h_key)]
                    m0, tm, h0, th = A_prop(V_p)
                    dm = -(1.0 / tm) * (m - m0)
                    dh = -(1.0 / th) * (h - h0)
                    I_A_pop = cp["g"] * m ** 4 * h * (V_p - cp["E"])
                    dV_channel_contribution = dV_channel_contribution - I_A_pop
                    dgate_parts.append((dm, layout.get(m_key)))
                    dgate_parts.append((dh, layout.get(h_key)))

                elif ch.kind == "Ca":
                    m_key  = f"{pop.name}.Ca.m"
                    h_key  = f"{pop.name}.Ca.h"
                    ca_key = f"{pop.name}.Ca.extra"
                    m  = X[layout.get(m_key)]
                    h  = X[layout.get(h_key)]
                    Ca = X[layout.get(ca_key)]
                    m0, tm, h0, th = Ca_prop(V_p)
                    dm = -(1.0 / tm) * (m - m0)
                    dh = -(1.0 / th) * (h - h0)
                    I_Ca_pop = cp["g"] * m ** 2 * h * (V_p - cp["E"])
                    A_Ca_val = float(cp.get("A_Ca", 2e-4))
                    Ca0_val  = float(cp.get("Ca0",  2.4e-4))
                    t_Ca_val = float(cp.get("t_Ca", 150.0))
                    dCa = -A_Ca_val * I_Ca_pop - (Ca - Ca0_val) / t_Ca_val
                    dV_channel_contribution = dV_channel_contribution - I_Ca_pop
                    dgate_parts.append((dm, layout.get(m_key)))
                    dgate_parts.append((dh, layout.get(h_key)))
                    dgate_parts.append((dCa, layout.get(ca_key)))

                elif ch.kind == "KCa":
                    m_key  = f"{pop.name}.KCa.m"
                    # Find Ca concentration from a Ca channel on the same population
                    ca_key = f"{pop.name}.Ca.extra"
                    m  = X[layout.get(m_key)]
                    Ca = X[layout.get(ca_key)]
                    m0, tm = KCa_prop(Ca)
                    dm = -(1.0 / tm) * (m - m0)
                    I_KCa_pop = cp["g"] * m * (V_p - cp["E"])
                    dV_channel_contribution = dV_channel_contribution - I_KCa_pop
                    dgate_parts.append((dm, layout.get(m_key)))

            I_channel = I_channel.at[pp["V_slice"]].add(dV_channel_contribution)

        # --- Injected current ------------------------------------------------
        t_idx = _input_index_at_time(t, input_scale, current_input_tensor.shape[0])
        I_inj = current_input_tensor[t_idx] * (V - E_input)

        # --- Synaptic currents -----------------------------------------------
        I_syn = jnp.zeros(n_n)

        ach_state_offset = 0
        for i, proj in enumerate(ach_projs):
            n_syn, row_ids, col_ids = ach_layouts[i]
            alp, bet, g, E = ach_syn_params[i]
            tag = f"syn.{proj.src}->{proj.dst}.ach"
            o = X[layout.get(tag)]
            T = jnp.where(
                jnp.logical_and(t > fire_t + t_delay, t < fire_t + t_max + t_delay),
                A_vec,
                jnp.zeros_like(A_vec),
            )
            T_col = T[col_ids]
            do = alp * (1.0 - o) * T_col - bet * o
            dgate_parts.append((do, layout.get(tag)))
            I_syn = I_syn + _sum_synaptic_current(o, row_ids, V, E, g, n_n)

        for i, proj in enumerate(fgaba_projs):
            n_syn, row_ids, col_ids = fgaba_layouts[i]
            alp, bet, V0, sigma, g, E = fgaba_syn_params[i]
            tag = f"syn.{proj.src}->{proj.dst}.fgaba"
            o = X[layout.get(tag)]
            T_col = (1.0 / (1.0 + jnp.exp(-(V - V0) / sigma)))[col_ids]
            do = alp * (1.0 - o) * T_col - bet * o
            dgate_parts.append((do, layout.get(tag)))
            I_syn = I_syn + _sum_synaptic_current(o, row_ids, V, E, g, n_n)

        for i, proj in enumerate(sgaba_projs):
            n_syn, row_ids, col_ids = sgaba_layouts[i]
            K_s, r1, r2, r3, r4, G, E = sgaba_syn_params[i]
            tag_r = f"syn.{proj.src}->{proj.dst}.sgaba.r"
            tag_g = f"syn.{proj.src}->{proj.dst}.sgaba.g"
            r_s = X[layout.get(tag_r)]
            g_s = X[layout.get(tag_g)]
            T = jnp.where(
                jnp.logical_and(t > fire_t + t_delay, t < fire_t + t_max + t_delay),
                A_vec,
                jnp.zeros_like(A_vec),
            )
            T_col = T[col_ids]
            dr = r1 * (1.0 - r_s) * T_col - r2 * r_s
            dg = -r4 * g_s + r3 * r_s
            G4 = jnp.power(g_s, 4) / (jnp.power(g_s, 4) + K_s)
            dgate_parts.append((dr, layout.get(tag_r)))
            dgate_parts.append((dg, layout.get(tag_g)))
            I_syn = I_syn + _sum_synaptic_current(G4, row_ids, V, E, G, n_n)

        # --- Assemble dV/dt (vectorised over all neurons) --------------------
        C_m_all = jnp.concatenate([
            _to_jax_array(np.full(pop.n, pop.C_m))
            for pop in model.populations
        ])
        dV = (-I_inj + I_channel - I_syn) / C_m_all

        # --- Build the full derivative vector --------------------------------
        dX = jnp.zeros_like(X)
        dX = dX.at[layout.get("V")].set(dV)
        for (deriv, sl) in dgate_parts:
            dX = dX.at[sl].set(deriv)
        # fire_t derivative is always 0 (updated by the integrator spike logic)
        return dX

    return dXdt


# ---------------------------------------------------------------------------
# Wrapper for carrying a dynamics_builder through ExperimentSpec
# ---------------------------------------------------------------------------

class _CustomDynamicsExperimentSpec:
    """Thin wrapper around ExperimentSpec that carries a custom dynamics builder.

    The ``prepare()`` method returns a ``PreparedExperiment`` with the
    ``dynamics_builder`` field set so that ``runtime.py`` uses the custom ODE.
    """

    def __init__(self, spec, dynamics_builder: Callable):
        self._spec = spec
        self._dynamics_builder = dynamics_builder

    # Delegate attribute access to the inner spec
    def __getattr__(self, name):
        return getattr(self._spec, name)

    def prepare(self, **overrides):
        prepared = self._spec.prepare(**overrides)
        from dataclasses import asdict as _asdict

        # Re-create PreparedExperiment with dynamics_builder set
        from intermittent_odors.experiment import PreparedExperiment
        return PreparedExperiment(
            config=dict(prepared.config),
            thresholds=prepared.thresholds,
            input_dt=prepared.input_dt,
            sample_stride=prepared.sample_stride,
            sample_neurons=prepared.sample_neurons,
            time_batches=prepared.time_batches,
            metadata=dict(prepared.metadata),
            dynamics_builder=self._dynamics_builder,
        )

    def __repr__(self):
        return f"_CustomDynamicsExperimentSpec({self._spec!r})"


# ---------------------------------------------------------------------------
# Standard PN/LN network factory
# ---------------------------------------------------------------------------

def pnln_network(
    p_n: int,
    l_n: int,
    ach_mat: Any,
    fgaba_mat: Any,
    sgaba_mat: Any,
    *,
    g_ach: Any | None = None,
    g_fgaba: Any | None = None,
    G_sgaba: Any | None = None,
    normalize_conductances: bool = True,
    # Per-neuron conductance overrides (if you want non-default values)
    g_K_pn: float = 3.6,
    g_K_ln: float = 36.0,
    g_Na: float = 7.15,
    g_A: float = 1.43,
    g_Ca: float = 5.0,
    g_KCa: float = 0.045,
    E_K: float = -95.0,
    E_L_pn: float = -64.0,
    E_L_ln: float = -50.0,
    E_Na: float = 50.0,
    E_A: float = -95.0,
    E_Ca: float = 140.0,
    E_KCa: float = -95.0,
    # Synapse globals
    A: float = 0.5,
    t_max: float = 0.3,
    t_delay: float = 0.0,
    # Extra global params
    input_scale: float = 100.0,
) -> "NetworkModel":
    """Create a standard PN/LN network model using the paper-published parameters.

    This factory produces a ``NetworkModel`` with ``_is_standard_pnln=True``,
    meaning ``build_dynamics_fn()`` routes through the unchanged
    ``build_dynamics_core`` in ``jax_network.py`` for guaranteed numerical
    parity with previously published results.

    Parameters
    ----------
    p_n, l_n : int
        Number of projection neurons and local neurons.
    ach_mat, fgaba_mat, sgaba_mat : array-like, shape (p_n+l_n, p_n+l_n)
        Connectivity matrices.
    g_ach, g_fgaba, G_sgaba : array-like, optional
        Per-destination conductance arrays (already normalised).  If *None*
        the caller is responsible for setting them later (or using
        ``build_fig2_experiment_spec`` / ``trial_case_to_experiment_spec``).
    normalize_conductances : bool
        If True and the conductance arrays look like un-normalised population
        values (shape == (n_n,)), divide by in-degree.
    """
    n_n = p_n + l_n
    ach_mat   = np.asarray(ach_mat,   dtype=np.float64)
    fgaba_mat = np.asarray(fgaba_mat, dtype=np.float64)
    sgaba_mat = np.asarray(sgaba_mat, dtype=np.float64)

    n_syn_ach   = int(np.sum(ach_mat   != 0))
    n_syn_fgaba = int(np.sum(fgaba_mat != 0))
    n_syn_sgaba = int(np.sum(sgaba_mat != 0))

    def _normalise(arr, mat):
        arr = np.asarray(arr, dtype=np.float64)
        if not normalize_conductances:
            return arr
        indegree = np.sum(mat, axis=1)
        out = np.zeros_like(arr)
        np.divide(arr, indegree, out=out, where=indegree != 0)
        return out

    if g_ach is None:
        _g_ach = np.zeros(n_n, dtype=np.float64)
    else:
        _g_ach = _normalise(np.asarray(g_ach, dtype=np.float64), ach_mat) if normalize_conductances else np.asarray(g_ach, np.float64)

    if g_fgaba is None:
        _g_fgaba = np.zeros(n_n, dtype=np.float64)
    else:
        _g_fgaba = _normalise(np.asarray(g_fgaba, dtype=np.float64), fgaba_mat) if normalize_conductances else np.asarray(g_fgaba, np.float64)

    if G_sgaba is None:
        _G_sgaba = np.zeros(n_n, dtype=np.float64)
    else:
        _G_sgaba = _normalise(np.asarray(G_sgaba, dtype=np.float64), sgaba_mat) if normalize_conductances else np.asarray(G_sgaba, np.float64)

    def _config_builder():
        return {
            "n_n": n_n,
            "p_n": p_n,
            "l_n": l_n,
            "C_m": [1.0] * n_n,
            "g_K": [g_K_pn] * p_n + [g_K_ln] * l_n,
            "g_L": [0.3] * n_n,
            "E_K": [E_K] * n_n,
            "E_L": [E_L_pn] * p_n + [E_L_ln] * l_n,
            "g_Na": [g_Na] * p_n,
            "g_A":  [g_A]  * p_n,
            "E_Na": [E_Na] * p_n,
            "E_A":  [E_A]  * p_n,
            "g_Ca":  [g_Ca]  * l_n,
            "g_KCa": [g_KCa] * l_n,
            "E_Ca":  [E_Ca]  * l_n,
            "E_KCa": [E_KCa] * l_n,
            "A_Ca": 2e-4,
            "Ca0":  2.4e-4,
            "t_Ca": 150.0,
            "input_scale": input_scale,
            "ach_mat": ach_mat,
            "alp_ach": [10.0] * n_syn_ach,
            "bet_ach": [0.2]  * n_syn_ach,
            "t_max":   t_max,
            "t_delay": t_delay,
            "A":       [A] * n_n,
            "g_ach":   _g_ach.tolist(),
            "E_ach":   [0.0] * n_n,
            "fgaba_mat": fgaba_mat,
            "alp_fgaba": [10.0] * n_syn_fgaba,
            "bet_fgaba": [0.16] * n_syn_fgaba,
            "V0":    [-20.0] * n_n,
            "sigma": [1.5]   * n_n,
            "g_fgaba": _g_fgaba.tolist(),
            "E_fgaba": [-70.0] * n_n,
            "sgaba_mat": sgaba_mat,
            "K_sgaba":  [100e-12] * n_syn_sgaba,
            "r1_sgaba": [1.0]   * n_syn_sgaba,
            "r2_sgaba": [0.025] * n_syn_sgaba,
            "r3_sgaba": [0.1]   * n_syn_sgaba,
            "r4_sgaba": [0.06]  * n_syn_sgaba,
            "G_sgaba": _G_sgaba.tolist(),
            "E_sgaba": [-95.0] * n_n,
        }

    # Build population objects (for layout / state_vector API)
    pn_pop = Population(
        "PN", p_n,
        channels=(
            K_channel(g_K_pn, E_K),
            Na_channel(g_Na, E_Na),
            A_channel(g_A, E_A),
        ),
        C_m=1.0, g_L=0.3, E_L=E_L_pn, threshold=0.0,
    )
    ln_pop = Population(
        "LN", l_n,
        channels=(
            K_channel(g_K_ln, E_K),
            Ca_channel(g_Ca, E_Ca),
            KCa_channel(g_KCa, E_KCa),
        ),
        C_m=1.0, g_L=0.3, E_L=E_L_ln, threshold=-20.0,
    )

    ach_syn_spec   = ach_synapse(reversal=0.0,   conductance=_g_ach,   A=A, t_max=t_max,  t_delay=t_delay)
    fgaba_syn_spec = fgaba_synapse(reversal=-70.0, conductance=_g_fgaba)
    sgaba_syn_spec = sgaba_synapse(reversal=-95.0, conductance=_G_sgaba)

    projs = [
        Projection("PN", "LN", ach_syn_spec,   ach_mat),
        Projection("LN", "PN", fgaba_syn_spec, fgaba_mat),
        Projection("LN", "PN", sgaba_syn_spec, sgaba_mat),
    ]

    return NetworkModel(
        populations=[pn_pop, ln_pop],
        projections=projs,
        global_params={"A": A, "t_max": t_max, "t_delay": t_delay, "input_scale": input_scale},
        _is_standard_pnln=True,
        _config_builder=_config_builder,
    )

