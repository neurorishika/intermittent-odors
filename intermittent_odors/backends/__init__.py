"""Backend implementations for JAX and TensorFlow integrators.

This sub-package contains the ODE kinetics, integrators, and precision
configuration for the two supported compute backends.  Most users should
not need to import from here directly — the public ``CompiledExperimentRunner``
and ``compile_experiment`` API in ``intermittent_odors.runtime`` handles
backend selection automatically.

Power-user entry points
-----------------------
``intermittent_odors.backends.jax_network``
    JAX channel kinetics and ``build_dynamics_core`` / ``build_dynamics``.
``intermittent_odors.backends.jax_integrator``
    RK4 integrators with spike-time tracking: ``odeint``, ``odeint_sampled``.
``intermittent_odors.backends.jax_precision``
    JAX dtype configuration (float64 / float32 / bfloat16) controlled by
    the ``IODOR_JAX_PRECISION`` environment variable.
``intermittent_odors.backends.tf_network``
    TensorFlow 1.x channel kinetics and ``build_dynamics``.
``intermittent_odors.backends.tf_integrator``
    TensorFlow 1.x RK4 integrator: ``odeint``.
"""

from intermittent_odors.backends.jax_network import \
    build_dynamics as build_jax_dynamics
from intermittent_odors.backends.jax_network import \
    build_dynamics_core as build_jax_dynamics_core

__all__ = [
    'build_jax_dynamics',
    'build_jax_dynamics_core',
]
