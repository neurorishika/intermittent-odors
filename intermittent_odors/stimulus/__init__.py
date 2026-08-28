"""Parametric stimulus builders for HH network experiments.

Submodules
----------
core       : StimulusData container and shared internal helpers.
odor       : IntermittentOdorParams and intermittent-odor trial builders.
patterns   : ConstantStimulusParams, StepStimulusParams, and their builders.
"""
from .core import StimulusData
from .odor import (
    IntermittentOdorParams,
    build_odor_stimulus,
    build_connectivity,
    build_odor_trial,
)
from .patterns import (
    ConstantStimulusParams,
    build_constant_stimulus,
    build_constant_trial,
    StepStimulusParams,
    build_step_stimulus,
)

__all__ = [
    "StimulusData",
    "IntermittentOdorParams",
    "build_odor_stimulus",
    "build_connectivity",
    "build_odor_trial",
    "ConstantStimulusParams",
    "build_constant_stimulus",
    "build_constant_trial",
    "StepStimulusParams",
    "build_step_stimulus",
]
