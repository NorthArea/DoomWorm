"""Connectome loading and the internal graph format (Neuron[], Connection[]). Dataset-agnostic."""

from broomworm.connectome.build import build_network
from broomworm.connectome.loader import DATA_DIR, load_cook2019
from broomworm.connectome.mappings import (
    MotorMapping,
    Route,
    SensoryMapping,
    default_motor_mapping,
    default_sensory_mapping,
)
from broomworm.connectome.model import Connection, ConnectionType, Connectome, NeuronInfo
from broomworm.connectome.neurotransmitters import GABA_NEURONS
from broomworm.connectome.variants import (
    VARIANTS,
    dense_connectome,
    make_variant,
    random_connectome,
    shuffled_connectome,
)

__all__ = [
    "DATA_DIR",
    "GABA_NEURONS",
    "VARIANTS",
    "Connection",
    "ConnectionType",
    "Connectome",
    "MotorMapping",
    "NeuronInfo",
    "Route",
    "SensoryMapping",
    "build_network",
    "default_motor_mapping",
    "default_sensory_mapping",
    "dense_connectome",
    "load_cook2019",
    "make_variant",
    "random_connectome",
    "shuffled_connectome",
]
