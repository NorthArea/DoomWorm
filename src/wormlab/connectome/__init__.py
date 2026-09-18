"""Connectome loading and the internal graph format (Neuron[], Connection[]). Dataset-agnostic."""

from wormlab.connectome.build import build_network
from wormlab.connectome.loader import DATA_DIR, load_cook2019
from wormlab.connectome.mappings import (
    MotorMapping,
    Route,
    SensoryMapping,
    default_motor_mapping,
    default_sensory_mapping,
)
from wormlab.connectome.model import Connection, ConnectionType, Connectome, NeuronInfo
from wormlab.connectome.neurotransmitters import GABA_NEURONS
from wormlab.connectome.variants import (
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
