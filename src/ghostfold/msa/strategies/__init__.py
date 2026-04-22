from ghostfold.msa.strategies.base import BaseStrategy
from ghostfold.msa.strategies.diverse_beam import DiverseBeamStrategy
from ghostfold.msa.strategies.encoder_perturb import EncoderPerturbStrategy
from ghostfold.msa.strategies.round_trip import RoundTripStrategy
from ghostfold.msa.strategies.threedipperturb import ThreeDiPerturbStrategy

STRATEGIES: dict[str, type[BaseStrategy]] = {
    "encoder_perturb": EncoderPerturbStrategy,
    "diverse_beam": DiverseBeamStrategy,
    "round_trip": RoundTripStrategy,
    "3di_perturb": ThreeDiPerturbStrategy,
}

__all__ = [
    "BaseStrategy",
    "DiverseBeamStrategy",
    "EncoderPerturbStrategy",
    "RoundTripStrategy",
    "ThreeDiPerturbStrategy",
    "STRATEGIES",
]
