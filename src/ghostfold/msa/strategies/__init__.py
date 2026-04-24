from ghostfold.msa.strategies.base import BaseStrategy
from ghostfold.msa.strategies.baseline import BaselineStrategy
from ghostfold.msa.strategies.diverse_beam import DiverseBeamStrategy
from ghostfold.msa.strategies.embedding_walk_encoder import EmbeddingWalkEncoderStrategy
from ghostfold.msa.strategies.embedding_walk_full import EmbeddingWalkFullStrategy
from ghostfold.msa.strategies.encoder_only_3di_sub import EncoderOnly3DiSubStrategy
from ghostfold.msa.strategies.encoder_perturb import EncoderPerturbStrategy
from ghostfold.msa.strategies.round_trip import RoundTripStrategy
from ghostfold.msa.strategies.temperature_sweep import TemperatureSweepStrategy
from ghostfold.msa.strategies.threedipperturb import ThreeDiPerturbStrategy

STRATEGIES: dict[str, type[BaseStrategy]] = {
    "baseline": BaselineStrategy,
    "encoder_only_3di_sub": EncoderOnly3DiSubStrategy,
    "temperature_sweep": TemperatureSweepStrategy,
    "embedding_walk_full": EmbeddingWalkFullStrategy,
    "embedding_walk_encoder": EmbeddingWalkEncoderStrategy,
    "encoder_perturb": EncoderPerturbStrategy,
    "round_trip": RoundTripStrategy,
    "3di_perturb": ThreeDiPerturbStrategy,
    # diverse_beam excluded: OOM-prone on all tested hardware
}

__all__ = [
    "BaseStrategy",
    "BaselineStrategy",
    "DiverseBeamStrategy",
    "EmbeddingWalkEncoderStrategy",
    "EmbeddingWalkFullStrategy",
    "EncoderOnly3DiSubStrategy",
    "EncoderPerturbStrategy",
    "RoundTripStrategy",
    "TemperatureSweepStrategy",
    "ThreeDiPerturbStrategy",
    "STRATEGIES",
]
