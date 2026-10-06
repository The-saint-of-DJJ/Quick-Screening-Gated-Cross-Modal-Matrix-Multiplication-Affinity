"""QS-GCMM-AF sequence/chemistry affinity ranking core.

The package contains only model inputs, the dual encoder, checkpoint loading,
training utilities, and batched screening.  Digest, target acquisition,
structure preparation, molecular dynamics, manuscript, and plotting workflows
remain outside this package.
"""

from .features import (
    AA,
    DEFAULT_HASH_DIM,
    chemistry_features,
    clean_sequence,
    peptide_smiles,
    sequence_features,
)
from .model import GCMMAffinity, load_checkpoint
from .screening import SCORE_FIELDS, screen

MODEL_NAME = "QS-GCMM-AF"
LEGACY_MODEL_NAME = "GCMM-AF"

__all__ = [
    "AA",
    "DEFAULT_HASH_DIM",
    "GCMMAffinity",
    "LEGACY_MODEL_NAME",
    "MODEL_NAME",
    "SCORE_FIELDS",
    "chemistry_features",
    "clean_sequence",
    "load_checkpoint",
    "peptide_smiles",
    "screen",
    "sequence_features",
]
