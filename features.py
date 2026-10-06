"""Deterministic sequence and peptide chemistry features for QS-GCMM-AF."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {letter: index for index, letter in enumerate(AA)}
DEFAULT_HASH_DIM = 256


def clean_sequence(value: str) -> str:
    """Remove FASTA/punctuation characters and normalize to uppercase."""
    return re.sub(r"[^A-Za-z]", "", value or "").upper()


def peptide_smiles(sequence: str) -> str:
    """Return an RDKit canonical peptide SMILES, or the explicit ``NA`` marker."""
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSequence(sequence)
        return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else "NA"
    except Exception:
        return "NA"


def _hash_index(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("ascii"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, 1.0 if (value >> 8) & 1 else -1.0


def sequence_features(sequences: Sequence[str], hash_dim: int = DEFAULT_HASH_DIM) -> np.ndarray:
    """Build residue-composition, hashed 2/3/4-mer, and descriptor features.

    The output width is ``20 + hash_dim + 8``.  Hashing is fixed with BLAKE2b,
    so feature generation is independent of Python's process hash seed.
    """
    if hash_dim < 32:
        raise ValueError("hash_dim must be >= 32")
    output = np.zeros((len(sequences), 20 + hash_dim + 8), dtype=np.float32)
    hydrophobic = set("AILMFWVY")
    charged = set("DEKR")
    for row_index, sequence in enumerate(sequences):
        if not sequence or not set(sequence).issubset(AA):
            raise ValueError(f"invalid amino-acid sequence at index {row_index}")
        length = float(len(sequence))
        for residue in sequence:
            output[row_index, AA_INDEX[residue]] += 1.0 / length
        for size in (2, 3, 4):
            for start in range(len(sequence) - size + 1):
                index, sign = _hash_index(sequence[start : start + size], hash_dim)
                output[row_index, 20 + index] += sign / max(1.0, length - size + 1)
        output[row_index, 20 + hash_dim :] = np.asarray(
            [
                math.log1p(length) / math.log(41.0),
                sum(residue in hydrophobic for residue in sequence) / length,
                sum(residue in charged for residue in sequence) / length,
                (sequence.count("K") + sequence.count("R") - sequence.count("D") - sequence.count("E")) / length,
                sequence.count("G") / length,
                sequence.count("P") / length,
                sequence.count("C") / length,
                sequence.count("W") / length,
            ],
            dtype=np.float32,
        )
    return output


def chemistry_features(smiles: Sequence[str]) -> np.ndarray:
    """Return eight bounded RDKit descriptors; unavailable/invalid values are zero."""
    output = np.zeros((len(smiles), 8), dtype=np.float32)
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except Exception:
        return output
    for index, value in enumerate(smiles):
        if value == "NA":
            continue
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            continue
        output[index] = np.asarray(
            [
                min(1.0, Descriptors.MolWt(molecule) / 2500.0),
                min(1.0, max(0.0, Crippen.MolLogP(molecule) / 12.0)),
                min(1.0, rdMolDescriptors.CalcTPSA(molecule) / 1000.0),
                min(1.0, Lipinski.NumHDonors(molecule) / 30.0),
                min(1.0, Lipinski.NumHAcceptors(molecule) / 50.0),
                min(1.0, Lipinski.NumRotatableBonds(molecule) / 60.0),
                min(1.0, molecule.GetNumAtoms() / 500.0),
                min(1.0, molecule.GetNumHeavyAtoms() / 400.0),
            ],
            dtype=np.float32,
        )
    return output
