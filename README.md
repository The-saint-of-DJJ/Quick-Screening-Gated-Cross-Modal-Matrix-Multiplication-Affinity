# QS-GCMM-AF code package

`QS-GCMM-AF` is the maintained code name for the sequence-based peptide-target
ranking model. Existing checkpoints and output tables retain the historical
`GCMM-AF-v1`/`GCMM-AF` identifiers for compatibility.

The package contains:

- `features.py`: deterministic amino-acid, hashed 2/3/4-mer, descriptor, and RDKit chemistry features.
- `model.py`: peptide/target encoders, gated pair head, dropout inference, and legacy checkpoint loading.
- `training.py`: grouped splits, regression plus ranking loss, metrics, and checkpoint training.
- `screening.py`: encoded matrix coarse screening, batched pair scoring, MC-dropout uncertainty, and TSV output.

The package does not include digestion, target acquisition, structure preparation,
docking, molecular dynamics, manuscript generation, or plotting.

## Commands

Train with the stable historical entry point:

```bash
python scripts/train_gcmm_af.py \
  --labels data/processed/peptide_target_affinity_v2/peptide_target_affinity.tsv \
  --checkpoint models/gcmm_af_human_kd_v2.pt \
  --report results/rapid_pipeline/gcmm_af_human_kd_v2_training.json
```

Run the existing computational candidate workflow; its screening calls now use
`qs_gcmm_af.screen`:

```bash
python scripts/run_computational_md_candidate_v1.py --help
```

Load a checkpoint directly:

```python
from pathlib import Path
from qs_gcmm_af import load_checkpoint, sequence_features

model, metadata, device = load_checkpoint(Path("models/gcmm_af_human_kd_v2.pt"), 284, "cpu")
```

Predictions remain computational ranking hypotheses. A checkpoint with
`NOT_EXTERNAL_CALIBRATED` must not be described as experimentally calibrated Kd.
