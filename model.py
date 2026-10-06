"""QS-GCMM-AF gated dual encoder and legacy checkpoint compatibility."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _torch():
    import torch

    return torch


class GCMMAffinity:
    """Dual sequence/chemistry encoder with a gated pair regression head.

    The class deliberately retains the historical ``GCMMAffinity`` name and
    ``module`` layout because existing ``GCMM-AF-v1`` checkpoints serialize
    these state-dict keys.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.12):
        import torch.nn as nn

        self.module = nn.ModuleDict(
            {
                "pep": nn.Sequential(
                    nn.Linear(input_dim + 8, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "target": nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "gate": nn.Sequential(
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                ),
                "head": nn.Sequential(
                    nn.Linear(hidden_dim * 4 + 1, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                ),
            }
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def parameters(self):
        return self.module.parameters()

    def to(self, device):
        self.module.to(device)
        return self

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        return self.module.load_state_dict(state)

    def encode(self, pep, target, chem):
        torch = _torch()
        p = self.module["pep"](torch.cat([pep, chem], dim=-1))
        t = self.module["target"](target)
        p = torch.nn.functional.normalize(p, dim=-1)
        t = torch.nn.functional.normalize(t, dim=-1)
        product = p * t
        difference = torch.abs(p - t)
        gate_input = torch.cat([p, t, product, difference], dim=-1)
        gate = torch.sigmoid(self.module["gate"](gate_input))
        return p, t, gate

    def forward_pairs(self, pep, target, chem):
        torch = _torch()
        p, t, gate = self.encode(pep, target, chem)
        pair = torch.cat([p, t, p * t, torch.abs(p - t), gate], dim=-1)
        return self.module["head"](pair).squeeze(-1)


def device(requested: str | None = None):
    torch = _torch()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, input_dim: int, device_name: str | None = None):
    """Load a ``GCMM-AF-v1`` or QS-GCMM-AF checkpoint without rewriting it."""
    torch = _torch()
    target_device = device(device_name)
    try:
        payload = torch.load(path, map_location=target_device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=target_device)
    expected_dim = int(payload.get("input_dim", input_dim))
    if expected_dim != input_dim:
        raise RuntimeError(
            f"checkpoint input_dim={expected_dim} does not match current features={input_dim}; "
            "use the same --hash-dim used during training"
        )
    model = GCMMAffinity(
        input_dim,
        int(payload.get("hidden_dim", 128)),
        float(payload.get("dropout", 0.12)),
    ).to(target_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload, target_device
