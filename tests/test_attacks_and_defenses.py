from __future__ import annotations

import torch

from ai_fragility.attacks import fgsm_attack, pgd_attack
from ai_fragility.defenses import DefenseConfig, apply_defense
from ai_fragility.metrics import compute_security_gap


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.view(x.size(0), -1))


def test_fgsm_linf_bound() -> None:
    torch.manual_seed(0)
    model = TinyModel()
    images = torch.rand(2, 3, 4, 4)
    labels = torch.tensor([0, 1])
    epsilon = 0.1

    adv = fgsm_attack(model.forward, images, labels, epsilon=epsilon)
    delta = (adv - images).abs().max().item()

    assert delta <= epsilon + 1e-6


def test_pgd_linf_bound() -> None:
    torch.manual_seed(0)
    model = TinyModel()
    images = torch.rand(2, 3, 4, 4)
    labels = torch.tensor([1, 2])
    epsilon = 0.1

    adv = pgd_attack(model.forward, images, labels, epsilon=epsilon, alpha=0.02, steps=5)
    delta = (adv - images).abs().max().item()

    assert delta <= epsilon + 1e-6


def test_defense_deterministic_gaussian() -> None:
    torch.manual_seed(0)
    images = torch.rand(1, 3, 8, 8)
    cfg = DefenseConfig(kind="gaussian", sigma=1.0)
    out1 = apply_defense(images, cfg, device=torch.device("cpu"))
    out2 = apply_defense(images, cfg, device=torch.device("cpu"))

    assert torch.allclose(out1, out2, atol=1e-6)


def test_security_gap_monotonicity() -> None:
    clean_logits = torch.tensor([[2.0, 0.0]])
    attacked_logits = torch.tensor([[0.5, 1.5]])
    defended_logits = torch.tensor([[1.5, 0.5]])

    gap = compute_security_gap(clean_logits, attacked_logits, defended_logits, reference_class=0)

    assert gap.clean_confidence > gap.defended_confidence > gap.attacked_confidence
    assert gap.delta == gap.defended_confidence - gap.attacked_confidence
