"""End-to-end attack/defense pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image

from ai_fragility.attacks import fgsm_attack, pgd_attack
from ai_fragility.defenses import DefenseConfig, apply_defense
from ai_fragility.metrics import SecurityGap, compute_security_gap
from ai_fragility.models import ModelBundle, load_model
from ai_fragility.seed import set_global_seed


AttackMethod = Literal["fgsm", "pgd"]


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""

    method: AttackMethod = "fgsm"
    epsilon: float = 0.02
    alpha: float = 0.005
    steps: int = 20


@dataclass
class NoiseConfig:
    """Configuration for additive random noise to visualize non-adversarial perturbations."""

    sigma: float = 0.02
    enabled: bool = True


@dataclass
class PipelineResult:
    """Structured outputs from a single attack/defense pass."""

    clean_logits: torch.Tensor
    noisy_logits: torch.Tensor | None
    attacked_logits: torch.Tensor
    defended_logits: torch.Tensor
    clean_image: torch.Tensor
    noisy_image: torch.Tensor | None
    attacked_image: torch.Tensor
    defended_image: torch.Tensor
    perturbation: torch.Tensor
    gradient: torch.Tensor
    security_gap: SecurityGap
    reference_class: int


class AttackDefensePipeline:
    """Coordinates model inference, attack, defense, and metrics."""

    def __init__(
        self,
        attack_config: AttackConfig | None = None,
        defense_config: DefenseConfig | None = None,
        noise_config: NoiseConfig | None = None,
        model_name: str = "resnet18",
        seed: int = 42,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_global_seed(seed)
        self.attack_config = attack_config or AttackConfig()
        self.defense_config = defense_config or DefenseConfig()
        self.noise_config = noise_config or NoiseConfig()
        self.bundle: ModelBundle = load_model(model_name=model_name, device=self.device)

    def _forward(self, raw_images: torch.Tensor) -> torch.Tensor:
        """Run the model on raw images by applying normalization first."""

        normalized = self.bundle.normalize(raw_images)
        return self.bundle.model(normalized)

    def run(
        self,
        image: Image.Image | Path,
        attack_config: AttackConfig | None = None,
        defense_config: DefenseConfig | None = None,
        noise_config: NoiseConfig | None = None,
    ) -> PipelineResult:
        """Execute the attack/defense pipeline for one image.

        Args:
            image: Input image object or path.
            attack_config: Optional override for attack settings.
            defense_config: Optional override for defense settings.

        Returns:
            PipelineResult with logits, tensors, gradients, and metrics.
        """

        active_attack = attack_config or self.attack_config
        active_defense = defense_config or self.defense_config
        active_noise = noise_config or self.noise_config

        pil_image = Image.open(image) if isinstance(image, Path) else image
        raw_tensor = self.bundle.to_tensor(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            clean_logits = self.bundle.model(self.bundle.normalize(raw_tensor))
        reference_label = clean_logits.argmax(dim=1)

        noisy_raw: torch.Tensor | None = None
        noisy_logits: torch.Tensor | None = None
        if active_noise.enabled and active_noise.sigma > 0:
            noise = torch.randn_like(raw_tensor) * active_noise.sigma
            noisy_raw = torch.clamp(raw_tensor + noise, 0.0, 1.0)
            noisy_logits = self.bundle.model(self.bundle.normalize(noisy_raw))

        # Gradient map on clean input for explainability
        normalized_for_grad = self.bundle.normalize(raw_tensor.clone().detach()).requires_grad_(True)
        logits_for_grad = self.bundle.model(normalized_for_grad)
        loss_for_grad = F.cross_entropy(logits_for_grad, reference_label)
        loss_for_grad.backward()
        gradient = normalized_for_grad.grad.detach()[0]

        if active_attack.method == "fgsm":
            attacked_raw = fgsm_attack(
                forward_fn=self._forward,
                images=raw_tensor,
                labels=reference_label,
                epsilon=active_attack.epsilon,
            )
        elif active_attack.method == "pgd":
            attacked_raw = pgd_attack(
                forward_fn=self._forward,
                images=raw_tensor,
                labels=reference_label,
                epsilon=active_attack.epsilon,
                alpha=active_attack.alpha,
                steps=active_attack.steps,
            )
        else:
            msg = f"Unsupported attack method {active_attack.method}"
            raise ValueError(msg)

        attacked_logits = self.bundle.model(self.bundle.normalize(attacked_raw))

        defended_raw = apply_defense(attacked_raw, config=active_defense, device=self.device)
        defended_logits = self.bundle.model(self.bundle.normalize(defended_raw))

        security_gap = compute_security_gap(
            clean_logits=clean_logits,
            attacked_logits=attacked_logits,
            defended_logits=defended_logits,
            reference_class=int(reference_label.item()),
        )

        perturbation = attacked_raw - raw_tensor

        return PipelineResult(
            clean_logits=clean_logits.detach(),
            noisy_logits=noisy_logits.detach() if noisy_logits is not None else None,
            attacked_logits=attacked_logits.detach(),
            defended_logits=defended_logits.detach(),
            clean_image=raw_tensor.detach(),
            noisy_image=noisy_raw.detach() if noisy_raw is not None else None,
            attacked_image=attacked_raw.detach(),
            defended_image=defended_raw.detach(),
            perturbation=perturbation.detach(),
            gradient=gradient,
            security_gap=security_gap,
            reference_class=int(reference_label.item()),
        )


__all__: list[str] = ["AttackDefensePipeline", "AttackConfig", "NoiseConfig", "PipelineResult"]
