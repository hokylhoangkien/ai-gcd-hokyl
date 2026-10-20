from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelOutput:
    image_features: torch.Tensor
    attribute_features: torch.Tensor
    attribute_summary: torch.Tensor
    reliabilities: torch.Tensor
    fused_features: torch.Tensor
    logits: torch.Tensor
    visual_logits: torch.Tensor
    attribute_logits: torch.Tensor
    debiased_logits: torch.Tensor
    distribution_logits: torch.Tensor


class AttributeReliabilityHead(nn.Module):
    """Predict per-sample, per-attribute reliability.

    OCR confidence is evidence, not the final reliability: the head also sees
    visual/text agreement and whether an attribute is present.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * embedding_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        attribute_features: torch.Tensor,
        ocr_confidence: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_attributes, _ = attribute_features.shape
        images = image_features[:, None, :].expand(-1, num_attributes, -1)
        agreement = F.cosine_similarity(images, attribute_features, dim=-1).unsqueeze(-1)
        evidence = torch.cat(
            [
                images,
                attribute_features,
                agreement,
                ocr_confidence.unsqueeze(-1),
                present.unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.sigmoid(self.network(evidence).squeeze(-1)) * present


class ReliabilityAwareGCD(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        clip_tokenizer,
        num_clusters: int,
        clip_dim: int,
        projection_dim: int = 256,
        reliability_hidden_dim: int = 128,
        fusion_mode: str = "proposed",
        freeze_text_encoder: bool = True,
        attribute_names: list[str] | None = None,
        num_known_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.clip = clip_model
        self.tokenizer = clip_tokenizer
        if fusion_mode not in {"visual", "attributes", "simple", "oracle", "proposed"}:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.freeze_text_encoder = freeze_text_encoder
        self.attribute_names = attribute_names
        self.image_projection = nn.Linear(clip_dim, projection_dim)
        self.text_projection = nn.Linear(clip_dim, projection_dim)
        self.reliability = AttributeReliabilityHead(projection_dim, reliability_hidden_dim)
        self.cluster_head = nn.utils.parametrizations.weight_norm(
            nn.Linear(projection_dim, num_clusters, bias=False)
        )
        self.debiased_head = nn.utils.parametrizations.weight_norm(
            nn.Linear(projection_dim, num_clusters, bias=False)
        )
        self.num_known_classes = num_known_classes or num_clusters
        self.distribution_projector = nn.Sequential(
            nn.Linear(projection_dim, reliability_hidden_dim),
            nn.GELU(),
            nn.Linear(reliability_hidden_dim, projection_dim),
        )
        self.distribution_head = nn.utils.parametrizations.weight_norm(
            nn.Linear(projection_dim, 2 * self.num_known_classes, bias=False)
        )
        if freeze_text_encoder:
            for name, parameter in self.clip.named_parameters():
                if not name.startswith("visual."):
                    parameter.requires_grad = False

    def set_visual_trainable(self, trainable: bool) -> None:
        for parameter in self.clip.visual.parameters():
            parameter.requires_grad = trainable

    def encode_attributes(
        self, attribute_values: list[list[str]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Collation is sample-major; flatten, encode, then restore [B, A, D].
        batch_size = len(attribute_values)
        num_attributes = len(attribute_values[0])
        names = self.attribute_names or [f"attribute_{index}" for index in range(num_attributes)]
        if len(names) != num_attributes:
            raise ValueError(f"Expected {len(names)} attributes, received {num_attributes}")
        flat = [
            f"{names[index].replace('_', ' ')}: {value}" if value else "unknown product attribute"
            for sample in attribute_values for index, value in enumerate(sample)
        ]
        present = torch.tensor(
            [[bool(value) for value in sample] for sample in attribute_values],
            dtype=torch.float32,
            device=device,
        )
        tokens = self.tokenizer(flat).to(device)
        context = torch.no_grad() if self.freeze_text_encoder else nullcontext()
        with context:
            text = self.clip.encode_text(tokens).float()
        text = self.text_projection(text).reshape(batch_size, num_attributes, -1)
        return F.normalize(text, dim=-1), present

    def forward(
        self,
        images: torch.Tensor,
        attribute_values: list[list[str]],
        attribute_confidences: torch.Tensor,
        temperature: float = 0.1,
        encoded_attributes: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> ModelOutput:
        image = self.clip.encode_image(images).float()
        image = F.normalize(self.image_projection(image), dim=-1)
        attributes, present = encoded_attributes or self.encode_attributes(attribute_values, images.device)
        predicted_reliability = self.reliability(image, attributes, attribute_confidences, present)
        if self.fusion_mode == "oracle":
            reliability = present
        elif self.fusion_mode in {"attributes", "simple"}:
            reliability = attribute_confidences * present
        else:
            reliability = predicted_reliability
        weights = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attribute_summary = (weights.unsqueeze(-1) * attributes).sum(dim=1)
        if self.fusion_mode == "visual":
            fused = image
        elif self.fusion_mode == "attributes":
            fused = F.normalize(attribute_summary, dim=-1)
        elif self.fusion_mode in {"simple", "oracle"}:
            fused = F.normalize(image + attribute_summary, dim=-1)
        else:
            text_strength = reliability.mean(dim=1, keepdim=True)
            fused = F.normalize(image + text_strength * attribute_summary, dim=-1)
        logits = self.cluster_head(fused) / temperature
        visual_logits = self.cluster_head(image) / temperature
        attribute_logits = self.cluster_head(F.normalize(attribute_summary, dim=-1)) / temperature
        debiased_logits = self.debiased_head(fused) / temperature
        distribution_features = F.normalize(self.distribution_projector(fused), dim=-1)
        distribution_logits = self.distribution_head(distribution_features) / temperature
        return ModelOutput(
            image, attributes, attribute_summary, reliability, fused, logits,
            visual_logits, attribute_logits, debiased_logits, distribution_logits,
        )


def create_model(
    config: dict,
    num_clusters: int,
    device: torch.device,
    attribute_names: list[str] | None = None,
    num_known_classes: int | None = None,
) -> ReliabilityAwareGCD:
    import open_clip

    name = config["clip_model"]
    pretrained = config["clip_pretrained"]
    clip_model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(name)
    clip_dim = int(clip_model.text_projection.shape[-1])
    model = ReliabilityAwareGCD(
        clip_model=clip_model,
        clip_tokenizer=tokenizer,
        num_clusters=num_clusters,
        clip_dim=clip_dim,
        projection_dim=config["projection_dim"],
        reliability_hidden_dim=config["reliability_hidden_dim"],
        fusion_mode=config.get("fusion_mode", "proposed"),
        freeze_text_encoder=config.get("freeze_text_encoder", True),
        attribute_names=attribute_names,
        num_known_classes=num_known_classes,
    )
    return model.to(device)

