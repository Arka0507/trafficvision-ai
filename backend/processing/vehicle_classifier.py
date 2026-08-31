from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from PIL import Image


@dataclass(frozen=True)
class VehiclePrediction:
    label: str
    confidence: float
    make: str | None = None
    model: str | None = None
    year: str | None = None


@dataclass
class VehicleConsensus:
    scores: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, VehiclePrediction] = field(default_factory=dict)
    samples: int = 0

    def add(self, prediction: VehiclePrediction) -> None:
        self.scores[prediction.label] = self.scores.get(prediction.label, 0.0) + prediction.confidence
        self.counts[prediction.label] = self.counts.get(prediction.label, 0) + 1
        self.metadata[prediction.label] = prediction
        self.samples += 1

    def best(self) -> VehiclePrediction | None:
        if not self.scores:
            return None
        label = max(self.scores, key=self.scores.get)
        confidence = self.scores[label] / max(self.counts[label], 1)
        metadata = self.metadata[label]
        return VehiclePrediction(label, confidence, metadata.make, metadata.model, metadata.year)


class FineGrainedVehicleClassifier:
    """Lazy EfficientNet-B3 classifier for 196 Stanford Cars categories."""

    CHECKPOINT_FILENAME = "efficientnet_b3_stanford300_augv2_best.pt"

    def __init__(self, model_name: str, confidence_threshold: float = 0.38, device: str = "auto") -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device_request = device
        self.transform = None
        self.model = None
        self.torch = None
        self.device = "cpu"
        self.class_names: list[str] = []
        self.label_mappings: list[dict] = []
        self.load_error: str | None = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.transform is not None and bool(self.class_names)

    def load(self) -> None:
        try:
            import timm
            import torch
            from huggingface_hub import hf_hub_download
            from torchvision import transforms

            self.torch = torch
            if self.device_request == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            elif self.device_request.isdigit():
                self.device = f"cuda:{self.device_request}"
            else:
                self.device = self.device_request

            configured_path = Path(self.model_name).expanduser()
            if configured_path.is_file():
                checkpoint_path = configured_path
            else:
                checkpoint_path = Path(
                    hf_hub_download(self.model_name, filename=self.CHECKPOINT_FILENAME)
                )

            # weights_only=True prevents arbitrary pickled code from running.
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            architecture = str(checkpoint.get("model_arch") or checkpoint.get("backbone") or "efficientnet_b3")
            number_of_classes = int(checkpoint.get("num_classes", 196))
            self.class_names = [str(name) for name in checkpoint["class_names"]]
            self.label_mappings = list(checkpoint.get("label_mappings", []))
            self.model = timm.create_model(architecture, pretrained=False, num_classes=number_of_classes)
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model_state")
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
            image_size = int(checkpoint.get("image_size", 300))
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )
        except Exception as exc:  # noqa: BLE001 - optional classifier failure must not stop detection.
            self.load_error = f"Vehicle classifier unavailable: {exc}"
            self.transform = None
            self.model = None

    def predict_batch(self, bgr_crops: Iterable) -> list[VehiclePrediction]:
        crops = list(bgr_crops)
        if not crops or not self.available or self.torch is None:
            return []
        images = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in crops]
        inputs = self.torch.stack([self.transform(image) for image in images]).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(inputs)
            probabilities = self.torch.softmax(logits, dim=-1)
            confidence, indices = probabilities.max(dim=-1)

        predictions: list[VehiclePrediction] = []
        for score, index in zip(confidence.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            label = self.class_names[index] if 0 <= index < len(self.class_names) else str(index)
            if 0 <= index < len(self.label_mappings):
                mapping = self.label_mappings[index]
                predictions.append(
                    VehiclePrediction(
                        label=label,
                        confidence=float(score),
                        make=str(mapping.get("make")) if mapping.get("make") else None,
                        model=str(mapping.get("model")) if mapping.get("model") else None,
                        year=str(mapping.get("year")) if mapping.get("year") else None,
                    )
                )
            else:
                predictions.append(self.parse_label(label, float(score)))
        return predictions

    @staticmethod
    def parse_label(label: str, confidence: float) -> VehiclePrediction:
        cleaned = label.replace("_", " ").strip()
        year_match = re.search(r"\b(19|20)\d{2}\b", cleaned)
        year = year_match.group(0) if year_match else None
        without_year = re.sub(r"\b(19|20)\d{2}\b", "", cleaned).strip(" -,")
        parts = without_year.split()
        make = parts[0] if parts else None
        model = " ".join(parts[1:]) if len(parts) > 1 else None
        return VehiclePrediction(cleaned, confidence, make, model, year)
