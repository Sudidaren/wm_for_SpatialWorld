"""FP32 full-frame RF-DETR detector with the existing 117-class taxonomy."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from phase_b.zoom import merge_detections


class RFDETRDetector:
    def __init__(self, checkpoint, labels, device="cpu"):
        self.labels = list(labels)
        if len(self.labels) != len(set(self.labels)) or len(self.labels) != 117:
            raise ValueError("Expected the fixed 117-class vocabulary")
        self.device = torch.device(device)
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RF-DETR checkpoint not found: {checkpoint}")
        # Upstream import changes this process-wide setting. Preserve the
        # caller's arithmetic policy, including when model loading fails.
        precision = torch.get_float32_matmul_precision()
        try:
            from rfdetr import RFDETRSmall
            self.wrapper = RFDETRSmall(pretrain_weights=str(checkpoint),
                num_classes=len(self.labels), device=str(self.device), amp=False)
        finally:
            torch.set_float32_matmul_precision(precision)
        if list(self.wrapper.class_names) != self.labels:
            raise ValueError("Checkpoint class names differ from evaluation vocabulary")
        self.model = self.wrapper.model.model.float().eval()
        if self.wrapper.model.resolution != 512:
            raise ValueError("This experiment fixes RF-DETR input resolution to 512")
        self.parameters = sum(p.numel() for p in self.model.parameters())
        if self.parameters != 32208316:
            raise ValueError(f"Unexpected detector parameter count: {self.parameters}")

    @torch.inference_mode()
    def __call__(self, rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected HWC uint8 RGB")
        # Request all native top-300 proposals so cache threshold comparisons can
        # use >= consistently, including scores exactly at the smallest grid value.
        with torch.autocast(self.device.type, enabled=False):
            result = self.wrapper.predict(Image.fromarray(rgb), threshold=0.0, include_source_image=False)
        detections = []
        background = invalid = 0
        for box, score, class_id in zip(result.xyxy, result.confidence, result.class_id):
            class_id, score = int(class_id), float(score)
            if class_id == len(self.labels):
                background += 1
                continue
            if not 0 <= class_id < len(self.labels):
                raise ValueError(f"Unknown model class ID: {class_id}")
            coords = [float(value) for value in box]
            if not math.isfinite(score) or not all(math.isfinite(value) for value in coords):
                raise ValueError("Nonfinite RF-DETR output")
            if not 0 <= score <= 1:
                raise ValueError("RF-DETR confidence outside [0,1]")
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                invalid += 1
                continue
            detections.append({"bbox": coords, "type": self.labels[class_id], "score": score})
        # As in DINO, suppress float boxes before integer pixel conversion.
        # Keep vocabulary classes with excluded GT as possible false positives;
        # only the separate reserved background slot is removed above.
        kept = merge_detections(detections, threshold=.5, max_det=100)
        for detection in kept:
            detection["bbox"] = [int(value) for value in detection["bbox"]]
        return {"detections": kept, "native_proposals": len(result),
                "reserved_background_dropped": background, "invalid_float_boxes_dropped": invalid}
