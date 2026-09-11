"""Full-frame RF-DETR plus trained monocular depth for the WM observation API.

This adapter does not select a checkpoint or threshold, or enable task runs.
Callers must supply their validated detector and explicit score threshold.
"""
import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from phase_b.perception_runtime import PerceptionRuntime
from phase_b.rfdetr_runtime import RFDETRDetector


class RFDETRDepthRuntime:
    def __init__(self, depth_checkpoint, detector_checkpoint, threshold, device="cpu"):
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Expected a finite score threshold in [0,1]")
        base = PerceptionRuntime(str(depth_checkpoint), device=str(device), zoom=False)
        self.depth_model = base.model
        for name in list(self.depth_model._modules):
            if name not in {"encoder", "depth_head"}:
                delattr(self.depth_model, name)
        self.depth_model.float().eval().requires_grad_(False)
        self.resolution = base.resolution
        self.device = torch.device(base.device)
        self.type_names = list(base.type_names)
        self.type_ids = {label: i for i, label in enumerate(self.type_names)}
        self.detector = RFDETRDetector(detector_checkpoint, self.type_names, device=self.device)
        self.obj_thr = threshold
        self.parameters = self.detector.parameters + sum(p.numel() for p in self.depth_model.parameters())
        if self.parameters != 54962173:
            raise ValueError(f"Unexpected combined parameter count: {self.parameters}")

    @torch.inference_mode()
    def __call__(self, rgb):
        raw = self.detector(rgb)  # Validates uint8 HWC RGB before depth preprocessing.
        h, w = rgb.shape[:2]
        image = Image.fromarray(rgb).resize((self.resolution, self.resolution))
        x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)[None].to(self.device)
        with torch.autocast(self.device.type, enabled=False):
            depth = self.depth_model.depth(x)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)[0, 0].float().cpu().numpy()
        if not np.isfinite(depth).all():
            raise ValueError("Nonfinite monocular depth output")
        detections = []
        for detection in raw["detections"]:
            if detection["score"] < self.obj_thr:
                continue
            a, b, c, d = detection["bbox"]
            detections.append({**detection, "class_id": self.type_ids[detection["type"]],
                               "center": [(a+c)/2, (b+d)/2], "size": [c-a, d-b]})
        return {"detections": detections, "depth": depth, "rgb_shape": (h, w),
                "detection_views": 1, "encoder_forwards": 2,
                "backend": "rfdetr_small_depth", "precision": "fp32"}
