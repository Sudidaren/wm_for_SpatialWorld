"""Full-frame RF-DETR plus trained monocular depth for the WM observation API.

This adapter does not select a checkpoint or threshold, or enable task runs.
Callers must supply their validated detector and explicit score threshold.
"""
import math
import os
from pathlib import Path

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

    @torch.inference_mode()
    def embed(self, rgb):
        """Place fingerprint: L2-normalised DINOv2 CLS of this frame.

        Reuses the depth head's frozen encoder, so loop closure costs one extra
        encoder forward and no extra weights.  Used only for "have I been here
        before" -- it never reads anything but the RGB the agent was shown.
        """
        h, w = rgb.shape[:2]
        image = Image.fromarray(rgb).resize((self.resolution, self.resolution))
        x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255)
        x = x.permute(2, 0, 1)[None].to(self.device)
        with torch.autocast(self.device.type, enabled=False):
            _, cls = self.depth_model._patch_features(x)
        v = cls[0].float().cpu().numpy()
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v


class RFDETRDA2Runtime:
    """Same interface as :class:`RFDETRDepthRuntime`, but the depth comes from
    the public Depth-Anything-V2 *metric indoor* model instead of our head.

    Measured on 250 held-out evaluation frames with the project's own depth
    tool (tools/eval_depth_calibration.py), median object error:

        our v2 head   1.070 m      our v3 head   0.989 m
        DA2-Small     0.444 m

    and with a *known* floor mask, DA2 reaches 0.110 m where our head reaches
    0.19-0.24 m -- its relative depth is simply better.  It is 99 MB, 24.8 M
    parameters and ~18 ms/frame at 518 on the local laptop GPU, so this is a
    drop-in swap rather than a new dependency; the detector is unchanged.

    The model is a public checkpoint trained on Hypersim/NYU-style indoor
    data.  Nothing about the evaluation rooms is used: no fine-tuning, no
    calibration, no eval frames enter training.
    """

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, detector_checkpoint, threshold, device="cpu",
                 da2_name="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                 size=518, type_names=None):
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Expected a finite score threshold in [0,1]")
        from transformers import DepthAnythingForDepthEstimation

        self.device = torch.device(device)
        # Prefer the copy that ships with the repo: a deployment must not
        # depend on the HF cache layout, HF_HOME, or a reachable mirror.
        local = os.environ.get("LIGHTWM_DA2_PATH") or str(
            Path(__file__).resolve().parents[1]
            / "checkpoints" / "da2_metric_indoor_small")
        source = local if Path(local, "model.safetensors").is_file() else da2_name
        try:
            self.model = DepthAnythingForDepthEstimation.from_pretrained(
                source).to(self.device).eval().requires_grad_(False)
        except Exception as exc:                       # pragma: no cover
            raise SystemExit(
                f"DA2 weights unavailable ({exc}). Ship them to "
                f"{local} (config.json + model.safetensors + "
                f"preprocessor_config.json, 99 MB) or point LIGHTWM_DA2_PATH "
                f"at a copy.")
        self.resolution = int(size)
        self.obj_thr = threshold
        # DA2's metric bias.  Measured on the *non-evaluation* rooms only
        # (pooled pred/gt = 1.3816 over n=1251 objects, IQR 1.294-1.482) and
        # the same ratio holds on the held-out rooms (1.32-1.43), so a single
        # constant transfers: median object error on the held-out rooms falls
        # from 0.658 m to 0.210 m with everything else unchanged.  It is a
        # property of the public checkpoint, not of any evaluation room.
        # LIGHTWM_DA2_SCALE overrides it (1.0 = leave the model untouched).
        try:
            self.scale_const = float(os.environ.get("LIGHTWM_DA2_SCALE", "1.3816"))
        except ValueError:
            self.scale_const = 1.3816
        if not math.isfinite(self.scale_const) or self.scale_const <= 0:
            raise ValueError("LIGHTWM_DA2_SCALE must be a positive number")
        if type_names is None:
            # the detector's own fixed 117-class vocabulary; the inventory is
            # the cheap source (the index fallback would load 250k frames)
            from phase_b.perception_runtime import _default_types
            type_names = _default_types()
        self.type_names = list(type_names)
        self.type_ids = {label: i for i, label in enumerate(self.type_names)}
        self.detector = RFDETRDetector(detector_checkpoint, self.type_names,
                                       device=self.device)
        self.parameters = (sum(p.numel() for p in self.model.parameters())
                           + self.detector.parameters)

    @torch.inference_mode()
    def __call__(self, rgb):
        raw = self.detector(rgb)
        h, w = rgb.shape[:2]
        image = Image.fromarray(rgb).resize((self.resolution, self.resolution),
                                            Image.BILINEAR)
        x = (np.asarray(image, np.float32) / 255.0 - self.MEAN) / self.STD
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        depth = self.model(pixel_values=x).predicted_depth
        depth = F.interpolate(depth[:, None].float(), size=(h, w),
                              mode="bilinear", align_corners=False)[0, 0]
        depth = depth.cpu().numpy() / self.scale_const
        if not np.isfinite(depth).all():
            raise ValueError("Nonfinite monocular depth output")
        detections = []
        for detection in raw["detections"]:
            if detection["score"] < self.obj_thr:
                continue
            a, b, c, d = detection["bbox"]
            detections.append({**detection,
                               "class_id": self.type_ids.get(detection["type"], -1),
                               "center": [(a + c) / 2, (b + d) / 2],
                               "size": [c - a, d - b]})
        return {"detections": detections, "depth": depth, "rgb_shape": (h, w),
                "detection_views": 1, "encoder_forwards": 1,
                "backend": "rfdetr_da2", "precision": "fp32"}

    @torch.inference_mode()
    def embed(self, rgb):
        """Place fingerprint from DA2's encoder.

        Its backbone returns feature maps (no CLS token), so the fingerprint is
        the global average of the deepest map -- still a pure function of the
        RGB the agent was shown, which is all loop closure is allowed to use.
        """
        image = Image.fromarray(rgb).resize((self.resolution, self.resolution),
                                            Image.BILINEAR)
        x = (np.asarray(image, np.float32) / 255.0 - self.MEAN) / self.STD
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        out = self.model.backbone(pixel_values=x)
        maps = getattr(out, "feature_maps", None)
        if not maps:
            return None
        fm = maps[-1]
        # (B, tokens, C) with token 0 = CLS; average the patch tokens
        v = (fm[0, 1:].mean(dim=0) if fm.dim() == 3
             else fm.mean(dim=(2, 3))[0]).float().cpu().numpy()
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v
