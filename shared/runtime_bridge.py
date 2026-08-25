"""Runtime bridge: how the learned spatial memory + VoI gate plug into the
LightWM runner (memory_probe.py).  Kept behind config flags; the running
config only enables it once offline validation passes.

Contract with memory_probe:
  - WorldModel owns odometry pose + object slots.
  - SpatialMemoryAdapter consumes detections + (predicted or GT) depth and
    maintains persistent 3D anchors.
  - HintRenderer turns anchors + occupancy into text candidates.
  - GateAdapter scores candidates and emits 0-1 hints per step.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from geometry import rel_direction  # noqa: E402
from spatial_memory import SpatialMemory  # noqa: E402

_DIR_ZH = {
    "front": "正前方", "back": "正后方", "left": "正左方", "right": "正右方",
}
_DIR_ZH_MIX = {
    "front-left": "左前方", "front-right": "右前方",
    "back-left": "左后方", "back-right": "右后方",
}
_HEIGHT_ZH = {"high": "偏高（在你上方）", "mid": "大致与视线同高", "low": "偏低（在你下方）"}
_DIST_ZH = {"near": "很近(<1m)", "mid": "中等距离(1-3m)", "far": "较远(>3m)"}


class HintRenderer:
    """Render anchor queries into detour-aware Chinese/English hints.

    Distinguishes straight-line distance from path distance: during a detour
    the straight-line distance may first grow then shrink; the hint says so
    instead of reporting "you are moving away".
    """

    def __init__(self, lang: str = "zh"):
        self.lang = lang

    def render(self, q: Dict[str, Any], detour: Optional[bool] = None) -> str:
        d = q["direction"]
        yaw = q["yaw_deg"]
        if abs(yaw) <= 30:
            dirw = "正前方"
        elif abs(yaw) >= 150:
            dirw = "正后方"
        elif yaw > 0:
            dirw = "右前方" if abs(yaw) <= 60 else "正右方"
        else:
            dirw = "左前方" if abs(yaw) <= 60 else "正左方"
        if q["pitch_deg"] > 12:
            height = "靠上"
        elif q["pitch_deg"] < -12:
            height = "靠下"
        else:
            height = "同高"
        base = (f"{q['type']} 在{dirw}约 {q['distance_m']:.1f} m"
                f"（{height}，{q['height_band']}位），"
                f"置信度 {q['confidence']:.0%}（看过 {q['n_views']} 次）")
        if detour:
            base += ("；当前被遮挡，需要绕行。绕行时直线距离会先增大后减小，"
                     "这是正常的，请继续绕行而不是折返")
        return base


class SpatialMemoryAdapter:
    def __init__(self, lang: str = "zh"):
        self.mem = SpatialMemory()
        self.renderer = HintRenderer(lang)

    def update(self, detections, agent_pos, yaw, depth, step, horizon=0.0):
        return self.mem.update(detections, agent_pos, yaw, depth, step, horizon)

    def hint_for(self, obj_type: str, agent_pos, yaw, detour: bool = False,
                 horizon: float = 0.0
                 ) -> Optional[str]:
        q = self.mem.query(obj_type, agent_pos, yaw, horizon)
        if q is None:
            return None
        return self.renderer.render(q, detour)

    def hint_all(self, agent_pos, yaw, targets: List[str],
                 horizon: float = 0.0,
                 detour: bool = False) -> List[str]:
        out = []
        for t in targets:
            h = self.hint_for(t, agent_pos, yaw, detour, horizon)
            if h:
                out.append(h)
        return out


class GateAdapter:
    """Learned VoI gate: scores candidate hints, emits at most one.

    Candidates are strings from SpatialMemoryAdapter + rule pool.  The gate
    network (phase_c) outputs P(prevents-error); threshold is configurable.
    Falls back to the rule baseline when no model file is provided.
    """

    def __init__(self, ckpt: Optional[str] = None, threshold: float = 0.3):
        self.threshold = threshold
        self.model = None
        if ckpt and os.path.exists(ckpt):
            import torch
            from phase_c.train_gate import GateNet
            state = torch.load(ckpt, map_location="cpu")
            dim = state["mlp.0.weight"].shape[1]
            self.model = GateNet(dim)
            self.model.load_state_dict(state)
            self.model.eval()

    def decide(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """candidates: [{text, features (np.ndarray), type}] -> top-1 if
        score > threshold, else []."""
        if not candidates:
            return []
        if self.model is None:
            return [candidates[0]]  # rule fallback: first priority candidate
        import torch
        X = torch.from_numpy(np.stack([c["features"] for c in candidates]))
        with torch.no_grad():
            scores = torch.sigmoid(self.model(X)).numpy()
        for c, s in zip(candidates, scores):
            c["score"] = float(s)
        best = max(candidates, key=lambda c: c.get("score", 0))
        return [best] if best.get("score", 0) >= self.threshold else []
