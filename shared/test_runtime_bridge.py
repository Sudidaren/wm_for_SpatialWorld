"""Smoke test: hint rendering incl. detour wording, and gate fallback."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from runtime_bridge import GateAdapter, HintRenderer, SpatialMemoryAdapter  # noqa


def main():
    r = HintRenderer()
    q = {"type": "Microwave", "distance_m": 2.1, "yaw_deg": 35,
         "pitch_deg": 15, "height_band": "high", "confidence": 0.8,
         "n_views": 3, "direction": "right-up"}
    print(r.render(q))
    print(r.render(q, detour=True))
    assert "绕行" in r.render(q, detour=True)
    g = GateAdapter(ckpt=None)
    cands = [{"text": "hand occupied", "features": [0.0] * 14, "type": 0}]
    assert g.decide(cands) == cands
    print("OK")


if __name__ == "__main__":
    main()
