"""Noisy observer: degrade simulator ground truth into what a real world model
would know from noisy perception + its own action log.

Sources replaced (layer 1):
  - agent pose        : odometry drift (accumulating gaussian noise)
  - object positions  : only objects the world model has SEEN are known;
                        estimates carry gaussian noise; unseen objects are absent
  - visibility        : detector miss rate (visible -> not seen)
  - forgetting        : objects unseen for N steps get re-noised (uncertain)
  - hand state        : inferred from the action log (Pickup/Put/Drop success),
                        with an optional error rate (state-hallucination risk)
  - container content : inferred from the action log (PutObject success)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional


class NoisyObserver:
    def __init__(
        self,
        *,
        position_sigma: float = 0.1,
        odom_drift: float = 0.02,
        visibility_miss_rate: float = 0.2,
        forget_steps: int = 10,
        hand_from_action_log: bool = True,
        hand_error_rate: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.position_sigma = position_sigma
        self.odom_drift = odom_drift
        self.visibility_miss_rate = visibility_miss_rate
        self.forget_steps = forget_steps
        self.hand_from_action_log = hand_from_action_log
        self.hand_error_rate = hand_error_rate
        self._rng = random.Random(seed)
        self._step = 0
        self._odom_error = [0.0, 0.0, 0.0]
        self._map: Dict[str, dict] = {}  # objectId -> {"type", "pos", "last_seen"}
        self._holding: Optional[str] = None
        self._contents: Dict[str, set] = {}  # container objectId -> {objectId}

    # ------------------------------------------------------------------
    def observe(
        self,
        metadata: Dict,
        action: Optional[Dict] = None,
        error_message: Optional[str] = None,
    ) -> Dict:
        self._step += 1
        action = action or {}
        action_name = action.get("action_name")
        ok = not error_message

        # --- agent pose with odometry drift ---
        agent = dict(metadata.get("agent") or {})
        pos = dict(agent.get("position") or {})
        if self.odom_drift > 0:
            self._odom_error = [
                e + self._rng.gauss(0, self.odom_drift) for e in self._odom_error
            ]
        agent["position"] = {
            k: (pos.get(k) or 0) + self._odom_error[i] for i, k in enumerate(("x", "y", "z"))
        }

        real_objects = metadata.get("objects", []) or []

        # --- update hand / container contents from the action log ---
        if self.hand_from_action_log:
            self._update_from_action(
                action_name, ok, action.get("object_type"), real_objects
            )

        # --- object registry: only seen objects enter the map ---
        out_objects = []
        for o in real_objects:
            oid = o.get("objectId")
            otype = o.get("objectType")
            vis = bool(o.get("visible"))
            if vis and self._rng.random() < self.visibility_miss_rate:
                vis = False  # detector miss
            o2 = dict(o)
            o2["visible"] = vis
            if oid and vis:
                p = o.get("position") or {}
                if oid not in self._map:
                    self._map[oid] = {"type": otype, "pos": None, "last_seen": self._step}
                noisy = [
                    (p.get(k) or 0) + self._rng.gauss(0, self.position_sigma)
                    for k in ("x", "y", "z")
                ]
                self._map[oid]["pos"] = noisy
                self._map[oid]["last_seen"] = self._step

            if oid in self._map and self._map[oid]["pos"] is not None:
                entry = self._map[oid]
                unseen = self._step - entry["last_seen"]
                if unseen > self.forget_steps and not vis:
                    # memory decay: re-noise the estimate (uncertain, not deleted)
                    entry["pos"] = [
                        v + self._rng.gauss(0, self.position_sigma * (1 + unseen - self.forget_steps))
                        for v in entry["pos"]
                    ]
                o2["position"] = {
                    "x": entry["pos"][0],
                    "y": entry["pos"][1],
                    "z": entry["pos"][2],
                }
                # container contents from the action log (if any)
                contents = self._contents.get(oid)
                if contents:
                    o2["parentReceptacles"] = list(contents)
                out_objects.append(o2)

        metadata2 = dict(metadata)
        metadata2["agent"] = agent
        metadata2["objects"] = out_objects

        if self.hand_from_action_log:
            inv = []
            if self._holding is not None:
                inv.append(
                    {
                        "objectId": self._holding,
                        "objectType": str(self._holding).split("|")[0],
                    }
                )
            metadata2["inventoryObjects"] = inv
        return metadata2

    # ------------------------------------------------------------------
    def _update_from_action(self, action_name, ok, object_type, real_objects) -> None:
        if not action_name:
            return
        # optional hand-state hallucination error
        if self.hand_error_rate > 0 and self._rng.random() < self.hand_error_rate:
            self._holding = None

        if action_name == "PickupObject" and ok:
            # picked object = nearest visible object of the action's type
            vis = [
                o for o in real_objects
                if o.get("objectType") == object_type and o.get("visible")
            ]
            if vis:
                self._holding = min(
                    vis, key=lambda o: float(o.get("distance") or 1e9)
                ).get("objectId")
        elif action_name in ("PutObject", "DropHandObject", "ThrowObject") and ok:
            if action_name == "PutObject" and self._holding is not None:
                vis = [
                    o for o in real_objects
                    if o.get("objectType") == object_type and o.get("visible")
                ]
                if vis:
                    target = min(
                        vis, key=lambda o: float(o.get("distance") or 1e9)
                    ).get("objectId")
                    self._contents.setdefault(target, set()).add(self._holding)
            self._holding = None
