"""A real (non-oracle) world model built from sensor streams.

Perception inputs (the world model's own sensors):
  - instance segmentation frame + a class-color vocabulary (metadata.colors),
    i.e. "a segmentation model that knows object classes";
  - metric depth frame;
  - agent pose from odometry (sim agent position/rotation for now).

The world model does NOT read per-step semantic metadata (objects/positions/
visibility/inventory). It builds and maintains:
  - persistent object slots (seen objects, refined on re-observation);
  - causal state from its own action log (hand holding, container contents);
  - the agent pose.

Its output is a metadata-like dict consumed by MemoryProbe, so the whole
injection / directive / waypoint / progress / done-gate pipeline is reused.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

_FURNITURE_TYPES = frozenset({
    "CounterTop", "Cabinet", "Drawer", "Shelf", "Fridge",
    "StoveBurner", "Sink", "Dresser", "Desk", "Table",
})


def _similarity2d(src, dst):
    """2-D similarity (scale, R, t) with dst ~= scale*R*src + t (Umeyama)."""
    ms = src.mean(axis=0)
    md = dst.mean(axis=0)
    src0 = src - ms
    dst0 = dst - md
    var_src = float(np.mean(np.sum(src0 * src0, axis=1)))
    if var_src < 1e-9:
        return 1.0, np.eye(2), (0.0, 0.0)
    H = (dst0.T @ src0) / len(src)
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    scale = float(np.sum(S)) / var_src
    t = md - scale * (R @ ms)
    return scale, R, t


class WorldModel:
    def __init__(
        self,
        targets: Optional[List[str]] = None,
        *,
        fov: float = 60.0,
        width: int = 800,
        height: int = 600,
        hand_from_action_log: bool = True,
        pose_from_action_log: bool = True,
        pose_initial: str = "origin",
        move_magnitudes: Optional[Dict[str, float]] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        self.targets = targets or []
        self.fov = fov
        self.width = width
        self.height = height
        self.hand_from_action_log = hand_from_action_log
        self.pose_from_action_log = pose_from_action_log
        self.pose_initial = pose_initial
        self.move_magnitudes = move_magnitudes or {}
        self.checkpoint_dir = checkpoint_dir
        self._vocab: Dict[tuple, str] = {}   # (r,g,b) -> objectId
        self._furniture_vocab: Dict[tuple, str] = {}  # (r,g,b) -> furniture instance
        self._furniture: Dict[str, dict] = {}  # furniture oid -> slot
        self._floor_colors: set = set()      # colors of Floor masks (skip from occupancy)
        self._slots: Dict[str, dict] = {}    # objectId -> slot
        self._slot_seq = 0
        self._holding: Optional[str] = None
        self._contents: Dict[str, set] = {}  # container oid -> {oid}
        self._states: Dict[str, dict] = {}  # oid -> {state_field: bool}
        self._affordances: Dict[str, dict] = {}  # type -> action -> {ok, fail}
        self._occupancy: Dict[tuple, dict] = {}  # (cx,cz) -> {"count","last"}
        self._walkable: Dict[tuple, dict] = {}  # (cx,cz) -> observed floor cells
        self._visited: set = set()               # agent cells (walkable)
        self._pose: Optional[List[float]] = None  # [x, y, z, yaw] self-built
        self._blocked_from_moves: set = set()    # move-failure target cells
        self._prev_agent: Optional[tuple] = None
        self._step = 0
        self._pose_sigma = 0.0          # odometry uncertainty (meters)
        self.landmark_correction = True
        self._perception = None         # PerceptionRuntime (dense+depth)
        if not pose_from_action_log and \
                os.environ.get("LIGHTWM_ALLOW_SIM_POSE") != "1":
            raise ValueError(
                "pose_from_action_log=False reads simulator pose; runtime is "
                "not allowed. Set LIGHTWM_ALLOW_SIM_POSE=1 only for offline "
                "measurement scripts.")

    def attach_perception(self, perception) -> None:
        """Attach the DINOv2 dense+depth runtime used as the model's eyes."""
        self._perception = perception
        self._vocab = {}  # sim color vocabulary is no longer used
        self._furniture_vocab = {}
        self._floor_colors = set()

    # ------------------------------------------------------------------
    def observe(
        self,
        observation: Any,
        action: Optional[Dict] = None,
        error_message: Optional[str] = None,
        env: Any = None,
    ) -> Dict:
        self._step += 1
        action = action or {}
        action_name = action.get("action_name")
        ok = not error_message
        controller = getattr(env, "controller", None)
        if controller is None:
            return {}

        meta = getattr(observation, "metadata", None) or {}
        agent_pos, agent_rot = self._resolve_pose(meta, action, error_message)
        ax, az = agent_pos.get("x"), agent_pos.get("z")
        if ax is not None and az is not None:
            self._visited.add((round(ax / 0.25), round(az / 0.25)))
            if not self.pose_from_action_log:
                cur_agent = (round(ax / 0.25), round(az / 0.25))
                # a failed move (position unchanged) marks the target cell blocked
                if (
                    self._prev_agent == cur_agent
                    and error_message
                    and action_name in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
                ):
                    dcx, dcz = self._move_dir(action_name, float(agent_rot.get("y", 0.0)))
                    self._blocked_from_moves.add(
                        (cur_agent[0] + dcx, cur_agent[1] + dcz)
                    )
                self._prev_agent = cur_agent

        # build the class-color vocabulary once
        if not self._vocab:
            self._build_vocab(meta)

        ev = controller.last_event
        seg = getattr(ev, "instance_segmentation_frame", None)
        depth = getattr(ev, "depth_frame", None)

        visible_ids = set()
        if self._perception is not None:
            frame = getattr(ev, "frame", None)
            if frame is None:
                frame = getattr(ev, "cv2img", None)
            if frame is not None:
                res = self._perception(frame)
                visible_ids = self._perceive_detections(
                    res["detections"], res["depth"], agent_pos, agent_rot)
                # walkable evidence is optional in dense mode; skip oracle seg
        elif seg is not None and depth is not None:
            raise RuntimeError(
                "sim instance-segmentation perception is disabled at runtime; "
                "attach a PerceptionRuntime (dense+depth) instead.")

        # causal state from the action log
        if self.hand_from_action_log:
            self._update_affordances(
                action_name, action.get("object_type"), ok, error_message
            )
            self._update_action_log(
                action_name, ok, action.get("object_type"), visible_ids
            )

        # move held objects to the hand pose
        if self._holding is not None and self._holding in self._slots:
            hand = self._hand_position(agent_pos, agent_rot)
            self._slots[self._holding]["pos"] = hand

        meta = self._to_metadata(agent_pos, agent_rot, visible_ids)

        # periodic disk checkpoint so long tasks can resume after an OOM
        if self.checkpoint_dir and self._step % 50 == 0:
            try:
                self.save(os.path.join(self.checkpoint_dir, "world_model_ckpt.json"))
            except Exception as exc:
                print(f"⚠️ WorldModel checkpoint failed at step {self._step}: {exc}")

        return meta

    def _perceive_detections(
        self, detections, depth_map, agent_pos, agent_rot
    ) -> set:
        """Build/refine anchors from dense detections + monocular depth only."""
        H, W = depth_map.shape[:2]
        matched = []  # (stored_pos, measured_pos) for landmark correction
        visible_ids = set()
        for det in detections:
            cx, cy = det["center"]
            u, v = int(round(cx)), int(round(cy))
            if not (0 <= u < W and 0 <= v < H):
                continue
            z = float(depth_map[v, u])
            if z <= 0 or not math.isfinite(z):
                continue
            pos = self._unproject(u, v, z, agent_pos, agent_rot)
            oid = self._match_or_create_slot(det["type"], pos)
            if oid is None:
                continue
            slot = self._slots[oid]
            if slot.get("seen", 0) > 0:
                matched.append((list(slot["pos"]), pos))
            a = 0.5
            slot["pos"] = [a * slot["pos"][i] + (1 - a) * pos[i]
                           for i in range(3)]
            slot["seen"] = slot.get("seen", 0) + 1
            slot["last_seen"] = self._step
            obs_noise = 0.12 + 0.05 * z
            slot["sigma"] = 0.5 * slot.get("sigma", 0.3) + 0.5 * obs_noise
            slot["score"] = max(slot.get("score", 0.0), det.get("score", 0.0))
            visible_ids.add(oid)
        # odometry uncertainty grows every step; shrink after correction
        self._pose_sigma += 0.02
        if self.landmark_correction and len(matched) >= 2:
            self._correct_pose_with_landmarks(matched)
        return visible_ids

    def _match_or_create_slot(self, otype: str, pos) -> Optional[str]:
        """Match a detection to the nearest existing slot of the same type
        (instance IDs are unavailable), otherwise create a new slot."""
        best_id, best_d = None, 2.0  # 2 m association radius
        for oid, s in self._slots.items():
            if s["type"] != otype:
                continue
            d = math.hypot(s["pos"][0] - pos[0], s["pos"][2] - pos[2])
            if d < best_d:
                best_id, best_d = oid, d
        if best_id is not None:
            return best_id
        self._slot_seq += 1
        oid = f"det|{otype}|{self._slot_seq}"
        self._slots[oid] = {
            "type": otype, "pos": list(pos), "seen": 1,
            "last_seen": self._step, "sigma": 0.25,
            "score": 0.0,
        }
        return oid

    def _correct_pose_with_landmarks(self, pairs) -> None:
        """Solve a bounded (x, z, yaw) correction from re-observed anchors.
        stored anchors (trusted reference) vs newly measured positions."""
        src = np.array([[p[1][0], p[1][2]] for p in pairs], dtype=np.float64)
        dst = np.array([[p[0][0], p[0][2]] for p in pairs], dtype=np.float64)
        if len(src) < 2:
            return
        sc, R, t = _similarity2d(src, dst)  # dst ~ sc*R*src + t
        # scale must stay ~1 (objects don't move)
        if abs(sc - 1.0) > 0.25:
            return
        dx, dz = float(t[0]), float(t[1])
        dyaw = float(math.degrees(math.atan2(R[1, 0], R[0, 0])))
        # bound one-step correction (prevent jumps from noisy detections)
        mag = math.hypot(dx, dz)
        if mag > 0.5 or abs(dyaw) > 12.0:
            return
        self._pose[0] += dx
        self._pose[2] += dz
        self._pose[3] = (self._pose[3] + dyaw) % 360
        self._pose_sigma = max(0.0, self._pose_sigma - 0.05)

    # ------------------------------------------------------------------
    def _update_affordances(self, action_name, object_type, ok, error_message) -> None:
        """Learn affordances from interaction outcomes (no static tables)."""
        if not action_name or not object_type:
            return
        if not ok and self._is_non_affordance_error(error_message):
            return  # positioning / prerequisite / state problem, not affordance
        e = self._affordances.setdefault(
            object_type, {}
        ).setdefault(action_name, {"ok": 0, "fail": 0})
        e["ok" if ok else "fail"] += 1

    @staticmethod
    def _is_non_affordance_error(error_message) -> bool:
        e = (error_message or "").lower()
        return any(
            kw in e
            for kw in (
                # positioning problems
                "not in view",
                "not visible",
                "too far",
                "out of reach",
                "not in range",
                "approach",
                # prerequisite / hand-state problems
                "hand already has",
                "already holding",
                "hand is empty",
                "not holding",
                "isn't holding",
                "nothing to drop",
                # state problems
                "must be off",
                "already on",
                "already open",
                "already closed",
                "can't look",
            )
        )

    def affordance(self, object_type: str, action_name: str) -> Optional[bool]:
        """Learned belief about whether an action works on an object type."""
        e = (self._affordances.get(object_type) or {}).get(action_name)
        if not e:
            return None  # unknown: not yet tried
        return e["ok"] > e["fail"]

    def affordance_summary(self) -> str:
        if not self._affordances:
            return ""
        parts = []
        for otype, acts in sorted(self._affordances.items()):
            for act, e in sorted(acts.items()):
                parts.append(f"{otype}·{act}:{'可' if e['ok'] > e['fail'] else '不可'}({e['ok']}/{e['fail']})")
        return "；".join(parts)

    # ------------------------------------------------------------------
    def _build_vocab(self, meta: Dict) -> None:
        self._floor_colors = set()
        for item in meta.get("colors") or []:
            name = item.get("name") or ""
            color = tuple(item.get("color") or ())
            if not name or len(color) != 3:
                continue
            otype = name.split("|")[0]
            if otype == "Floor":
                # ground plane: never an obstacle (matches instance or type-only rows)
                self._floor_colors.add(color)
                continue
            if "|" not in name:
                continue  # keep only instance-level ids (type-only rows skipped)
            if self.targets and otype not in self.targets:
                if otype in _FURNITURE_TYPES:
                    self._furniture_vocab[color] = name
                continue
            self._vocab[color] = name

    def _perceive(self, seg, depth, agent_pos, agent_rot) -> set:
        visible_ids = set()
        h, w = seg.shape[:2]
        unique = np.unique(seg.reshape(-1, 3), axis=0)
        for color in unique:
            mask = np.all(seg == color, axis=2)
            area = int(mask.sum())
            if area < 10:
                continue  # tiny speck
            ckey = (int(color[0]), int(color[1]), int(color[2]))
            if ckey in self._floor_colors:
                self._mark_walkable(mask, depth, agent_pos, agent_rot)
                continue  # ground plane: observed walkable, not an obstacle
            self._mark_occupancy(mask, depth, agent_pos, agent_rot)
            fname = self._furniture_vocab.get(ckey)
            if fname is not None:
                fy, fx = np.nonzero(mask)
                fu, fv = float(fx.mean()), float(fy.mean())
                fz = float(depth[int(fv), int(fu)])
                if fz > 0 and math.isfinite(fz):
                    fpos = self._unproject(fu, fv, fz, agent_pos, agent_rot)
                    fslot = self._furniture.get(fname)
                    if fslot is None:
                        self._furniture[fname] = {
                            "type": fname.split("|")[0],
                            "pos": fpos,
                            "seen": 1,
                            "last_seen": self._step,
                        }
                    else:
                        a = 0.5
                        fslot["pos"] = [
                            a * fslot["pos"][i] + (1 - a) * fpos[i]
                            for i in range(3)
                        ]
                        fslot["seen"] += 1
                        fslot["last_seen"] = self._step
                continue
            oid = self._vocab.get(ckey)
            if oid is None:
                continue
            ys, xs = np.nonzero(mask)
            u, v = float(xs.mean()), float(ys.mean())
            z = float(depth[int(v), int(u)])
            if z <= 0 or not math.isfinite(z):
                continue
            pos = self._unproject(u, v, z, agent_pos, agent_rot)
            slot = self._slots.get(oid)
            if slot is None:
                self._slots[oid] = {
                    "type": oid.split("|")[0],
                    "pos": pos,
                    "seen": 1,
                    "last_seen": self._step,
                }
            else:
                # refine on re-observation (moving average)
                a = 0.5
                slot["pos"] = [
                    a * slot["pos"][i] + (1 - a) * pos[i] for i in range(3)
                ]
                slot["seen"] += 1
                slot["last_seen"] = self._step
            visible_ids.add(oid)
        return visible_ids

    def _mark_occupancy(self, mask, depth, agent_pos, agent_rot) -> None:
        """Unproject mask extent pixels onto the 0.25m grid; repeat observations
        at the same cell build confidence that it is a static obstacle."""
        ys, xs = np.nonzero(mask)
        u0, u1, v0, v1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        samples = [
            (u0, v0), (u1, v0), (u0, v1), (u1, v1),
            ((u0 + u1) // 2, (v0 + v1) // 2),
        ]
        cells = set()
        for (u, v) in samples:
            z = float(depth[int(v), int(u)])
            if z <= 0 or not math.isfinite(z):
                continue
            wx, _wy, wz = self._unproject(u, v, z, agent_pos, agent_rot)
            cells.add((round(wx / 0.25), round(wz / 0.25)))
        for c in cells:
            e = self._occupancy.setdefault(
                c, {"count": 0, "last": 0, "consec": 0}
            )
            e["count"] += 1
            if e["last"] == self._step:
                pass  # duplicate within the same step: keep consec as-is
            elif e["last"] == self._step - 1:
                e["consec"] += 1  # observed in consecutive steps
            else:
                e["consec"] = 1
            e["last"] = self._step

    def _resolve_pose(self, meta: Dict, action: Optional[Dict], error_message: Optional[str]):
        """Agent pose from the action log (dead reckoning) by default, so no
        per-step simulator truth is read.  ``pose_initial=='sim'`` seeds the
        relative frame once from the first frame (offline eval only)."""
        if not self.pose_from_action_log:
            agent = meta.get("agent") or {}
            return (agent.get("position") or {}), (agent.get("rotation") or {})
        if self._pose is None:
            if self.pose_initial == "sim":
                agent = meta.get("agent") or {}
                pos = agent.get("position") or {}
                rot = agent.get("rotation") or {}
                self._pose = [
                    float(pos.get("x") or 0.0),
                    float(pos.get("y") or 0.0),
                    float(pos.get("z") or 0.0),
                    float(rot.get("y") or 0.0),
                ]
            else:
                self._pose = [0.0, 0.0, 0.0, 0.0]
        else:
            self._integrate_pose(action, error_message)
        x, y, z, yaw = self._pose
        return {"x": x, "y": y, "z": z}, {"y": yaw}

    def _integrate_pose(self, action: Optional[Dict], error_message: Optional[str]) -> None:
        """Dead-reckon the pose from the last action; a failed move marks the
        target cell blocked and does not move."""
        if self._pose is None:
            return
        action = action or {}
        name = action.get("action_name")
        if not name:
            return
        x, _y, z, yaw = self._pose
        if name in ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight"):
            mag = self._action_magnitude(name, action)
            dcx, dcz = self._move_dir(name, yaw)
            if error_message:
                self._blocked_from_moves.add(
                    (round(x / 0.25) + dcx, round(z / 0.25) + dcz)
                )
                return
            self._pose[0] = x + dcx * mag
            self._pose[2] = z + dcz * mag
        elif name in ("RotateLeft", "RotateRight"):
            try:
                deg = float(action.get("degrees") or action.get("magnitude") or 90.0)
            except (TypeError, ValueError):
                deg = 90.0
            # AI2-THOR convention: RotateRight increases yaw (clockwise)
            self._pose[3] = (yaw + (deg if name == "RotateRight" else -deg)) % 360

    def _action_magnitude(self, name: str, action: Dict) -> float:
        mag = action.get("magnitude")
        if mag is not None:
            try:
                return float(mag)
            except (TypeError, ValueError):
                pass
        m = self.move_magnitudes.get(name)
        if m:
            return float(m)
        return 0.5 if name in ("MoveAhead", "MoveBack") else 0.25

    def _mark_walkable(
        self, mask, depth, agent_pos, agent_rot, stride: int = 24
    ) -> None:
        """Record visible floor pixels as observed walkable cells, so the map
        grows exactly with what the agent has explored (no pre-built regions).
        The clear line of sight to each floor cell is also marked walkable."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return
        ax = float(agent_pos.get("x") or 0.0)
        az = float(agent_pos.get("z") or 0.0)
        cells = set()
        for u, v in zip(xs[::stride], ys[::stride]):
            z = float(depth[int(v), int(u)])
            if z <= 0 or not math.isfinite(z):
                continue
            wx, _wy, wz = self._unproject(int(u), int(v), z, agent_pos, agent_rot)
            cells.add((round(wx / 0.25), round(wz / 0.25)))
            self._mark_ray_free(ax, az, wx, wz)
        for c in cells:
            self._bump_walkable(c)

    def _mark_walkable_from_depth(
        self, depth, agent_pos, agent_rot, stride: int = 24
    ) -> None:
        """Depth-based floor evidence with a corrected y sign.  Only used when
        the pose carries an absolute camera height (sim-seeded offline eval);
        in relative-frame runs the segmentation floor path is authoritative."""
        h, w = depth.shape[:2]
        ay = float(agent_pos.get("y") or 0.0)
        if ay < 0.8:
            return
        ax = float(agent_pos.get("x") or 0.0)
        az = float(agent_pos.get("z") or 0.0)
        for v in range(0, h, stride):
            for u in range(0, w, stride):
                z = float(depth[v, u])
                if z <= 0 or not math.isfinite(z):
                    continue
                wx, wy, wz = self._unproject(u, v, z, agent_pos, agent_rot)
                wy_corr = 2 * ay - wy
                if wy_corr > 0.35:
                    continue  # above floor level -> not walkable ground
                c = (round(wx / 0.25), round(wz / 0.25))
                if c in self._occupancy:
                    continue  # an obstacle was seen there
                self._bump_walkable(c)
                self._mark_ray_free(ax, az, wx, wz)

    def _bump_walkable(self, c: tuple) -> None:
        e = self._walkable.setdefault(
            c, {"count": 0, "last": 0, "consec": 0}
        )
        e["count"] += 1
        if e["last"] == self._step:
            pass
        elif e["last"] == self._step - 1:
            e["consec"] += 1
        else:
            e["consec"] = 1
        e["last"] = self._step

    def _mark_ray_free(self, ax: float, az: float, tx: float, tz: float) -> None:
        """Mark the line of sight from the agent to a visible floor cell as
        observed walkable: seeing continuous floor means the ray is clear.
        Stops at the first cell already marked as an obstacle."""
        dist = math.hypot(tx - ax, tz - az)
        steps = max(1, int(round(dist / 0.25)))
        for i in range(1, steps + 1):
            t = i / steps
            cx = ax + (tx - ax) * t
            cz = az + (tz - az) * t
            c = (round(cx / 0.25), round(cz / 0.25))
            if c in self._occupancy:
                break
            self._bump_walkable(c)

    @staticmethod
    def _move_dir(name: str, yaw: float):
        rad = math.radians(yaw)
        fwd = (math.sin(rad), math.cos(rad))
        right = (math.cos(rad), -math.sin(rad))
        if name == "MoveAhead":
            return (round(fwd[0]), round(fwd[1]))
        if name == "MoveBack":
            return (-round(fwd[0]), -round(fwd[1]))
        if name == "MoveRight":
            return (round(right[0]), round(right[1]))
        if name == "MoveLeft":
            return (-round(right[0]), -round(right[1]))
        return (0, 0)

    def reachable_cells(self, max_margin: int = 32) -> set:
        """Explored-only walkable region: BFS from visited cells through cells
        the agent has actually seen (walked, or observed as floor).  Unknown
        cells are NOT traversable: the map grows exactly with exploration, so
        the planner never routes through territory GPT-5 has not seen."""
        start = list(self._visited)
        if not start:
            return set()
        xs = [c[0] for c in start]
        zs = [c[1] for c in start]
        x0, x1 = min(xs) - max_margin, max(xs) + max_margin
        z0, z1 = min(zs) - max_margin, max(zs) + max_margin

        evidence = set(self._visited)
        for c, e in self._walkable.items():
            if e.get("count", 0) >= 1:
                evidence.add(c)
        static = self.static_cells()
        reach = set(start)
        seen = set(start)
        frontier = list(start)
        while frontier:
            c = frontier.pop()
            for dcx, dcz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + dcx, c[1] + dcz)
                if n in seen:
                    continue
                if not (x0 <= n[0] <= x1 and z0 <= n[1] <= z1):
                    continue
                seen.add(n)
                if n in static:
                    continue
                if n in self._blocked_from_moves:
                    continue
                if n not in evidence:
                    continue  # unexplored: not traversable
                reach.add(n)
                frontier.append(n)
        return reach

    def static_cells(self, min_consec: int = 2) -> set:
        """Cells observed in >= ``min_consec`` consecutive steps: confirmed
        static obstacles. Consecutive (not merely repeated) observations make
        depth-edge speckle noise far less likely to be mislabeled as static.
        """
        return {
            c
            for c, e in self._occupancy.items()
            if e.get("consec", 0) >= min_consec
        }

    def known_walkable(self) -> set:
        """Cells with direct evidence of being traversable: walked cells plus
        observed floor, minus confirmed obstacles and move-blocked cells."""
        known = set(self._visited)
        for c, e in self._walkable.items():
            if e.get("count", 0) >= 1:
                known.add(c)
        for c, e in self._occupancy.items():
            if e.get("consec", 0) < 2:
                known.add(c)
        return known - self._blocked_from_moves

    def _unproject(self, u, v, z, agent_pos, agent_rot) -> List[float]:
        fx = (self.width / 2.0) / math.tan(math.radians(self.fov / 2.0))
        fy = fx
        cx, cy = self.width / 2.0, self.height / 2.0
        # scale pixel coords to the actual seg frame size
        x_rel = (u - cx) * z / fx
        y_rel = (v - cy) * z / fy
        z_rel = z
        yaw = math.radians(float(agent_rot.get("y", 0.0)))
        fwd = (math.sin(yaw), math.cos(yaw))
        right = (math.cos(yaw), -math.sin(yaw))
        return [
            (agent_pos.get("x") or 0) + right[0] * x_rel + fwd[0] * z_rel,
            (agent_pos.get("y") or 0) + y_rel,
            (agent_pos.get("z") or 0) + right[1] * x_rel + fwd[1] * z_rel,
        ]

    def _hand_position(self, agent_pos, agent_rot) -> List[float]:
        yaw = math.radians(float(agent_rot.get("y", 0.0)))
        fwd = (math.sin(yaw), math.cos(yaw))
        return [
            (agent_pos.get("x") or 0) + fwd[0] * 0.3,
            (agent_pos.get("y") or 0) + 0.4,
            (agent_pos.get("z") or 0) + fwd[1] * 0.3,
        ]

    def _update_action_log(self, action_name, ok, object_type, visible_ids) -> None:
        if not action_name:
            return
        # object states are not readable from vision: infer from outcomes
        if ok and action_name in (
            "OpenObject", "CloseObject", "ToggleObjectOn",
            "ToggleObjectOff", "SliceObject",
        ):
            oid = None
            for c, s in self._slots.items():
                if s["type"] == object_type and c in visible_ids:
                    oid = c
                    break
            if oid is not None:
                st = self._states.setdefault(oid, {})
                if action_name == "OpenObject":
                    st["isOpen"] = True
                elif action_name == "CloseObject":
                    st["isOpen"] = False
                elif action_name == "ToggleObjectOn":
                    st["isToggled"] = True
                elif action_name == "ToggleObjectOff":
                    st["isToggled"] = False
                elif action_name == "SliceObject":
                    st["isSliced"] = True
        if action_name == "PickupObject" and ok:
            # pick the nearest visible slot of the target type
            cands = [
                (oid, s)
                for oid, s in self._slots.items()
                if s["type"] == object_type and oid in visible_ids
            ]
            if cands:
                self._holding = cands[0][0]
        elif action_name in ("PutObject", "DropHandObject", "ThrowObject") and ok:
            if action_name == "PutObject" and self._holding is not None:
                cands = [
                    (oid, s)
                    for oid, s in self._slots.items()
                    if s["type"] == object_type and oid in visible_ids
                ]
                if cands:
                    target = cands[0][0]
                    self._contents.setdefault(target, set()).add(self._holding)
                    # the placed object moves to the target's position
                    self._slots[self._holding]["pos"] = list(self._slots[target]["pos"])
            self._holding = None

    def _to_metadata(self, agent_pos, agent_rot, visible_ids) -> Dict:
        objects_out = []
        ax, az = agent_pos.get("x"), agent_pos.get("z")
        for oid, slot in self._slots.items():
            pos = slot["pos"]
            dist = math.hypot(pos[0] - ax, pos[2] - az) if ax is not None else 0.0
            objects_out.append(
                {
                    "objectId": oid,
                    "objectType": slot["type"],
                    "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                    "visible": oid in visible_ids,
                    "distance": dist,
                    "distance_err": float(slot.get("sigma", 0.25)
                                          + self._pose_sigma),
                    "parentReceptacles": [
                        c
                        for c, contents in self._contents.items()
                        if oid in contents
                    ],
                    "states": dict(self._states.get(oid, {})),
                }
            )
        furn_out = []
        for oid, slot in self._furniture.items():
            fpos = slot["pos"]
            fdist = (
                math.hypot(fpos[0] - ax, fpos[2] - az)
                if ax is not None
                else 0.0
            )
            furn_out.append(
                {
                    "objectId": oid,
                    "objectType": slot["type"],
                    "position": {"x": fpos[0], "y": fpos[1], "z": fpos[2]},
                    "distance": fdist,
                    "seen": slot.get("seen", 1),
                    "last_seen": slot.get("last_seen", 0),
                }
            )
        inv = []
        if self._holding is not None:
            inv.append(
                {
                    "objectId": self._holding,
                    "objectType": str(self._holding).split("|")[0],
                }
            )
        return {
            "agent": {
                "position": dict(agent_pos),
                "rotation": dict(agent_rot),
            },
            "objects": objects_out,
            "furniture": furn_out,
            "inventoryObjects": inv,
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialize all learnable state to JSON (safe to reload later).

        All internal structures are plain dicts/sets of JSON-able values, so
        no pickling is needed.  ``path`` may include a directory that does not
        exist yet.
        """
        import json

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "targets": self.targets,
            "fov": self.fov,
            "width": self.width,
            "height": self.height,
            "hand_from_action_log": self.hand_from_action_log,
            "_step": self._step,
            "_floor_colors": [list(c) for c in sorted(self._floor_colors)],
            "_vocab": [
                {"color": list(color), "name": name}
                for color, name in sorted(self._vocab.items())
            ],
            "_slots": self._slots,
            "_holding": self._holding,
            "_contents": {k: sorted(v) for k, v in self._contents.items()},
            "_states": {k: dict(v) for k, v in self._states.items()},
            "_furniture": self._furniture,
            "_affordances": self._affordances,
            "_occupancy": {
                f"{x},{z}": e for (x, z), e in self._occupancy.items()
            },
            "_visited": sorted(self._visited),
            "_walkable": {
                f"{x},{z}": e for (x, z), e in self._walkable.items()
            },
            "_blocked_from_moves": sorted(self._blocked_from_moves),
            "_pose": list(self._pose) if self._pose else None,
            "_prev_agent": list(self._prev_agent) if self._prev_agent else None,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        targets: Optional[List[str]] = None,
        fov: float = 60.0,
        width: int = 800,
        height: int = 600,
        hand_from_action_log: bool = True,
        checkpoint_dir: Optional[str] = None,
    ) -> "WorldModel":
        """Restore a world model from a JSON checkpoint produced by save()."""
        import json

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        wm = cls(
            targets=targets if targets is not None else data.get("targets") or [],
            fov=fov,
            width=width,
            height=height,
            hand_from_action_log=hand_from_action_log,
            checkpoint_dir=checkpoint_dir,
        )
        wm._vocab = {tuple(c["color"]): c["name"] for c in data.get("_vocab") or []}
        wm._floor_colors = {
            tuple(c) for c in data.get("_floor_colors") or []
        }
        wm._slots = data.get("_slots") or {}
        wm._holding = data.get("_holding")
        wm._contents = {
            k: set(v) for k, v in (data.get("_contents") or {}).items()
        }
        wm._states = {
            k: dict(v) for k, v in (data.get("_states") or {}).items()
        }
        wm._furniture = data.get("_furniture") or {}
        wm._affordances = data.get("_affordances") or {}
        wm._occupancy = {
            tuple(int(x) for x in k.split(",")): e
            for k, e in (data.get("_occupancy") or {}).items()
        }
        wm._visited = {tuple(c) for c in data.get("_visited") or []}
        wm._walkable = {
            tuple(int(x) for x in k.split(",")): e
            for k, e in (data.get("_walkable") or {}).items()
        }
        wm._pose = (
            [float(v) for v in data["_pose"]] if data.get("_pose") else None
        )
        wm._blocked_from_moves = {
            tuple(c) for c in data.get("_blocked_from_moves") or []
        }
        wm._prev_agent = (
            tuple(data["_prev_agent"]) if data.get("_prev_agent") else None
        )
        wm._step = int(data.get("_step") or 0)
        return wm
