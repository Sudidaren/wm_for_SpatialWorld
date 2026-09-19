"""A real (non-oracle) world model built from sensor streams.

Perception inputs (the world model's own sensors):
  - RGB frames processed by an attached perception runtime that returns
    detected object boxes plus a metric monocular depth map;
  - agent pose from odometry (action log + landmark correction).

The world model does NOT read simulator segmentation frames, the simulator
class-color vocabulary, or per-step semantic metadata (objects/positions/
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

try:                                     # package import (normal case)
    from .self_observation import MOVE_ACTIONS
except ImportError:                      # script / flat import fallback
    from self_observation import MOVE_ACTIONS

try:
    from . import state_variants
except ImportError:                      # script / flat import fallback
    import state_variants


#: AI2-THOR default camera height above the agent's body root.  The agent
#: never changes height (no vertical movement), so this is a constant, not
#: something to integrate -- ``_pose[1]`` stays 0 and the camera sits at
#: ``CAMERA_Y`` above it.
CAMERA_Y = 0.675

#: AI2-THOR clamps cameraHorizon to this range (degrees).
HORIZON_LIMIT = 60.0

#: LookUp/LookDown without an explicit angle move by this much.
DEFAULT_LOOK_DEGREES = 30.0


def camera_basis(yaw: float, horizon: float = 0.0):
    """Return (forward, right, up) unit vectors in the world frame.

    Mirrors ``shared/geometry.camera_basis`` (the project's validated
    projection convention) so the runtime and the offline libraries agree.

    ``horizon`` is AI2-THOR's ``cameraHorizon`` in degrees: **positive =
    looking down**, negative = looking up.
    """
    rad = math.radians(float(yaw))
    fwd_h = np.array([math.sin(rad), 0.0, math.cos(rad)])
    right = np.array([math.cos(rad), 0.0, -math.sin(rad)])
    up = np.array([0.0, 1.0, 0.0])
    th = math.radians(-float(horizon or 0.0))
    if abs(th) > 1e-9:
        c, s = math.cos(th), math.sin(th)
        return fwd_h * c + up * s, right, up * c - fwd_h * s
    return fwd_h, right, up


def _load_rgb(image_path: Optional[str]):
    """Load the exact RGB frame the frozen MLLM was shown for this step."""
    if not image_path or not os.path.isfile(image_path):
        return None
    try:
        from PIL import Image

        return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None

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


#: AI2-THOR reports its ``fieldOfView`` with Unity semantics, i.e. it is the
#: VERTICAL fov, so for the 800x600 rig with square pixels
#: ``fx = fy = (height/2)/tan(fov/2) = 519.6``.  The legacy code used
#: ``(width/2)/tan(fov/2) = 692.8`` for *both* axes, which stretches every ray
#: vertically and shrinks the horizontal offsets.
#:
#: Measured on 3088 recorded object views with the ground-truth depth (so the
#: detector and the depth head cannot confound it), median localisation error:
#:
#:     legacy 692.8 :  3D 0.356 m   horizontal 0.260 m   vertical 0.191 m
#:     correct 519.6:  3D 0.189 m   horizontal 0.145 m   vertical 0.040 m
#:
#: Reproduce with ``tools/eval_projection_accuracy.py --fov-convention {horizontal,vertical}``.
FOV_CONVENTION = os.environ.get("LIGHTWM_FOV_CONVENTION", "vertical")


def camera_intrinsics(width: int, height: int, fov: float,
                      convention: Optional[str] = None):
    """``(fx, fy, cx, cy)`` for the AI2-THOR rig.

    ``convention="vertical"`` (default, correct) treats the reported fov as the
    vertical one; ``"horizontal"`` reproduces the legacy value for A/B tests.
    """
    conv = (convention or FOV_CONVENTION or "vertical").lower()
    if conv == "horizontal":
        f = (width / 2.0) / math.tan(math.radians(fov) / 2.0)
    else:
        f = (height / 2.0) / math.tan(math.radians(fov) / 2.0)
    return f, f, width / 2.0, height / 2.0


class WorldModel:
    def __init__(
        self,
        *,
        fov: float = 60.0,
        width: int = 800,
        height: int = 600,
        fov_convention: Optional[str] = None,
        hand_from_action_log: bool = True,
        pose_from_action_log: bool = True,
        pose_initial: str = "origin",
        move_magnitudes: Optional[Dict[str, float]] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        self.fov = fov
        self.width = width
        self.height = height
        # "vertical" (correct for AI2-THOR) or "horizontal" (legacy A/B)
        self.fov_convention = (fov_convention or FOV_CONVENTION or
                               "vertical").lower()
        self.hand_from_action_log = hand_from_action_log
        self.pose_from_action_log = pose_from_action_log
        self.pose_initial = pose_initial
        self.move_magnitudes = move_magnitudes or {}
        self.checkpoint_dir = checkpoint_dir
        self._slots: Dict[str, dict] = {}    # objectId -> slot
        self._slot_seq = 0
        self._holding: Optional[str] = None
        self._contents: Dict[str, set] = {}  # container oid -> {oid}
        self._states: Dict[str, dict] = {}  # oid -> {state_field: bool}
        #: The state-variant spelling the agent itself used for a base type
        #: (``Lettuce`` -> ``LettuceSliced``).  The environment renames an
        #: object when it is sliced, and the agent must use the new name in
        #: its next action, so the hint repeats that spelling back.
        self._variants: Dict[str, str] = {}
        self._visited: set = set()               # agent cells (walkable)
        self._pose: Optional[List[float]] = None  # [x, y, z, yaw] self-built
        #: AI2-THOR cameraHorizon in degrees (positive = looking down).
        #: Without this the agent's pitch was silently dropped and every
        #: anchor seen while looking down came out too far / too high.
        self._horizon = 0.0
        self._blocked_from_moves: set = set()    # move-failure target cells
        self._step = 0
        self._pose_sigma = 0.0          # odometry uncertainty (meters)
        # ---- loop closure (2026-09-15) -----------------------------------
        # Dead-reckoned odometry drifts; revisiting a known place lets us
        # snap everything back.  A "place" is the L2-normalised DINOv2 CLS of
        # the frame the agent was shown -- RGB only, no simulator data.
        self.loop_closure = True
        # Guards against the two ways this goes wrong: while the agent spins in
        # place the same view recurs with cos -> 1.000 and a meaningless pose
        # gap, and a look-alike room can match just as well as a real revisit.
        # A correction is therefore only accepted when the agent has genuinely
        # travelled away and come back, and only for a bounded amount.
        self.lc_sim_thr = 0.985       # cosine similarity that declares a revisit
        #: m of *path* (odometry travel) that must separate the two visits.
        #: Measuring "have I really been away and come back?" by travelled
        #: path -- rather than by the pose gap to the stored keyframe, which is
        #: the very drift being corrected -- keeps a spin-in-place from
        #: triggering a correction.
        self.lc_min_path = 2.0
        self.lc_min_step_gap = 12
        self.lc_keyframe_dist = 0.75  # m; min travel before storing a keyframe
        self.lc_max_correction = 1.0  # m; bigger "drift" is implausible
        self.lc_max_frac = 0.35       # correction must be small vs the odom gap
        self.lc_cooldown = 8          # steps between accepted corrections
        self._last_closure_step = -10 ** 9
        self._places: List[tuple] = []   # (fingerprint, odom_xy, step, path)
        self.n_closure = 0
        self.last_correction: Optional[tuple] = None
        self._path_len = 0.0             # odometry travel since the episode start
        self.landmark_correction = True
        self._perception = None         # PerceptionRuntime (dense+depth)
        self._last_outcome_source = "none"
        # ---- how much to trust multi-view triangulation over one frame -----
        # The monocular depth head is compressed on rooms it never trained on
        # (predictions come out at 0.59-0.62x the true distance), and that bias
        # is *systematic*: it does not average out.  The triangulated position,
        # by contrast, is metric on its own -- the ray through a pixel depends
        # only on the pixel, and the baseline comes from the agent's own
        # 0.25 m/step odometry.  Blending 0.6*tri + 0.4*depth therefore keeps
        # ~0.4*(1-0.62) = 16 % of the distance error, which is 0.32 m on a 2 m
        # object -- comparable to the whole triangulation error.
        #
        # Measured offline on 707 held-out objects seen from >=0.5 m baselines
        # (tools: the pair-median triangulation reported in
        # results/depth_head_ab_20260918/SCALE_CALIBRATION_20260918.md):
        # geometry-only 0.236 m vs 0.8 m for the depth-derived position, so
        # with two well-separated views the triangulation is the better term.
        # Set LIGHTWM_TRI_WEIGHT=0.6 to get the old behaviour back for an A/B.
        try:
            self.tri_weight = float(os.environ.get("LIGHTWM_TRI_WEIGHT", "1.0"))
        except ValueError:
            self.tri_weight = 1.0
        self.tri_weight = min(1.0, max(0.0, self.tri_weight))
        # How to read the depth for one detection: the bbox-centre pixel
        # (shipped default) or the median over the inner 50% of the box.  The
        # centre pixel is a single sample of a 32x32 depth map: on a thin or
        # partly occluded object it can land on the background.  Measured with
        # tools/eval_wm_positions.py; the shipped default is unchanged.
        self.depth_probe = os.environ.get("LIGHTWM_DEPTH_PROBE", "center").lower()
        # ---- label-free metric scale from the WM's own triangulation -------
        # See _perceive_detections for the derivation.  Off by default until a
        # run shows it helps; LIGHTWM_SCALE_ANCHOR=1 turns it on.
        self.scale_anchor = os.environ.get("LIGHTWM_SCALE_ANCHOR", "0") == "1"
        self._scale_r: Optional[float] = None   # smoothed correction factor
        self._scale_n: int = 0                  # frames that produced a ratio
        # Minimum camera separation before a second ray is accepted for
        # triangulation.  0.30 m (the shipped value) makes the intersection
        # extremely sensitive to bounding-box jitter: with a 0.3 m baseline at
        # 3 m, a 20 px centre offset moves the intersection by ~1 m, and the
        # measured effect is a *systematically too-near* anchor (the median
        # triangulated distance came out at 0.46x the depth-derived one).
        # Measured sweep with tools/eval_wm_positions.py (12 classic-family
        # non-evaluation episodes, median anchor error): 1.0 m -> 1.005 m,
        # 2.0 m -> 0.960 m, 4.0 m -> 0.843 m, 6.0 m -> 0.803 m, 8.0 m -> 0.803 m.
        # LIGHTWM_TRI_BASELINE retunes it.
        try:
            self.tri_baseline = float(os.environ.get("LIGHTWM_TRI_BASELINE", "6.0"))
        except ValueError:
            self.tri_baseline = 0.30
        if not pose_from_action_log and \
                os.environ.get("LIGHTWM_ALLOW_SIM_POSE") != "1":
            raise ValueError(
                "pose_from_action_log=False reads simulator pose; runtime is "
                "not allowed. Set LIGHTWM_ALLOW_SIM_POSE=1 only for offline "
                "measurement scripts.")

    def attach_perception(self, perception) -> None:
        """Attach the detector+depth runtime used as the model's eyes."""
        self._perception = perception

    # ------------------------------------------------------------------
    def observe(
        self,
        observation: Any,
        action: Optional[Dict] = None,
        moved: Optional[bool] = None,
        action_ok: Optional[bool] = None,
        frame: Any = None,
    ) -> Dict:
        """Fuse one step into the world model.

        ``action_ok`` / ``moved`` come from the frame difference between the
        two images the agent was shown (``self_observation.estimate_success``):
        ``mse > 1`` -> True (the action did something), ``mse < 1`` -> False.
        ``None`` means the frames were unavailable.

        The environment object is deliberately not a parameter: the only
        things this method may look at are the image the agent was shown
        (``observation.image_path``) and the action it took.
        """
        self._step += 1
        action = action or {}
        action_name = action.get("action_name")
        agent_pos, agent_rot = self._resolve_pose(action, moved)
        ax, az = agent_pos.get("x"), agent_pos.get("z")
        if ax is not None and az is not None:
            self._visited.add((round(ax / 0.25), round(az / 0.25)))
            # A locomotion action whose frame did not change is treated as
            # blocked: the cell directly ahead is remembered so the hint can
            # tell the model to go around.
            if (moved is False and action_name in MOVE_ACTIONS):
                cur_agent = (round(ax / 0.25), round(az / 0.25))
                dcx, dcz = self._move_dir(action_name, float(agent_rot.get("y", 0.0)))
                self._blocked_from_moves.add((cur_agent[0] + dcx, cur_agent[1] + dcz))

        if self._perception is None:
            raise RuntimeError(
                "world model requires a perception runtime (object detector + "
                "monocular depth); simulator segmentation perception is not "
                "supported.")

        visible_ids = set()
        rgb = frame
        if rgb is None:
            rgb = _load_rgb(getattr(observation, "image_path", None))
        if rgb is not None:
            res = self._perception(rgb)
            visible_ids = self._perceive_detections(
                res["detections"], res["depth"], agent_pos, agent_rot)
            if self.loop_closure:
                fp = None
                if hasattr(self._perception, "embed"):
                    try:
                        fp = self._perception.embed(rgb)
                    except Exception:
                        fp = None
                if fp is not None:
                    self._maybe_close_loop(fp)

        # A target that is not fully inside the frame can change state outside
        # the image, so the frame diff is blind there: leave the outcome
        # unknown instead of calling it success or failure.
        action_ok = self._resolve_outcome(
            action_name, action.get("object_type"), action_ok)

        # causal state from the action log + the (possibly overridden) outcome
        if self.hand_from_action_log:
            self._update_action_log(
                action_name, action.get("object_type"), visible_ids, agent_pos,
                action_ok=action_ok,
            )

        # move held objects to the hand pose
        if self._holding is not None and self._holding in self._slots:
            hand = self._hand_position(agent_pos, agent_rot)
            self._slots[self._holding]["pos"] = hand

        meta = self._to_metadata(agent_pos, agent_rot, visible_ids)
        meta["action_outcome"] = {
            "ok": action_ok,
            "source": self._last_outcome_source,
        }

        # periodic disk checkpoint so long tasks can resume after an OOM
        if self.checkpoint_dir and self._step % 50 == 0:
            try:
                self.save(os.path.join(self.checkpoint_dir, "world_model_ckpt.json"))
            except Exception as exc:
                print(f"⚠️ WorldModel checkpoint failed at step {self._step}: {exc}")

        return meta

    # ------------------------------------------------------------------
    # Loop closure
    # ------------------------------------------------------------------
    def _maybe_close_loop(self, fp) -> None:
        """Detect a revisit and, if found, snap the whole frame back.

        Algorithm mirrors ``lightwm_phases/shared/loop_closure.py`` (place
        fingerprint -> cosine match -> odom->world offset re-alignment), but is
        implemented here so it can act on this runtime's own anchor store.
        Translation-only on purpose: the yaw drift over a household trajectory
        is small compared to the positional one, and a pure translation keeps
        the occupancy grid rigid.
        """
        xy = np.array([self._pose[0], self._pose[2]], dtype=float)
        if (not self._places
                or np.linalg.norm(xy - self._places[-1][1]) >= self.lc_keyframe_dist):
            self._places.append((fp, xy.copy(), self._step, self._path_len))
        best, best_sim = None, self.lc_sim_thr
        for nfp, nxy, nstep, npath in self._places[:-1]:
            if self._step - nstep < self.lc_min_step_gap:
                continue
            travel = self._path_len - npath
            if travel < self.lc_min_path:
                continue                # we have not really been away and back
            sim = float(np.dot(fp, nfp))
            if sim > best_sim:
                best, best_sim = (nxy, travel), sim
        if best is None:
            return
        nxy, travel = best
        if self._step - self._last_closure_step < self.lc_cooldown:
            return                      # do not chain corrections
        delta = nxy - xy
        mag = float(np.linalg.norm(delta))
        # The correction may only explain a *modest* fraction of the path
        # actually travelled since that keyframe, and never more than the
        # absolute bound; anything bigger is far more likely to be a
        # look-alike view than real drift.
        if mag > self.lc_max_correction or mag > self.lc_max_frac * travel:
            return
        self._last_closure_step = self._step
        # odometry says `xy`, but this place was first seen at `best`; the
        # difference is the accumulated drift -> shift everything back.
        self._apply_drift_correction(delta, best_sim)

    def _apply_drift_correction(self, delta_xy, similarity: float) -> None:
        dx, dz = float(delta_xy[0]), float(delta_xy[1])
        self._pose[0] += dx
        self._pose[2] += dz
        for slot in self._slots.values():
            slot["pos"][0] += dx
            slot["pos"][2] += dz
        di, dj = int(round(dx / 0.25)), int(round(dz / 0.25))
        if di or dj:
            self._visited = {(i + di, j + dj) for (i, j) in self._visited}
        self.n_closure += 1
        self.last_correction = (self._step, round(dx, 2), round(dz, 2),
                                round(similarity, 3))
        print(f"↻ loop closure #{self.n_closure} @step {self._step}: "
              f"drift 修正 dx={dx:+.2f}m dz={dz:+.2f}m "
                      f"(cos={similarity:.3f})", flush=True)

    def _perceive_detections(
        self, detections, depth_map, agent_pos, agent_rot
    ) -> set:
        """Build/refine anchors from dense detections + monocular depth only."""
        H, W = depth_map.shape[:2]
        matched = []  # (stored_pos, measured_pos) for landmark correction
        visible_ids = set()

        # ---- self-anchoring the depth scale from the WM's own geometry -----
        # The monocular head is compressed on rooms it never saw (predictions
        # at 0.59-0.62x the truth), and that bias is systematic, so it is not
        # averaged away by fusing many frames.  The multi-view triangulation,
        # by contrast, is metric on its own: the ray through a pixel is fixed
        # by the pixel and the intrinsics, and the baseline comes from the
        # agent's own odometry.  So for objects that already have a
        # triangulated anchor, the ratio
        #     r = ||triangulated - camera|| / ||depth-derived - camera||
        # measures this frame's depth scale directly, with no labels, no
        # ground plane and no simulator state.  Applying r to the depth of the
        # remaining (single-view) detections transfers the metric scale to the
        # objects the agent has seen from one pose only.
        cam = self._camera_xyz(agent_pos, agent_rot)
        ratios = []
        if self.scale_anchor:
            for det in detections:
                u, v = self._det_uv(det, W, H)
                if u is None:
                    continue
                z = self._probe_depth(depth_map, det, u, v)
                if z <= 0 or not math.isfinite(z):
                    continue
                oid = self._match_slot_only(det["type"], self._unproject(
                    u, v, z, agent_pos, agent_rot))
                if oid is None:
                    continue
                slot = self._slots[oid]
                if len(slot.get("obs") or []) < 2:
                    continue
                pd = np.array(self._unproject(u, v, z, agent_pos, agent_rot))
                nd = float(np.linalg.norm(pd - cam))
                nt = float(np.linalg.norm(np.array(slot["pos"]) - cam))
                if nd > 0.15 and nt > 0.15:
                    ratios.append(nt / nd)
            if len(ratios) >= 3:
                r = float(np.median(ratios))
                r = min(2.5, max(0.4, r))
                self._scale_r = (r if self._scale_r is None
                                 else 0.7 * self._scale_r + 0.3 * r)
                self._scale_n += 1
        r_now = self._scale_r if (self.scale_anchor and self._scale_r) else 1.0

        for det in detections:
            u, v = self._det_uv(det, W, H)
            if u is None:
                continue
            z = self._probe_depth(depth_map, det, u, v)
            if z <= 0 or not math.isfinite(z):
                continue
            z = z * r_now
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
            # 2D box of this detection: needed to know whether the object is
            # only partially inside the frame (state changes then happen
            # outside the view and the frame diff is blind to them).
            box = det.get("bbox")
            if box:
                x0, y0, x1, y1 = [float(v) for v in box]
                slot["box"] = [x0, y0, x1, y1]
                slot["edge_px"] = min(x0, y0, W - x1, H - y1)
                slot["area_frac"] = ((x1 - x0) * (y1 - y0)) / float(W * H)
            # multi-view triangulation: fuse this frame's ray with earlier ones
            # taken from a sufficiently different pose (see the method's doc).
            self._add_view_and_triangulate(oid, u, v, z, agent_pos, agent_rot, pos)
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
            "obs": [],            # multi-view rays for triangulation
        }
        return oid

    # ------------------------------------------------------------------
    # Multi-view triangulation
    # ------------------------------------------------------------------
    def _add_view_and_triangulate(self, oid: str, u: float, v: float, z: float,
                                  agent_pos, agent_rot, measured_pos) -> None:
        """Keep a bounded set of observation rays per object and fuse them.

        A single-frame "bbox centre x depth" has ~0.5 m median error (measured
        over 444 objects).  The same object seen from a different pose gives a
        second ray; the least-squares intersection of the rays averages out
        both the monocular-depth noise and the bbox-centre bias.  Only rays from
        a pose at least ``baseline`` away from the stored ones are kept, and the
        result is blended with the running EMA rather than replacing it.
        """
        cam = self._camera_xyz(agent_pos, agent_rot)
        fx, fy, cx, cy = camera_intrinsics(self.width, self.height, self.fov,
                                           self.fov_convention)
        fwd, right, up = camera_basis(float(agent_rot.get("y", 0.0)),
                                      float(agent_rot.get("horizon", 0.0) or 0.0))
        x_rel = (u - cx) * z / fx
        y_rel = (cy - v) * z / fy
        ray = right * x_rel + up * y_rel + fwd * z
        n = float(np.linalg.norm(ray))
        if n <= 1e-9:
            return
        ray = ray / n

        slot = self._slots.get(oid)
        if slot is None:
            return
        obs = slot.setdefault("obs", [])
        # accept a new ray once the camera has moved a little (below that the
        # parallax carries no information at all) ...
        if any(float(np.linalg.norm(cam - c)) < 0.30 for c, _ in obs):
            return
        obs.append((cam.copy(), ray))
        del obs[:-10]
        if len(obs) < 2:
            return
        tri = self._triangulate(obs)
        if tri is None:
            return
        # Blend with the running estimate.  ``tri_weight`` defaults to 1.0
        # because the single-frame term carries the depth head's systematic
        # scale error (see __init__); the running estimate is the better
        # fallback than this frame's raw unprojection.
        # ... but trust the intersection in proportion to how well the views
        # actually bracket it: a ray's depth precision is ~b^2 / z^2, so a
        # 0.5 m pair says almost nothing while a 2 m pair is what makes the
        # anchor metric.  `tri_baseline` is that reference separation; the
        # trust ramps quadratically from 0 to 1 across it.
        b_max = max((float(np.linalg.norm(c_i - c_j))
                     for i, (c_i, _) in enumerate(obs)
                     for j, (c_j, _) in enumerate(obs) if i < j), default=0.0)
        trust = min(1.0, (b_max / max(1e-6, self.tri_baseline)) ** 2)
        w = trust * self.tri_weight
        slot["tri_trust"] = float(trust)
        slot["tri_baseline_m"] = float(b_max)
        slot["pos"] = [w * tri[i] + (1.0 - w) * float(slot["pos"][i])
                       for i in range(3)]
        # keep the ray-fit residual so a run can be audited afterwards
        slot["tri_resid"] = self._ray_residual(tri, obs)
        slot["tri_views"] = len(obs)
        slot["sigma"] = max(0.10, float(slot.get("sigma", 0.25)) * 0.8)

    @staticmethod
    def _ray_residual(point, obs) -> float:
        """Median distance from ``point`` to the observation rays (metres)."""
        d = []
        for c, r in obs:
            v = np.asarray(point, dtype=float) - c
            along = float(np.dot(v, r))
            d.append(float(np.linalg.norm(v - r * along)))
        return float(np.median(d)) if d else float("nan")

    def _camera_xyz(self, agent_pos, agent_rot) -> np.ndarray:
        return np.array([float(agent_pos.get("x") or 0.0),
                         float(agent_pos.get("y") or 0.0) + CAMERA_Y,
                         float(agent_pos.get("z") or 0.0)])

    def _probe_depth(self, depth_map, det, u: int, v: int) -> float:
        """Depth (m) for one detection -- see ``self.depth_probe``."""
        if self.depth_probe == "boxmedian":
            box = det.get("bbox")
            if box and len(box) == 4:
                H, W = depth_map.shape[:2]
                x1, y1, x2, y2 = [float(t) for t in box]
                ix1, iy1 = int(x1 + 0.25 * (x2 - x1)), int(y1 + 0.25 * (y2 - y1))
                ix2, iy2 = int(x1 + 0.75 * (x2 - x1)), int(y1 + 0.75 * (y2 - y1))
                ix1, iy1 = max(0, ix1), max(0, iy1)
                ix2, iy2 = min(W, max(ix1 + 1, ix2)), min(H, max(iy1 + 1, iy2))
                patch = depth_map[iy1:iy2, ix1:ix2]
                patch = patch[np.isfinite(patch) & (patch > 0.05)]
                if patch.size >= 4:
                    return float(np.median(patch))
        return float(depth_map[v, u])

    @staticmethod
    def _det_uv(det, W: int, H: int):
        """Bounding-box centre in pixels, or ``(None, None)`` if outside."""
        cx, cy = det["center"]
        u, v = int(round(cx)), int(round(cy))
        if not (0 <= u < W and 0 <= v < H):
            return None, None
        return u, v

    def _match_slot_only(self, otype: str, pos):
        """Like :meth:`_match_or_create_slot` but never creates a new slot."""
        best_id, best_d = None, 2.0
        for oid, s in self._slots.items():
            if s["type"] != otype:
                continue
            d = math.hypot(s["pos"][0] - pos[0], s["pos"][2] - pos[2])
            if d < best_d:
                best_id, best_d = oid, d
        return best_id

    @staticmethod
    def _triangulate(obs):
        """Least-squares intersection of rays ``(c_i, d_i)``.

        Minimises ``sum_i w_i || (I - d_i d_i^T)(x - c_i) ||^2`` with
        ``A = sum w_i (I - d_i d_i^T)``, ``b = sum w_i (I - d_i d_i^T) c_i``.

        The weight matters more than it looks.  A ray's geometric depth
        precision at range z is ``z^2 * sigma_theta / baseline``, so a pair of
        views 0.3 m apart pins the point ~7x worse than a pair 2 m apart --
        and the ray through a *bounding-box centre* carries a large
        sigma_theta (the box wobbles as the viewpoint changes).  Unweighted,
        three short-baseline rays outvote one long-baseline ray and the
        intersection is dragged toward the camera (measured: the median
        triangulated distance came out at 0.46x the depth-derived one).
        Weighting by the ray's own parallax ``w_i = b_i^2`` (b_i = the largest
        camera separation available for that ray) keeps every observation but
        lets the informative ones decide.  Measured with
        tools/eval_wm_positions.py on 12 classic-family non-evaluation
        episodes: anchors seen from >=2 poses go from 1.054 m to 0.57 m of 3D
        error without starving objects of rays.
        """
        A = np.zeros((3, 3))
        b = np.zeros(3)
        cams = [c for c, _ in obs]
        for i, (c, d) in enumerate(obs):
            b_i = max((float(np.linalg.norm(c - o)) for j, o in enumerate(cams)
                       if j != i), default=0.0)
            w = max(b_i * b_i, 1e-4)
            P = np.eye(3) - np.outer(d, d)
            A += w * P
            b += w * (P @ c)
        try:
            x = np.linalg.solve(A + 1e-6 * np.eye(3), b)
        except np.linalg.LinAlgError:
            return None
        return x if np.all(np.isfinite(x)) else None

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
    def _resolve_pose(self, action: Optional[Dict], moved: Optional[bool] = None):
        """Agent pose by dead reckoning from the action log ONLY.

        Information isolation: the pose is never seeded from, or corrected by,
        simulator metadata.  ``pose_initial`` must be ``origin``: the episode
        starts at the origin of the world model's own frame.  Whether a
        locomotion action actually advanced is taken from the purely visual
        ``moved`` estimate.
        """
        # The first observation both seeds the pose and integrates the action
        # that produced it, so the agent's first move is never dropped.
        if self._pose is None:
            self._pose = [0.0, 0.0, 0.0, 0.0]
        self._integrate_pose(action, moved)
        x, y, z, yaw = self._pose
        return ({"x": x, "y": y, "z": z},
                {"y": yaw, "horizon": self._horizon})

    def _integrate_pose(self, action: Optional[Dict],
                        moved: Optional[bool] = None) -> None:
        """Dead-reckon the pose from the last action.

        ``moved is False`` (a locomotion action whose frame did not change)
        means the agent did not advance.  The verdict comes from the frames
        alone; the simulator's error message is never consulted.
        """
        action = action or {}
        name = action.get("action_name")
        if not name:
            return
        if name in ("LookUp", "LookDown"):
            # The pitch does not translate the agent, but it *does* rotate the
            # camera, and every unprojection depends on it: 79% of the AI2-THOR
            # tasks begin with a LookDown.  Positive horizon = looking down.
            try:
                deg = float(action.get("degrees")
                            or action.get("magnitude")
                            or DEFAULT_LOOK_DEGREES)
            except (TypeError, ValueError):
                deg = DEFAULT_LOOK_DEGREES
            delta = deg if name == "LookDown" else -deg
            self._horizon = max(-HORIZON_LIMIT,
                                min(HORIZON_LIMIT, self._horizon + delta))
            return
        if self._pose is None:
            return
        x, _y, z, yaw = self._pose
        if name in ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight"):
            mag = self._action_magnitude(name, action)
            dcx, dcz = self._move_dir(name, yaw)
            if moved is False:
                # visually blocked (frames identical): remember it, do not move
                self._blocked_from_moves.add(
                    (round(x / 0.25) + dcx, round(z / 0.25) + dcz)
                )
                return
            self._pose[0] = x + dcx * mag
            self._pose[2] = z + dcz * mag
            self._path_len += float(mag)
        elif name in ("RotateLeft", "RotateRight"):
            try:
                deg = float(action.get("degrees") or action.get("magnitude") or 90.0)
            except (TypeError, ValueError):
                deg = 90.0
            # AI2-THOR convention: RotateRight increases yaw (clockwise)
            self._pose[3] = (yaw + (deg if name == "RotateRight" else -deg)) % 360

    def _action_magnitude(self, name: str, action: Dict) -> float:
        """Mirror the AI2-THOR wrapper's resolution EXACTLY.

        The parser emits ``granularity`` ("Small"/"Medium"/"Large") for
        ``MoveAhead(Small)`` and an explicit ``magnitude`` only for a numeric
        argument; a bare ``MoveAhead`` carries neither.  The wrapper resolves
        ``magnitude -> granularity -> move_small_magnitude (0.25)``.

        Odometry is only as good as this resolution: a bare ``MoveAhead`` moves
        0.25 m while ``MoveAhead(Large)`` moves 1.0 m, so the fallback must be
        the *small* magnitude and never the configured ``move_ahead_magnitude``.
        """
        mag = action.get("magnitude")
        if isinstance(mag, (int, float)):
            return float(mag)
        g = str(action.get("granularity") or "").strip().lower()
        if g == "small":
            return float(self.move_magnitudes.get("MoveSmall", 0.25))
        if g == "medium":
            return float(self.move_magnitudes.get("MoveMedium", 0.5))
        if g == "large":
            return float(self.move_magnitudes.get("MoveLarge", 1.0))
        if mag is not None:
            try:
                return float(mag)
            except (TypeError, ValueError):
                pass
        # wrapper default when neither magnitude nor granularity is given
        return float(self.move_magnitudes.get("MoveSmall", 0.25))

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

    def _unproject(self, u, v, z, agent_pos, agent_rot) -> List[float]:
        """Pixel + metric depth -> world point.

        Uses the same camera model as ``shared/geometry.unproject``:
        camera at ``agent + (0, CAMERA_Y, 0)``, basis rotated by the agent's
        yaw **and** its camera horizon, and the vertical pixel offset measured
        *upwards* from the principal point (``cy - v``).

        All three parts are required: AI2-THOR's depth is the distance along
        the optical axis, 79% of the household tasks start with a LookDown, and
        the metric depth map is meaningless without the matching orientation.
        ``tools/eval_projection_accuracy.py`` measures this against recorded
        ground truth.
        """
        fx, fy, cx, cy = camera_intrinsics(self.width, self.height, self.fov,
                                           self.fov_convention)
        x_rel = (u - cx) * z / fx
        y_rel = (cy - v) * z / fy
        fwd, right, up = camera_basis(float(agent_rot.get("y", 0.0)),
                                      float(agent_rot.get("horizon", 0.0) or 0.0))
        cam = self._camera_xyz(agent_pos, agent_rot)
        point = cam + right * x_rel + up * y_rel + fwd * z
        return [float(point[0]), float(point[1]), float(point[2])]

    def _hand_position(self, agent_pos, agent_rot) -> List[float]:
        yaw = math.radians(float(agent_rot.get("y", 0.0)))
        fwd = (math.sin(yaw), math.cos(yaw))
        return [
            (agent_pos.get("x") or 0) + fwd[0] * 0.3,
            (agent_pos.get("y") or 0) + 0.4,
            (agent_pos.get("z") or 0) + fwd[1] * 0.3,
        ]

    #: a detection box closer than this to the image border counts as clipped
    EDGE_CLIP_PX = 5.0

    # -- naming of state variants ------------------------------------------
    def _matches_slot_type(self, slot_type: str, action_type: str) -> bool:
        """Does an action target name this slot?

        The environment renames an object once it is sliced (``Lettuce`` ->
        ``LettuceSliced``) and the agent then acts on the new name, while the
        perception head keeps labelling it ``Lettuce``.  Both spellings are
        the same object, so both have to match the same slot.  The rename
        rule is parsed from the prompt every arm receives (``state_variants``
        reads "SliceObject(X) produces XSliced"), so this adds no vocabulary.
        """
        if str(slot_type) == str(action_type):
            return True
        base = state_variants.base_of(action_type)
        return bool(base) and base != str(action_type) and base == str(slot_type)

    def _slots_of_type(self, action_type: str,
                       visible_ids: Optional[set] = None) -> List[str]:
        """Slot ids an action target addresses, newest state knowledge last."""
        out = []
        for oid, slot in self._slots.items():
            if not self._matches_slot_type(slot["type"], action_type):
                continue
            if visible_ids is not None and oid not in visible_ids:
                continue
            out.append(oid)
        return out

    def _remember_variant(self, action_type: str) -> None:
        """Keep the agent's own spelling of a state variant.

        Called only when the action was attributed to a slot we already have,
        so a hallucinated name cannot introduce a variant that does not exist.
        """
        if state_variants.is_variant(action_type):
            self._variants[state_variants.base_of(action_type)] = str(action_type)

    def _resolve_outcome(self, action_name, object_type, action_ok):
        """Drop the frame-diff verdict when the target is not fully in view.

        If the acted-on object only shows a sliver at the frame border (or is
        not detected at all), the state change may happen outside the image,
        so the frame difference cannot decide success or failure.  In that
        case the outcome is left ``None`` (unknown): no verdict is published
        and the hint does not claim success or failure.
        """
        self._last_outcome_source = ("frame_diff" if action_ok is not None
                                     else "none")
        if not object_type:
            return action_ok
        cands = [self._slots[oid] for oid in self._slots_of_type(object_type)]
        if not cands:
            # target never perceived -> nothing visible to judge from
            self._last_outcome_source = "not_visible"
            return None
        slot = max(cands, key=lambda s: s.get("last_seen", -1))
        edge = slot.get("edge_px")
        if edge is not None and edge <= self.EDGE_CLIP_PX:
            self._last_outcome_source = "not_fully_visible"
            return None
        return action_ok

    def _update_action_log(self, action_name, object_type, visible_ids,
                           agent_pos=None, action_ok=None) -> None:
        """Update causal state from the action log + the frame-diff outcome.

        Rule (2026-09-15): the step succeeded iff the frame changed
        (``mse > 1``, see ``self_observation.estimate_success``).  State
        updates are applied only when ``action_ok`` is True; ``action_ok``
        None (frames unavailable, or the target is not fully in view) falls
        back to the "target visible" guard.
        """
        if not action_name:
            return
        if action_ok is False:
            # the frame did not change -> the action did not take effect
            return
        ax = (agent_pos or {}).get("x")
        az = (agent_pos or {}).get("z")

        # ---- 0) learn the environment's rename, if it used one ---------------
        # The agent acts on "LettuceSliced" while the slot is "Lettuce"; we
        # ground the name against a slot we already hold, so an invented name
        # changes nothing, and a step whose frame did not change never gets
        # here at all.
        if object_type and self._slots_of_type(object_type):
            self._remember_variant(object_type)

        # ---- 1) state actions: assume they took effect (visibility-guarded) --
        state_delta = {
            "OpenObject": ("isOpen", True),
            "CloseObject": ("isOpen", False),
            "ToggleObjectOn": ("isToggled", True),
            "ToggleObjectOff": ("isToggled", False),
            "SliceObject": ("isSliced", True),
            "CookObject": ("isCooked", True),
            "DirtyObject": ("isDirty", True),
            "CleanObject": ("isDirty", False),
            "FillObjectWithLiquid": ("isFilledWithLiquid", True),
            "EmptyLiquidFromObject": ("isFilledWithLiquid", False),
            "UseUpObject": ("isUsedUp", True),
        }
        if action_name in state_delta and object_type:
            field, value = state_delta[action_name]
            cands = self._slots_of_type(object_type, visible_ids)
            if cands:
                self._states.setdefault(cands[0], {})[field] = value
                self._remember_variant(object_type)

        # ---- 2) hand state ---------------------------------------------------
        if action_name == "PickupObject" and object_type:
            nearest, best = None, 1e9
            for oid in self._slots_of_type(object_type, visible_ids):
                s = self._slots.get(oid)
                if s is None:
                    continue
                if ax is None or az is None:
                    continue
                d = math.hypot(s["pos"][0] - ax, s["pos"][2] - az)
                if d < best:
                    nearest, best = oid, d
            # an object held in the hand is ~0.3-0.6 m from the agent body
            if nearest is not None and best <= 0.7:
                self._holding = nearest
                self._remember_variant(object_type)
        elif action_name in ("PutObject", "DropHandObject", "ThrowObject"):
            if (action_name == "PutObject" and self._holding is not None
                    and object_type):
                cands = self._slots_of_type(object_type, visible_ids)
                if cands:
                    target = cands[0]
                    self._contents.setdefault(target, set()).add(self._holding)
                    self._slots[self._holding]["pos"] = list(self._slots[target]["pos"])
            self._holding = None

    def _to_metadata(self, agent_pos, agent_rot, visible_ids) -> Dict:
        objects_out = []
        ax, az = agent_pos.get("x"), agent_pos.get("z")
        for oid, slot in self._slots.items():
            pos = slot["pos"]
            dist = math.hypot(pos[0] - ax, pos[2] - az) if ax is not None else 0.0
            state = dict(self._states.get(oid) or {})
            #: The name the *agent* has to use for this object.  The same as
            #: ``objectType`` until the environment renames it (slicing),
            #: after which the next action would address a name that no
            #: longer exists unless the hint switches to the new spelling.
            #: Matching keeps using the base ``objectType``.
            display = state_variants.display_name(
                slot["type"], state, self._variants.get(slot["type"]))
            inside = []
            for hid in self._contents.get(oid, ()):  # what was put inside
                htype = self._slots.get(hid, {}).get("type")
                if htype:
                    inside.append(htype)
            objects_out.append(
                {
                    "objectId": oid,
                    "objectType": slot["type"],
                    "objectTypeDisplay": display,
                    "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                    "visible": oid in visible_ids,
                    "distance": dist,
                    "edge_px": float(slot.get("edge_px", -1.0)),
                    "box": list(slot.get("box") or []),
                    # memory bookkeeping for the per-step target hint
                    "last_seen_step": int(slot.get("last_seen") or 0),
                    "seen_count": int(slot.get("seen") or 0),
                    "sigma": float(slot.get("sigma") or 0.0),
                    "state": dict(self._states.get(oid) or {}),
                    "contents": inside,
                }
            )
        inv = []
        if self._holding is not None:
            # The perception-backed slots are named "det|<Type>|<n>", so a
            # naive split("|")[0] published the literal string "det" as the
            # held object's type and the hint read "手持：det" (bug found in
            # the ai2thor05529 smoke run, 2026-09-15).
            _hid = str(self._holding)
            _seg = _hid.split("|")
            _htype = (_seg[1] if (len(_seg) > 1 and _seg[0] == "det")
                      else _seg[0])
            inv.append(
                {
                    "objectId": self._holding,
                    "objectType": _htype,
                    "objectTypeDisplay": state_variants.display_name(
                        _htype, self._states.get(self._holding),
                        self._variants.get(_htype)),
                }
            )
        return {
            "agent": {
                "position": dict(agent_pos),
                "rotation": dict(agent_rot),
            },
            "objects": objects_out,
            "inventoryObjects": inv,
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialize all learnable state to JSON (safe to reload later).

        All internal structures are plain dicts/sets of JSON-able values, so
        no pickling is needed.  ``path`` may include a directory that does not
        exist yet.

        Two guarantees:

        * **numpy is converted**, not passed through.  A slot's ``obs`` holds
          ``(camera, ray)`` numpy pairs, and ``json.dump`` writes
          *incrementally*, so a non-serializable value would abort the dump
          half way and leave a truncated, unparseable file behind.
        * **the write is atomic**: payload goes to ``<path>.tmp`` first and is
          then ``os.replace``d, so a failure at any point leaves the previous
          checkpoint intact instead of a half-written one.
        """
        import json

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "fov": self.fov,
            "fov_convention": self.fov_convention,
            "width": self.width,
            "height": self.height,
            "hand_from_action_log": self.hand_from_action_log,
            "_step": self._step,
            "_slot_seq": self._slot_seq,
            "_slots": self._slots,
            "_holding": self._holding,
            "_contents": {k: sorted(v) for k, v in self._contents.items()},
            "_states": {k: dict(v) for k, v in self._states.items()},
            "_variants": dict(self._variants),
            "_visited": sorted(self._visited),
            "_blocked_from_moves": sorted(self._blocked_from_moves),
            "_pose": list(self._pose) if self._pose else None,
            "_horizon": self._horizon,
            "_path_len": self._path_len,
        }
        payload = self._jsonable(data)
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1)
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        os.replace(tmp, path)

    @staticmethod
    def _jsonable(value):
        """Recursively turn numpy / set / tuple structures into JSON types."""
        if isinstance(value, dict):
            return {str(k): WorldModel._jsonable(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [WorldModel._jsonable(v) for v in value]
        if isinstance(value, (set, frozenset)):
            return [WorldModel._jsonable(v) for v in sorted(value, key=repr)]
        return value

    @classmethod
    def load(
        cls,
        path: str,
        *,
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
            fov=fov,
            width=width,
            height=height,
            hand_from_action_log=hand_from_action_log,
            checkpoint_dir=checkpoint_dir,
        )
        wm._slots = data.get("_slots") or {}
        # Without this a resumed run restarts the id counter at 0 and can
        # collide with restored "det|<Type>|<n>" ids.  Fall back to the ids
        # actually present when an older checkpoint has no _slot_seq.
        seq = data.get("_slot_seq")
        if seq is None:
            seen = []
            for key in wm._slots:
                tail = str(key).rsplit("|", 1)[-1]
                if str(key).startswith("det|") and tail.isdigit():
                    seen.append(int(tail))
            seq = max(seen) if seen else 0
        wm._slot_seq = int(seq)
        wm._holding = data.get("_holding")
        wm._contents = {
            k: set(v) for k, v in (data.get("_contents") or {}).items()
        }
        wm._states = {
            k: dict(v) for k, v in (data.get("_states") or {}).items()
        }
        wm._variants = {
            str(k): str(v) for k, v in (data.get("_variants") or {}).items()
        }
        wm._visited = {tuple(c) for c in data.get("_visited") or []}
        wm._pose = (
            [float(v) for v in data["_pose"]] if data.get("_pose") else None
        )
        wm._horizon = float(data.get("_horizon") or 0.0)
        wm._path_len = float(data.get("_path_len") or 0.0)
        wm._blocked_from_moves = {
            tuple(c) for c in data.get("_blocked_from_moves") or []
        }
        wm._step = int(data.get("_step") or 0)
        return wm
