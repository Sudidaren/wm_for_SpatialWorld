"""Per-step spatial-memory hint injected in front of the image.

Contents (2026-09-16):

  * 手持 / 移动被挡 / 上一个动作成败 -- world-model state;
  * 可选的目标物提示 (``WM_TARGET_HINT=1``) -- the objects *the model itself*
    named at the start of the episode, reported only where the world model
    actually detected them (see ``target_priority``).

No object vocabulary and no alias table is used anywhere here: the baseline
arm never sees one, so the WM arm must not either.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple


_DIR_WORDS = ("正前方", "右前方", "正右方", "右后方",
              "正后方", "左后方", "正左方", "左前方")

def _direction_word(dx: float, dz: float, yaw: float) -> str:
    """World-space delta -> egocentric 8-way direction label."""
    rad = math.radians(float(yaw))
    fwd = dx * math.sin(rad) + dz * math.cos(rad)
    right = dx * math.cos(rad) - dz * math.sin(rad)
    ang = math.degrees(math.atan2(right, fwd))
    return _DIR_WORDS[int(round(ang / 45.0)) % 8]


class MemoryProbe:
    """Per-step hint block injected in front of the image.

    Target-free readout of what the world model owns (hand, blocked move,
    previous action outcome) plus, when ``WM_TARGET_HINT=1``, the objects the
    model itself named at the start of the episode.
    """

    def __init__(self, task_description: Optional[str] = None) -> None:
        self._task_desc = task_description or ""
        self._pending: List[str] = []
        #: 目标物优先级提示（WM_TARGET_HINT=1）：每步只报最重要的前 K 个物体
        self._target_hint: Dict[str, Any] = {}
        self._acts: Dict[str, Tuple[int, str, Optional[bool]]] = {}
        self._tick: int = 0
        self._hinter = None
        self._vlm = None                # 模型自己；用于开局自述任务物品

    def set_target_hint(self, cfg: Optional[Dict[str, Any]]) -> None:
        self._target_hint = dict(cfg or {})

    def set_vlm(self, vlm) -> None:
        """The agent's own VLM, used once per task to name the task objects."""
        self._vlm = vlm

    def update(
        self,
        *,
        wm_metadata: Dict,
        action_name: Optional[str],
        object_type: Optional[str] = None,
        blocked: bool = False,
        action_ok: Optional[bool] = None,
        action_ok_source: Optional[str] = None,
    ) -> str:
        """This step's hint block: hand, blocked move, previous action outcome.

        Nothing here names a task target; ``action_ok`` is the frame-difference
        outcome (``mse > 1``), and it is reported only when the world model
        could actually judge it.
        """
        if os.environ.get("WM_NO_INJECT") == "1":
            # Ablation: run the whole world model but say nothing.  The WM's
            # only channel to the agent is this block, so this arm must come
            # out identical to the plain baseline -- that is exactly the point:
            # it proves there is no hidden path by which the memory reaches the
            # model (no side effects on the action space, no state it can query).
            self._pending = []
            return ""
        parts: List[str] = []
        try:
            self._tick += 1
            if action_name and object_type:
                self._acts[str(object_type)] = (self._tick, str(action_name),
                                                action_ok)
            if self._target_hint.get("enabled"):
                from mllm_base_agent.agent import target_priority as tp

                inv0 = wm_metadata.get("inventoryObjects") or []
                held0 = str((inv0[0] or {}).get("objectType") or "") if inv0 else ""
                hinter = self._hinter
                if hinter is None:
                    _mf = self._target_hint.get("memory_frames")
                    hinter = tp.TargetHinter(
                        task_description=self._task_desc or "",
                        vlm=self._vlm,
                        limit=int(self._target_hint.get("limit", 6)),
                        names_limit=int(self._target_hint.get("names_limit", 5)),
                        max_dist=float(self._target_hint.get("max_dist", 3.0)),
                        max_sigma=float(self._target_hint.get("max_sigma", 0.5)),
                        memory_frames=(None if _mf is None else int(_mf)),
                    )
                    self._hinter = hinter
                block = hinter.update(wm_metadata, self._acts, held0, self._tick)
                if block:
                    parts.append(block)
            inv = wm_metadata.get("inventoryObjects") or []
            held = ""
            if inv:
                held = str((inv[0] or {}).get("objectType") or "")
            parts.append(f"手持：{held}" if held else "手持：空手")
            if blocked:
                parts.append(
                    "移动提示：上一步的移动没有让画面发生变化（多半被挡）。"
                    "换方向绕行——先 RotateLeft(90)/RotateRight(90) 再 MoveAhead，"
                    "不要连续重复同一个被挡的动作"
                )
            if action_name:
                if action_ok is True:
                    parts.append(f"上一个动作：{action_name}（成功：画面已变化）")
                elif action_ok is False:
                    parts.append(f"上一个动作：{action_name}（失败：画面未变化）")
                else:
                    parts.append(f"上一个动作：{action_name}")
        except Exception:
            import os as _os
            if _os.environ.get("WM_DEBUG"):
                import traceback as _tb
                print("[WM-DEBUG] MemoryProbe.update raised:\n"
                      + _tb.format_exc(), flush=True)
            return ""
        self._pending = ["；".join(parts)] if parts else []
        return self.pending_text()

    def pending_text(self) -> str:
        if not self._pending:
            return ""
        body = "\n".join(f"- {h}" for h in self._pending)
        return (
            "🧠 [SpatialMemory] 世界模型提供的当前状态（基于自身感知与动作日志，请直接采信）：\n"
            f"{body}"
        )
