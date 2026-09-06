"""Failure Detection v0 for the SpatialWorld runner.

Detects actionable failure signals from vision + action history + the
environment's action feedback, and produces short Chinese hints injected into
the next model prompt.

Signals:
  - blocked move/rotate  : frame difference ~ 0 after a move/rotate
  - visual stagnation    : several consecutive near-identical frames
  - action loop          : repeated same action or small-period cycles
  - repeat after failure : re-issuing the exact action that just failed
  - environment errors   : classify error_message into actionable hints
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

DIFF_THRESHOLD = 0.00001

MOVES = {"moveahead", "moveback", "moveleft", "moveright"}
ROTATES = {"rotateleft", "rotateright", "lookup", "lookdown"}
PROGRESS_ACTIONS = MOVES | ROTATES


def _norm_action(action: Any) -> str:
    if action is None:
        return ""
    if isinstance(action, dict):
        action = action.get("action_name") or action.get("name") or ""
    raw = str(action).strip()
    if "(" in raw:
        raw = raw.split("(", 1)[0]
    return raw.lower().replace("_", "").replace(" ", "").replace("-", "")


def _frame_mse(a_path: str, b_path: str) -> Optional[float]:
    try:
        with Image.open(a_path) as ia, Image.open(b_path) as ib:
            a = np.asarray(ia.convert("RGB").resize((200, 150)), dtype=np.float32) / 255.0
            b = np.asarray(ib.convert("RGB").resize((200, 150)), dtype=np.float32) / 255.0
        return float(np.mean((a - b) ** 2))
    except Exception:
        return None


class FailureDetector:
    def __init__(
        self,
        *,
        diff_threshold: float = DIFF_THRESHOLD,
        stagnation_steps: int = 3,
        same_action_repeat: int = 4,
        loop_periods: tuple = (2, 3),
        min_cycles: int = 2,
        max_hints: int = 2,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.stagnation_steps = stagnation_steps
        self.same_action_repeat = same_action_repeat
        self.loop_periods = loop_periods
        self.min_cycles = min_cycles
        self.max_hints = max_hints
        self._history: List[str] = []
        self._stagnation = 0
        self._pending: List[str] = []
        self._prev_image: Optional[str] = None

    # ------------------------------------------------------------------
    def update(
        self,
        *,
        action: Any,
        action_success: bool,
        error_message: Optional[str],
        prev_image_path: Optional[str],
        cur_image_path: Optional[str],
    ) -> None:
        an = _norm_action(action)
        hints: List[str] = []

        # 1) blocked move/rotate (frame did not change)
        if an in PROGRESS_ACTIONS and prev_image_path and cur_image_path:
            diff = _frame_mse(prev_image_path, cur_image_path)
            if diff is not None and diff < self.diff_threshold:
                self._stagnation += 1
                if not action_success:
                    hints.append(
                        f"刚才的 {an} 没有生效（画面几乎没变，可能被挡住）。"
                        "不要原样重试，请先转向或换个方向。"
                    )
                if self._stagnation >= self.stagnation_steps:
                    hints.append(
                        f"连续 {self._stagnation} 步画面几乎没有变化，疑似卡住。"
                        "请转身、后退或换一条路线。"
                    )
            else:
                self._stagnation = 0

        # 2) action loop
        loop = self._detect_loop()
        if loop:
            hints.append(f"检测到动作循环（{loop}）。请跳出循环，尝试新的动作。")

        # 3) repeat right after failure
        if (
            not action_success
            and self._history
            and _norm_action(action) == self._history[-1]
        ):
            hints.append(f"动作 {an} 刚刚失败过，不要原样重试。")

        # 4) classify environment error message into an actionable hint
        env_hint = self._classify_error(error_message)
        if env_hint:
            hints.append(env_hint)

        self._history.append(an)
        if len(self._history) > 12:
            self._history = self._history[-12:]
        self._prev_image = cur_image_path
        self._pending = (self._pending + hints)[: self.max_hints]

    # ------------------------------------------------------------------
    def _detect_loop(self) -> Optional[str]:
        tail = self._history
        if len(tail) < 2:
            return None
        last = tail[-1]
        if last:
            run = 0
            for a in reversed(tail):
                if a == last:
                    run += 1
                else:
                    break
            if run >= self.same_action_repeat:
                return f"同一动作 {last} 连续 {run} 次"
        for period in self.loop_periods:
            need = period * self.min_cycles
            if len(tail) < need:
                continue
            segment = tail[-need:]
            first = segment[:period]
            if all(segment[i] == first[i % period] for i in range(need)):
                return f"周期 {period} 重复：{' → '.join(first)}"
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _classify_error(error_message: Optional[str]) -> Optional[str]:
        e = (error_message or "").lower()
        if not e:
            return None
        if "not in view" in e:
            return "目标不在当前视野中。请先移动到能看到它的位置，再执行交互。"
        if "hand already has an object" in e:
            return "你手里已经有物体，再次 PickupObject 会失败。请先 DropHandObject 或 PutObject。"
        if "not holding anything" in e or "isn't holding" in e or "agent isn't holding" in e:
            return "你手里没有物体，无法执行该动作。"
        if "no valid positions to place" in e:
            return "放置失败：没有合适的位置。请换个容器或调整位置后再放。"
        if "not a receptacle" in e:
            return "目标不是容器，无法放置。请选择真正的容器/台面。"
        if "receptacle is closed" in e or "is closed, can't place" in e:
            return "容器是关闭的，请先打开再放置。"
        if "can't look down beyond" in e or "can't look up beyond" in e:
            return "视角已达上下极限，请勿继续 LookUp/LookDown。"
        if "blocking agent" in e or "blocking the agent" in e:
            return "前方被物体挡住，移动失败。请转向或绕路。"
        if "collide" in e:
            return "手持物体可能发生碰撞，请先放下或换方向。"
        return None

    # ------------------------------------------------------------------
    def pending_text(self) -> str:
        if not self._pending:
            return ""
        body = "\n".join(f"- {h}" for h in self._pending)
        return (
            "⚠️ [FailureDetector] 以下为基于画面/动作/反馈的提醒，请优先参考：\n"
            f"{body}"
        )

    def consume_pending(self) -> str:
        text = self.pending_text()
        self._pending = []
        return text
