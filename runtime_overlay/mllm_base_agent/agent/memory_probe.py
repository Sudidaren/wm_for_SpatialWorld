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

#: 移动/转向类动作（用于区分"被挡在原地打转"和"交互没成功"）
_MOVE_ACTIONS = frozenset({"MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
                           "RotateLeft", "RotateRight", "LookUp", "LookDown"})
#: 模拟器的交互距离上限（环境规格，不是任务答案）
_REACH_M = 1.0

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
        #: 最近几个 (动作, 是否改变了画面)，用于识别"卡死在重复同一个动作"
        self._recent: List[Tuple[str, Optional[bool]]] = []
        self._stuck_for: Optional[str] = None
        #: 连续被挡的步数。用来在"转身"的两个方向之间交替，避免连续两次
        #: 建议互相抵消（先左后右回到原地）。
        self._blocked_streak: int = 0

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
                # 被挡住的那一步不给"往哪走"的建议：脱困指令和"建议先转向它
                # 再靠近"是两条打架的移动指令（实测 36% 的被挡提示同时挂了
                # 它）。只压这一步；记忆行里的位置与距离照常保留。
                block = hinter.update(wm_metadata, self._acts, held0, self._tick,
                                      suppress_nav=bool(blocked))
                if block:
                    parts.append(block)
            inv = wm_metadata.get("inventoryObjects") or []
            held = ""
            if inv:
                # 环境改名后（Lettuce -> LettuceSliced）报模型能用的那个名字。
                held = str((inv[0] or {}).get("objectTypeDisplay")
                           or (inv[0] or {}).get("objectType") or "")
            # 2026-09-19：空手这一行是零信息（模型自己看得见手），不再注入；
            # 只有真的拿着东西时才报——那时它才影响下一步能做什么。
            if held:
                parts.append(f"手持：{held}")
            if blocked:
                # 语气要硬、只给一个动作：原来的"先 RotateLeft(90)/
                # RotateRight(90) 再 MoveAhead"一次给三个动作让模型自己拼，
                # 实测 301 次里只有 36% 下一步真的转身。方向按连续被挡次数
                # 交替，免得两次建议互相抵消。
                self._blocked_streak += 1
                turn = "RotateRight" if self._blocked_streak % 2 else "RotateLeft"
                parts.append(
                    f"移动提示：⚠ 上一步被挡住了——画面完全没有发生变化，"
                    f"那个方向走不通，继续走是浪费时间。"
                    f"下一步只做这一件事：{turn}(90)，"
                    f"转过身再从别的方向走"
                )
            else:
                self._blocked_streak = 0
            # 同理：成功的那一步画面自己会说明，只有失败才值得占 token。
            if action_name and action_ok is False:
                parts.append(f"上一个动作：{action_name}（失败：画面未变化）")

            # ③ 卡死纠错：同一个动作连续 3 次没让画面变化 -> 换策略。
            #    每次"卡死"只提示一次（动作变了或成功了就重新武装），并且
            #    只给一句、不引入新概念，避免变成噪声。
            if action_name and os.environ.get("LIGHTWM_STUCK_HINT", "1") != "0":
                name = str(action_name)
                self._recent.append((name, action_ok))
                self._recent = self._recent[-6:]
                same = [ok for a, ok in self._recent if a == name]
                stuck = len(same) >= 3 and all(ok is False for ok in same[-3:])
                if stuck and self._stuck_for != name:
                    self._stuck_for = name
                    if name in _MOVE_ACTIONS:
                        parts.append(
                            f"重复提示：{name} 已连续 3 次没有改变画面——"
                            f"换成先 RotateLeft(90)/RotateRight(90) 环视，再从别的方向靠近，不要继续重复")
                    else:
                        parts.append(
                            f"重复提示：{name} 已连续 3 次没有成功——"
                            f"先确认目标就在视野内并走到 1m 以内，或者换一个目标物")
                elif not stuck:
                    self._stuck_for = None

            # ④ 交互失败且目标还太远 -> 直接告诉它还差多少（可执行的数字）
            if (action_name and action_ok is False
                    and str(action_name) not in _MOVE_ACTIONS
                    and not blocked
                    and os.environ.get("LIGHTWM_REACH_HINT", "1") != "0"
                    and self._hinter is not None):
                gap = self._hinter.nearest(wm_metadata, visible=True)
                if gap is not None and gap[1] > _REACH_M:
                    parts.append(
                        f"距离提示：{gap[0]} 在{gap[2]}约 {gap[1]:.1f}m，"
                        f"要先走到 1m 以内才能交互（还差约 {gap[1] - _REACH_M:.1f}m）")
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

    def relevant_types(self) -> set:
        """任务相关物体集合（供 CheckState 汇总收窄用；未开提示时为空）。"""
        hinter = self._hinter
        return set(getattr(hinter, "_relevant", None) or set())
