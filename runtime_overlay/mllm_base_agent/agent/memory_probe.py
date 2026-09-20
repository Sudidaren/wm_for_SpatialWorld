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
import sys
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
        #: 提示装配过程中抛出的异常次数。**这个数不能永远是 0 而不被发现**：
        #: 任何内部异常都会让这一步的提示变成空串，表现是"WM 在跑但一个字都
        #: 不说"。2026-09-20 就发生过一次（删属性时漏删了两处使用），当时靠
        #: 二十多个单测同时报错才抓到。见 update() 末尾的处理。
        self.errors: int = 0
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
            reach_gap = None
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
                # 先算这一步要不要发"距离提示"（它需要 hinter），再决定要不要
                # 压掉导航建议 —— 顺序不能反，否则第一次调用时 hinter 还没建，
                # 距离提示会被整条丢掉。
                # nearest() 依赖"任务相关物体"集合，而那个集合是 relevant()
                # 建立/缓存的（update() 内部也是调它）。这里先调一次幂等的，
                # 让判断在新一步也能工作。
                hinter.relevant([str(o.get("objectType"))
                                 for o in (wm_metadata.get("objects") or [])
                                 if o.get("objectType")])
                reach_gap = self._reach_gap(
                    wm_metadata, action_name, action_ok, blocked)
                # 不给"往哪走"的两种情形 —— 都只压这一步，记忆行里的位置与
                # 距离照常保留，压掉的是**指令**不是情报：
                #   a) 被挡住：脱困指令与"建议先转向它再靠近"打架（实测 36%
                #      的被挡提示同时挂了它）；
                #   b) 马上要发"距离提示"（交互失败 + 目标可见但还太远）：
                #      那是"去够眼前这个东西"，而 nav 建议只针对**看不见**的
                #      目标 —— 两者一旦同时出现必然指向不同物体，就是两条移动
                #      指令。例：Apple 在 1.6m 要走近，Fridge 在 2.2m 要转向。
                block = hinter.update(
                    wm_metadata, self._acts, held0, self._tick,
                    suppress_nav=bool(blocked) or reach_gap is not None)
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
                # 事实说硬（继续走没有效果、别重复），走法全列出来，不下
                # "某个动作没用"的断言 —— WM 报事实，不替模型判定手段。
                #
                # "也可以做对任务有利的交互"这一条必须**带条件**，否则会误导：
                # 实测被挡时模型选交互 62 次，只有 26% 真的改变了画面（74% 空转），
                # 其中 PickupObject 更是 28 次空转 / 5 次生效 —— 因为它手里已经
                # 有东西（"手持"那行就在旁边写着），或者目标根本不在视野里。
                # 但反过来，被挡经常意味着"你已经站到那件家具面前了"（撞上冰箱
                # 门，正确动作就是 OpenObject(Fridge)），所以不能只教绕路。
                # 判据就写成"目标已到眼前、伸手可及"——这个条件 WM 自己知道。
                parts.append(
                    "移动提示：⚠ 上一步被挡住了，画面完全没有发生变化——"
                    "继续朝那个方向走不会有任何效果，不要重复同一个动作。"
                    "换一种做法再试：转向（RotateLeft/RotateRight）、"
                    "左移或右移（MoveLeft/MoveRight）、后退（MoveBack）；"
                    "如果目标已经就在眼前、伸手可及，那就直接对它做该做的交互"
                    "（拿起来 / 打开 / 放进去），不必先绕路"
                )
            # 同理：成功的那一步画面自己会说明，只有失败才值得占 token。
            if action_name and action_ok is False:
                parts.append(f"上一个动作：{action_name}（失败：画面未变化）")

            # ③ 卡死纠错（`重复提示`）已于 2026-09-20 删除。证据：
            #   * 模型本来就在换动作 —— 触发时近 4 步已经用过 2.86 个不同动作
            #     （普通失败是 3.10），它不是死磕一个动作，是在试；
            #   * 控制住停滞程度后没有收益 —— 按"近 3 步有效动作数"分层，
            #     主流层 62%(n=103) vs 对照 60%(n=297)，总体 65% vs 63%；
            #   * 有副作用 —— 触发后模型下一步转向 MoveBack 19% / LookDown 15%
            #     / Stand 9%，而 LookDown 的脱困率实测是 0%(n=7)，等于每触发
            #     一次就有约 1/4 的概率诱发一个无效动作；
            #   * 它原来还谎称"已连续 3 次"（137 次触发里 0 次真连续）。
            # 教训：加提示 ≠ 有帮助，得量。这条通道的全部状态机（_recent /
            # _stuck_for）随它一起删除。
            # ④ 交互失败且目标还太远 -> 直接告诉它还差多少（可执行的数字）
            if reach_gap is not None:
                parts.append(
                    f"距离提示：{reach_gap[0]} 在{reach_gap[2]}约 {reach_gap[1]:.1f}m，"
                    f"要先走到 1m 以内才能交互"
                    f"（还差约 {reach_gap[1] - _REACH_M:.1f}m）")
        except Exception as exc:                      # noqa: BLE001
            # 兜底不能是"静音"。这里吞掉的任何异常都等于这一步不注入，而
            # 外部看起来完全正常（进程健在、任务照跑、只是 WM 哑了）——
            # 那是最难查的一类故障。所以：
            #   * self.errors 记数，调用方/测试可以直接断言它必须是 0；
            #   * stderr 上留一行（前 3 次都留，之后每 50 次留一次，避免刷屏）；
            #   * WM_DEBUG=1 时照旧打印完整 traceback。
            self.errors += 1
            if self.errors <= 3 or self.errors % 50 == 0:
                print(f"[WM-ERROR] MemoryProbe.update 第 {self.errors} 次异常，"
                      f"这一步的提示为空：{type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
            if os.environ.get("WM_DEBUG"):
                import traceback as _tb
                print("[WM-DEBUG] MemoryProbe.update raised:\n"
                      + _tb.format_exc(), flush=True)
            return ""
        self._pending = ["；".join(parts)] if parts else []
        return self.pending_text()

    def _reach_gap(self, wm_metadata, action_name, action_ok, blocked):
        """这一步会不会发「距离提示」？会的话返回 (类型, 距离, 方向词)。

        必须**在** hinter.update() 之前判断，因为一旦要发距离提示，就要同时
        压掉这一步的导航建议：nav 建议只挑**看不见**的目标（记住的位置那几行），
        而距离提示只挑**可见**的 —— 两者同时出现必然指向不同物体，就是两条
        移动指令。例子：Apple 在眼前 1.6m 要走近，Fridge 在记忆里 2.2m 要转向。
        """
        if (not action_name or action_ok is not False
                or str(action_name) in _MOVE_ACTIONS
                or blocked
                or os.environ.get("LIGHTWM_REACH_HINT", "1") == "0"
                or self._hinter is None):
            return None
        gap = self._hinter.nearest(wm_metadata, visible=True)
        if gap is not None and gap[1] > _REACH_M:
            return gap
        return None

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
