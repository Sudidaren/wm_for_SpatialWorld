"""Experiment B: perfect-memory injection probe.

At every step, build a concise Chinese "copilot memory" hint from simulator
ground truth (agent pose, held object, target positions relative to the agent,
last action result) and inject it into the next prompt. This measures whether
GPT-5 can USE a perfect spatial memory when it is given, versus having to build
it itself (official baseline).
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

_DIRS = [
    "正前方",
    "右前方",
    "正右方",
    "右后方",
    "正后方",
    "左后方",
    "正左方",
    "左前方",
]

_INTERACTION_ACTIONS = {
    "PickupObject",
    "PutObject",
    "DropHandObject",
    "ThrowObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
    "BreakObject",
    "CookObject",
    "DirtyObject",
    "CleanObject",
    "FillObjectWithLiquid",
    "EmptyLiquidFromObject",
    "UseUpObject",
    "PushObject",
    "PullObject",
}

# common-sense kitchen priors: where an object is usually found
_PRIORS: Dict[str, list] = {
    "Egg": ["CounterTop", "Plate", "Fridge", "StoveBurner"],
    "Potato": ["CounterTop", "Plate"],
    "Lettuce": ["CounterTop", "Fridge"],
    "Tomato": ["CounterTop", "Fridge"],
    "Apple": ["CounterTop", "Fridge", "Bowl", "Plate"],
    "Bread": ["CounterTop", "Toaster", "Cabinet"],
    "Plate": ["Cabinet", "CounterTop", "Sink"],
    "Pot": ["Cabinet", "CounterTop", "StoveBurner"],
    "Pan": ["Cabinet", "CounterTop", "StoveBurner"],
    "Mug": ["Cabinet", "CounterTop", "Shelf", "Sink"],
    "Knife": ["CounterTop", "Cabinet", "Drawer"],
    "Microwave": ["CounterTop", "Shelf", "Cabinet"],
    "KeyChain": ["Cabinet", "Drawer", "Shelf", "Box", "Desk"],
    "CreditCard": ["Cabinet", "Drawer", "Desk", "Box", "Shelf"],
    "CellPhone": ["Desk", "CounterTop", "Cabinet", "Shelf"],
    "Book": ["Shelf", "Desk", "Cabinet", "Table"],
    "AlarmClock": ["Desk", "Shelf", "Table"],
    "GarbageCan": ["Floor"],
}

_ZH_ALIASES: Dict[str, list] = {
    "Egg": ["鸡蛋"], "Potato": ["土豆"], "Pot": ["锅"], "Pan": ["平底锅"],
    "Plate": ["盘子"], "Lettuce": ["生菜"], "Tomato": ["番茄", "西红柿"],
    "Apple": ["苹果"], "KeyChain": ["钥匙"], "CreditCard": ["银行卡", "信用卡"],
    "Mug": ["杯子", "马克杯"], "Knife": ["刀"], "Bread": ["面包"],
    "CellPhone": ["手机"], "Book": ["书"], "GarbageCan": ["垃圾桶"],
    "Microwave": ["微波炉"], "Fridge": ["冰箱"], "Cabinet": ["柜子"],
}

_LOCATION_WORDS = {
    "counter": "台面", "countertop": "台面", "table": "桌上", "desk": "桌上",
    "fridge": "冰箱里", "refrigerator": "冰箱里", "cabinet": "柜子里",
    "drawer": "抽屉里", "shelf": "架子上", "plate": "盘子里", "pot": "锅里",
    "pan": "锅里", "microwave": "微波炉里", "box": "盒子里", "sink": "水槽里",
    "stove": "灶台上", "burner": "灶台上", "bed": "床上", "toaster": "烤面包机旁",
    "bowl": "碗里", "dresser": "柜子上", "chest": "柜子上",
}

_DIR_TO_MOVE = {
    "正前方": "MoveAhead",
    "右前方": "MoveRight 再 MoveAhead",
    "正右方": "MoveRight",
    "右后方": "MoveRight 再 MoveBack",
    "正后方": "MoveBack",
    "左后方": "MoveLeft 再 MoveBack",
    "正左方": "MoveLeft",
    "左前方": "MoveLeft 再 MoveAhead",
}

_DIR_TO_FACE = {
    "正左方": "RotateLeft(90)",
    "正右方": "RotateRight(90)",
    "正后方": "RotateLeft(180)",
    "左前方": "RotateLeft(90)",
    "右前方": "RotateRight(90)",
    "左后方": "RotateLeft(90)",
    "右后方": "RotateRight(90)",
}


def _rel_direction(dx: float, dz: float, yaw: float) -> str:
    """Map a world-space delta to an egocentric direction label."""
    rad = math.radians(yaw)
    fwd = dx * math.sin(rad) + dz * math.cos(rad)
    right = dx * math.cos(rad) - dz * math.sin(rad)
    ang = math.degrees(math.atan2(right, fwd))  # -180..180, 0 = front
    idx = int(round(ang / 45.0)) % 8
    return _DIRS[idx]


class MemoryProbe:
    def __init__(
        self,
        targets: Optional[List[str]] = None,
        navigation_directive: bool = False,
        approach_threshold: float = 0.95,
        interact_soft_dist: float = 1.2,
        success_conditions: Optional[List[Dict]] = None,
        map_provider=None,
        task_description: Optional[str] = None,
        oracle_reachable: bool = False,
    ) -> None:
        self.targets = targets or ["Lettuce", "GarbageCan"]
        self.navigation_directive = navigation_directive
        self.approach_threshold = approach_threshold
        self.interact_soft_dist = interact_soft_dist
        self._conditions = success_conditions or []
        self._map_provider = map_provider
        self._oracle_reachable = oracle_reachable
        self._task_desc = task_description or ""
        self._task_location = self._parse_task_locations()
        self._known_provider = None
        self._prev_dist: Dict[str, float] = {}
        self._stuck: Dict[str, int] = {}
        self._interact_fail: Dict[str, int] = {}
        self._act_hist: List[tuple] = []
        self._not_in_view_fails: Dict[tuple, int] = {}
        self._place_fail: int = 0
        self._reachable: Optional[set] = None
        self._blocked_cells: set = set()
        self._blocked_moves: set = set()
        self._prev_pos = None
        self._prev_yaw = None
        self._last_blocked = False
        self._pending: List[str] = []

    def set_map_provider(self, provider) -> None:
        self._map_provider = provider

    def set_known_provider(self, provider) -> None:
        self._known_provider = provider
        self._reachable = None

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

    def update(
        self,
        *,
        metadata: Dict,
        action_name: Optional[str],
        error_message: Optional[str],
        object_type: Optional[str] = None,
        env: Any = None,
    ) -> str:
        hints: List[str] = []
        try:
            agent = metadata.get("agent") or {}
            pos = agent.get("position") or {}
            rot = agent.get("rotation") or {}
            x, z, yaw = pos.get("x"), pos.get("z"), rot.get("y")
            if x is None or z is None or yaw is None:
                return ""
            cur_cell = (round(x / 0.25), round(z / 0.25))
            # track interaction failures for soft-zone escalation
            if action_name in _INTERACTION_ACTIONS:
                key = object_type or action_name
                if error_message:
                    self._interact_fail[key] = self._interact_fail.get(key, 0) + 1
                else:
                    self._interact_fail[key] = 0
            # success-action history for loop detection (last 12)
            if action_name and not error_message:
                self._act_hist.append((action_name, object_type or ""))
                del self._act_hist[:-12]
                if action_name in ("LookUp", "LookDown"):
                    self._not_in_view_fails.clear()
            # consecutive not-in-view failures -> pitch-angle advice
            if action_name in _INTERACTION_ACTIONS:
                nkey = (action_name, object_type or "")
                emsg = (error_message or "").lower()
                if error_message and ("not in view" in emsg or "not visible" in emsg):
                    self._not_in_view_fails[nkey] = self._not_in_view_fails.get(nkey, 0) + 1
                elif not error_message:
                    self._not_in_view_fails[nkey] = 0
            # PutObject placement failures -> reposition advice
            if action_name == "PutObject":
                pmsg = (error_message or "").lower()
                if error_message and "no valid positions" in pmsg:
                    self._place_fail += 1
                elif not error_message:
                    self._place_fail = 0
            # record failed moves as locally blocked (position did not change)
            if (
                self._prev_pos is not None
                and cur_cell == self._prev_pos
                and error_message
                and action_name in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
            ):
                self._last_blocked = True
                dcx, dcz = self._move_dir(action_name, self._prev_yaw or 0.0)
                self._blocked_moves.add((self._prev_pos, (dcx, dcz)))
                self._blocked_cells.add((self._prev_pos[0] + dcx, self._prev_pos[1] + dcz))
            else:
                self._last_blocked = False
            self._prev_pos = cur_cell
            self._prev_yaw = float(yaw)
            objects = metadata.get("objects", []) or []
            inv = metadata.get("inventoryObjects", []) or []
            held_ids = {o.get("objectId") for o in inv if o.get("objectId")}

            # positions of held containers (for proximity-based content detection)
            held_pos = {}
            # NOTE: inventoryObjects entries carry only objectId/objectType (no
            # position); read positions from the full objects list instead.
            for o in objects:
                p = o.get("position") or {}
                if o.get("objectId") in held_ids and p.get("x") is not None:
                    held_pos[o["objectId"]] = (p["x"], p["y"], p["z"])

            # objects carried ON held containers (e.g. Potato on the held Plate):
            # AI2-THOR marks none of this (isPickedUp=False, parentReceptacles=null),
            # so detect via parent link OR position coincidence with a held item.
            # containers OF the held item are not carried by it
            held_parent_ids = set()
            for o in objects:
                if o.get("objectId") in held_ids:
                    held_parent_ids.update(o.get("parentReceptacles") or [])
            carried_objs = []
            for o in objects:
                oid = o.get("objectId")
                if not oid or oid in held_ids or oid in held_parent_ids:
                    continue
                linked = any(p in held_ids for p in (o.get("parentReceptacles") or []))
                p = o.get("position") or {}
                if p.get("x") is None:
                    continue
                near_held = any(
                    math.hypot(p["x"] - hx, p["z"] - hz) < 0.35 and abs(p["y"] - hy) < 0.2
                    for (hx, hy, hz) in held_pos.values()
                )
                if linked or near_held:
                    carried_objs.append(o)
            carried_ids = {o.get("objectId") for o in carried_objs if o.get("objectId")}
            contents_by_held: Dict[str, set] = {}
            for o in carried_objs:
                p = o.get("position") or {}
                owner = None
                for hid in held_ids:
                    if hid in (o.get("parentReceptacles") or []):
                        owner = hid
                        break
                if owner is None and p.get("x") is not None:
                    # attribute to the nearest held container within range
                    best = None
                    best_d = 1e9
                    for hid, (hx, hy, hz) in held_pos.items():
                        d = math.hypot(p["x"] - hx, p["z"] - hz) + abs(p["y"] - hy)
                        if d < best_d:
                            best, best_d = hid, d
                    if best is not None and best_d < 0.8:
                        owner = best
                if owner is not None:
                    contents_by_held.setdefault(owner, set()).add(
                        str(o.get("objectType") or "?")
                    )

            # hand description with container contents
            hand_parts: List[str] = []
            for o in inv:
                hid = o.get("objectId")
                t = str(hid).split("|")[0] if hid else "?"
                contents = sorted(contents_by_held.get(hid, set()))
                hand_parts.append(f"{t}（上面有{'、'.join(contents)}）" if contents else t)
            held_s = "、".join(hand_parts) if hand_parts else "空手"

            parts = [f"你位于 ({x:.2f},{z:.2f})，朝向 {int(yaw) % 360}°，手持：{held_s}"]
            excluded = held_ids | carried_ids
            target_infos: List[tuple] = []
            for t in self.targets:
                objs = [
                    o
                    for o in objects
                    if o.get("objectType") == t and o.get("objectId") not in excluded
                ]
                if not objs:
                    continue
                nearest = min(
                    objs,
                    key=lambda o: float(o.get("distance") or 1e9),
                )
                p = nearest.get("position") or {}
                dx = float(p.get("x", x)) - float(x)
                dz = float(p.get("z", z)) - float(z)
                dist = math.hypot(dx, dz)
                direction = _rel_direction(dx, dz, float(yaw))
                vis = "可见" if nearest.get("visible") else "不可见"
                err = float(nearest.get("distance_err") or 0.15)
                dist_s = f"{max(0.0, dist - err):.2f}–{dist + err:.2f}m" \
                    if err >= 0.05 else f"{dist:.2f}m"
                parts.append(f"{t} 在{direction}约 {dist_s}（{vis}）")
                target_infos.append(
                    (t, dist, direction, bool(nearest.get("visible")),
                     p.get("x"), p.get("z"), err)
                )

            # move-then-scan navigation directives
            if self.navigation_directive and target_infos:
                directives: List[str] = []
                for (t, dist, direction, visible, tx, tz, err) in target_infos:
                    prev = self._prev_dist.get(t)
                    dist_s = f"{max(0.0, dist - err):.2f}–{dist + err:.2f}m" \
                        if err >= 0.05 else f"{dist:.2f}m"
                    if dist > self.interact_soft_dist:
                        if self._last_blocked:
                            wp = self._find_aligned_waypoint(t, tx, tz, metadata, env)
                            if wp is not None:
                                wx, wz, face, route = wp
                                route_txt = f"；下一步：{route}" if route else ""
                                d = (
                                    f"{t} 在{direction}约 {dist_s}，且刚才移动被挡："
                                    f"绕道到航点 ({wx:.2f},{wz:.2f}) 面朝 {face}°，目标在正前方"
                                    f"{route_txt}"
                                )
                            else:
                                d = (
                                    f"{t} 在{direction}约 {dist_s}，且刚才移动被挡："
                                    "换方向绕道靠近，先别急着交互"
                                )
                        else:
                            move = _DIR_TO_MOVE.get(direction, "MoveAhead")
                            face = _DIR_TO_FACE.get(direction)
                            if visible:
                                d = (
                                    f"{t} 在{direction}约 {dist_s}（可见），还比较远："
                                    f"向{direction}移动靠近（{move}），被挡就绕道"
                                )
                            elif face:
                                d = (
                                    f"{t} 在{direction}约 {dist_s}（不可见）："
                                    f"先{face}面向它，再{move}靠近；被挡就绕道，别原地反复转"
                                )
                            else:
                                d = (
                                    f"{t} 在{direction}约 {dist_s}（不可见），还比较远："
                                    f"向{direction}移动靠近（{move}），被挡就绕道"
                                )
                        if prev is not None and dist > prev + 0.05:
                            d += f"（注意：你正在远离{t}，上一步 {prev:.2f}m → 现在 {dist:.2f}m）"
                        self._stuck[t] = 0
                        directives.append(d)
                    elif visible:
                        d = (
                            f"{t} 在{direction}约 {dist_s}（可见）：已很近，"
                            "可以尝试交互；若失败，再靠近或绕道"
                        )
                        self._stuck[t] = 0
                        directives.append(d)
                    else:
                        self._stuck[t] = self._stuck.get(t, 0) + 1
                        if (
                            self._stuck[t] >= 2
                            or self._interact_fail.get(t, 0) >= 1
                        ):
                            wp = self._find_aligned_waypoint(t, tx, tz, metadata, env)
                            if wp is not None:
                                wx, wz, face, route = wp
                                route_txt = f"；下一步：{route}" if route else ""
                                directives.append(
                                    f"{t} 在你{direction}约 {dist_s}（当前不可见），"
                                    f"且多次尝试未成功：请走到航点 ({wx:.2f},{wz:.2f}) "
                                    f"面朝 {face}°，目标在正前方"
                                    f"{route_txt}；"
                                    "到位后若仍不可见，先 LookUp/LookDown 再交互"
                                )
                            else:
                                directives.append(
                                    f"{t} 在{direction}约 {dist_s} 但不可见："
                                    "可尝试交互，失败则 LookUp/LookDown 或原地旋转；"
                                    "仍不行就靠近或绕道"
                                )
                        else:
                            directives.append(
                                f"{t} 在{direction}约 {dist_s} 但不可见："
                                "可尝试交互，失败则调整视角；仍不行就靠近"
                            )
                for (t, dist, _dir, _vis, _tx, _tz) in target_infos:
                    self._prev_dist[t] = dist
                parts.append("建议：" + "；".join(directives))

            # coach: action-loop and pitch-angle advice
            if self._act_hist:
                recent = self._act_hist[-8:]
                counts: Dict[tuple, int] = {}
                for a in recent:
                    counts[a] = counts.get(a, 0) + 1
                loop = next(
                    ((name, otype) for (name, otype), n in sorted(counts.items())
                     if n >= 3 and (
                         name in _INTERACTION_ACTIONS
                         or name in ("RotateLeft", "RotateRight")
                     )),
                    None,
                )
                if loop is not None:
                    lname, lotype = loop
                    parts.append(
                        f"⚠️ 检测到你在原地转圈/重复动作（{lname} 已执行 {counts[loop]} 次）"
                        "没有推进任务：停止原地反复旋转，按目标方位直接移动靠近"
                    )
            target_dist = {t: d for (t, d, *_rest) in target_infos}
            for (fname, fotype), n in sorted(self._not_in_view_fails.items()):
                if n < 2:
                    continue
                near = (
                    fotype in target_dist
                    and target_dist[fotype] <= self.interact_soft_dist
                )
                if near or n >= 3:
                    parts.append(
                        f"⚠️ {fname}({fotype}) 已连续失败 {n} 次（not in view）且目标很近："
                        "相机俯仰角可能不对，先 LookUp 或 LookDown 调整视角再交互，"
                        "仍失败则移动站位"
                    )
            if self._place_fail >= 1:
                parts.append(
                    f"⚠️ 放置失败（No valid positions）{self._place_fail} 次："
                    "你离目标太近或角度不对——先往前走一小步（MoveAhead 0.25）调整站位，"
                    "再左转（RotateLeft 90°），再试 PutObject；若被挡就横移一步（MoveRight/MoveLeft）再试"
                )

            # systematic search for required targets never seen
            if self._conditions:
                srch = self._search_directive(objects)
                if srch:
                    parts.append("搜索：" + "；".join(srch))

            # task-progress checklist from success conditions
            if self._conditions:
                held_types = {
                    str(o.get("objectId") or "").split("|")[0] for o in inv
                }
                carried_types = {str(o.get("objectType") or "") for o in carried_objs}
                progress = self._check_progress(objects, held_types, carried_types)
                if progress:
                    parts.append(progress)

            if action_name:
                if error_message:
                    parts.append(f"上一个动作 {action_name} 失败：{str(error_message)[:80]}")
                else:
                    parts.append(f"上一个动作 {action_name} 成功")
            hints.append("；".join(parts))
        except Exception:
            return ""
        self._pending = hints
        return self.pending_text()

    def _check_progress(self, objects: list, held_types: set = None, carried_types: set = None) -> str:
        done = []
        todo = []
        for c in self._conditions:
            if not isinstance(c, dict):
                continue
            if self._check_condition(c, objects, held_types, carried_types):
                done.append(self._cond_label(c))
            else:
                todo.append(self._cond_hint(c))
        if not done and not todo:
            return ""
        text = "进度：" + ("；".join(done) if done else "（暂无完成项）")
        if todo:
            text += "；还差：" + "；".join(todo)
        else:
            text += "（全部满足）"
        return text

    @staticmethod
    def _check_condition(c: Dict, objects: list, held_types=None, carried_types=None) -> bool:
        ct = c.get("type")
        if ct == "object_in_receptacle":
            ot = c.get("object_type")
            rt = c.get("receptacle_type")
            if any(
                o.get("objectType") == ot
                and any(str(p).split("|")[0] == rt for p in (o.get("parentReceptacles") or []))
                for o in objects
            ):
                return True
            # held container carrying the object counts as "in receptacle"
            if (carried_types and ot in carried_types) and (held_types and rt in held_types):
                return True
            return False
        if ct == "object_state":
            ot = c.get("object_type")
            st = c.get("state")
            val = c.get("value")
            return any(
                o.get("objectType") == ot and ((o.get("states") or {}).get(st) == val)
                for o in objects
            )
        return True

    @staticmethod
    def _cond_label(c: Dict) -> str:
        desc = c.get("description")
        if desc:
            return str(desc)
        if c.get("type") == "object_in_receptacle":
            return f"{c.get('object_type')} 在 {c.get('receptacle_type')} 中"
        return f"{c.get('object_type')}.{c.get('state')}={c.get('value')}"

    @staticmethod
    def _cond_hint(c: Dict) -> str:
        if c.get("type") == "object_in_receptacle":
            return f"把 {c.get('object_type')} 放进 {c.get('receptacle_type')}"
        ot = c.get("object_type")
        st = c.get("state")
        val = c.get("value")
        if st == "isToggled" and val is True:
            return (
                f"任务未完成：{ot} 未开启（isToggled=false）。不要 DONE，"
                f"下一步：先 CloseObject({ot}) 再 ToggleObjectOn({ot})"
            )
        if st == "isOpen":
            return f"{ot} 门{'未开：OpenObject' if val is True else '未关：CloseObject'}({ot})"
        action = {
            "isSliced": "SliceObject",
            "isCooked": "CookObject",
            "isDirty": "CleanObject",
            "isBroken": "BreakObject",
            "isUsedUp": "UseUpObject",
            "isFilledWithLiquid": "FillObjectWithLiquid",
        }.get(st)
        if action:
            return f"{ot} 需 {action}({ot})"
        return f"{ot}.{st} 应为 {val}"

    def _find_aligned_waypoint(
        self,
        target_type: str,
        tx: float,
        tz: float,
        metadata: Dict,
        env: Any,
    ):
        """Nearest reachable aligned cell + a BFS route from the agent to it.

        Returns (wx, wz, facing_yaw, route_text) or None.
        """
        reachable = self._reachable_cells(env)
        if not reachable:
            return None
        pos = (metadata.get("agent") or {}).get("position") or {}
        ax, az = pos.get("x"), pos.get("z")
        yaw = float(((metadata.get("agent") or {}).get("rotation") or {}).get("y", 0.0))
        gx = round(float(tx) / 0.25) * 0.25
        gz = round(float(tz) / 0.25) * 0.25
        candidates = [
            (gx, gz + 0.75, 180),    # waypoint south of target -> face north (-z)
            (gx, gz - 0.75, 0),      # waypoint north of target -> face south (+z)
            (gx + 0.75, gz, 270),    # waypoint east of target -> face west (-x)
            (gx - 0.75, gz, 90),     # waypoint west of target -> face east (+x)
        ]
        best = None
        best_d = float("inf")
        for (wx, wz, face) in candidates:
            if (round(wx / 0.25), round(wz / 0.25)) not in reachable:
                continue
            d = math.hypot(wx - ax, wz - az) if ax is not None and az is not None else 0.0
            if d < best_d:
                best_d, best = d, (wx, wz, face)
        if best is None:
            return None
        wx, wz, face = best
        from_cell = (round(ax / 0.25), round(az / 0.25)) if ax is not None else None
        to_cell = (round(wx / 0.25), round(wz / 0.25))
        route = None
        if from_cell is not None:
            path = self._plan_route(
                from_cell, to_cell, reachable, self._known_cells()
            )
            if path:
                acts = self._path_to_actions(path, yaw, face)
                route = acts[0] if acts else None
        return (wx, wz, face, route)

    def _plan_route(self, from_cell, to_cell, reachable, known=None):
        """Dijkstra over reachable grid cells; returns list of cells or None.

        Known-walkable cells cost 1, unknown cells cost 8, so the planner
        prefers the already-walked footprint and only cuts through unseen
        cells when there is no cheaper alternative (avoids walking through
        walls that have not been observed yet).
        """
        import heapq

        if from_cell not in reachable:
            return None
        known = known or set()
        UNKNOWN_COST = 8
        q = [(0, from_cell)]
        dist = {from_cell: 0}
        prev = {from_cell: None}
        while q:
            d, cur = heapq.heappop(q)
            if d != dist.get(cur):
                continue
            if cur == to_cell:
                break
            for dcx, dcz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cur[0] + dcx, cur[1] + dcz)
                if (
                    nxt not in reachable
                    or nxt in self._blocked_cells
                    or (cur, (dcx, dcz)) in self._blocked_moves
                ):
                    continue
                nd = d + (1 if nxt in known else UNKNOWN_COST)
                if nd >= dist.get(nxt, float("inf")):
                    continue
                dist[nxt] = nd
                prev[nxt] = cur
                heapq.heappush(q, (nd, nxt))
        if to_cell not in prev:
            return None
        path = []
        cur = to_cell
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    @staticmethod
    def _path_to_actions(path, yaw, face):
        """Convert a cell path to move actions relative to the current yaw."""
        actions = []
        rad = math.radians(yaw)
        for i in range(len(path) - 1):
            dcx = path[i + 1][0] - path[i][0]
            dcz = path[i + 1][1] - path[i][1]
            wx, wz = dcx * 0.25, dcz * 0.25
            fwd = wx * math.sin(rad) + wz * math.cos(rad)
            right = wx * math.cos(rad) - wz * math.sin(rad)
            if abs(fwd) >= abs(right):
                actions.append(f"Move{'Ahead' if fwd > 0 else 'Back'}(0.25)")
            else:
                actions.append(f"Move{'Right' if right > 0 else 'Left'}(0.25)")
        diff = (face - yaw) % 360
        if diff == 90:
            actions.append("RotateRight(90)")
        elif diff == 270:
            actions.append("RotateLeft(90)")
        elif diff == 180:
            actions.append("RotateLeft(180)")
        elif diff != 0:
            actions.append(f"Rotate{'Left' if diff <= 180 else 'Right'}({min(diff, 360 - diff)})")
        return actions

    def _reachable_cells(self, env: Any) -> set:
        """Cached reachable grid cells from the simulator (query action)."""
        if self._map_provider is not None:
            # world-model occupancy grows every step: always re-query
            try:
                return set(self._map_provider()) or set()
            except Exception:
                return set()
        if self._reachable is not None:
            return self._reachable
        if not self._oracle_reachable:
            # runtime is not allowed to query simulator ground truth
            return set()
        controller = getattr(env, "controller", None)
        if controller is None:
            return set()
        try:
            ev = controller.step(action="GetReachablePositions")
            ret = (ev.metadata or {}).get("actionReturn") or []
            self._reachable = {
                (round(p.get("x", 0) / 0.25), round(p.get("z", 0) / 0.25))
                for p in ret
                if "x" in p and "z" in p
            }
        except Exception:
            self._reachable = set()
        return self._reachable

    def _parse_task_locations(self) -> Dict[str, str]:
        """Mine explicit "X is on/in Y" statements from the task description.
        Goal phrases like "put the egg in the pot" are ignored."""
        out: Dict[str, str] = {}
        desc = self._task_desc or ""
        if not desc:
            return out
        for t in self.targets:
            names = [t, t.lower()] + _ZH_ALIASES.get(t, [])
            for name in names:
                pat_en = re.compile(
                    r"\b" + re.escape(name) + r"[^.]{0,40}?\b(?:is|are)\s+(?:on|in|at|inside)\s+the\s+([a-z ]+?)(?=[,. ]|$)",
                    re.IGNORECASE,
                )
                m = pat_en.search(desc)
                if m:
                    place = m.group(1).strip().split()[0].lower()
                    out[t] = _LOCATION_WORDS.get(place, place)
                    break
                pat_zh = re.compile(
                    re.escape(name) + r"\s*在\s*([^，。；]{1,10}?)(?:上|里|旁边|附近)"
                )
                mz = pat_zh.search(desc)
                if mz:
                    out[t] = mz.group(1).strip()
                    break
        return out

    def _needed_targets(self) -> set:
        needed = set()
        for c in self._conditions:
            ct = c.get("type")
            if ct in ("object_in_receptacle", "object_in_hand"):
                if c.get("object_type"):
                    needed.add(c.get("object_type"))
                if c.get("receptacle_type"):
                    needed.add(c.get("receptacle_type"))
            elif ct == "object_state" and c.get("object_type"):
                needed.add(c.get("object_type"))
        return needed

    def _nearest_furniture(self, types, furniture, x, z, yaw):
        if not types:
            return None
        best = None
        best_d = 1e9
        for f in furniture:
            if int(f.get("seen") or 0) < 2:
                continue  # avoid one-frame furniture flashes
            ft = f.get("objectType") or ""
            if ft not in types:
                continue
            fp = f.get("position") or {}
            fx, fz = fp.get("x"), fp.get("z")
            if fx is None or fz is None:
                continue
            d = math.hypot(fx - x, fz - z)
            if d < best_d:
                best_d = d
                best = (ft, fx - x, fz - z, d)
        if best is None:
            return None
        ft, dx, dz, d = best
        return {"type": ft, "direction": _rel_direction(dx, dz, yaw), "dist": d}

    def _exploration_advice(self, objects, furniture, x, z, yaw) -> List[str]:
        """Human-like exploration hints for targets the task needs but the
        world model has not seen yet: task-text location > common-sense
        priors + nearest furniture > generic explore advice."""
        if x is None or z is None:
            return []
        known = {o.get("objectType") for o in objects if o.get("objectType")}
        needed = self._needed_targets()
        parts: List[str] = []
        for t in sorted(needed):
            if t in known:
                continue
            if t in self._task_location:
                parts.append(
                    f"{t} 还没找到，但任务描述提到它在{self._task_location[t]}，直接去那里找"
                )
                continue
            priors = _PRIORS.get(t, [])
            priors_txt = "、".join(priors) if priors else "台面/柜子/抽屉等容器"
            cand = self._nearest_furniture(priors, furniture, x, z, yaw)
            if cand is not None:
                parts.append(
                    f"{t} 还没找到：常识上 {t} 常出现在{priors_txt}；"
                    f"{cand['direction']}约 {cand['dist']:.1f}m 有个还没确认内容的"
                    f"{cand['type']}，建议先去看看"
                )
            else:
                parts.append(
                    f"{t} 还没找到：常识上 {t} 常出现在{priors_txt}；"
                    "建议去还没探索过的台面/柜子/抽屉附近找找"
                )
        return parts

    def _search_directive(self, objects) -> List[str]:
        """Systematic 3-level visual search for required targets never seen:
        look down + full circle, level + full circle, look up + full circle,
        always rotating in the same direction to cover as much as possible.
        Wandering while staring down misses objects above the horizon."""
        known = {o.get("objectType") for o in objects if o.get("objectType")}
        missing = [
            t for t in sorted(self._needed_targets()) if t not in known
        ]
        if not missing:
            return []
        names = "、".join(missing)
        return [
            f"⚠️ {names} 还没找到：按三遍巡视找——"
            "先 LookDown 后连续同一方向转 4 次（全左转或全右转），"
            "再平视连续转 4 次，最后 LookUp 连续转 4 次；"
            f"看到 {names} 就停下直接过去拿"
        ]

    def _known_cells(self) -> set:
        """Known-walkable cells from the world model, if provided."""
        if self._known_provider is not None:
            try:
                return set(self._known_provider()) or set()
            except Exception:
                return set()
        return set()

    def pending_text(self) -> str:
        if not self._pending:
            return ""
        body = "\n".join(f"- {h}" for h in self._pending)
        return (
            "🧠 [SpatialMemory] 世界模型提供的当前状态（基于自身感知与动作日志，请直接采信）：\n"
            f"{body}"
        )

    def consume_pending(self) -> str:
        text = self.pending_text()
        self._pending = []
        return text
