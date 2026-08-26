"""Hidden-state / containment inference (causal memory).

Builds container->contents from the action log (PutObject / PickupObject /
DropHandObject), then answers "where might X be" by combining:
  1. causal truth  : "KeyChain was put into Box at step 12, Box not opened since"
  2. scene graph   : geometric on/in edges ("Potato on CounterTop#3")
  3. nearest anchor: "last seen near Cabinet#1"
"""

from __future__ import annotations

from typing import Dict, List, Optional


class HiddenState:
    def __init__(self):
        self.contents: Dict[str, Dict[str, object]] = {}  # container_key -> info
        self.obj_location: Dict[str, Dict[str, object]] = {}  # obj -> {container, step}
        self.container_states: Dict[str, bool] = {}       # container -> isOpen
        self.holding: Optional[str] = None

    def note_action(self, name: str, args: Dict, success: bool,
                    step: int) -> None:
        if not success:
            return
        oid = (args or {}).get("objectId") or ""
        if name == "PickupObject" and oid:
            self.holding = oid
            self.obj_location.pop(oid, None)
        elif name in ("PutObject",) and oid and self.holding:
            self.contents.setdefault(oid, {})["objs"] = \
                list(self.contents.get(oid, {}).get("objs", []))
            if self.holding not in self.contents[oid]["objs"]:
                self.contents[oid]["objs"].append(self.holding)
            self.contents[oid]["since_step"] = step
            self.obj_location[self.holding] = {"container": oid, "step": step}
            self.holding = None
        elif name in ("DropHandObject", "ThrowObject") and self.holding:
            # object left the hand; exact location unknown
            self.obj_location.pop(self.holding, None)
            self.holding = None
        elif name in ("OpenObject", "CloseObject") and oid:
            self.container_states[oid] = (name == "OpenObject")

    def where_is(self, obj_type: str) -> Optional[Dict]:
        """obj_type here is the base type (e.g. 'KeyChain')."""
        for obj, loc in self.obj_location.items():
            base = obj.split("_")[0].split("|")[0]
            if base == obj_type:
                c = loc["container"]
                opened_since = self.container_states.get(c, False)
                return {"container": c, "step": loc["step"],
                        "opened_since": opened_since}
        return None

    def inside(self, container_key: str) -> List[str]:
        return list(self.contents.get(container_key, {}).get("objs", []))

    def infer_location(self, obj_type: str, scene_graph, anchors) -> str:
        """Combine causal truth, scene-graph containment and last-known anchor."""
        causal = self.where_is(obj_type)
        if causal:
            st = "，容器之后未打开过" if not causal["opened_since"] else \
                "（容器之后被打开过，可能已取出）"
            return (f"{obj_type} 在 {causal['container'].split('|')[0]} 里"
                    f"（step {causal['step']} 放入{st}）")
        # scene graph: anchor of this type with an on/in parent
        for aid, a in anchors.items():
            if a.obj_type == obj_type:
                for rel in scene_graph.adj.get(aid, {}).values():
                    if rel["relation"] in ("on", "in"):
                        parent = next(
                            (x for x in scene_graph.adj[aid]
                             if scene_graph.adj[aid][x] is rel), None)
                        if parent:
                            return (f"{obj_type} 在 {parent.split('#')[0]} "
                                    f"{'上' if rel['relation'] == 'on' else '里'}")
                return f"{obj_type} 最后在锚点 {aid} 附近"
        return f"{obj_type} 从未见过"
