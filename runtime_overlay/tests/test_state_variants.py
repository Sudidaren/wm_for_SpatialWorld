#!/usr/bin/env python
"""State variants: the environment renames an object, the WM must follow.

The environment turns ``Lettuce`` into ``LettuceSliced`` when it is sliced,
and the agent then has to act on the new name, while the perception head keeps
labelling the pixels ``Lettuce``.  Before this change that disagreement cost
three things at once -- the hand state, the frame-diff verdict and the
relevant-object match -- and on ``ai2thor03001`` ("slice all the vegetables")
it meant the whole episode ran with an empty memory block.

The rename rules are parsed from the prompt that both arms receive, so nothing
here is looked up in an external dictionary.

    python tests/test_state_variants.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mllm_base_agent.agent import state_variants as sv          # noqa: E402
from mllm_base_agent.agent.target_priority import TargetHinter   # noqa: E402
from mllm_base_agent.agent.world_model import WorldModel         # noqa: E402


def _meta(objects, held=None, x=0.0, z=0.0, yaw=0.0):
    return {
        "agent": {"position": {"x": x, "y": 0.0, "z": z},
                  "rotation": {"x": 0.0, "y": yaw, "z": 0.0}},
        "objects": objects,
        "inventoryObjects": ([{"objectId": "h", "objectType": held,
                               "objectTypeDisplay": held}] if held else []),
    }


def _obj(tp, dist=1.0, visible=True, step=1, sigma=0.2, x=0.0, z=1.0,
         display=None, state=None):
    pos = {"x": x, "y": 1.0, "z": z}
    return {
        "objectId": f"det|{tp}|0", "objectType": tp,
        "objectTypeDisplay": display or tp,
        "position": pos, "visible": visible, "distance": dist,
        "edge_px": 200.0, "box": [0, 0, 10, 10],
        "last_seen_step": step, "seen_count": 1, "sigma": sigma,
        "state": state or {}, "contents": [],
    }


# --------------------------------------------------------------------------
# the rules come from the prompt
# --------------------------------------------------------------------------
def test_rules_are_read_out_of_the_shipped_prompt():
    suffixes, explicit = sv.rules()
    assert "Sliced" in suffixes, suffixes
    # the prompt's one-off: BreakObject(Egg) produces EggCracked
    assert explicit.get("Egg") == "EggCracked", explicit
    prompt = (REPO / "mllm_base_agent" / "prompts" / "ai2thor.py").read_text(
        encoding="utf-8")
    assert "produces **XSliced**" in prompt


def test_variant_names_map_back_to_the_base_type():
    for variant, base in (("LettuceSliced", "Lettuce"),
                          ("BreadSliced", "Bread"),
                          ("TomatoSliced", "Tomato"),
                          ("EggCracked", "Egg"),
                          ("PotatoSliced", "Potato")):
        assert sv.base_of(variant) == base, variant
        assert sv.is_variant(variant)
    for plain in ("Lettuce", "Fridge", "CounterTop", "SinkBasin"):
        assert sv.base_of(plain) == plain
        assert not sv.is_variant(plain)


def test_state_word_and_derived_name():
    assert sv.state_word({"isSliced": True}) == "已切"
    assert sv.state_word({"isOpen": True, "isDirty": False}) == "已打开、已洗干净"
    assert sv.state_word({}) == ""
    assert sv.variant_name("Lettuce", {"isSliced": True}) == "LettuceSliced"
    # a state the prompt does not rename must not invent a name
    assert sv.variant_name("Potato", {"isCooked": True}) == "Potato"


# --------------------------------------------------------------------------
# the world model
# --------------------------------------------------------------------------
def _wm_with_lettuce() -> WorldModel:
    wm = WorldModel(fov=60.0, width=800, height=600)
    wm._slots["det|Lettuce|0"] = {
        "type": "Lettuce", "pos": [0.0, 1.0, 0.6], "last_seen": 3,
        "seen": 2, "sigma": 0.2, "edge_px": 300.0, "box": [0, 0, 10, 10],
    }
    return wm


def test_pickup_of_the_variant_is_attributed_to_the_base_slot():
    wm = _wm_with_lettuce()
    # the agent says LettuceSliced, the slot is Lettuce: same object
    wm._update_action_log("PickupObject", "LettuceSliced",
                          visible_ids={"det|Lettuce|0"},
                          agent_pos={"x": 0.0, "z": 0.0}, action_ok=True)
    assert wm._holding == "det|Lettuce|0", wm._holding
    md = wm._to_metadata({"x": 0.0, "y": 0.0, "z": 0.0},
                         {"x": 0.0, "y": 0.0, "z": 0.0},
                         {"det|Lettuce|0"})
    held = md["inventoryObjects"][0]
    # matching still uses the base name, the hint uses the agent's own word
    assert held["objectType"] == "Lettuce"
    assert held["objectTypeDisplay"] == "LettuceSliced", held


def test_outcome_can_be_judged_through_a_variant_name():
    wm = _wm_with_lettuce()
    assert wm._resolve_outcome("PickupObject", "LettuceSliced", True) is True
    assert wm._resolve_outcome("PickupObject", "BananaSliced", True) is None


def test_slicing_marks_the_state_and_renames_the_readout():
    wm = _wm_with_lettuce()
    wm._update_action_log("SliceObject", "Lettuce",
                          visible_ids={"det|Lettuce|0"},
                          agent_pos={"x": 0.0, "z": 0.0}, action_ok=True)
    md = wm._to_metadata({"x": 0.0, "y": 0.0, "z": 0.0},
                         {"x": 0.0, "y": 0.0, "z": 0.0},
                         {"det|Lettuce|0"})
    obj = md["objects"][0]
    assert obj["state"].get("isSliced") is True
    assert obj["objectType"] == "Lettuce"
    assert obj["objectTypeDisplay"] == "LettuceSliced", obj


def test_variant_survives_a_checkpoint_round_trip(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "wm.json")
        wm = _wm_with_lettuce()
        wm._update_action_log("PickupObject", "LettuceSliced",
                              visible_ids={"det|Lettuce|0"},
                              agent_pos={"x": 0.0, "z": 0.0}, action_ok=True)
        wm.save(path)
        back = WorldModel.load(path)
        assert back._variants.get("Lettuce") == "LettuceSliced"


def test_a_name_we_cannot_place_creates_no_variant():
    wm = _wm_with_lettuce()
    wm._update_action_log("PickupObject", "BananaSliced",
                          visible_ids={"det|Lettuce|0"},
                          agent_pos={"x": 0.0, "z": 0.0}, action_ok=True)
    assert wm._variants == {}, wm._variants


# --------------------------------------------------------------------------
# the hint
# --------------------------------------------------------------------------
def _with_action_channel(fn):
    """The action channel is an ablation: it is off unless switched on."""
    os.environ["LIGHTWM_TARGET_FROM_ACTION"] = "1"
    try:
        return fn()
    finally:
        del os.environ["LIGHTWM_TARGET_FROM_ACTION"]


def test_action_targets_join_the_relevant_set():
    """'slice all the vegetables' names no object; the action stream does."""
    def run():
        hinter = TargetHinter(
            task_description="slice all the vegetables in front of me")
        acts = {"Lettuce": (1, "SliceObject", True)}
        objects = [_obj("Lettuce", 1.2, visible=False, step=3),
                   _obj("Bowl", 2.0, visible=False)]
        return hinter.update(_meta(objects), acts, "", 4)

    block = _with_action_channel(run)
    assert "Lettuce" in block, block
    assert "Bowl" not in block, block


def test_a_variant_action_target_reaches_the_base_slot():
    def run():
        hinter = TargetHinter(
            task_description="I want to slice all the vegetables.")
        acts = {"LettuceSliced": (2, "PickupObject", True)}
        objects = [_obj("Lettuce", 1.0, visible=False, step=3,
                        display="LettuceSliced", state={"isSliced": True})]
        return hinter.update(_meta(objects), acts, "LettuceSliced", 3)

    block = _with_action_channel(run)
    assert "LettuceSliced" in block, block
    assert "已切" in block, block


def test_put_destination_is_not_promoted():
    def run():
        hinter = TargetHinter(task_description="put the apple in the fridge")
        acts = {"CounterTop": (3, "PutObject", True)}
        objects = [_obj("CounterTop", 1.0, visible=False, step=3)]
        return hinter.update(_meta(objects), acts, "", 4)

    block = _with_action_channel(run)
    assert "CounterTop" not in block, block


def test_action_channel_is_off_unless_asked_for():
    """The released configuration must not promote an object just because the
    agent touched it: it knows what it just did, and a chance interaction
    would otherwise be re-reported (and walked back to) for the whole episode."""
    hinter = TargetHinter(task_description="slice all the vegetables")
    block = hinter.update(_meta([_obj("Lettuce", 1.0, visible=False)]),
                          {"Lettuce": (1, "SliceObject", True)}, "", 2)
    assert "Lettuce" not in block, block
    assert hinter.from_action is False


def test_the_ai2thor03001_case_is_no_longer_silent():
    """End to end on the task that lost with an empty memory block."""
    def run():
        hinter = TargetHinter(
            task_description="I'm hungry, so I want to slice all the "
                             "vegetables in front of me.")
        acts = {"Lettuce": (1, "SliceObject", True)}
        return hinter.update(
            _meta([_obj("Lettuce", 0.9, visible=False, step=1,
                        display="LettuceSliced", state={"isSliced": True}),
                   _obj("Tomato", 0.9, visible=True, step=1)]),
            acts, "", 2)

    block = _with_action_channel(run)
    assert block.strip(), "the block is still empty"
    assert "LettuceSliced" in block, block
    # 视野内那一行只列任务相关物体：Tomato 被检测到了但不是相关物体。
    assert "Tomato" not in block, block


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"ok   {name}")
        except AssertionError as exc:
            fails += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:                        # noqa: BLE001
            fails += 1
            print(f"ERR  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'FAILED' if fails else 'passed'}")
    raise SystemExit(1 if fails else 0)
