from pathlib import Path

from mllm_base_agent.agent.hidden_location_advisor import HiddenLocationAdvisor


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "phase_c/hidden_world_belief/checkpoints/spatialworld_v2_finetune_mixed_balanced.pt"


def test_advisor_is_deterministic_and_filters_search_history():
    advisor = HiddenLocationAdvisor(CHECKPOINT)
    furniture = [
        {"objectId": "Shelf|a", "objectType": "Shelf", "seen": 2},
        {"objectId": "Shelf|b", "objectType": "Shelf", "seen": 3},
        {"objectId": "Desk|c", "objectType": "Desk", "seen": 2},
        {"objectId": "flash", "objectType": "Sofa", "seen": 1},
    ]
    first = advisor.rank("Book", "living_room", furniture)
    second = advisor.rank("Book", "living_room", furniture)
    assert first == second
    assert all(x["receptacle_id"] != "flash" for x in first)
    searched = advisor.rank("Book", "living_room", furniture,
                            searched_absent=[first[0]["receptacle_id"]])
    assert first[0]["receptacle_id"] not in [x["receptacle_id"] for x in searched]


def test_hint_contains_only_ranked_perceived_candidates():
    advisor = HiddenLocationAdvisor(CHECKPOINT)
    furniture = [{"objectId": "Desk|c", "objectType": "Desk", "seen": 2}]
    hint = advisor.render_hint("CellPhone", "living_room", furniture)
    assert "CellPhone" in hint and "Desk|c" in hint


def test_persistent_belief_updates_and_projects_unsafe_learned_evidence():
    advisor = HiddenLocationAdvisor(CHECKPOINT)
    furniture = [
        {"objectId": "Cabinet|a", "objectType": "Cabinet", "seen": 2},
        {"objectId": "Fridge|b", "objectType": "Fridge", "seen": 2},
        {"objectId": "Drawer|c", "objectType": "Drawer", "seen": 2},
    ]
    belief = advisor.new_belief("Egg", "kitchen")
    first, _ = advisor.update_belief(
        belief, furniture, observation_id="frame-1", top_k=3)
    first_ids = [item["receptacle_id"] for item in first]

    # An arbitrary learned likelihood is computed and retained in shadow mode,
    # but exact projection prevents it from changing the deployed order.
    learned = {first_ids[-1]: 100.0}
    second, diag = advisor.update_belief(
        belief, [], observation_id="frame-2",
        learned_log_likelihood=learned, learned_alpha=1.0,
        projection="exact", top_k=3)
    assert [item["receptacle_id"] for item in second] == first_ids
    assert not diag["learned_update_accepted"]
    assert diag["shadow_order"][0] == first_ids[-1]

    # Reliable negative evidence removes only the searched candidate; a true
    # remaining candidate therefore cannot move down the list.
    belief.confirm_absent(first_ids[0])
    third, _ = advisor.update_belief(
        belief, [], observation_id="frame-3", top_k=3)
    assert first_ids[0] not in [item["receptacle_id"] for item in third]
    assert first_ids[1] == third[0]["receptacle_id"]
