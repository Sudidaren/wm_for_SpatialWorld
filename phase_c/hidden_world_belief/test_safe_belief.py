from phase_c.hidden_world_belief.safe_belief import SafeSequentialBelief


def _belief():
    belief = SafeSequentialBelief("Egg", "kitchen")
    belief.observe([
        {"receptacle_id": "cabinet-a", "receptacle_type": "Cabinet", "prior_logit": 3.0},
        {"receptacle_id": "fridge-a", "receptacle_type": "Fridge", "prior_logit": 2.0},
        {"receptacle_id": "drawer-a", "receptacle_type": "Drawer", "prior_logit": 1.0},
    ], observation_id="frame-1")
    return belief


def test_alpha_zero_is_bit_exact_safe_order():
    belief = _belief()
    belief.observe([], observation_id="frame-2", learned_log_likelihood={
        "drawer-a": 100.0, "cabinet-a": -100.0,
    })
    ranked, diag = belief.rank(learned_alpha=0.0)
    assert [item.receptacle_id for item in ranked] == diag["safe_order"]
    assert diag["learned_update_accepted"]


def test_exact_projection_rejects_ranking_regression_but_keeps_shadow():
    belief = _belief()
    belief.observe([], observation_id="frame-2",
                   learned_log_likelihood={"drawer-a": 10.0})
    ranked, diag = belief.rank(learned_alpha=1.0, projection="exact")
    assert [item.receptacle_id for item in ranked] == [
        "cabinet-a", "fridge-a", "drawer-a"]
    assert diag["shadow_order"][0] == "drawer-a"
    assert not diag["learned_update_accepted"]


def test_confirmed_negative_cannot_worsen_true_candidate_rank():
    belief = _belief()
    before, _ = belief.rank()
    true_id = "drawer-a"
    old_rank = [item.receptacle_id for item in before].index(true_id)
    belief.confirm_absent("cabinet-a")
    after, _ = belief.rank()
    new_rank = [item.receptacle_id for item in after].index(true_id)
    assert new_rank <= old_rank
    assert "cabinet-a" not in [item.receptacle_id for item in after]


def test_candidates_persist_across_temporary_misses_and_frames_are_deduped():
    belief = _belief()
    assert not belief.observe([], observation_id="frame-1",
                              learned_log_likelihood={"drawer-a": 10.0})
    ranked, diag = belief.rank()
    assert len(ranked) == 3
    assert diag["observation_count"] == 1
