"""Safety-constrained sequential hidden-location belief updates.

For candidate ``c`` the deployed posterior is

    log b_t(c) = prior(c) + sum_j log P(o_j | z=c).

Confirmed-empty candidates receive zero probability.  Learned observation
likelihoods are accumulated in a shadow posterior and are projected back to
the safe posterior whenever they violate the configured ranking invariant.
The default ``exact`` projection therefore cannot change any ranking metric;
it is intended for shadow-mode deployment until a residual passes validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RankedCandidate:
    receptacle_id: str
    receptacle_type: str
    probability: float
    logit: float
    prior_logit: float


class SafeSequentialBelief:
    """Persistent task-local posterior with a no-regression projection."""

    def __init__(self, target_type: str, room_type: str, *,
                 score_decimals: int = 5, protected_top_k: int = 3):
        self.target_type = target_type
        self.room_type = room_type
        self.score_decimals = score_decimals
        self.protected_top_k = protected_top_k
        self._prior: dict[str, float] = {}
        self._types: dict[str, str] = {}
        self._arrival: dict[str, int] = {}
        self._learned_log_likelihood: dict[str, float] = {}
        self._confirmed_absent: set[str] = set()
        self._observation_ids: set[str] = set()
        self._next_arrival = 0

    @property
    def confirmed_absent(self) -> frozenset[str]:
        return frozenset(self._confirmed_absent)

    def observe(self, candidates: Iterable[Mapping], *,
                observation_id: str | None = None,
                learned_log_likelihood: Mapping[str, float] | None = None) -> bool:
        """Merge newly perceived candidates and optional learned evidence.

        Candidate tracks persist when temporarily missed.  Reusing an
        ``observation_id`` is a no-op, preventing repeated frames from being
        counted as independent evidence.
        """
        if observation_id is not None:
            key = str(observation_id)
            if key in self._observation_ids:
                return False
            self._observation_ids.add(key)
        for item in candidates:
            rid = str(item["receptacle_id"])
            if not rid:
                continue
            if rid not in self._arrival:
                self._arrival[rid] = self._next_arrival
                self._next_arrival += 1
            self._types[rid] = str(item["receptacle_type"])
            self._prior[rid] = float(item["prior_logit"])
            self._learned_log_likelihood.setdefault(rid, 0.0)
        for rid, value in (learned_log_likelihood or {}).items():
            rid = str(rid)
            if rid in self._prior and math.isfinite(float(value)):
                self._learned_log_likelihood[rid] += float(value)
        return True

    def confirm_absent(self, receptacle_id: str) -> None:
        """Apply reliable negative evidence: P(empty | z=c)=0."""
        rid = str(receptacle_id)
        if rid in self._prior:
            self._confirmed_absent.add(rid)

    def _order(self, scores: Mapping[str, float]) -> list[str]:
        allowed = [rid for rid in self._prior
                   if rid not in self._confirmed_absent]
        return sorted(allowed, key=lambda rid: (
            -round(float(scores[rid]), self.score_decimals),
            self._arrival[rid], rid,
        ))

    def rank(self, *, learned_alpha: float = 0.0,
             projection: str = "exact", top_k: int | None = None
             ) -> tuple[list[RankedCandidate], dict]:
        """Rank the posterior and return projection diagnostics.

        ``exact`` preserves the complete safe ordering. ``top_k`` permits
        reordering only when the protected Top-k membership is unchanged.
        ``none`` exposes the learned posterior and is evaluation-only.
        """
        if projection not in {"exact", "top_k", "none"}:
            raise ValueError("projection must be exact, top_k, or none")
        safe_scores = dict(self._prior)
        shadow_scores = {
            rid: value + float(learned_alpha) *
            self._learned_log_likelihood.get(rid, 0.0)
            for rid, value in self._prior.items()
        }
        safe_order = self._order(safe_scores)
        shadow_order = self._order(shadow_scores)
        accepted = True
        if projection == "exact" and shadow_order != safe_order:
            accepted = False
        elif projection == "top_k":
            k = min(self.protected_top_k, len(safe_order))
            accepted = set(shadow_order[:k]) == set(safe_order[:k])
        deployed_scores = shadow_scores if accepted or projection == "none" \
            else safe_scores
        deployed_order = shadow_order if accepted or projection == "none" \
            else safe_order
        logits = [deployed_scores[rid] for rid in deployed_order]
        if logits:
            peak = max(logits)
            weights = [math.exp(value - peak) for value in logits]
            normalizer = sum(weights)
        else:
            weights, normalizer = [], 1.0
        ranked = [RankedCandidate(
            receptacle_id=rid,
            receptacle_type=self._types[rid],
            probability=weights[index] / normalizer,
            logit=deployed_scores[rid],
            prior_logit=self._prior[rid],
        ) for index, rid in enumerate(deployed_order)]
        if top_k is not None:
            ranked = ranked[:max(int(top_k), 0)]
        diagnostics = {
            "learned_alpha": float(learned_alpha),
            "projection": projection,
            "learned_update_accepted": bool(accepted or projection == "none"),
            "safe_order": safe_order,
            "shadow_order": shadow_order,
            "deployed_order": deployed_order,
            "confirmed_absent": sorted(self._confirmed_absent),
            "observation_count": len(self._observation_ids),
        }
        return ranked, diagnostics
