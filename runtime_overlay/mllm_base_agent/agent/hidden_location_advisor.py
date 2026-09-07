"""Tiny deterministic hidden-object advisor for the SpatialWorld runtime.

Inputs are perceived furniture tracks only.  The advisor never reads the
target object's simulator metadata or ground-truth position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from phase_c.hidden_world_belief.safe_belief import SafeSequentialBelief
from phase_c.hidden_world_belief.train_object_ranker_v5 import BilinearTypeRanker


_OPENABLE = {"Cabinet", "Drawer", "Fridge", "Microwave", "Safe", "Box"}
_CONTAINERS = _OPENABLE | {"Bowl", "Cup", "Mug", "Pan", "Pot", "SinkBasin"}


class _Ranker(nn.Module):
    def __init__(self, nt, nr, nc):
        super().__init__()
        self.target = nn.Embedding(nt, 32)
        self.room = nn.Embedding(nr, 8)
        self.receptacle = nn.Embedding(nc, 32)
        self.relation = nn.Embedding(2, 4)
        self.scorer = nn.Sequential(
            nn.Linear(77, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, target, room, receptacle, relation, numeric):
        n = len(receptacle)
        features = torch.cat((self.target(target).expand(n, -1),
                              self.room(room).expand(n, -1),
                              self.receptacle(receptacle),
                              self.relation(relation), numeric), dim=-1)
        return self.scorer(features).squeeze(-1)


class HiddenLocationAdvisor:
    def __init__(self, checkpoint: str | Path, score_decimals: int = 5):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.vocab = payload["vocabs"]
        self.architecture = payload.get("architecture", "legacy_candidate_mlp")
        if self.architecture == "bilinear_type_v5":
            self.model = BilinearTypeRanker(
                len(self.vocab["target"]), len(self.vocab["room"]),
                len(self.vocab["receptacle"]))
            self.model.load_state_dict(payload["model"])
        elif self.architecture == "legacy_candidate_mlp":
            self.model = _Ranker(
                len(self.vocab["target"]), len(self.vocab["room"]),
                len(self.vocab["receptacle"]))
            self.model.load_state_dict(payload["model"])
        else:
            raise ValueError(f"unsupported advisor checkpoint: {self.architecture}")
        self.model.eval()
        self.score_decimals = score_decimals

    def _prior_candidates(self, target_type: str, room_type: str,
                          furniture: Iterable[Dict]) -> List[Dict]:
        """Encode stable perceived tracks and attach their prior logits."""
        candidates, seen = [], set()
        for item in furniture:
            object_id = str(item.get("objectId") or item.get("id") or "")
            object_type = str(item.get("objectType") or item.get("type") or "")
            if not object_id or not object_type or object_id in seen:
                continue
            if int(item.get("seen") or 0) < 2:
                continue
            seen.add(object_id)
            candidates.append((object_id, object_type, item))
        if not candidates:
            return []
        vocab = self.vocab
        target = torch.tensor(vocab["target"].get(target_type, 0))
        room = torch.tensor(vocab["room"].get(room_type, 0))
        types = [kind for _, kind, _ in candidates]
        receptacle = torch.tensor([
            vocab["receptacle"].get(kind, 0) for kind in types])
        relation = torch.tensor([int(kind in _CONTAINERS) for kind in types])
        numeric = torch.tensor([[float(kind in _OPENABLE)] for kind in types])
        with torch.no_grad():
            if self.architecture == "bilinear_type_v5":
                logits = self.model(target[None], room[None])[0][receptacle]
            else:
                logits = self.model(target, room, receptacle, relation, numeric)
        return [{"receptacle_id": object_id, "receptacle_type": object_type,
                 "prior_logit": float(logits[index])}
                for index, (object_id, object_type, _) in enumerate(candidates)]

    def new_belief(self, target_type: str, room_type: str,
                   protected_top_k: int = 3) -> SafeSequentialBelief:
        return SafeSequentialBelief(
            target_type, room_type, score_decimals=self.score_decimals,
            protected_top_k=protected_top_k)

    def update_belief(self, belief: SafeSequentialBelief,
                      furniture: Iterable[Dict], *,
                      observation_id: str | None = None,
                      learned_log_likelihood: Dict[str, float] | None = None,
                      learned_alpha: float = 0.0,
                      projection: str = "exact", top_k: int = 3
                      ) -> tuple[List[Dict], dict]:
        belief.observe(
            self._prior_candidates(
                belief.target_type, belief.room_type, furniture),
            observation_id=observation_id,
            learned_log_likelihood=learned_log_likelihood,
        )
        ranked, diagnostics = belief.rank(
            learned_alpha=learned_alpha, projection=projection, top_k=top_k)
        return [{"receptacle_id": item.receptacle_id,
                 "receptacle_type": item.receptacle_type,
                 "probability": item.probability,
                 "utility": item.logit}
                for item in ranked], diagnostics

    def rank(self, target_type: str, room_type: str,
             furniture: Iterable[Dict], searched_absent: Iterable[str] = (),
             top_k: int = 3) -> List[Dict]:
        belief = self.new_belief(target_type, room_type, top_k)
        belief.observe(self._prior_candidates(target_type, room_type, furniture))
        for receptacle_id in searched_absent:
            belief.confirm_absent(str(receptacle_id))
        ranked, _ = belief.rank(top_k=top_k)
        return [{"receptacle_id": item.receptacle_id,
                 "receptacle_type": item.receptacle_type,
                 "probability": item.probability}
                for item in ranked]

    @staticmethod
    def render_ranked_hint(target_type: str, ranked: Iterable[Dict]) -> str:
        ranked = list(ranked)
        if not ranked:
            return ""
        choices = "、".join(
            f"{item['receptacle_type']}({item['receptacle_id']})"
            for item in ranked)
        return f"{target_type} 尚未找到；隐藏位置模型建议依次检查：{choices}。"

    def render_hint(self, target_type: str, room_type: str,
                    furniture: Iterable[Dict], searched_absent: Iterable[str] = (),
                    top_k: int = 3) -> str:
        ranked = self.rank(target_type, room_type, furniture, searched_absent, top_k)
        return self.render_ranked_hint(target_type, ranked)
