"""Unified memory manager for the LightWM world model.

Every piece of long-term knowledge is a RECORD with the same lifecycle:

    {kind, key, content, confidence, importance, last_access,
     created_step, source_step}

  - confidence : how sure we are (0..1); rises on consistent re-observation,
                 falls on conflict.
  - importance : task/type-level priority (objects > furniture > trivia).
  - freshness  : decays with wall-clock steps since last access.
  - score      = confidence * importance * exp(-age / TAU)
  - eviction   : fixed budget; lowest-score records are evicted first.

Kinds wired in:
  spatial   -> 3D object anchors (SpatialMemory)
  place     -> loop-closure place keyframes (LoopClosure)
  episodic  -> event log ("opened Cabinet#3", "saw Potato right-front")
  semantic  -> common-sense priors / learned layout stats
  causal    -> container->contents / object states
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional


class MemoryRecord:
    __slots__ = ("kind", "key", "content", "confidence", "importance",
                 "last_access", "created_step", "source_step", "conflicts")

    def __init__(self, kind: str, key: str, content: Dict[str, Any],
                 confidence: float, importance: float, step: int,
                 source_step: Optional[int] = None):
        self.kind = kind
        self.key = key
        self.content = content
        self.confidence = float(confidence)
        self.importance = float(importance)
        self.last_access = step
        self.created_step = step
        self.source_step = source_step if source_step is not None else step
        self.conflicts = 0

    def score(self, step: int, tau: float = 200.0) -> float:
        age = max(0, step - self.last_access)
        return self.confidence * self.importance * math.exp(-age / tau)


class MemoryManager:
    def __init__(self, budget: int = 2000, tau: float = 200.0):
        self.budget = budget
        self.tau = tau
        self.records: Dict[str, MemoryRecord] = {}
        self.step = 0
        self.evicted: List[str] = []
        # indexes: kind -> set(keys); content fields for fast query
        self._by_kind: Dict[str, set] = {}
        self._events: List[Dict[str, Any]] = []

    # -- lifecycle ----------------------------------------------------------
    def step_forward(self, n: int = 1):
        self.step += n

    def put(self, kind: str, key: str, content: Dict[str, Any],
            confidence: float = 0.8, importance: float = 0.5,
            step: Optional[int] = None) -> MemoryRecord:
        step = self.step if step is None else step
        rec = self.records.get(key)
        if rec is None:
            rec = MemoryRecord(kind, key, content, confidence, importance,
                               step)
            self.records[key] = rec
            self._by_kind.setdefault(kind, set()).add(key)
        else:
            # consistent re-observation: confidence moves up, content merged
            if rec.confidence >= 0:
                rec.confidence = min(0.95,
                                     rec.confidence + 0.1 * confidence)
            rec.importance = max(rec.importance, importance)
            rec.last_access = step
            rec.content.update(content)
        self._maybe_evict(step)
        return rec

    def touch(self, key: str, step: Optional[int] = None):
        rec = self.records.get(key)
        if rec:
            rec.last_access = self.step if step is None else step

    def conflict(self, key: str, step: Optional[int] = None) -> MemoryRecord:
        """Record a contradictory observation: confidence drops sharply."""
        rec = self.records.get(key)
        if rec is None:
            raise KeyError(key)
        step = self.step if step is None else step
        rec.conflicts += 1
        rec.confidence = max(0.1, rec.confidence * 0.5)
        rec.last_access = step
        return rec

    def _maybe_evict(self, step: int):
        if len(self.records) <= self.budget:
            return
        scored = [(r.score(step, self.tau), k, r) for k, r in self.records.items()]
        scored.sort()
        n_remove = len(scored) - self.budget
        for _, k, r in scored[:n_remove]:
            self.records.pop(k, None)
            self._by_kind.get(r.kind, set()).discard(k)
            self.evicted.append(k)

    # -- queries ------------------------------------------------------------
    def get(self, key: str, touch: bool = True) -> Optional[MemoryRecord]:
        rec = self.records.get(key)
        if rec and touch:
            rec.last_access = self.step
        return rec

    def query(self, kind: Optional[str] = None,
              content_key: Optional[str] = None,
              content_value=None, topk: int = 20) -> List[MemoryRecord]:
        out = []
        keys = self._by_kind.get(kind, self.records.keys()) if kind else \
            self.records.keys()
        for k in keys:
            rec = self.records.get(k)
            if rec is None:
                continue
            if content_key is not None and \
                    rec.content.get(content_key) != content_value:
                continue
            out.append(rec)
        out.sort(key=lambda r: -r.score(self.step, self.tau))
        return out[:topk]

    def stats(self) -> Dict[str, Any]:
        by_kind = {k: len(v) for k, v in self._by_kind.items()}
        return {"records": len(self.records), "by_kind": by_kind,
                "evicted": len(self.evicted)}

    # -- episodic events ----------------------------------------------------
    def add_event(self, text: str, importance: float = 0.3,
                  meta: Optional[Dict[str, Any]] = None,
                  step: Optional[int] = None) -> None:
        step = self.step if step is None else step
        self._events.append({"step": step, "text": text,
                             "importance": importance, "meta": meta or {}})

    def events(self, since_step: int = 0, topk: int = 50) -> List[Dict]:
        evs = [e for e in self._events if e["step"] >= since_step]
        return evs[-topk:]

    def recent_events_text(self, since_step: int = 0, topk: int = 5) -> str:
        return "\n".join(f"step {e['step']}: {e['text']}"
                         for e in self.events(since_step, topk))
