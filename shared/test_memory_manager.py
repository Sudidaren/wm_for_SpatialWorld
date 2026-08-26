"""Memory manager: confidence lifecycle, eviction, episodic events."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from memory_manager import MemoryManager  # noqa: E402


def test():
    mm = MemoryManager(budget=10, tau=50)
    # add 15 records -> 5 must be evicted
    for i in range(15):
        mm.step_forward()
        mm.put("spatial", f"obj{i}",
               {"type": "Obj", "pos": [i, 1.0, 0]},
               confidence=0.8,
               importance=0.2 if i < 10 else 0.9)  # later ones important
    s = mm.stats()
    assert s["records"] <= 10, s
    assert s["evicted"] >= 5, s
    # important records survive
    assert any(k.startswith("obj1") for k in mm.records)
    # consistent re-observation raises confidence
    mm.put("spatial", "obj14", {"type": "Obj"}, confidence=0.8,
           importance=0.9)
    c1 = mm.get("obj14").confidence
    mm.put("spatial", "obj14", {"type": "Obj"}, confidence=0.8,
           importance=0.9)
    c2 = mm.get("obj14").confidence
    assert c2 > c1
    # conflict drops it
    mm.conflict("obj14")
    assert mm.get("obj14").confidence < c2
    # episodic events
    mm.step_forward()
    mm.add_event("opened Cabinet#3", importance=0.6)
    mm.add_event("saw Potato right-front", importance=0.4)
    txt = mm.recent_events_text()
    assert "opened Cabinet#3" in txt and "saw Potato" in txt
    # queries
    found = mm.query(kind="spatial", content_key="type",
                     content_value="Obj", topk=3)
    assert found
    print(s, "| conf lifecycle OK | events OK | query OK")
    print("OK")


if __name__ == "__main__":
    test()
