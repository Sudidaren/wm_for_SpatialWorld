"""Action outcome from two consecutive RGB frames (frame difference).

Rule (2026-09-15, user decision): a step succeeded when the frame changed.

    mse > 1.0  -> success
    mse < 1.0  -> failure

Applies to every action type: locomotion, rotation and interaction.
Measured on the real run corpus (AI2-THOR, 4,194 interaction steps): a failed
interaction never moved the frame by more than MSE 3.11; 91% of the successful
interactions that show no frame change have their target box hugging the image
border (the state change falls outside the view).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

MOVE_ACTIONS = frozenset({
    "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
    "MoveSmall", "MoveMedium", "MoveLarge",
})
ROTATE_ACTIONS = frozenset({
    "RotateLeft", "RotateRight", "LookUp", "LookDown", "Crouch", "Stand",
})
#: actions that change an object's state rather than the agent's pose.
#: The runtime's own policy never branches on this list (outcomes come from
#: the frame difference); it is the shared action taxonomy used by the
#: offline frame-census / evidence-collection scripts in the eval repo.
INTERACTION_ACTIONS = frozenset({
    "PickupObject", "PutObject", "DropHandObject", "ThrowObject",
    "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff",
    "SliceObject", "BreakObject", "CookObject", "DirtyObject", "CleanObject",
    "FillObjectWithLiquid", "EmptyLiquidFromObject", "UseUpObject",
    "PushObject", "PullObject",
})

#: frame MSE above this = the action did something = success
DEFAULT_MSE_THRESHOLD = 1.0


def frame_mse(a_path: Optional[str], b_path: Optional[str]) -> Optional[float]:
    """Mean squared error between two saved frames; ``None`` if unavailable."""
    if not a_path or not b_path or not os.path.isfile(a_path) or not os.path.isfile(b_path):
        return None
    try:
        import numpy as np
        from PIL import Image

        a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.float32)
        if a.shape != b.shape:
            return None
        return float(np.mean((a - b) ** 2))
    except Exception:
        return None


def estimate_success(
    action_name: Optional[str],
    prev_image_path: Optional[str],
    cur_image_path: Optional[str],
    *,
    threshold: float = DEFAULT_MSE_THRESHOLD,
) -> Tuple[Optional[bool], Optional[float]]:
    """``(success, mse)`` for every action type; ``(None, mse)`` if unknown.

    ``success = mse > threshold``.  ``Pass`` (the wrapper's no-op) and a
    missing action return ``None``.
    """
    mse = frame_mse(prev_image_path, cur_image_path)
    if not action_name or action_name == "Pass":
        return None, mse
    if mse is None:
        return None, None
    return (mse > threshold), mse
