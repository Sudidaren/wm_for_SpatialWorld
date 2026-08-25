"""Evaluate the feasibility head: action success prediction + error class."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dataset import FeasibilityDataset, collate_feasibility, load_rgb  # noqa
from model import PerceptionModel  # noqa: E402


def main(ckpt: str, n: int = 800):
    from shared.data_index import load_index
    index = load_index()
    ds = FeasibilityDataset(index, use_fd=True)
    rng = np.random.RandomState(1)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    model = PerceptionModel(num_types=len(index["object_types"]),
                            num_actions=len(index["actions"]),
                            num_errors=len(index["error_classes"]),
                            device="cpu").cuda()
    state = torch.load(ckpt, map_location="cuda")
    cur = model.state_dict()
    filt = {k: v for k, v in state.items()
            if k in cur and cur[k].shape == v.shape}
    model.load_state_dict(filt, strict=False)
    model.eval()
    ok_succ = ok_err = tot = 0
    err_ok = succ_ok = 0
    with torch.no_grad():
        for i in idx:
            b = ds[i]
            rgb = b["rgb"].unsqueeze(0).cuda()
            act = b["action"].unsqueeze(0).cuda()
            arg = b["arg_type"].unsqueeze(0).cuda()
            s_logit, e_logits = model.feasibility(rgb, act, arg)
            p_succ = torch.sigmoid(s_logit).item() > 0.5
            p_err = int(e_logits.argmax(-1).item())
            tot += 1
            ok_succ += p_succ == bool(b["success"].item())
            if bool(b["success"].item()):
                succ_ok += p_err == 0
            else:
                err_ok += p_err == int(b["error_class"].item())
            ok_err += p_err == int(b["error_class"].item())
    print(f"n={tot}")
    print(f"success prediction acc: {ok_succ/tot:.3f}")
    print(f"error-class acc (all):  {ok_err/tot:.3f}")
    if tot - succ_ok > 0:
        print(f"error-class acc (on failures): "
              f"{err_ok / (tot - succ_ok):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/home/sudidaren/lightwm_phases/checkpoints/feasibility_best.pt")
    args = ap.parse_args()
    main(args.ckpt)
