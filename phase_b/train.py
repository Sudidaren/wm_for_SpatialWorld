"""Train Phase B heads.

Usage:
  python phase_b/train.py --task perception --epochs 3 --batch 24 --limit 3000
  python phase_b/train.py --task depth     --epochs 3 --batch 24 --limit 3000
  python phase_b/train.py --task feasibility --epochs 5 --batch 64 --limit 8000

--limit 0 means full dataset.  Checkpoints go to --out/<task>_best.pt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dataset import (  # noqa: E402
    FeasibilityDataset,
    PerceptionDataset,
    collate_feasibility,
    collate_perception,
)
from model import (  # noqa: E402
    PerceptionModel,
    decode_dense,
    dense_targets,
    giou,
    hungarian_match,
)


def eval_dense_sample(model, index, device, n=60, obj_thr=0.4):
    """Quick detection P/R on held-out frames (for training-time monitoring)."""
    import dataset as ds_mod
    from dataset import load_rgb
    from model import decode_dense
    type2id = {t: i for i, t in enumerate(index["object_types"])}
    exclude = {type2id[t] for t in ("Floor", "Wall", "Ceiling", "Window")
               if t in type2id}
    frames = [f for f in index["frames"][::97]
              if f.get("rgb") and not f["rgb"].endswith("/")][:n]
    tp = fp = fn = 0
    model.eval()
    with torch.no_grad():
        for fr in frames:
            rgb = torch.from_numpy(load_rgb(fr["rgb"])).permute(2, 0, 1)
            rgb = rgb.unsqueeze(0).to(device)
            d = model.dense_detect(rgb)
            boxes, clss, scores = decode_dense(
                d["objectness"], d["offset"], d["size"], d["class_logits"],
                obj_thr=obj_thr, nms_thr=0.5)
            IMG = ds_mod.IMG_SIZE
            bx = boxes[0].cpu().numpy() * IMG
            preds = [(b[0] - b[2] / 2, b[1] - b[3] / 2,
                      b[0] + b[2] / 2, b[1] + b[3] / 2)
                     for b in bx if b[2] > 0.01]
            gts = []
            for v in fr.get("visible", []) or []:
                t = type2id.get(v["type"])
                if t is None or t in exclude or v.get("bbox") is None:
                    continue
                b = v["bbox"]
                sx, sy = IMG / 800.0, IMG / 600.0
                gts.append((b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy))
            used = set()
            for p in preds:
                best = max(((iou2(p, g), i) for i, g in enumerate(gts)
                            if i not in used), default=(0, -1))
                if best[1] >= 0 and best[0] > 0.5:
                    tp += 1
                    used.add(best[1])
                else:
                    fp += 1
            fn += len(gts) - len(used)
    model.train()
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def iou2(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0):
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1 - p)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return ((1 - pt) ** gamma * ce).mean()


def scale_invariant_depth(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
    d = torch.log(pred.clamp(min=0.05)) - torch.log(gt.clamp(min=0.05))
    d = d * mask
    n = mask.sum().clamp(min=1)
    si = ((d ** 2).sum() / n - (d.sum() / n) ** 2)
    l1 = (torch.abs(pred - gt) * mask).sum() / n
    return si + 0.5 * l1


def train_perception(model, loader, epochs, lr, out, device, log_every=50):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    best = 1e9
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot_loss = 0.0
        n_batches = 0
        for bi, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            boxes = batch["boxes"].to(device)
            classes = batch["classes"].to(device)
            num = batch["num_boxes"].to(device)
            out_d = model.perception(rgb)
            assign = hungarian_match(out_d["boxes"], out_d["objectness"],
                                     boxes, num)
            B, S, _ = out_d["boxes"].shape
            mask = assign >= 0
            # box loss on matched slots
            loss_box, loss_giou, loss_cls, loss_obj = 0.0, 0.0, 0.0, 0.0
            for b in range(B):
                idx = assign[b]
                gt_idx = idx[mask[b]]
                if len(gt_idx) > 0:
                    gb = boxes[b, gt_idx]
                    pb = out_d["boxes"][b][mask[b]]
                    loss_box = loss_box + F.l1_loss(pb, gb)
                    loss_giou = loss_giou + (1 - giou(pb.unsqueeze(0), gb.unsqueeze(0))).mean()
                    loss_cls = loss_cls + F.cross_entropy(
                        out_d["class_logits"][b][mask[b]], classes[b, gt_idx])
            loss_box = loss_box / B
            loss_giou = loss_giou / B
            loss_cls = loss_cls / B
            obj_t = mask.float()
            loss_obj = focal_loss(out_d["objectness"], obj_t)
            loss = loss_box + 2.0 * loss_giou + loss_cls + 0.5 * loss_obj
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            tot_loss += float(loss)
            n_batches += 1
            if (bi + 1) % log_every == 0:
                print(f"  ep{ep} b{bi+1}/{len(loader)} loss={loss.item():.3f} "
                      f"box={loss_box.item():.3f} giou={loss_giou.item():.3f} "
                      f"cls={loss_cls.item():.3f} obj={loss_obj.item():.3f}")
        avg = tot_loss / max(n_batches, 1)
        print(f"[perception] epoch {ep} avg_loss={avg:.3f} "
              f"time={time.time()-t0:.0f}s")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(out, "perception_best.pt"))


def train_depth(model, loader, epochs, lr, out, device, log_every=50):
    opt = torch.optim.AdamW(
        [p for p in model.depth_head.parameters()], lr=lr, weight_decay=1e-4)
    best = 1e9
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for bi, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            gt = batch["depth"].to(device)          # (B,1,56,56)
            pred = model.depth(rgb)                 # (B,1,64,64)
            pred = F.interpolate(pred, size=(56, 56), mode="bilinear",
                                 align_corners=False)
            mask = (gt > 0.1).float()
            loss = scale_invariant_depth(pred, gt, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.depth_head.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            if (bi + 1) % log_every == 0:
                print(f"  ep{ep} b{bi+1}/{len(loader)} loss={loss.item():.3f}")
        avg = tot / max(bi + 1, 1)
        print(f"[depth] epoch {ep} avg_loss={avg:.3f} time={time.time()-t0:.0f}s")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(out, "depth_best.pt"))


def train_feasibility(model, loader, epochs, lr, out, device, num_errors,
                      log_every=50):
    opt = torch.optim.AdamW(
        [p for p in model.feas_head.parameters()]
        + [p for p in model.action_emb.parameters()]
        + [p for p in model.arg_emb.parameters()],
        lr=lr, weight_decay=1e-4)
    best = 1e9
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for bi, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            act = batch["action"].to(device)
            arg = batch["arg_type"].to(device)
            succ = batch["success"].to(device)
            err = batch["error_class"].to(device)
            s_logit, e_logits = model.feasibility(rgb, act, arg)
            loss = F.binary_cross_entropy_with_logits(s_logit, succ)
            # ignore error-class loss when success (class = none)
            none_id = 0  # 'none' is first error class in index
            ce = F.cross_entropy(e_logits, err, reduction="none")
            loss = loss + 0.3 * ce.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.feas_head.parameters()], 5.0)
            opt.step()
            tot += float(loss)
            if (bi + 1) % log_every == 0:
                print(f"  ep{ep} b{bi+1}/{len(loader)} loss={loss.item():.3f}")
        avg = tot / max(bi + 1, 1)
        print(f"[feasibility] epoch {ep} avg_loss={avg:.3f} "
              f"time={time.time()-t0:.0f}s")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(out, "feasibility_best.pt"))


def train_dense(model, loader, epochs, lr, out, device, grid=16,
                log_every=100, amp=False, eval_every=0, index=None):
    params = ([p for p in model.dense_head.parameters()]
              + [p for p in model.dense_obj.parameters()]
              + [p for p in model.dense_off.parameters()]
              + [p for p in model.dense_size.parameters()]
              + [p for p in model.dense_cls.parameters()])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    best = 1e9
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.05)
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        for bi, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            boxes = batch["boxes"].to(device)
            classes = batch["classes"].to(device)
            num = batch["num_boxes"].to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                out_d = model.dense_detect(rgb)
                obj_t, off_t, size_t, cls_t, npos = dense_targets(
                    boxes, classes, num, grid, device)
                obj_loss = focal_loss(out_d["objectness"], obj_t)
                mask = (obj_t > 0.5).float()
                n = mask.sum().clamp(min=1)
                off_loss = ((torch.abs(out_d["offset"] - off_t) * mask).sum() / n)
                size_loss = ((torch.abs(out_d["size"] - size_t) * mask).sum() / n)
                cls_loss = torch.zeros((), device=device)
                if npos.sum() > 0:
                    cls_logits = out_d["class_logits"].permute(0, 2, 3, 1)
                    cls_loss = (F.cross_entropy(
                        cls_logits.reshape(-1, cls_logits.shape[-1]),
                        cls_t.reshape(-1), reduction="none") *
                        mask.reshape(-1)).sum() / n
                loss = obj_loss + 5.0 * off_loss + 5.0 * size_loss + 0.5 * cls_loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            tot += float(loss)
            if (bi + 1) % log_every == 0:
                print(f"  ep{ep} b{bi+1}/{len(loader)} loss={loss.item():.3f} "
                      f"obj={obj_loss.item():.3f} off={off_loss.item():.3f} "
                      f"size={size_loss.item():.3f} cls={cls_loss.item():.3f}")
        avg = tot / max(bi + 1, 1)
        print(f"[dense] epoch {ep} avg_loss={avg:.3f} time={time.time()-t0:.0f}s")
        if eval_every and (ep + 1) % eval_every == 0 and index is not None:
            p, r = eval_dense_sample(model, index, device)
            print(f"  [dense] epoch {ep} val detection: precision={p:.3f} "
                  f"recall={r:.3f}")
        scheduler.step()
        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(out, "dense_best.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["perception", "dense", "depth", "feasibility"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/home/sudidaren/lightwm_phases/checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=["small", "base"], default="small")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="eval detection P/R every N epochs (dense task)")
    ap.add_argument("--class-balanced", action="store_true",
                    help="class-balanced frame sampling (detection tasks)")
    ap.add_argument("--copy-paste", action="store_true",
                    help="copy-paste augmentation for rare objects (dense)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = args.device

    from shared.data_index import load_index
    index = load_index()
    num_types = len(index["object_types"])
    num_actions = len(index["actions"])
    num_errors = len(index["error_classes"])

    print(f"[{args.task}] types={num_types} actions={num_actions} "
          f"errors={num_errors} device={device}")
    import dataset as ds_mod
    ds_mod.IMG_SIZE = args.resolution
    dinov2_name = ("dinov2_vits14" if args.variant == "small"
                   else "dinov2_vitb14")
    model = PerceptionModel(
        num_types=num_types, num_actions=num_actions, num_errors=num_errors,
        device="cpu", img_size=args.resolution,
        head_width=args.width, dinov2_name=dinov2_name).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train}")

    if args.task == "perception":
        ds = PerceptionDataset(index, limit=args.limit, seed=args.seed,
                               class_balanced=args.class_balanced,
                               copy_paste=args.copy_paste)
        loader = _make_loader(ds, args, collate_perception)
        train_perception(model, loader, args.epochs, args.lr, args.out, device)
    elif args.task == "depth":
        ds = PerceptionDataset(index, limit=args.limit, seed=args.seed,
                               class_balanced=args.class_balanced,
                               require_depth=True)
        loader = _make_loader(ds, args, collate_perception)
        train_depth(model, loader, args.epochs, args.lr, args.out, device)
    elif args.task == "dense":
        ds = PerceptionDataset(index, limit=args.limit, seed=args.seed,
                               class_balanced=args.class_balanced,
                               copy_paste=args.copy_paste)
        loader = _make_loader(ds, args, collate_perception)
        train_dense(model, loader, args.epochs, args.lr, args.out, device,
                    grid=args.resolution // 14, amp=args.amp,
                    eval_every=args.eval_every, index=index)
    else:
        ds = FeasibilityDataset(index, limit=args.limit, seed=args.seed)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            num_workers=6, collate_fn=collate_feasibility,
                            pin_memory=True)
        train_feasibility(model, loader, args.epochs, args.lr, args.out, device,
                          num_errors)


def _make_loader(ds, args, collate):
    from torch.utils.data import WeightedRandomSampler
    if ds.weights is not None:
        sampler = WeightedRandomSampler(ds.weights, len(ds.weights),
                                        replacement=True)
        return DataLoader(ds, batch_size=args.batch, sampler=sampler,
                          num_workers=6, collate_fn=collate, pin_memory=True)
    return DataLoader(ds, batch_size=args.batch, shuffle=True,
                      num_workers=6, collate_fn=collate, pin_memory=True)


if __name__ == "__main__":
    main()
