"""Low-variance bilinear type ranker with candidate hard-negative training."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler



def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_vocab(rows: list[dict], field: str,
               candidate_field: str | None = None,
               extra_values=()) -> dict:
    values = set(extra_values) - {"<unk>"}
    for row in rows:
        if candidate_field:
            values.update(item[candidate_field] for item in row["candidates"])
        else:
            values.add(row[field])
    return {"<unk>": 0,
            **{value: index + 1
               for index, value in enumerate(sorted(values))}}


def initialize_from_checkpoint(model: nn.Module, vocab: dict, payload: dict) -> None:
    """Copy v5 weights by token while allowing an expanded vocabulary."""
    if payload.get("architecture") != "bilinear_type_v5":
        raise ValueError("initial checkpoint is not bilinear_type_v5")
    old_vocab = payload["vocabs"]
    old_model = BilinearTypeRanker(
        len(old_vocab["target"]), len(old_vocab["room"]),
        len(old_vocab["receptacle"]))
    old_model.load_state_dict(payload["model"])
    with torch.no_grad():
        for name in ("target", "room", "receptacle"):
            old_weight = getattr(old_model, name).weight
            new_weight = getattr(model, name).weight
            for token, new_index in vocab[name].items():
                old_index = old_vocab[name].get(token)
                if old_index is not None:
                    new_weight[new_index].copy_(old_weight[old_index])
        for token, new_index in vocab["receptacle"].items():
            old_index = old_vocab["receptacle"].get(token)
            if old_index is not None:
                model.bias[new_index].copy_(old_model.bias[old_index])
        model.context.load_state_dict(old_model.context.state_dict())
        model.log_scale.copy_(old_model.log_scale)


class Rows(Dataset):
    def __init__(self, rows: list[dict], vocab: dict):
        self.rows, self.vocab = rows, vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        types = list(dict.fromkeys(item["receptacle_type"] for item in row["candidates"]))
        parent = self.vocab["receptacle"].get(row["parent_type"], 0)
        return {
            "target": self.vocab["target"].get(row["target_type"], 0),
            "room": self.vocab["room"].get(row["room_type"], 0),
            "candidate_types": [self.vocab["receptacle"].get(x, 0) for x in types],
            "parent": parent,
            "label": next(i for i, item in enumerate(row["candidates"]) if item["label"]),
            "instance_types": [self.vocab["receptacle"].get(item["receptacle_type"], 0)
                               for item in row["candidates"]],
        }


def collate(items: list[dict]) -> dict:
    return {key: [item[key] for item in items] for key in items[0]}


class BilinearTypeRanker(nn.Module):
    def __init__(self, n_target: int, n_room: int, n_receptacle: int):
        super().__init__()
        self.target = nn.Embedding(n_target, 48)
        self.room = nn.Embedding(n_room, 16)
        self.context = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 48),
        )
        self.receptacle = nn.Embedding(n_receptacle, 48)
        self.bias = nn.Parameter(torch.zeros(n_receptacle))
        self.log_scale = nn.Parameter(torch.tensor(math.log(4.0)))

    def forward(self, target: torch.Tensor, room: torch.Tensor) -> torch.Tensor:
        context = F.normalize(self.context(torch.cat([
            self.target(target), self.room(room)
        ], dim=-1)), dim=-1)
        receptacle = F.normalize(self.receptacle.weight, dim=-1)
        return context @ receptacle.T * self.log_scale.exp().clamp(max=20) + self.bias


def move_ids(values: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long, device=device)


def metrics(model, rows: list[dict], vocab: dict, device: torch.device) -> dict:
    model.eval()
    total = Counter()
    with torch.no_grad():
        for row in rows:
            logits = model(
                move_ids([vocab["target"].get(row["target_type"], 0)], device),
                move_ids([vocab["room"].get(row["room_type"], 0)], device),
            )[0]
            candidate_ids = [vocab["receptacle"].get(x["receptacle_type"], 0)
                             for x in row["candidates"]]
            scores = logits[move_ids(candidate_ids, device)]
            label = next(i for i, item in enumerate(row["candidates"]) if item["label"])
            order = torch.argsort(scores, descending=True, stable=True).tolist()
            rank = order.index(label) + 1
            total["instance_hit1"] += rank == 1
            total["instance_hit3"] += rank <= 3
            total["instance_mrr"] += 1 / rank
            unique = list(dict.fromkeys(candidate_ids))
            type_order = sorted(unique, key=lambda x: float(logits[x]), reverse=True)
            type_rank = type_order.index(candidate_ids[label]) + 1
            total["type_hit1"] += type_rank == 1
            total["type_hit3"] += type_rank <= 3
            total["type_mrr"] += 1 / type_rank
    return {key: total[key] / len(rows) for key in (
        "instance_hit1", "instance_hit3", "instance_mrr",
        "type_hit1", "type_hit3", "type_mrr",
    )}


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "data/object_location_curated")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument(
        "--spatial-fraction", type=float, default=None,
        help="weighted-sampling mass assigned to spatialworld_* train rows")
    parser.add_argument("--target-balance-strength", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--gate-checkpoint", type=Path, default=None,
        help="require validation no-regression versus this frozen v5 checkpoint")
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train = read_jsonl(args.data / "train.jsonl")
    val = read_jsonl(args.data / "val.jsonl")
    # A validation-only training run must not even read the test file.
    test = [] if args.skip_test else read_jsonl(args.data / "test.jsonl")
    initial = (torch.load(args.init_checkpoint, map_location="cpu",
                          weights_only=False)
               if args.init_checkpoint else None)
    initial_vocabs = initial["vocabs"] if initial else {}
    vocab = {
        "target": make_vocab(train, "target_type", extra_values=
                             initial_vocabs.get("target", ())),
        "room": make_vocab(train, "room_type", extra_values=
                           initial_vocabs.get("room", ())),
        "receptacle": make_vocab(train, "", "receptacle_type", extra_values=
                                 initial_vocabs.get("receptacle", ())),
    }
    sampler = None
    if args.spatial_fraction is not None:
        if not 0.0 < args.spatial_fraction < 1.0:
            raise ValueError("--spatial-fraction must be strictly between 0 and 1")
        source = ["spatial" if row.get("source", "").startswith("spatialworld_")
                  else "procthor" for row in train]
        source_mass = {"spatial": args.spatial_fraction,
                       "procthor": 1.0 - args.spatial_fraction}
        group_count = Counter((kind, row["target_type"])
                              for kind, row in zip(source, train))
        raw = [group_count[(kind, row["target_type"])] **
               (-args.target_balance_strength)
               for kind, row in zip(source, train)]
        normalizer = {kind: sum(value for value, item_kind in zip(raw, source)
                                if item_kind == kind)
                      for kind in source_mass}
        weights = [source_mass[kind] * value / normalizer[kind]
                   for value, kind in zip(raw, source)]
        sampler = WeightedRandomSampler(weights, len(train), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
    elif args.balanced:
        source_count = Counter(row.get("source", "base") for row in train)
        target_count = Counter(row["target_type"] for row in train)
        weights = [1 / source_count[row.get("source", "base")] /
                   math.sqrt(target_count[row["target_type"]]) for row in train]
        sampler = WeightedRandomSampler(weights, len(train), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(Rows(train, vocab), batch_size=args.batch,
                        shuffle=sampler is None, sampler=sampler, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BilinearTypeRanker(len(vocab["target"]), len(vocab["room"]),
                               len(vocab["receptacle"])).to(device)
    if initial:
        initialize_from_checkpoint(model, vocab, initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    gate_metrics = None
    if args.gate_checkpoint:
        gate_payload = torch.load(args.gate_checkpoint, map_location=device,
                                  weights_only=False)
        gate_vocab = gate_payload["vocabs"]
        gate_model = BilinearTypeRanker(
            len(gate_vocab["target"]), len(gate_vocab["room"]),
            len(gate_vocab["receptacle"])).to(device)
        gate_model.load_state_dict(gate_payload["model"])
        gate_metrics = metrics(gate_model, val, gate_vocab, device)
        print("gate_val", gate_metrics, flush=True)
    best, stale = None, 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for batch in loader:
            target = move_ids(batch["target"], device)
            room = move_ids(batch["room"], device)
            parent = move_ids(batch["parent"], device)
            logits = model(target, room)
            loss = F.cross_entropy(logits, parent, label_smoothing=0.03)
            hard = []
            for i, candidates in enumerate(batch["candidate_types"]):
                ids = move_ids(candidates, device)
                negatives = ids[ids != parent[i]]
                if len(negatives):
                    hard.append(F.softplus(logits[i, negatives].max() - logits[i, parent[i]]))
            if hard:
                loss = loss + args.hard_weight * torch.stack(hard).mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            running += float(loss.detach()) * len(batch["target"])
        result = metrics(model, val, vocab, device)
        if gate_metrics is None:
            key = (result["type_mrr"] + 0.25 * result["instance_mrr"],)
            safe = True
        else:
            safe = all(result[name] >= gate_metrics[name] - 1e-12 for name in (
                "instance_hit1", "instance_hit3", "instance_mrr",
                "type_hit1", "type_hit3", "type_mrr"))
            key = (result["instance_hit1"], result["instance_mrr"],
                   result["instance_hit3"], result["type_hit1"],
                   result["type_mrr"])
        if safe and (best is None or key > best):
            best, stale = key, 0
            torch.save({"model": model.state_dict(), "vocabs": vocab,
                        "val_metrics": result, "epoch": epoch,
                        "architecture": "bilinear_type_v5",
                        "initialized_from": (str(args.init_checkpoint)
                                             if args.init_checkpoint else None),
                        "spatial_fraction": args.spatial_fraction,
                        "target_balance_strength": args.target_balance_strength,
                        "gate_checkpoint": (str(args.gate_checkpoint)
                                            if args.gate_checkpoint else None),
                        "gate_metrics": gate_metrics,
                        "validation_gate_passed": safe,
                        "test_data_read": not args.skip_test}, args.out)
        else:
            stale += 1
        print(f"epoch={epoch:03d} loss={running/len(train):.4f} "
              f"val={result} safe={safe}", flush=True)
        if stale >= args.patience:
            break
    if best is None:
        raise RuntimeError("no validation checkpoint passed the safety gate")
    payload = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    print("best_epoch", payload["epoch"])
    print("best_val", metrics(model, val, vocab, device))
    if not args.skip_test:
        print("test", metrics(model, test, vocab, device))


if __name__ == "__main__":
    main()
