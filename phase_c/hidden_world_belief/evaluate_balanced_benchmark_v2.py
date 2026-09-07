"""Evaluate a frozen v5 ranker under the balanced benchmark v2 protocol."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from phase_c.hidden_world_belief.train_object_ranker_v5 import (
    BilinearTypeRanker,
    move_ids,
    read_jsonl,
)


def summarize(ranks: list[int]) -> dict:
    return {
        "queries": len(ranks),
        "hit_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "hit_at_3": sum(rank <= 3 for rank in ranks) / len(ranks),
        "hit_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank for rank in ranks) / len(ranks),
        "mean_search_steps": sum(ranks) / len(ranks),
    }


def evaluate(model, rows: list[dict], vocab: dict,
             device: torch.device) -> dict:
    instance_by_type = defaultdict(list)
    receptacle_type_by_target = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for row in rows:
            logits = model(
                move_ids([vocab["target"].get(row["target_type"], 0)], device),
                move_ids([vocab["room"].get(row["room_type"], 0)], device),
            )[0]
            candidate_ids = [vocab["receptacle"].get(
                candidate["receptacle_type"], 0) for candidate in row["candidates"]]
            scores = logits[move_ids(candidate_ids, device)]
            label = next(i for i, item in enumerate(row["candidates"])
                         if item["label"])
            order = torch.argsort(scores, descending=True, stable=True).tolist()
            instance_rank = order.index(label) + 1
            instance_by_type[row["target_type"]].append(instance_rank)

            type_order = []
            for index in order:
                candidate_type = row["candidates"][index]["receptacle_type"]
                if candidate_type not in type_order:
                    type_order.append(candidate_type)
            type_rank = type_order.index(row["parent_type"]) + 1
            receptacle_type_by_target[row["target_type"]].append(type_rank)

    all_instance = [rank for ranks in instance_by_type.values() for rank in ranks]
    all_types = [rank for ranks in receptacle_type_by_target.values()
                 for rank in ranks]
    return {
        "independent_query_key": ["scene", "target_id", "parent_id"],
        "instance_ranking": summarize(all_instance),
        "receptacle_type_ranking": summarize(all_types),
        "per_target_instance_ranking": {
            target: summarize(ranks)
            for target, ranks in sorted(instance_by_type.items())
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path,
                        default=root / "data/object_location_benchmark_v2")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab = payload["vocabs"]
    model = BilinearTypeRanker(len(vocab["target"]), len(vocab["room"]),
                               len(vocab["receptacle"])).to(device)
    model.load_state_dict(payload["model"])
    result = {
        "benchmark": args.data.name,
        "split": args.split,
        "checkpoint_selected_by": "validation_only",
        **evaluate(model, read_jsonl(args.data / f"{args.split}.jsonl"),
                   vocab, device),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
