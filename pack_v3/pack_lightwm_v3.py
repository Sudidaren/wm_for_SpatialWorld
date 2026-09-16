"""Validate a LightWM-format dataset root, build manifest + tar, upload to HF.

Usage:
  HF_TOKEN=<token> HF_ENDPOINT=https://hf-mirror.com \
    python upload_tars/pack_upload_lightwm.py \
      --name lightwm_data_virtualhome \
      --root /mnt/d/lightwm_data_virtualhome \
      --repo Sudidaren/lightwm-data [--upload]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile

from huggingface_hub import HfApi


def validate(root: str, name: str) -> list[dict]:
    ep_root = os.path.join(root, "episodes")
    ep_dirs = sorted(
        d for d in os.listdir(ep_root)
        if os.path.isdir(os.path.join(ep_root, d))
    )
    assert ep_dirs, f"no episode dirs under {ep_root}"
    rows = []
    total = 0
    for ep in ep_dirs:
        ep_dir = os.path.join(ep_root, ep)
        jp = os.path.join(ep_dir, "episode.json")
        with open(jp) as f:
            meta = json.load(f)
        frames = meta["frames"]
        n = len(frames)
        total += n
        for fr in frames:
            for key, suffix in (("rgb", "_rgb.png"), ("seg", "_seg.png")):
                rel = fr.get(key)
                if not rel:
                    continue
                assert os.path.isfile(os.path.join(ep_dir, rel)), \
                    f"missing {rel}"
            if fr.get("depth"):
                assert os.path.isfile(
                    os.path.join(ep_dir, fr["depth"])), f"missing {fr['depth']}"
        rows.append({
            "episode_id": meta["episode_id"],
            "scene": meta["scene"],
            "mode": meta.get("mode", name),
            "num_frames": n,
            "dir": os.path.join(name, "episodes", ep),
        })
        print(f"OK {ep}: {n} frames", flush=True)
    print(f"validated {len(rows)} episodes, total frames {total}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--repo", default="Sudidaren/lightwm-data")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.root),
                                           f"{args.name}_upload")
    rows = validate(args.root, args.name)
    os.makedirs(out_dir, exist_ok=True)
    manifest_p = os.path.join(out_dir, "manifest.jsonl")
    with open(manifest_p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tar_p = os.path.join(out_dir, f"{args.name}.tar")
    if os.path.exists(tar_p):
        os.remove(tar_p)
    with tarfile.open(tar_p, "w") as tf:
        tf.add(os.path.join(args.root, "episodes"),
               arcname=f"{args.name}/episodes")
        if os.path.isdir(os.path.join(args.root, "scene_gt")):
            tf.add(os.path.join(args.root, "scene_gt"),
                   arcname=f"{args.name}/scene_gt")
        tf.add(manifest_p, arcname=f"{args.name}/manifest.jsonl")
    print(f"tar -> {tar_p} ({os.path.getsize(tar_p)/1e9:.2f} GB)",
          flush=True)
    if not args.upload:
        print("pack done (no upload)", flush=True)
        return
    if not args.token:
        sys.exit("HF_TOKEN required")
    api = HfApi(token=args.token)
    api.upload_file(path_or_fileobj=tar_p,
                    path_in_repo=f"tarballs/{args.name}.tar",
                    repo_id=args.repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=manifest_p,
                    path_in_repo=f"{args.name}/manifest.jsonl",
                    repo_id=args.repo, repo_type="dataset")
    files = {f.path: f.size for f in api.list_repo_tree(
        args.repo, repo_type="dataset", recursive=True)
        if getattr(f, "size", None) is not None}
    for remote, local in [(f"tarballs/{args.name}.tar", tar_p),
                          (f"{args.name}/manifest.jsonl", manifest_p)]:
        ok = files.get(remote) == os.path.getsize(local)
        print(f"verify {remote}: {'OK' if ok else 'MISMATCH'}", flush=True)


if __name__ == "__main__":
    main()
