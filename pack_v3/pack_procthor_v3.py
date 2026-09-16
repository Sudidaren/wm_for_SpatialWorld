"""Validate ProcTHOR episodes, build manifest, tar and upload to HuggingFace.

Usage:
  HF_TOKEN=<token> python upload_tars/pack_upload_procthor.py \
      --root /mnt/d/lightwm_data_procthor \
      --out-dir /mnt/d/lightwm_data_procthor_upload \
      --repo Sudidaren/lightwm-data

Steps: validate -> manifest.jsonl -> tar (root lightwm_data_procthor/) ->
upload tarballs/lightwm_data_procthor.tar -> upload manifest -> verify sizes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile

from huggingface_hub import HfApi


def validate(root: str) -> list[dict]:
    ep_root = os.path.join(root, "episodes")
    ep_dirs = sorted(
        d for d in os.listdir(ep_root)
        if os.path.isdir(os.path.join(ep_root, d))
    )
    assert ep_dirs, f"no episode dirs under {ep_root}"
    rows = []
    for name in ep_dirs:
        if ".partial_" in name:
            print(f"skip {name}: partial dir (not an episode)", flush=True)
            continue
        ep_dir = os.path.join(ep_root, name)
        jp = os.path.join(ep_dir, "episode.json")
        if not os.path.isfile(jp):
            print(f"skip {name}: no episode.json", flush=True)
            continue
        with open(jp) as f:
            meta = json.load(f)
        frames = meta["frames"]
        n = len(frames)
        for fr in frames:
            for suffix in ("_rgb.png", "_depth.png", "_seg.png"):
                p = os.path.join(ep_dir, fr["rgb"].replace("_rgb.png", suffix))
                assert os.path.isfile(p), f"missing {p}"
        rows.append({
            "episode_id": meta["episode_id"],
            "scene": meta["scene"],
            "mode": meta.get("mode", "procthor_coverage"),
            "num_frames": n,
            "dir": os.path.join("lightwm_data_procthor", "episodes", name),
        })
        print(f"OK {name}: {n} frames ({3*n} pngs)", flush=True)
    print(f"validated {len(rows)} episodes, total frames "
          f"{sum(r['num_frames'] for r in rows)}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/d/lightwm_data_procthor")
    ap.add_argument("--out-dir",
                    default="/mnt/d/lightwm_project/upload_tars_v3")
    ap.add_argument("--repo", default="Sudidaren/lightwm-data")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--upload", action="store_true",
                    help="upload after packing (otherwise local pack only)")
    args = ap.parse_args()

    rows = validate(args.root)
    os.makedirs(args.out_dir, exist_ok=True)

    # manifest (mirrors existing lightwm_data_cov/manifest.jsonl format)
    manifest_p = os.path.join(args.out_dir, "manifest.jsonl")
    with open(manifest_p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("manifest ->", manifest_p, flush=True)

    tar_p = os.path.join(args.out_dir, "lightwm_data_procthor.tar")
    if os.path.exists(tar_p):
        os.remove(tar_p)
    with tarfile.open(tar_p, "w") as tf:
        episodes_arc = os.path.join("lightwm_data_procthor", "episodes")
        for r in rows:
            ep_name = os.path.basename(r["dir"])
            tf.add(os.path.join(args.root, "episodes", ep_name),
                   arcname=os.path.join(episodes_arc, ep_name))
        tf.add(manifest_p,
               arcname="lightwm_data_procthor/manifest.jsonl")
    size = os.path.getsize(tar_p)
    print(f"tar -> {tar_p} ({size/1e9:.2f} GB)", flush=True)

    if not args.upload:
        print("pack done (no upload; pass --upload to upload)", flush=True)
        return

    if not args.token:
        sys.exit("HF_TOKEN required for upload")
    api = HfApi(token=args.token)
    print("uploading tarball ...", flush=True)
    api.upload_file(
        path_or_fileobj=tar_p,
        path_in_repo="tarballs/lightwm_data_procthor.tar",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print("uploading manifest ...", flush=True)
    api.upload_file(
        path_or_fileobj=manifest_p,
        path_in_repo="lightwm_data_procthor/manifest.jsonl",
        repo_id=args.repo,
        repo_type="dataset",
    )
    # verify
    found = {
        f.path: f.size for f in api.list_repo_tree(
            args.repo, repo_type="dataset", recursive=True)
        if getattr(f, "size", None) is not None
    }
    for remote, local in [
        ("tarballs/lightwm_data_procthor.tar", tar_p),
        ("lightwm_data_procthor/manifest.jsonl", manifest_p),
    ]:
        ok = found.get(remote) == os.path.getsize(local)
        print(f"verify {remote}: {'OK' if ok else 'MISMATCH ' + str(found.get(remote))}",
              flush=True)


if __name__ == "__main__":
    main()
