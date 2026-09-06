"""M0 step 1: data inventory from frame_index.pkl (+ fd_index.pkl).

Extracts per-pool scene/episode/frame/trajectory-type stats, AI2-THOR scene
family mapping, per-pool per-class coverage (highlighting the 14 rarest
classes), and verifies referenced frame files exist on the local data roots.

Outputs:
  data/inventory.json        (structured, machine readable)
  data/inventory_report.md   (human readable tables)

Run from the repo root:
  python shared/make_inventory.py [--index data/frame_index.pkl]
                                  [--fd-index data/fd_index.pkl]
                                  [--no-file-check]
The same script can be run on the AutoDL instance to compare inventories
(index paths there are absolute under /root/autodl-tmp).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys


def pool_of(path: str) -> str:
    if "_virtualhome" in path:
        return "virtualhome"
    if "_procthor" in path:
        return "procthor"
    if "_objviews" in path:
        return "objviews"
    if "_cov" in path:
        return "cov"
    if "fd_benchmark" in path:
        return "fd"
    return "lightwm_data"


def family_of(scene: str) -> str:
    """AI2-THOR family buckets; others return their own tag."""
    if scene is None:
        return "unknown"
    m = re.search(r"FloorPlan(\d+)", scene)
    if m:
        n = int(m.group(1))
        for lo, hi in ((1, 30), (201, 230), (301, 330), (401, 430)):
            if lo <= n <= hi:
                return f"FloorPlan {lo}-{hi}"
        return f"FloorPlan other({n})"
    if scene.startswith("procthor"):
        return "procthor-val"
    if scene.startswith("virtualhome"):
        return "virtualhome"
    return "other"


def base_scene(scene: str) -> str:
    """Strip trailing collectors' suffixes (e.g. FloorPlan315_physics)."""
    if scene is None:
        return None
    m = re.match(r"(FloorPlan\d+)", scene)
    return m.group(1) if m else scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/frame_index.pkl")
    ap.add_argument("--fd-index", default="data/fd_index.pkl")
    ap.add_argument("--no-file-check", action="store_true")
    ap.add_argument("--rare-n", type=int, default=14)
    args = ap.parse_args()

    def load(p):
        if not os.path.exists(p):
            print(f"[inventory] missing index {p}", file=sys.stderr)
            return None
        import pickle
        with open(p, "rb") as f:
            return pickle.load(f)

    idx = load(args.index)
    fd = load(args.fd_index)
    if idx is None:
        sys.exit(1)

    frames = idx.get("frames", [])
    vocab = set(idx.get("object_types", []))
    print(f"[inventory] frame_index frames={len(frames)} "
          f"object_types={len(idx.get('object_types', []))}")

    # ---------- per-pool stats ----------
    pool_scenes = collections.defaultdict(set)
    pool_eps = collections.defaultdict(set)
    pool_frames = collections.Counter()
    pool_mode_frames = collections.defaultdict(collections.Counter)
    pool_mode_eps = collections.defaultdict(
        lambda: collections.defaultdict(set))
    pool_vis = collections.defaultdict(collections.Counter)  # class counts
    pool_oov = collections.defaultdict(collections.Counter)
    pool_vis_frames = collections.Counter()
    n_rgb = n_vis = 0
    for fr in frames:
        if not fr.get("rgb"):
            continue
        p = pool_of(fr["rgb"])
        n_rgb += 1
        ep = fr.get("episode")
        sc = fr.get("scene")
        pool_frames[p] += 1
        if ep:
            pool_eps[p].add(ep)
        if sc:
            pool_scenes[p].add(sc)
        mode = fr.get("mode") or "unknown"
        pool_mode_frames[p][mode] += 1
        if ep:
            pool_mode_eps[p][mode].add(ep)
        if fr.get("visible"):
            n_vis += 1
            pool_vis_frames[p] += 1
            for o in fr["visible"]:
                t = o.get("type")
                if t and t in vocab:
                    pool_vis[p][t] += 1
                elif t:
                    pool_oov[p][t] += 1

    # ---------- fd pool ----------
    fd_frames = []
    fd_eps = collections.defaultdict(set)
    fd_scenes = collections.defaultdict(set)
    fd_mode_frames = collections.Counter()
    if fd:
        fd_frames = fd.get("frames", [])
        for fr in fd_frames:
            p = pool_of(fr.get("rgb", "fd"))
            if p == "lightwm_data":  # fd_index paths contain fd_benchmark
                p = "fd"
            sub = "ai2thor" if "/ai2thor/" in fr.get("rgb", "") else "procthor"
            fd_mode_frames[sub] += 1
            if fr.get("episode"):
                fd_eps[p].add(fr["episode"])
            if fr.get("scene"):
                fd_scenes[p].add(fr["scene"])
        pool_frames["fd"] += len(fd_frames)
        pool_eps["fd"] |= fd_eps.get("fd", set())
        pool_scenes["fd"] |= fd_scenes.get("fd", set())
        print(f"[inventory] fd_index frames={len(fd_frames)} "
              f"actions={len(fd.get('actions', []))}")

    # ---------- class coverage across visible frames ----------
    all_counts = collections.Counter()
    for c in pool_vis.values():
        all_counts.update(c)
    rare = [t for t, _ in all_counts.most_common()[:-args.rare_n - 1:-1]]
    rare = sorted(rare, key=lambda t: all_counts[t])

    # ---------- AI2-THOR scene registry (main pools) ----------
    ai2thor_scenes = collections.defaultdict(
        lambda: {"families": set(), "pools": set(), "episodes": set(),
                 "frames": 0})
    for fr in frames:
        if not fr.get("rgb") or not fr.get("scene"):
            continue
        sc = fr["scene"]
        if not sc.startswith("FloorPlan"):
            continue
        b = base_scene(sc)
        rec = ai2thor_scenes[b]
        rec["families"].add(family_of(sc))
        rec["pools"].add(pool_of(fr["rgb"]))
        if fr.get("episode"):
            rec["episodes"].add(fr["episode"])
        rec["frames"] += 1

    family_pool_scenes = collections.defaultdict(
        lambda: collections.defaultdict(set))
    for b, rec in ai2thor_scenes.items():
        for fam in rec["families"]:
            for p in rec["pools"]:
                family_pool_scenes[fam][p].add(b)

    # ---------- local file integrity check ----------
    missing = []
    if not args.no_file_check:
        paths = [fr.get("rgb") for fr in frames if fr.get("rgb")]
        paths += [fr.get("rgb") for fr in fd_frames if fr.get("rgb")]
        for i, p in enumerate(paths):
            if i % 20000 == 0 and i:
                print(f"[inventory] checked {i}/{len(paths)} frames ...")
            if not os.path.exists(p):
                missing.append(p)
                if len(missing) >= 20:
                    break
        print(f"[inventory] file check: {len(paths) - len(missing)}/"
              f"{len(paths)} rgb frames exist"
              + (f"; missing sample: {missing[:5]}" if missing else ""))

    # ---------- assemble JSON ----------
    out = {
        "meta": {
            "frame_index_frames": len(frames),
            "fd_index_frames": len(fd_frames),
            "total_frames": len(frames) + len(fd_frames),
            "object_types": len(idx.get("object_types", [])),
            "rare_class_n": args.rare_n,
            "rare_definition": "lowest visible-occurrence classes",
        },
        "pools": {},
        "ai2thor_families": {},
        "rare_classes": {t: all_counts[t] for t in rare},
        "file_check": {
            "performed": not args.no_file_check,
            "missing_count": len(missing),
            "missing_sample": missing[:10],
        },
    }
    order = ["lightwm_data", "cov", "objviews", "procthor", "virtualhome",
             "fd"]
    for p in order:
        if p not in pool_frames and p not in pool_eps:
            continue
        scenes = sorted(pool_scenes.get(p, set()))
        out["pools"][p] = {
            "scenes": len(scenes),
            "scene_names": scenes,
            "episodes": len(pool_eps.get(p, set())),
            "frames": pool_frames[p],
            "trajectory_frames": dict(
                sorted(pool_mode_frames.get(p, {}).items(),
                       key=lambda kv: -kv[1])),
            "trajectory_episodes": {
                k: len(v) for k, v in
                sorted(pool_mode_eps.get(p, {}).items(),
                       key=lambda kv: -len(kv[1]))},
            "visible_frames": pool_vis_frames[p],
            "class_coverage": dict(sorted(
                pool_vis.get(p, {}).items(), key=lambda kv: -kv[1])),
            "oov_types": dict(sorted(
                pool_oov.get(p, {}).items(), key=lambda kv: -kv[1])),
        }
    for fam, bypool in sorted(family_pool_scenes.items()):
        out["ai2thor_families"][fam] = {
            p: len(s) for p, s in sorted(bypool.items())}
    out["ai2thor_families"]["__all__"] = {
        p: len(set().union(*(bypool[p] for bypool in
                             family_pool_scenes.values()
                             if p in bypool)))
        for p in ("lightwm_data", "cov", "objviews")}

    os.makedirs("data", exist_ok=True)
    with open("data/inventory.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)

    # ---------- markdown report ----------
    md = ["# LightWM 数据盘点报告\n",
          f"> 生成时间：本地运行；frame_index 共 {len(frames)} 帧，"
          f"fd_index 共 {len(fd_frames)} 帧，合计 {len(frames) + len(fd_frames)} 帧。",
          "",
          "## 各数据池总览\n",
          "| 池 | 场景数 | episode 数 | 帧数 | 带 visible 帧数 |",
          "|---|---|---|---|---|"]
    for p in order:
        if p in out["pools"]:
            d = out["pools"][p]
            md.append(f"| {p} | {d['scenes']} | {d['episodes']} | "
                      f"{d['frames']} | {d['visible_frames']} |")
    md += ["", "## 轨迹类型分布（帧数）\n",
           "| 池 | 类型 | 帧数 | episode 数 |", "|---|---|---|---|"]
    for p in order:
        if p not in out["pools"]:
            continue
        d = out["pools"][p]
        rows = sorted(d["trajectory_frames"].items(),
                      key=lambda kv: -kv[1])
        for mode, n in rows:
            md.append(f"| {p} | {mode} | {n} | "
                      f"{d['trajectory_episodes'].get(mode, 0)} |")
    md += ["", "## AI2-THOR 场景族分布（FloorPlan 场景数）\n",
           "| 族 | lightwm_data | cov | objviews |",
           "|---|---|---|---|"]
    for fam in sorted(out["ai2thor_families"]):
        if fam == "__all__":
            continue
        d = out["ai2thor_families"][fam]
        md.append(f"| {fam} | {d.get('lightwm_data', 0)} | "
                  f"{d.get('cov', 0)} | {d.get('objviews', 0)} |")
    d = out["ai2thor_families"]["__all__"]
    md.append(f"| 合计(去重) | {d.get('lightwm_data', 0)} | "
              f"{d.get('cov', 0)} | {d.get('objviews', 0)} |")
    md += ["", "## 稀有类覆盖（全局 visible 出现次数最低的 "
              f"{args.rare_n} 类，按池分）\n",
           "| 类别 | 全局 | " + " | ".join(
               f"{p}" for p in order if p in out["pools"] and p != "fd")
           + " |", "|---|" + "---|" * (
               sum(1 for p in order if p in out["pools"] and p != "fd") + 1)]
    for t in rare:
        row = [t, str(all_counts[t])]
        for p in order:
            if p == "fd" or p not in out["pools"]:
                continue
            row.append(str(out["pools"][p]["class_coverage"].get(t, 0)))
        md.append("| " + " | ".join(row) + " |")
    md += ["", "## 词表外类型（未计入 117 类覆盖；多为结构件或未映射物体）\n",
           "| 池 | 类型 | 出现次数 |", "|---|---|---|"]
    for p in order:
        if p not in out["pools"]:
            continue
        for t, n in sorted(out["pools"][p]["oov_types"].items(),
                           key=lambda kv: -kv[1]):
            md.append(f"| {p} | {t} | {n} |")
    md += ["", "## 本地文件完整性\n",
           f"- 检查执行：{out['file_check']['performed']}",
           f"- 缺失 rgb 帧数：{out['file_check']['missing_count']}"]
    if out["file_check"]["missing_sample"]:
        md.append("- 缺失样例：")
        md += [f"  - `{p}`" for p in out["file_check"]["missing_sample"]]
    md.append("")
    with open("data/inventory_report.md", "w") as f:
        f.write("\n".join(md))
    print("[inventory] wrote data/inventory.json and "
          "data/inventory_report.md")


if __name__ == "__main__":
    main()
