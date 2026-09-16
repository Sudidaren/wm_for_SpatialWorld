import json, os, math
from collections import Counter

D = os.path.expanduser("~/wm_dev/pull_latest")
BASE_RUNS = os.path.expanduser("~/spatialworld_eval/runs")

def load(path):
    with open(path) as f: d = json.load(f)
    return {t["task_id"]: t for t in d.get("tasks", [])}

def decided(t): return t.get("status") in ("success", "failed_model")

def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)

def reason_bucket(r):
    r = (r or "").strip()
    if not r: return "other/unknown"
    if "success conditions not met" in r: return "PREM: 提前 DONE"
    if "maximum step limit" in r.lower() or "step limit" in r.lower(): return "SLIM: 步数耗尽"
    return "OTHER: " + r[:60]

PAIRS = [
    ("Qwen3-VL-8B-Instruct",
     f"{D}/wm_ai2thor311_qwen3vl8b_v1.state.json",
     f"{D}/qwen3vl8b_438_v1.state.json", "base438", "8B"),
    ("Kimi-VL-A3B-Instruct",
     f"{D}/wm_ai2thor311_kimivl_a3b_v1.state.json",
     f"{D}/kimivl_a3b_438_v1.state.json", "base438", "Kimi"),
    ("Qwen3-VL-30B-A3B",
     f"{D}/wm_ai2thor311_qwen3vl30b_v1.state.json",
     f"{BASE_RUNS}/qwen3vl30b_ai2thor_v1/state.json", "base311", "30B"),
]

print("=" * 118)
print("WM vs 基线（同任务集配对，AI2-THOR 311）".center(100))
print("=" * 118)
hdr = f"{'Model':<24}{'WM判定':>7}{'WM成功':>7}{'WM TSR':>9}{'基线成功':>9}{'基线TSR':>9}{'WM独有':>7}{'基线独有':>9}{'McNemar p':>11}"
print(hdr); print("-" * 118)

rows = []
for label, wm_path, base_path, kind, tag in PAIRS:
    wm = load(wm_path); base = load(base_path)
    if kind == "base438":
        base = {k: v for k, v in base.items() if v.get("env") == "ai2thor"}
    D_ids = [k for k, v in wm.items() if decided(v)]
    b_only = w_only = both = neither = 0
    for k in D_ids:
        w = bool(wm[k].get("success")); b = bool(base.get(k, {}).get("success"))
        if w and b: both += 1
        elif w: w_only += 1
        elif b: b_only += 1
        else: neither += 1
    n = len(D_ids)
    ws = w_only + both; bs = b_only + both
    p = mcnemar_exact(b_only, w_only)
    print(f"{label:<24}{n:>7}{ws:>7}{ws/n*100:>8.1f}%{bs:>9}{bs/n*100:>8.1f}%{w_only:>7}{b_only:>9}{p:>11.3f}")
    rows.append((tag, label, n, ws, bs, w_only, b_only, p, wm, base, D_ids))

print()
print("=" * 118)
print("失败结构（仅 WM 批次，decided 任务）".center(100))
print("=" * 118)
print(f"{'Model':<24}{'判定':>6}{'成功':>6}{'PREM提前DONE':>14}{'SLIM步数耗尽':>14}{'其他':>8}{'AvgTokens':>12}{'AvgSteps':>10}")
print("-" * 118)
for tag, label, n, ws, bs, wo, bo, p, wm, base, D_ids in rows:
    c = Counter()
    toks = []; steps = []
    for k in D_ids:
        t = wm[k]
        if t.get("success"): continue
        c[reason_bucket(t.get("fail_reason"))] += 1
    for k in D_ids:
        t = wm[k]
        if t.get("total_tokens"): toks.append(t["total_tokens"])
        if t.get("actual_steps"): steps.append(t["actual_steps"])
    prem = c.get("PREM: 提前 DONE", 0); slim = c.get("SLIM: 步数耗尽", 0)
    other = sum(v for k2, v in c.items() if k2.startswith("OTHER"))
    avg_tok = sum(toks)/len(toks) if toks else 0
    avg_st = sum(steps)/len(steps) if steps else 0
    print(f"{label:<24}{n:>6}{ws:>6}{prem:>8} ({prem/max(1,n-ws)*100:>4.0f}%){slim:>7} ({slim/max(1,n-ws)*100:>4.0f}%){other:>8}{avg_tok:>12,.0f}{avg_st:>10.1f}")

print()
print("=" * 118)
print("旧 WM 对照（438 批，仅 AI2-THOR 子集）".center(100))
print("=" * 118)
for f, lab in [("wingman_wm_plannew_qwen3vl30b_438_v1", "30B + WM(new, 旧词表版)"),
               ("wingman_wm_planold_qwen3vl30b_438_v1", "30B + WM(old, 旧词表版)")]:
    wm = load(f"{D}/{f}.state.json")
    sub = {k: v for k, v in wm.items() if v.get("env") == "ai2thor"}
    dd = [k for k, v in sub.items() if decided(v)]
    s = sum(1 for k in dd if sub[k].get("success"))
    print(f"{lab:<32} decided {len(dd):>3}/311  success {s:>3}  TSR {s/len(dd)*100:>5.1f}%")
bb = load(f"{BASE_RUNS}/qwen3vl30b_ai2thor_v1/state.json")
print(f"{'30B 基线 (纯, 无WM)':<32} decided {len(bb):>3}/311  success {sum(1 for v in bb.values() if v.get('success')):>3}  TSR {sum(1 for v in bb.values() if v.get('success'))/len(bb)*100:>5.1f}%")
