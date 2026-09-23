#!/usr/bin/env python3
"""干净版主表：5 个模型 × 有/无 WingmanWM × 2 个环境（2026-09-23）。

只从 main_table.md 里取这 10 行，不带标记、不带脚注堆叠；两个环境各一张表。
输出到桌面：WingmanWM_主表_干净版_20260923.{html,md,csv}
"""
from __future__ import annotations

import csv
import html
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DESKTOP = "/mnt/c/Users/29742/Desktop"
STAMP = "20260923"

#: (展示名, main_table.md 里的 base 行名, 该行的 +WM 行名)
MODELS = [
    ("Qwen3-VL-30B-A3B", "Qwen3-VL-30B-A3B (BF16, vLLM)", "Qwen3-VL-30B-A3B + WingmanWM"),
    ("Qwen3-VL-8B", "Qwen3-VL-8B (BF16, vLLM)", "Qwen3-VL-8B + WingmanWM"),
    ("Kimi-VL-A3B", "Kimi-VL-A3B (BF16, vLLM)", "Kimi-VL-A3B + WingmanWM"),
    ("Gemini 3.1 Pro", "Gemini 3.1 Pro", "Gemini 3.1 Pro + WingmanWM"),
    ("GPT-5", "GPT-5", "GPT-5 + WingmanWM"),
]
#: 某些行在 md 里名字略有不同（同一配置），这里给出别名
ALIASES = {"Gemini 3.1 Pro": ("Gemini 3.1 Pro (frozen v1)",)}
ENVS = ["AI2-THOR", "ProcTHOR"]
HEAD = ["Model", "WingmanWM", "N", "TSR", "Avg steps", "Avg invalid actions",
        "Avg tokens/task", "Coverage"]
#: 干净版不列 N（环境标题里已写样本量），也不带 Coverage 的注解后缀
SHOW = ["Model", "WingmanWM", "TSR", "Avg steps", "Avg invalid actions",
        "Avg tokens/task", "Coverage"]
#: main_table.md 里的列名（前两列不同）
MD_HEAD = ["Method", "Env", "N", "TSR", "Avg steps", "Avg invalid actions",
           "Avg tokens/task", "Coverage"]


def parse(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 8 or cells[0] in ("Method",) or set(cells[0]) <= set("-: "):
                continue
            cells = [re.sub(r"\*\*", "", c) for c in cells]
            cells[0] = re.sub(r"\s*\$[^$]*\$\s*", "", cells[0]).strip()
            rows.append(dict(zip(MD_HEAD, cells)))
    return rows


def clean(rows: list[dict]) -> dict[str, list[dict]]:
    out = {env: [] for env in ENVS}
    for env in ENVS:
        for label, base_name, wm_name in MODELS:
            for wm in (False, True):
                want = wm_name if wm else base_name
                names = (want,) + (ALIASES.get(want, ()) if not wm else ())
                hit = next((r for r in rows
                            if r["Env"] == env and r["Method"].strip() in names), None)
                if hit is None:
                    continue
                out[env].append({"Model": label,
                                 "WingmanWM": "✓" if wm else "—",
                                 **{k: re.sub(r"\s*(⚠️partial|·\s*ep\d+)\s*$", "", hit[k]).strip()
                                    for k in SHOW[2:]}})
    return out


def md_table(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(SHOW) + " |",
             "|" + "|".join(["---"] * len(SHOW)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in SHOW) + " |")
    return "\n".join(lines)


STYLE = """
body{margin:0;padding:26px 30px 50px;background:#f6f8fa;color:#1b1f23;
 font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",Helvetica,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;background:#fff;border:1px solid #d8dee4;border-radius:10px;
 padding:24px 28px 34px}
h1{font-size:22px;margin:4px 0 4px;padding-bottom:8px;border-bottom:2px solid #0b5cad}
h2{font-size:17px;margin:26px 0 8px;color:#0b5cad}
p.note{color:#57606a;font-size:13px;margin:6px 0 14px}
table{border-collapse:collapse;width:100%;margin:10px 0 6px;font-size:13.5px}
th,td{border:1px solid #d8dee4;padding:7px 10px;text-align:left}
th{background:#eef3f8;font-weight:600}
td:nth-child(n+3){text-align:right}
tbody tr:nth-child(even){background:#fbfcfd}
td:nth-child(1){font-weight:600}
"""


def main() -> int:
    rows = parse(os.path.join(HERE, "main_table.md"))
    data = clean(rows)
    parts_md = [f"# WingmanWM 主表（5 个模型 × 有/无 WM × 2 环境）", ""]
    body = [f"<h1>WingmanWM 主表（5 个模型 × 有/无 WM × 2 环境）</h1>",
            '<p class="note">样本：AI2-THOR 120 / ProcTHOR 20（分层抽样，与 311/127 同分布）。'
            'TSR = success / (success + failure)；failed_external、pending、env_error 不计入分母（见 Coverage）。</p>']
    for env in ENVS:
        label = f"{env}（N = {120 if env == 'AI2-THOR' else 20}）"
        body.append(f"<h2>{label}</h2>")
        body.append("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in SHOW)
                    + "</tr></thead><tbody>")
        for r in data[env]:
            body.append("<tr>" + "".join(f"<td>{html.escape(str(r[h]))}</td>" for h in SHOW)
                        + "</tr>")
        body.append("</tbody></table>")
        parts_md += [f"## {label}", "", md_table(data[env]), ""]
    doc = ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
           f"<title>WingmanWM 主表 {STAMP}</title><style>{STYLE}</style></head><body>"
           "<main>" + "\n".join(body) + "</main></body></html>")
    os.makedirs(DESKTOP, exist_ok=True)
    paths = {ext: os.path.join(DESKTOP, f"WingmanWM_主表_干净版_{STAMP}.{ext}")
             for ext in ("html", "md", "csv")}
    with open(paths["html"], "w", encoding="utf-8") as fh:
        fh.write(doc)
    with open(paths["md"], "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts_md) + "\n")
    with open(paths["csv"], "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        for env in ENVS:
            w.writerow([env])
            w.writerow(SHOW)
            for r in data[env]:
                w.writerow([r[h] for h in SHOW])
            w.writerow([])
    for p in paths.values():
        print("wrote", p, os.path.getsize(p))
    print()
    print("\n".join(parts_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
