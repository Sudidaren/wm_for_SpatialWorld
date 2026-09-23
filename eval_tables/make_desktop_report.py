#!/usr/bin/env python3
"""把主表/附表/消融表打成一版"给人看的"报告，放到桌面（2026-09-23）。

产出（默认写到 /mnt/c/Users/29742/Desktop/）：
    WingmanWM_主表_20260923.html   ← 右击用浏览器打开（含排版、脚注）
    WingmanWM_主表_20260923.md     ← 纯文本版
    WingmanWM_主表_20260923.csv    ← Excel 版（三张表拼在一个文件里，用空行分隔）
"""
from __future__ import annotations

import csv
import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DESKTOP = "/mnt/c/Users/29742/Desktop"
STAMP = "20260923"


def read(name: str) -> str:
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


def md_to_html(md: str) -> str:
    """够用的小转换：标题 / 表格 / 引用 / 段落（不依赖任何库）。"""
    def inline(text: str) -> str:
        """先转义，再做很轻量的内联格式化（粗体 / 代码 / LaTeX 标记）。"""
        s = html.escape(text)
        for latex, shown in (("$\\L$", "<sup>L</sup>"), ("$\\P$", "<sup>P</sup>"),
                             ("$\\G$", "<sup>G</sup>"), ("$\\S$", "<sup>S</sup>"),
                             ("$\\aleph$", "<sup>ℵ</sup>"), ("$\\star$", "★"),
                             ("$\\ddagger$", "‡"), ("$\\dagger$", "†")):
            s = s.replace(latex, shown)
        while "**" in s:
            s = s.replace("**", "<b>", 1)
            if "**" in s:
                s = s.replace("**", "</b>", 1)
            else:
                break
        while "`" in s:
            s = s.replace("`", "<code>", 1)
            if "`" in s:
                s = s.replace("`", "</code>", 1)
            else:
                break
        return s

    out, in_table, in_quote = [], False, False

    def close_blocks() -> None:
        nonlocal in_table, in_quote
        if in_table:
            out.append("</tbody></table>")
            in_table = False
        if in_quote:
            out.append("</blockquote>")
            in_quote = False

    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):      # 分隔行
                continue
            if not in_table:
                close_blocks()
                out.append("<table><thead><tr>"
                           + "".join(f"<th>{inline(c)}</th>" for c in cells)
                           + "</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        close_blocks()
        if not line.strip():
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{inline(line.lstrip('# ').strip())}</h{level}>")
        elif line.startswith(">"):
            out.append(f"<blockquote>{inline(line.lstrip('> ').strip())}</blockquote>")
        else:
            out.append(f"<p>{inline(line)}</p>")
    close_blocks()
    return "\n".join(out)


STYLE = """
:root{--ink:#1b1f23;--muted:#57606a;--line:#d8dee4;--acc:#0b5cad;--bg:#fff}
*{box-sizing:border-box}
body{margin:0;padding:28px 34px 60px;background:#f6f8fa;color:var(--ink);
     font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;background:var(--bg);border:1px solid var(--line);
     border-radius:10px;padding:26px 30px 40px;box-shadow:0 1px 3px rgba(27,31,35,.08)}
h1{font-size:24px;margin:8px 0 6px;padding-bottom:10px;border-bottom:2px solid var(--acc)}
h2{font-size:18px;margin:30px 0 10px;color:var(--acc)}
h3{font-size:15px;margin:20px 0 8px}
table{border-collapse:collapse;width:100%;margin:12px 0 18px;font-size:13.5px}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
th{background:#eef3f8;font-weight:600;white-space:nowrap}
tbody tr:nth-child(even){background:#fbfcfd}
blockquote{margin:6px 0;padding:8px 14px;background:#fff8e5;border-left:4px solid #e3b341;
           color:#4d3b00;font-size:13px;border-radius:0 6px 6px 0}
p{margin:6px 0}
code{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:12.5px}
.meta{color:var(--muted);font-size:13px}
"""


def table_rows(md: str) -> list[list[str]]:
    rows = []
    for line in md.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def main() -> int:
    sample = read("main_table.md")
    full = read("main_table_full.md")
    ablation = read("ablation_table.md")
    body = f"""
<h1>WingmanWM / SpatialWorld 评测结果（2026-09-23）</h1>
<p class="meta">统一样本主表 = AI2-THOR 120 + ProcTHOR 20（分层抽样，与 311/127 同分布）；
含全量表与 5 臂消融。数据来源：云端 4 张卡 + 本机补跑，逐格注明 run 名与日期。</p>
{md_to_html(sample)}
<h1>附表：各模型全量批次</h1>
{md_to_html(full.split(chr(10), 1)[1])}
<h1>5 臂消融（官方清单 + 各模型诊断层）</h1>
{md_to_html(ablation.split(chr(10), 1)[1])}
"""
    doc = (f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
           f"<title>WingmanWM 主表 {STAMP}</title><style>{STYLE}</style></head>"
           f"<body><main>{body}</main></body></html>")
    os.makedirs(DESKTOP, exist_ok=True)
    html_path = os.path.join(DESKTOP, f"WingmanWM_主表_{STAMP}.html")
    md_path = os.path.join(DESKTOP, f"WingmanWM_主表_{STAMP}.md")
    csv_path = os.path.join(DESKTOP, f"WingmanWM_主表_{STAMP}.csv")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(sample + "\n\n---\n\n" + full + "\n\n---\n\n" + ablation)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        for title, md in (("主表（统一样本）", sample),
                          ("附表（全量）", full),
                          ("5 臂消融", ablation)):
            w.writerow([title])
            w.writerows(table_rows(md))
            w.writerow([])
    for p in (html_path, md_path, csv_path):
        print("wrote", p, os.path.getsize(p), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
