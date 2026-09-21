"""Emit the main table as LaTeX (paste-ready) from the same run data.

Writes:
  main_table.tex            standalone, compilable document (PDF preview)
  main_table_snippet.tex    just the \\begin{table}...\\end{table} block
  main_table_zh.tex         Chinese variant (needs ctex / xeCJK)
"""

from __future__ import annotations

import os
from typing import List, Tuple

from build_main_table import BATCHES, _allowed, stats

HERE = os.path.dirname(os.path.abspath(__file__))

METHOD_TEX = {
    "Qwen3-VL-30B-A3B (BF16, vLLM)": "Qwen3-VL-30B-A3B",
    "Gemini 3.1 Pro": "Gemini 3.1 Pro",
    "Gemini 3.1 Pro (frozen v1)": "Gemini 3.1 Pro",
    "Gemini 3.1 Pro (legacy replay)": "Gemini 3.1 Pro (legacy replay)$^{\\ddagger}$",
    "Gemini 3.1 Pro (frozen v1, running)": "Gemini 3.1 Pro (frozen v1)",
    # 2026-09-21 夜新增：闭源主批次（120 + 20 分层样本）
    "Gemini 3.1 Pro + WingmanWM": "Gemini 3.1 Pro + WingmanWM",
    "GPT-5": "GPT-5",
    "GPT-5 + WingmanWM": "GPT-5 + WingmanWM",
}


def rate(v) -> str:
    return "--" if v is None else f"{v * 100:.1f}\\%"


def num(v, nd: int = 1) -> str:
    return "--" if v is None else f"{v:.{nd}f}"


def rows() -> List[Tuple[str, str, dict]]:
    out = []
    for b in BATCHES:
        method, env, runs, envdir, planned, complete, marker, filt = b
        out.append((method, env, stats(runs, envdir, planned, _allowed(filt))))
    return out


def table_block(zh: bool = False) -> str:
    data = rows()
    cap = ("Main results on the household subset of SpatialWorld (single-agent "
           "protocol). \\emph{Invalid actions} counts steps whose environment "
           "call returned an error (target not in view, blocked by an obstacle, "
           "object does not exist), i.e.\\ steps that changed nothing; the "
           "percentage in parentheses is its share of the average step count. "
           "Success rate excludes API/simulator failures ($=$ \\emph{Null}) from "
           "the denominator. $\\dagger$: batch still running, numbers will move."
           if not zh else
           "SpatialWorld 单智能体家庭任务主结果。\\emph{无效动作} = 该步环境返回错误"
           "（目标不在视野 / 被障碍物阻挡 / 物体不存在），即花了步数但什么也没改变；"
           "括号内为其占平均步数的比例。成功率分母不含 API/环境故障（Null）。"
           "$\\dagger$：批次仍在跑，数字会变。")
    L = []
    L.append("% ---- required packages ----------------------------------------")
    L.append("% \\usepackage{booktabs}")
    L.append("% \\usepackage{multirow}")
    L.append("% ---------------------------------------------------------------")
    L.append("\\begin{table*}[t]")
    L.append("  \\centering")
    L.append(f"  \\caption{{{cap}}}")
    L.append("  \\label{tab:main-results}")
    L.append("  \\small")
    L.append("  \\setlength{\\tabcolsep}{4pt}")
    L.append("  \\begin{tabular}{llrrrrrrr}")
    L.append("    \\toprule")
    L.append("    Method & Environment & $N$ & Succ. & Fail. & Null "
             "& Success rate & Avg steps & Avg invalid actions \\\\")
    L.append("    \\midrule")
    i = 0
    while i < len(data):
        method, env, s = data[i]
        # group the two environments of one method (multirow when consecutive)
        span = 1
        while (i + span < len(data) and data[i + span][0] == method):
            span += 1
        for k in range(span):
            method_k, env_k, s_k = data[i + k]
            dagger = "" if s_k["decided"] >= s_k["planned"] and not s_k["null"] else "$\\dagger$"
            label = METHOD_TEX.get(method_k, method_k)
            mcell = (label if span == 1
                     else (f"\\multirow{{{span}}}{{*}}{{{label}}}" if k == 0 else ""))
            ishare = ("--" if s_k["invalid_rate"] is None
                      else f" ({s_k['invalid_rate'] * 100:.0f}\\%)")
            L.append(
                f"    {mcell} & {env_k} & {s_k['planned']} & {s_k['success']} "
                f"& {s_k['failure']} & {s_k['null']} & {rate(s_k['success_rate'])}{dagger} "
                f"& {num(s_k['avg_steps'])} & {num(s_k['avg_invalid'])}{ishare} \\\\")
        if span == 2 and (i + span) < len(data):
            L.append("    \\cmidrule(l){1-9}")
        i += span
    L.append("    \\bottomrule")
    L.append("  \\end{tabular}")
    L.append("\\end{table*}")
    return "\n".join(L)


def standalone(block: str) -> str:
    return ("\\documentclass[10pt]{article}\n"
            "\\usepackage[margin=0.7in]{geometry}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{multirow}\n"
            "\\usepackage{graphicx}\n"
            "\\begin{document}\n\n"
            f"{block}\n\n"
            "\\end{document}\n")


def standalone_zh(block: str) -> str:
    # xelatex + xeCJK; the CJK font is taken from the Windows font directory
    # because this machine has no Linux CJK font installed.
    return ("\\documentclass[10pt]{ctexart}\n"
            "\\usepackage[margin=0.7in]{geometry}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{multirow}\n"
            "\\usepackage{xeCJK}\n"
            "\\setCJKmainfont[Path=/mnt/c/Windows/Fonts/]{simhei.ttf}\n"
            "\\begin{document}\n\n"
            f"{block}\n\n"
            "\\end{document}\n")


def overleaf_block(align: str = "r") -> str:
    """Inline (non-floating) results table, paste-ready for Overleaf.

    Rows whose measurement does not exist yet print an en dash, so the table
    never implies a number that has not been produced.

    ``align`` sets the alignment of the four numeric columns:
      ``"r"`` right (default, textbook style for numbers),
      ``"c"`` centred, ``"l"`` left.
    """
    numeric = align * 4
    groups = [
        ("Qwen3-VL-30B-A3B", "qwen3vl30b_ai2thor_v1", "qwen3vl30b_procthor_v1"),
        ("Gemini 3.1 Pro", "replay_legacy311_v1", None),
        ("GPT-5", None, None),
        ("Qwen3-VL-30B-A3B + WingmanWM", None, None),
        ("Gemini 3.1 Pro + WingmanWM", None, None),
        ("GPT-5 + WingmanWM", None, None),
    ]
    planned = {"AI2-THOR": 311, "ProcTHOR": 127}
    envdir = {"AI2-THOR": "ai2thor", "ProcTHOR": "procthor"}
    cap = ("Main results on the household subset of SpatialWorld (single-agent "
           "protocol). $N$ is the number of evaluated tasks and TSR is the "
           "task-level pass rate returned by the official terminal-state "
           "verifier. \\emph{Invalid actions} counts steps whose environment "
           "call returned an error (target not in view, blocked by an obstacle, "
           "object does not exist), i.e.\\ steps that changed nothing; the "
           "percentage in parentheses is its share of the average step count. "
           "Dashes mark measurements that have not been run yet.")

    L = []
    L.append("% =====================================================================")
    L.append("% Overleaf-ready main results table (inline, inside a section).")
    L.append(f"% Numeric columns are '{align}'-aligned "
             f"({'right' if align == 'r' else 'centred' if align == 'c' else 'left'}); "
             f"change the column spec below to llccc / llccc etc. to re-align.")
    L.append("% Required packages:")
    L.append("%   \\usepackage{booktabs}   % \\toprule \\midrule \\bottomrule")
    L.append("%   \\usepackage{multirow}   % only for the grouped Method column")
    L.append("% Compiler: pdfLaTeX (default on Overleaf).")
    L.append("% Not needed when it sits in running text: table/table* float,")
    L.append("% \\caption, \\label, [t], \\centering.  Wrap the tabular in those")
    L.append("% if you want a numbered float with the caption below.")
    L.append("% =====================================================================")
    L.append("\\begin{center}")
    L.append("  \\small")
    L.append("  \\setlength{\\tabcolsep}{4pt}")
    L.append(f"  \\begin{{tabular}}{{ll{numeric}}}")
    L.append("    \\toprule")
    L.append("    Method & Environment & $N$ & TSR "
             "& Avg steps & Avg invalid actions \\\\")
    L.append("    \\midrule")
    for method, run_ai, run_pc in groups:
        runs = {"AI2-THOR": run_ai, "ProcTHOR": run_pc}
        L.append(f"    \\multirow{{2}}{{*}}{{{method}}}")
        for env in ("AI2-THOR", "ProcTHOR"):
            run = runs[env]
            s = stats(run, envdir[env], planned[env]) if run else None
            if s and s.get("decided"):
                n = f"{s['decided']}"
                tsr = rate(s["success_rate"])
                stp = num(s["avg_steps"])
                inv = num(s["avg_invalid"])
                if s["invalid_rate"] is not None:
                    inv += f" ({s['invalid_rate'] * 100:.0f}\\%)"
            else:
                n = tsr = stp = inv = "--"
            head = "     " if env == "ProcTHOR" else "    "
            L.append(f"{head}& {env} & {n} & {tsr} & {stp} & {inv} \\\\")
        L.append("    \\midrule")
    if L and L[-1].strip() == "\\midrule":
        L.pop()
    L.append("    \\bottomrule")
    L.append("  \\end{tabular}")
    L.append("\\end{center}")
    return "\n".join(L)


def overleaf_test_doc(block: str, two_col: bool = True) -> str:
    """Local compile harness: simulates an ICLR-ish two-column text width."""
    if two_col:
        cls = ("\\documentclass[10pt,twocolumn]{article}\n"
               "\\usepackage[letterpaper,margin=0.75in]{geometry}\n")
    else:
        cls = ("\\documentclass[10pt]{article}\n"
               "\\usepackage[letterpaper,margin=1in]{geometry}\n")
    return (cls +
            "\\usepackage{booktabs}\n"
            "\\usepackage{multirow}\n"
            "\\begin{document}\n"
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\caption{width probe}\n"
            "\\begin{tabular}{ll}\n\\toprule\nA & B \\\\\n\\midrule\n1 & 2 \\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
            f"{block}\n"
            "\\newpage\n"
            "\\begin{table*}[t]\n\\centering\n\\caption{probe2}\n"
            "\\begin{tabular}{ll}\n\\toprule\nA & B \\\\\n\\bottomrule\n\\end{tabular}\n"
            "\\end{table*}\n"
            "\\end{document}\n")


def main() -> int:
    en = table_block(False)
    zh = table_block(True)
    for name, text in (("main_table_snippet.tex", en + "\n"),
                       ("main_table.tex", standalone(en)),
                       ("main_table_zh.tex", standalone_zh(zh))):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote", name)
    for name, block in (("main_table_overleaf.tex", overleaf_block("r")),
                        ("main_table_overleaf_center.tex", overleaf_block("c")),
                        ("main_table_overleaf_left.tex", overleaf_block("l"))):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as fh:
            fh.write(block + "\n")
        with open(os.path.join(HERE, name.replace(".tex", "_test.tex")),
                  "w", encoding="utf-8") as fh:
            fh.write(overleaf_test_doc(block))
        print("wrote", name)
    print()
    print(en)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
