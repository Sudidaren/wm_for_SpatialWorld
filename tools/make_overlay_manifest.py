#!/usr/bin/env python3
"""重新生成 runtime_overlay/MANIFEST.sha256。

规则（与已发布清单逐字一致，2026-09-19 交接文档 §9.6）：
  * 顶层 meta 文件排除：MANIFEST.sha256 / README.md / INVENTORY.md /
    setup_lightwm_runtime.sh / verify_delivery.sh
  * 排除 __pycache__ 与 *.pyc
  * ``.gitignore`` 用 ``./`` 前缀（历史格式如此，改了会与旧清单对不上）
  * 其余按相对路径字典序排列
  * 每行 ``<sha256>␣␣<path>``

    python3 tools/make_overlay_manifest.py [--check]

``--check`` 只比对不写盘，用来确认 manifest 是否与磁盘一致。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

OVERLAY = Path(__file__).resolve().parents[1] / 'runtime_overlay'
META = {'MANIFEST.sha256', 'README.md', 'INVENTORY.md',
        'setup_lightwm_runtime.sh', 'verify_delivery.sh'}


def build() -> list[str]:
    rows = []
    for path in sorted(OVERLAY.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(OVERLAY).as_posix()
        if rel in META or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        shown = f'./{rel}' if rel == '.gitignore' else rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f'{digest}  {shown}')
    return rows


def main() -> int:
    rows = build()
    target = OVERLAY / 'MANIFEST.sha256'
    want = '\n'.join(rows) + '\n'
    if '--check' in sys.argv:
        have = target.read_text(encoding='utf-8') if target.exists() else ''
        if have == want:
            print(f'MANIFEST 与磁盘一致（{len(rows)} 条）')
            return 0
        old = {ln.split(None, 1)[1] for ln in have.splitlines() if ln.strip()}
        new = {ln.split(None, 1)[1] for ln in rows}
        print('MANIFEST 不一致：')
        for extra in sorted(new - old):
            print(f'  + {extra}')
        for gone in sorted(old - new):
            print(f'  - {gone}')
        old_map = {ln.split(None, 1)[1]: ln.split()[0]
                   for ln in have.splitlines() if ln.strip()}
        new_map = {ln.split(None, 1)[1]: ln.split()[0] for ln in rows}
        for key in sorted(set(old_map) & set(new_map)):
            if old_map[key] != new_map[key]:
                print(f'  ~ {key} 内容变了')
        return 1
    target.write_text(want, encoding='utf-8')
    print(f'写入 {target}（{len(rows)} 条）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
