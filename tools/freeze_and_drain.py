#!/usr/bin/env python3
"""冻结 supervisor，但保证在跑的 worker 能跑完。

背景：spatialworld_eval 的 supervisor 用 PIPE 抓子进程 stdout/stderr。
   * 直接 KILL supervisor → 管道读端关闭，worker 写 stdout 会 EPIPE/崩；
   * 只 STOP supervisor    → 没人读管道，缓冲区（64KB）写满后 worker 阻塞。

所以这里：SIGSTOP 冻结 supervisor（= 不再提交任何新任务），
同时把 supervisor 的管道 fd 复制出来持续读掉，让还活着的 worker 正常收尾。
worker 全部退出后脚本自己结束，supervisor 保持冻结状态。

用法：python3 freeze_and_drain.py <supervisor_pid>
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time


def pipe_fds(pid: int) -> dict[int, str]:
    out = {}
    d = f'/proc/{pid}/fd'
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for name in names:
        try:
            target = os.readlink(os.path.join(d, name))
        except OSError:
            continue
        if target.startswith('pipe:'):
            out[int(name)] = target
    return out


def workers_alive() -> int:
    r = subprocess.run(['pgrep', '-cf', 'work.run_task'], capture_output=True,
                       text=True)
    return int(r.stdout.strip() or 0)


def main() -> None:
    pid = int(sys.argv[1])
    os.kill(pid, signal.SIGSTOP)
    print(f'[{time.strftime("%T")}] supervisor {pid} 已 SIGSTOP（不会再提交新任务）',
          flush=True)

    handles: dict[int, object] = {}
    for _ in range(6):
        for fd in pipe_fds(pid):
            if fd in handles:
                continue
            try:
                handles[fd] = os.fdopen(os.open(f'/proc/{pid}/fd/{fd}',
                                                os.O_RDONLY | os.O_NONBLOCK),
                                        'rb', buffering=0)
                print(f'[{time.strftime("%T")}] 接管 supervisor fd {fd}', flush=True)
            except OSError as exc:
                print(f'  fd {fd} 打不开: {exc}', flush=True)
        if workers_alive() == 0:
            break
        time.sleep(5)

    idle = 0
    while True:
        alive = workers_alive()
        if alive == 0:
            print(f'[{time.strftime("%T")}] 6 个 worker 全部退出，放掉管道',
                  flush=True)
            break
        try:
            ready, _, _ = select.select(list(handles.values()), [], [], 30)
        except (OSError, ValueError):
            ready = []
        for handle in ready:
            try:
                if not handle.read(1 << 16):
                    handles.pop(handle.fileno(), None)
                    handle.close()
            except OSError:
                pass
        idle += 1
        if idle % 10 == 0:
            print(f'[{time.strftime("%T")}] 在读管道；worker 还剩 {alive} 个',
                  flush=True)

    print(f'[{time.strftime("%T")}] 完成：supervisor {pid} 仍是 SIGSTOP 状态，'
          f'不会再启动任何任务', flush=True)


if __name__ == '__main__':
    main()
