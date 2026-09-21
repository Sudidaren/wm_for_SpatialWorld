#!/usr/bin/env bash
# 把 key / 任务清单 / 启动脚本铺到新卡，然后按 10 worker 起 ProCTHOR 的 GPT-5+WM。
set -uo pipefail
cd /home/sudidaren
cp lightwm_phases/plans/procthor_main20.txt /tmp/gpt5_wm_procthor_tasks.txt
umask 077
KEY=$(grep -m1 '^export OPENAI_API_KEY=' "$HOME/.bashrc" | cut -d'"' -f2)
[ -n "$KEY" ] || { echo "拿不到 key"; exit 2; }
printf 'export LLM_API_KEY=%s\n' "$KEY" > /tmp/.llm_env
chmod 600 /tmp/.llm_env

python3 - <<'PY'
import paramiko
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.westb.seetacloud.com', port=56266, username='root',
          password='/dgVLk3vXhc1', timeout=30, banner_timeout=60)
s = c.open_sftp()
s.put('/tmp/.llm_env', '/root/.llm_key')
s.put('/tmp/gpt5_wm_procthor_tasks.txt', '/root/gpt5_wm_procthor_tasks.txt')
s.put('/home/sudidaren/lightwm_phases/tools/card_run_gpt5_wm_procthor.sh',
      '/root/card_run_gpt5_wm_procthor.sh')
s.close()
_i, o, e = c.exec_command(
    "chmod 600 /root/.llm_key; echo -n '任务数='; wc -l < /root/gpt5_wm_procthor_tasks.txt; "
    "echo '前3条:'; head -3 /root/gpt5_wm_procthor_tasks.txt; "
    "echo -n 'PROFILE=wm 出现次数='; grep -c 'PROFILE=wm' /root/card_run_gpt5_wm_procthor.sh",
    timeout=60)
print(o.read().decode().strip())
c.close()
PY
