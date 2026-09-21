#!/usr/bin/env python3
"""Verify that the vLLM server on the card actually sees the images.

If the multimodal path is broken (images dropped, wrong field names, the
OpenAI-compatible request not carrying image_url), the agent is effectively
blind: it still emits plausible-looking actions, but every episode fails.
That would look like "the WM does not work" when in fact the model never saw
the frame.  This sends one recorded step PNG and asks the model to describe it.

Usage (on the card):
    /root/miniconda3/bin/python check_vision.py <served_name> <png_path> [base_url]
"""

from __future__ import annotations

import base64
import json
import sys
import urllib.request

name, png = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8000/v1"

with open(png, "rb") as fh:
    b64 = base64.b64encode(fh.read()).decode()

payload = {
    "model": name,
    "max_tokens": 256,
    "temperature": 0.0,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text",
             "text": "Describe this indoor scene in one sentence: which room is "
                     "it, and name two objects you can see."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }],
}
req = urllib.request.Request(
    base + "/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
)
with urllib.request.urlopen(req, timeout=180) as r:
    out = json.load(r)
print("=== 模型对这张图的描述 ===")
print(out["choices"][0]["message"]["content"][:600])
print("=== usage ===", out.get("usage"))
