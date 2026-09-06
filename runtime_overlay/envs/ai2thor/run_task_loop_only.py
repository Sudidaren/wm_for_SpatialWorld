#!/usr/bin/env python3
import os
import base64
import cv2
import time
from ai2thor.controller import Controller
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("❌ 请设置 OPENAI_API_KEY")
    exit(1)

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-vl-plus"
TASK = "You are in a kitchen. First walk to the front of the fridge, then open its door. If already open, say 'Done'. Do not say OpenFridge unless you are close to the fridge."
MAX_STEPS = 12

print("🚀 初始化 AI2-THOR...")
controller = Controller(
    scene="FloorPlan1",
    # 不指定 commit_id，让 AI2-THOR 使用默认版本（已在 test_connection 中验证通过）
)
print("✅ 环境初始化完成")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_action(frame, step):
    prompt = f"Step {step}. Task: {TASK}\nWhat action do you take? Respond with one command: MoveForward, MoveBackward, RotateLeft, RotateRight, OpenFridge, Done."
    _, buffer = cv2.imencode('.png', frame)
    img_b64 = base64.b64encode(buffer).decode()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        }],
        max_tokens=50,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

print("🎬 开始执行任务...")
for step in range(1, MAX_STEPS+1):
    print(f"--- Step {step} ---")
    event = controller.step("Pass")
    if event.frame is None:
        print("❌ 无法获取帧")
        break
    cv2.imwrite(f"step_{step:02d}.png", event.frame)
    action = get_action(event.frame, step)
    print(f"🤖 动作: {action}")
    if action == "Done":
        print("✅ 任务完成!")
        break
    # 执行动作
    if action == "MoveForward":
        controller.step("MoveAhead")
    elif action == "MoveBackward":
        controller.step("MoveBack")
    elif action == "RotateLeft":
        controller.step("RotateLeft")
    elif action == "RotateRight":
        controller.step("RotateRight")
    elif action == "OpenFridge":
        controller.step("OpenObject", objectId="Fridge")
    else:
        print(f"⚠️ 未知动作: {action}, 跳过")
    time.sleep(0.5)

print("🏁 任务结束")
controller.stop()
