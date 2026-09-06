import os, base64, cv2, sys
from ai2thor.controller import Controller
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("❌ 请设置 OPENAI_API_KEY")
    sys.exit(1)

print("⏳ 1. 启动 AI2-THOR (无图形界面模式)...")
try:
    controller = Controller(
        scene="FloorPlan1",
        
        # branch="v2.3.0"
    )
    print("✅ AI2-THOR 启动成功！")
except Exception as e:
    print(f"❌ AI2-THOR 启动失败: {e}")
    sys.exit(1)

print("⏳ 2. 获取当前视角截图...")
event = controller.step("Pass")
if event.frame is None:
    print("❌ 截图获取失败")
    sys.exit(1)
cv2.imwrite("test_screenshot.png", event.frame)
print("✅ 截图已保存为 test_screenshot.png")

print("⏳ 3. 调用 Qwen API 询问下一步动作...")
_, buffer = cv2.imencode('.png', event.frame)
img_b64 = base64.b64encode(buffer).decode()
client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
resp = client.chat.completions.create(
    model="qwen-vl-plus",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "看到厨房了，请输出一个动作，例如 Move forward"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
    }]
)
print(f"✅ API 回复: {resp.choices[0].message.content}")
controller.stop()
print("🏁 测试全部通过！环境与模型链路正常。")
