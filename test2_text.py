import os
import json
import cv2
import numpy as np
import shutil
import unicodedata
import base64
import requests
import re
from PIL import Image, ImageDraw, ImageFont  # 引入 PIL 用于绘制中文

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
OUTPUT_BASE = "output_result"
JSON_DIR = os.path.join(OUTPUT_BASE, "json")
IMG_DIR = os.path.join(OUTPUT_BASE, "image")
TEMP_DIR = "temp_processed"

# VLLM API 配置
API_URL = "http://127.0.0.1:8118/v1/chat/completions"
MODEL_NAME = "PaddleOCR-VL-0.9B"

# 字体配置 (请根据你的系统修改路径，Linux通常在 /usr/share/fonts)
# 尝试按顺序查找，如果都找不到将使用默认字体(不支持中文)
FONT_PATHS = [
    "simhei.ttf",                       # Windows 默认黑体
    "msyh.ttc",                         # Windows 微软雅黑
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", # Linux 常见
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"     # Linux Noto
]

for d in [JSON_DIR, IMG_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ================= 2. 图像处理与API工具 =================
def preprocess_image_enhanced(img_path, temp_save_path):
    """预处理：CLAHE增强 + USM锐化"""
    img = cv2.imread(img_path)
    if img is None: return False
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_contrast = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(img_contrast, (0, 0), 3.0)
    img_sharp = cv2.addWeighted(img_contrast, 1.5, gaussian, -0.5, 0, img_contrast)
    cv2.imwrite(temp_save_path, img_sharp)
    return True

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def extract_json(content):
    """正则提取 JSON"""
    content = content.strip()
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if match: content = match.group(1)
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1: return json.loads(content[start:end+1])
        return json.loads(content)
    except: return None

def call_openai_api(image_path):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "请对图片进行版面分析，提取所有可见的文字区域，准确输出坐标并合并语义连续的文本行。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
            ]}
        ],
        "temperature": 0.0, "max_tokens": 4096
    }
    try:
        resp = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"})
        if resp.status_code == 200:
            return extract_json(resp.json()['choices'][0]['message']['content'])
    except Exception as e: print(f"  [API错误] {e}")
    return None

# ================= 3. 可视化绘制 (新增文字回填) =================
def draw_visualization_with_text(img_path, json_data, save_path):
    """使用 PIL 绘制红框和对应的文本"""
    if not os.path.exists(img_path): return

    # 1. 加载图片 (PIL格式)
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # 2. 加载字体 (自动寻找可用中文字体)
    font = None
    font_size = max(20, int(image.width / 50)) # 根据图宽动态调整字号
    for path in FONT_PATHS:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except: continue
    if font is None:
        font = ImageFont.load_default() # 找不到则用默认(不支持中文)

    # 3. 绘制框和文字
    if "parsing_res_list" in json_data:
        for item in json_data["parsing_res_list"]:
            bbox = item.get("block_bbox", [])
            text = item.get("block_content", "")
            
            if len(bbox) == 4 and text:
                x1, y1, x2, y2 = map(int, bbox)
                
                # 画红框
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                # 计算文字背景大小
                bbox_text = draw.textbbox((0, 0), text, font=font)
                text_w = bbox_text[2] - bbox_text[0]
                text_h = bbox_text[3] - bbox_text[1]

                # 文字位置：放在框的上方，如果上方空间不够则放框内
                text_x = x1
                text_y = y1 - text_h - 5
                if text_y < 0: text_y = y1 + 5

                # 画文字背景(白色半透明效果需用RGBA，这里用实心白底)
                draw.rectangle([text_x, text_y, text_x + text_w + 4, text_y + text_h + 4], fill="white", outline="red", width=1)
                
                # 画文字(蓝色)
                draw.text((text_x + 2, text_y + 2), text, fill="blue", font=font)

    # 4. 保存
    image.save(save_path)

# ================= 4. 过滤逻辑 =================
PLACEHOLDER_CHARS = set("口□■▢▣▤▥▦▧▨▩▪▫◻◼◽◾☐☑☒")
def is_meaningful_text(text: str) -> bool:
    if not text: return False
    s = "".join(ch for ch in text if not ch.isspace())
    if not s or all(ch in PLACEHOLDER_CHARS for ch in s): return False
    if all(unicodedata.category(ch).startswith(("P", "S")) for ch in s): return False
    return any(unicodedata.category(ch).startswith(("L", "N")) for ch in s)

# ================= 5. 主循环 =================
if __name__ == "__main__":
    try:
        for filename in os.listdir(INPUT_DIR):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')): continue
            
            orig_path = os.path.join(INPUT_DIR, filename)
            temp_path = os.path.join(TEMP_DIR, filename)
            base_name = os.path.splitext(filename)[0]
            print(f"正在处理: {filename} ...")

            # 1. 预处理
            if not preprocess_image_enhanced(orig_path, temp_path): continue

            # 2. 调用模型
            data = call_openai_api(temp_path)
            if not data or "parsing_res_list" not in data:
                print("  [失败] 模型未返回有效数据")
                continue

            # 3. 过滤清洗
            clean_list = []
            for item in data["parsing_res_list"]:
                if "image" in item.get("block_label", "").lower(): continue
                if not is_meaningful_text(item.get("block_content", "")): continue
                
                # 清洗字段
                clean_item = {k: v for k, v in item.items() if k not in {"block_id", "block_order", "block_label", "group_id"}}
                clean_list.append(clean_item)

            final_json = {"input_path": orig_path, "parsing_res_list": clean_list}

            # 4. 保存 JSON
            with open(os.path.join(JSON_DIR, f"{base_name}.json"), 'w', encoding='utf-8') as f:
                json.dump(final_json, f, ensure_ascii=False, indent=4)

            # 5. 保存带文字的图片 (传入的是预处理后的清晰图 temp_path，也可以改传 orig_path)
            vis_save_path = os.path.join(IMG_DIR, f"{base_name}_vis.jpg")
            draw_visualization_with_text(temp_path, final_json, vis_save_path)
            
            print(f"  [完成] 结果已保存")

    finally:
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)