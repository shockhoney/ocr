import os
import json
import cv2
import numpy as np
import shutil
import unicodedata
import base64
import requests
import re
from PIL import Image, ImageDraw, ImageFont  # 新增：用于绘制中文

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
OUTPUT_BASE = "output_result"
JSON_DIR = os.path.join(OUTPUT_BASE, "json")
IMG_DIR = os.path.join(OUTPUT_BASE, "image")
TEMP_DIR = "temp_processed" 

API_URL = "http://127.0.0.1:8118/v1/chat/completions"
MODEL_NAME = "PaddleOCR-VL-0.9B"

for d in [JSON_DIR, IMG_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ================= 2. 图像预处理 (CLAHE + USM) =================
def preprocess_image_enhanced(img_path, temp_save_path):
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

# ================= 3. 新增功能：绘制文本和边框 =================
def save_visualization_with_text(image_path, json_data, save_path):
    """
    在图片上绘制红框，并在框上方填写识别到的文本。
    """
    if not os.path.exists(image_path): return

    try:
        # 1. 打开图片
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # 2. 加载字体 (自动尝试常见中文路径，找不到则用默认)
        font_path = None
        common_fonts = [
            "simhei.ttf", "msyh.ttc",  # Windows
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", # Linux
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        ]
        for font in common_fonts:
            if os.path.exists(font) or (os.name == 'nt' and font in ["simhei.ttf", "msyh.ttc"]):
                font_path = font
                break
        
        # 动态计算字号：约为图片宽度的 1/50
        font_size = max(15, int(image.width / 50))
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

        # 3. 遍历绘制
        if "parsing_res_list" in json_data:
            for item in json_data["parsing_res_list"]:
                bbox = item.get("block_bbox", [])
                text = item.get("block_content", "")
                
                if len(bbox) == 4 and text:
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # 画红框 (outline颜色, width线宽)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    
                    # 计算文字背景大小
                    if hasattr(font, "getbbox"): # 新版 Pillow
                        tb = font.getbbox(text)
                        text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
                    else: # 旧版 Pillow
                        text_w, text_h = font.getsize(text)

                    # 确定文字位置 (优先放框上方，防止遮挡内容)
                    tx, ty = x1, y1 - text_h - 5
                    if ty < 0: ty = y1 + 5 # 上方不够放就放下面

                    # 画文字白底背景 (增加可读性)
                    draw.rectangle([tx, ty, tx + text_w + 4, ty + text_h + 4], fill="white", outline="red", width=0)
                    # 画文字
                    draw.text((tx + 2, ty + 2), text, fill="blue", font=font)

        # 4. 保存
        image.save(save_path)
        
    except Exception as e:
        print(f"  [绘图警告] {e}")

# ================= 4. 工具函数 (API与清洗) =================
def encode_image(path):
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')

def extract_json(content):
    content = content.strip()
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if match: content = match.group(1)
    try:
        s, e = content.find('{'), content.rfind('}')
        if s!=-1 and e!=-1: return json.loads(content[s:e+1])
        return json.loads(content)
    except: return None

def call_openai_api(img_path):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": '''请对图片进行版面分析，识别并提取所有可见的文字区域，包括水平、垂直和倾斜排列的文字。
注意文字可能具有不一致的字体大小，需根据内容连续性进行合理合并。
输出时应准确标注每个文字区域的文本框坐标（bounding box），并确保语义连续的文字被包含在同一个文本框中。'''},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}}
            ]}
        ],
        "temperature": 0.0, "max_tokens": 4096
    }
    try:
        r = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"})
        if r.status_code==200: return extract_json(r.json()['choices'][0]['message']['content'])
    except Exception as e: print(f"  [API错误] {e}")
    return None

def is_meaningful_text(text):
    if not text: return False
    s = "".join(c for c in text if not c.isspace())
    if not s or all(c in "口□■▢▣▤▥▦▧▨▩▪▫◻◼◽◾☐☑☒" for c in s): return False
    if all(unicodedata.category(c).startswith(("P", "S")) for c in s): return False
    return any(unicodedata.category(c).startswith(("L", "N")) for c in s)

# ================= 5. 主循环 =================
if __name__ == "__main__":
    try:
        for filename in os.listdir(INPUT_DIR):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')): continue
            
            orig_path = os.path.join(INPUT_DIR, filename)
            temp_path = os.path.join(TEMP_DIR, filename)
            base_name = os.path.splitext(filename)[0]

            print(f"正在处理: {filename} ...")
            
            if not preprocess_image_enhanced(orig_path, temp_path): continue
            
            # 使用临时增强图进行预测
            data = call_openai_api(temp_path)
            
            if not data or "parsing_res_list" not in data:
                print("  [失败] 无效数据")
                continue

            clean_list = []
            for item in data["parsing_res_list"]:
                if "image" in item.get("block_label", "").lower(): continue
                if not is_meaningful_text(item.get("block_content", "")): continue
                clean_list.append({k: v for k, v in item.items() if k not in {"block_id", "block_order", "block_label", "group_id"}})

            final_json = {"input_path": orig_path, "parsing_res_list": clean_list}

            # 保存 JSON
            with open(os.path.join(JSON_DIR, f"{base_name}.json"), 'w', encoding='utf-8') as f:
                json.dump(final_json, f, ensure_ascii=False, indent=4)

            # 调用新函数：保存带文字的可视化图
            # 注意：这里传入 orig_path (原图) 或者 temp_path (增强图) 都可以
            # 建议用 temp_path 因为它更清晰，如果想看原图效果则改为 orig_path
            vis_save_path = os.path.join(IMG_DIR, f"{base_name}_vis.jpg")
            save_visualization_with_text(temp_path, final_json, vis_save_path)
            
            print(f"  [完成] 结果已保存至 {vis_save_path}")

    finally:
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)