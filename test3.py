import os
import json
import cv2
import numpy as np
import shutil
import unicodedata
import base64
import re
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

INPUT_DIR = "main_file"
OUTPUT_BASE = "output_result"
JSON_DIR = os.path.join(OUTPUT_BASE, "json")
IMG_DIR = os.path.join(OUTPUT_BASE, "image")
TEMP_DIR = "temp_processed"
FONT_PATH = "simhei.ttf"

VLLM_API_URL = "http://127.0.0.1:8118/v1"
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "PaddlePaddle/paddleocr_vl"

for d in (JSON_DIR, IMG_DIR, TEMP_DIR):
    os.makedirs(d, exist_ok=True)

# preprocess logic (unchanged)
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

# filtering logic (unchanged)
PLACEHOLDER_CHARS = set("口□■▢▣▤▥▦▧▨▩▪▫◻◼◽◾☐☑☒")
def is_meaningful_text(text: str) -> bool:
    if not text: return False
    s = "".join(ch for ch in text if not ch.isspace())
    if not s: return False
    if all(ch in PLACEHOLDER_CHARS for ch in s): return False
    if all(unicodedata.category(ch).startswith(("P", "S")) for ch in s): return False
    if any(unicodedata.category(ch).startswith(("L", "N")) for ch in s): return True
    return False

def draw_ocr_result(original_img_path, json_data, save_img_path):
    img_cv = cv2.imdecode(np.fromfile(original_img_path, dtype=np.uint8), -1)
    if img_cv is None: return

    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    if "parsing_res_list" in json_data:
        for item in json_data["parsing_res_list"]:
            text = item.get("block_content", "")
            bbox = item.get("block_bbox", [])

            if len(bbox) == 4 and text:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                h = y2 - y1

                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=1)

                font_size = max(12, min(40, int(h * 0.6)))
                try:
                    font = ImageFont.truetype(FONT_PATH, font_size)
                except:
                    font = ImageFont.load_default()

                text_bbox = draw.textbbox((x1, y1), text, font=font)
                text_pixel_height = text_bbox[3] - text_bbox[1]

                text_y = y1 - text_pixel_height - 5
                if text_y < 0: text_y = y1 + 5

                draw.text((x1-1, text_y), text, font=font, fill=(255,255,255))
                draw.text((x1+1, text_y), text, font=font, fill=(255,255,255))
                draw.text((x1, text_y-1), text, font=font, fill=(255,255,255))
                draw.text((x1, text_y+1), text, font=font, fill=(255,255,255))
                draw.text((x1, text_y), text, fill=(255, 0, 0), font=font)

    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imencode('.jpg', img_result)[1].tofile(save_img_path)

def extract_json_from_text(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception:
        return None

def save_raw_or_json(raw_text, base_name):
    file_path = os.path.join(JSON_DIR, f"{base_name}_res.json")
    data = extract_json_from_text(raw_text)
    if data:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return data, file_path
    print(f"  [ERROR] JSON解析失败，已保存原始返回内容到 {file_path}")
    print(f"  [DEBUG] 模型原始返回片段: {raw_text[:100]}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(raw_text)
    return None, file_path

def clean_result(raw_data, original_path):
    clean_data = {"input_path": original_path, "parsing_res_list": []}
    exclude_keys = {"block_id", "block_order", "group_id"}

    for item in raw_data.get("parsing_res_list", []):
        label = item.get("block_label", item.get("label", "text")).lower()
        content = item.get("block_content", item.get("text", ""))
        bbox = item.get("block_bbox", item.get("box", []))

        item["block_content"] = content
        item["block_bbox"] = bbox
        item["block_label"] = label

        if "image" in label: continue
        if not is_meaningful_text(content): continue

        clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
        clean_data["parsing_res_list"].append(clean_item)

    return clean_data

FULL_PROMPT = """你是一个专业的OCR助手。请分析这张图片，提取所有可见文字。
必须严格输出纯 JSON 格式，不要包含 ```json 标记。格式如下：
{
  "parsing_res_list": [
    {
      "block_content": "文字内容",
      "block_label": "text",
      "block_bbox": [x1, y1, x2, y2]
    }
  ]
}
注意：
1. block_bbox 必须是像素坐标整数 [xmin, ymin, xmax, ymax]。
2. 识别读取到的文本语义，语义连续的要合并为一个框
"""

client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_URL)
try:
    client.models.list()
    print(f"connected, using model {MODEL_NAME}")
except Exception as e:
    print(f"models.list failed, still using {MODEL_NAME}. error: {e}")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

try:
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            continue
        original_path = os.path.join(INPUT_DIR, filename)
        temp_path = os.path.join(TEMP_DIR, filename)
        print(f"\n正在处理: {filename}")

        if not preprocess_image_enhanced(original_path, temp_path):
            continue

        base64_image = encode_image(temp_path)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": FULL_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        result_text = response.choices[0].message.content or ""

        base_name = os.path.splitext(filename)[0]
        raw_data, json_path = save_raw_or_json(result_text, base_name)
        if not raw_data:
            continue

        clean_data = clean_result(raw_data, original_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)

        final_img_path = os.path.join(IMG_DIR, f"{base_name}_result.jpg")
        draw_ocr_result(original_path, clean_data, final_img_path)
        print(f"  [完成] {filename}")
finally:
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
