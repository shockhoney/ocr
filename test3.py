import os
import json
import cv2
import numpy as np
import unicodedata
import base64
import re
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

INPUT_DIR = "main_file"
OUTPUT_BASE = "output_result"
JSON_DIR = os.path.join(OUTPUT_BASE, "json")
IMG_DIR = os.path.join(OUTPUT_BASE, "image")
FONT_PATH = "simhei.ttf"

VLLM_API_URL = "http://127.0.0.1:8118/v1"
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "PaddleOCR-VL-0.9B"

for d in (JSON_DIR, IMG_DIR):
    os.makedirs(d, exist_ok=True)

# preprocess logic (unchanged)
def preprocess_image_enhanced(img_path):
    img = cv2.imread(img_path)
    if img is None: return None

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_contrast = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    gaussian = cv2.GaussianBlur(img_contrast, (0, 0), 3.0)
    img_sharp = cv2.addWeighted(img_contrast, 1.5, gaussian, -0.5, 0, img_contrast)

    return img_sharp

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

class MockPaddleResult:
    def __init__(self, raw_text, base_name, original_path):
        self.raw_text = raw_text
        self.base_name = base_name
        self.original_path = original_path

    def _parse_json(self):
        return extract_json_from_text(self.raw_text)

    def save_to_json(self, save_dir):
        file_path = os.path.join(save_dir, f"{self.base_name}_res.json")
        data = self._parse_json()
        if data:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            print(f"  [ERROR] JSON parse failed, saved raw output to {file_path}")
            print(f"  [DEBUG] Raw model output snippet: {self.raw_text[:100]}...")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.raw_text)
        return file_path

    def save_to_img(self, save_dir):
        data = self._parse_json()
        if not data:
            return
        ocr_img = os.path.join(save_dir, f"{self.base_name}_ocr_res.png")
        order_img = os.path.join(save_dir, f"{self.base_name}_layout_order_res.png")
        draw_ocr_result(self.original_path, data, ocr_img)
        draw_ocr_result(self.original_path, data, order_img)

def clean_result(raw_data, original_path):
    clean_data = {"input_path": original_path, "parsing_res_list": []}
    exclude_keys = {"block_id", "block_order", "block_label", "group_id"}

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

def process_single_result(res, filename, original_path):
    base_name = os.path.splitext(filename)[0]

    res.save_to_img(IMG_DIR)
    order_img_path = os.path.join(IMG_DIR, f"{base_name}_layout_order_res.png")
    if os.path.exists(order_img_path):
        os.remove(order_img_path)

    json_path = res.save_to_json(JSON_DIR)
    if not os.path.exists(json_path):
        json_path = os.path.join(JSON_DIR, f"{base_name}.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        clean_data = clean_result(raw_data, original_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)

        print(f"  [DONE] {filename}")
    except Exception as e:
        print(f"  [JSON ERROR] {e}")

FULL_PROMPT = """请对图片进行版面分析，识别并提取所有可见的文字区域，包括水平、垂直和倾斜排列的文字。注意文字可能具有不一致的字体大小，需根据内容连续性进行合理合并。输出时应准确标注每个文字区域的文本框坐标（bounding box），并确保语义连续的文字被包含在同一个文本框中"""

client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_URL)
try:
    client.models.list()
    print(f"connected, using model {MODEL_NAME}")
except Exception as e:
    print(f"models.list failed, still using {MODEL_NAME}. error: {e}")

def encode_image(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        continue
    original_path = os.path.join(INPUT_DIR, filename)
    print(f"\nProcessing: {filename}")

    img_processed = preprocess_image_enhanced(original_path)
    if img_processed is None:
        continue

    base64_image = encode_image(img_processed)
    if not base64_image:
        continue

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
    mock_res = MockPaddleResult(result_text, base_name, original_path)
    process_single_result(mock_res, filename, original_path)
