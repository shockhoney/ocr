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

# ================= 配置区域 =================
INPUT_DIR = "main_file"
OUTPUT_BASE = "output_result"
JSON_DIR = os.path.join(OUTPUT_BASE, "json")
IMG_DIR = os.path.join(OUTPUT_BASE, "image")
TEMP_DIR = "temp_processed"
FONT_PATH = "simhei.ttf"
VLLM_API_URL = "http://127.0.0.1:8118/v1"
VLLM_API_KEY = "EMPTY"

for d in [JSON_DIR, IMG_DIR, TEMP_DIR]: os.makedirs(d, exist_ok=True)

# ================= 辅助函数 =================
def preprocess_image_enhanced(img_path, temp_save_path):
    img = cv2.imread(img_path)
    if img is None: return False
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l)
    img_contrast = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(img_contrast, (0, 0), 3.0)
    img_sharp = cv2.addWeighted(img_contrast, 1.5, gaussian, -0.5, 0, img_contrast)
    cv2.imwrite(temp_save_path, img_sharp)
    return True

def is_meaningful_text(text: str) -> bool:
    if not text: return False
    s = "".join(ch for ch in text if not ch.isspace())
    if not s or all(ch in "口□■▢▣▤▥▦▧▨▩▪▫◻◼◽◾☐☑☒" for ch in s): return False
    if all(unicodedata.category(ch).startswith(("P", "S")) for ch in s): return False
    return any(unicodedata.category(ch).startswith(("L", "N")) for ch in s)

def draw_ocr_result(original_img_path, json_data, save_img_path):
    img_cv = cv2.imdecode(np.fromfile(original_img_path, dtype=np.uint8), -1)
    if img_cv is None: return
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for item in json_data.get("parsing_res_list", []):
        text, bbox = item.get("block_content", ""), item.get("block_bbox", [])
        if len(bbox) == 4 and text:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h = y2 - y1
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            try: font = ImageFont.truetype(FONT_PATH, max(12, min(40, int(h * 0.6))))
            except: font = ImageFont.load_default()
            
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            ty = y1 - (text_bbox[3] - text_bbox[1]) - 5
            ty = ty if ty >= 0 else y1 + 5
            
            for o in [(1,1), (-1,-1), (1,-1), (-1,1)]: # 描边
                draw.text((x1+o[0], ty+o[1]), text, font=font, fill=(255,255,255))
            draw.text((x1, ty), text, fill=(255, 0, 0), font=font)
            
    cv2.imencode('.jpg', cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))[1].tofile(save_img_path)

def encode_image(path):
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')

# ================= 主逻辑 =================
client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_URL)
try: MODEL_NAME = client.models.list().data[0].id
except: MODEL_NAME = "paddleocr/PP-DocBee-2B"

PROMPT = """请分析图片，提取所有可见文字。
必须输出纯JSON，根对象为 "parsing_res_list" 列表。
每个项包含:
- "block_content": 文字内容
- "block_bbox": [xmin, ymin, xmax, ymax] (像素坐标整数)
注意：语义连续的文字可能不在同一行，但也需要合并，不要输出markdown标记。"""

try:
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')): continue
        print(f"正在处理: {filename}")
        
        orig_path = os.path.join(INPUT_DIR, filename)
        temp_path = os.path.join(TEMP_DIR, filename)
        
        if preprocess_image_enhanced(orig_path, temp_path):
            try:
                # 1. 调用 OpenAI 接口
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"<image>\n{PROMPT}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(temp_path)}"}}
                        ]
                    }],
                    max_tokens=2048, temperature=0.01
                )
                res_text = response.choices[0].message.content

                # 2. 解析 JSON (处理可能的 markdown 包裹)
                try:
                    clean_str = re.sub(r'^```json\s*|\s*```$', '', res_text.strip(), flags=re.MULTILINE)
                    raw_data = json.loads(clean_str)
                except:
                    # 尝试正则提取
                    match = re.search(r'\{.*\}', res_text, re.DOTALL)
                    raw_data = json.loads(match.group(0)) if match else {"parsing_res_list": []}

                # 3. 数据清洗与过滤
                clean_data = {"input_path": orig_path, "parsing_res_list": []}
                # 兼容列表直接返回或字典返回
                items = raw_data if isinstance(raw_data, list) else raw_data.get("parsing_res_list", [])
                
                for item in items:
                    label = item.get("block_label", item.get("label", "text")).lower()
                    content = item.get("block_content", item.get("text", ""))
                    bbox = item.get("block_bbox", item.get("box", []))
                    
                    if "image" in label: continue
                    if not is_meaningful_text(content): continue

                    clean_data["parsing_res_list"].append({
                        "block_content": content,
                        "block_label": label,
                        "block_bbox": bbox
                    })

                # 4. 保存结果
                base_name = os.path.splitext(filename)[0]
                json_path = os.path.join(JSON_DIR, f"{base_name}.json")
                img_out_path = os.path.join(IMG_DIR, f"{base_name}_result.jpg")
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(clean_data, f, ensure_ascii=False, indent=4)
                
                draw_ocr_result(orig_path, clean_data, img_out_path)
                print(f"  [完成] -> {img_out_path}")

            except Exception as e:
                print(f"  [失败] {e}")
finally:
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)