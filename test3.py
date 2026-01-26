import os
import json
import cv2
import numpy as np
import shutil
import unicodedata
import base64
import re  # [新增] 用于正则提取JSON
from PIL import Image, ImageDraw, ImageFont 
from openai import OpenAI

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
OUTPUT_BASE = "output_result"
JSON_DIR = os.path.join(OUTPUT_BASE, "json")
IMG_DIR = os.path.join(OUTPUT_BASE, "image")
TEMP_DIR = "temp_processed" 
FONT_PATH = "simhei.ttf" 

# [重要] 请根据你右侧日志显示的实际端口修改此处！
# 截图右下角显示的似乎是 43262，右上角是 41362，请以启动命令为准
VLLM_API_URL = "http://127.0.0.1:8118/v1" 
VLLM_API_KEY = "EMPTY"

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ================= 2. 增强型预处理 (保持不变) =================
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

# ================= 3. 过滤与画框逻辑 (保持不变) =================
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
                w = x2 - x1
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

# ================= 4. [修改] MockPaddleResult 与 JSON 提取 =================
def extract_json_from_text(text):
    """
    使用正则表达式提取字符串中第一个有效的 JSON 对象
    """
    try:
        # 寻找第一个 { 和最后一个 }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            # 尝试直接解析
            return json.loads(text)
    except Exception:
        return None

class MockPaddleResult:
    def __init__(self, json_content_str, base_name):
        self.raw_text = json_content_str
        self.base_name = base_name

    def save_to_json(self, save_dir):
        file_path = os.path.join(save_dir, f"{self.base_name}_res.json")
        
        # 尝试提取 JSON
        data = extract_json_from_text(self.raw_text)
        
        if data:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            # [重要] 如果解析失败，把原始文本保存下来，方便你查看原因
            print(f"  [ERROR] JSON解析失败，已保存原始返回内容到 {file_path}")
            print(f"  [DEBUG] 模型原始返回片段: {self.raw_text[:100]}...") 
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.raw_text)

    def save_to_img(self, save_dir):
        pass

# ================= 5. OpenAI Client =================
client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_URL)

try:
    # 尝试获取模型列表
    models = client.models.list()
    MODEL_NAME = models.data[0].id
    print(f"✅ 连接成功，使用模型: {MODEL_NAME}")
except Exception as e:
    print(f"⚠️ 获取模型失败，使用默认名。错误: {e}")
    MODEL_NAME = "paddleocr/PP-DocBee-2B"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ================= 6. 处理流程 (增加调试打印) =================
def process_single_result(res, filename, original_path):
    base_name = os.path.splitext(filename)[0]
    
    # 1. 保存原始 JSON (这里会触发上面的 save_to_json 逻辑)
    res.save_to_json(JSON_DIR)
    
    json_path = os.path.join(JSON_DIR, f"{base_name}_res.json")
    if not os.path.exists(json_path): json_path = os.path.join(JSON_DIR, f"{base_name}.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            # 尝试读取，如果之前保存的是非JSON的文本，这里会再次报错
            file_content = f.read()
            
        try:
            raw_data = json.loads(file_content)
        except json.JSONDecodeError:
            print(f"  [跳过] 文件内容不是有效JSON，无法继续处理。")
            return

        # 后续清洗逻辑
        clean_data = {"input_path": original_path, "parsing_res_list": []}
        exclude_keys = {"block_id", "block_order", "group_id"} 

        if "parsing_res_list" in raw_data:
            for item in raw_data["parsing_res_list"]:
                label = item.get("block_label", item.get("label", "text")).lower()
                content = item.get("block_content", item.get("text", ""))
                bbox = item.get("block_bbox", item.get("box", []))
                
                # 规范化 item
                item["block_content"] = content
                item["block_bbox"] = bbox
                item["block_label"] = label

                if "image" in label: continue
                if not is_meaningful_text(content): continue

                clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
                clean_data["parsing_res_list"].append(clean_item)

        # 保存清洗结果
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
        
        # 画图
        final_img_path = os.path.join(IMG_DIR, f"{base_name}_result.jpg")
        draw_ocr_result(original_path, clean_data, final_img_path)
        print(f"  [完成] {filename}")

    except Exception as e:
        print(f"  [处理异常] {e}")

# ================= 7. 主循环 (合并 Prompt) =================
# [修改] 将所有指令放入 User Prompt，提高本地模型依从性
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
2. 合并同一行的文字。
"""

try:
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            original_path = os.path.join(INPUT_DIR, filename)
            temp_path = os.path.join(TEMP_DIR, filename)
            
            print(f"\n正在处理: {filename}")
            
            if preprocess_image_enhanced(original_path, temp_path):
                try:
                    base64_image = encode_image(temp_path)
                    
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            # 许多本地 VLM 忽略 system prompt，所以这里只用 user
                            {"role": "user", "content": [
                                {"type": "text", "text": FULL_PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]}
                        ],
                        temperature=0.1, # 极低温度，让输出更稳定
                        max_tokens=2048
                    )
                    
                    result_text = response.choices[0].message.content
                    
                    # [调试] 打印部分返回内容，确认是否为空
                    print(f"  [API返回] 长度: {len(result_text) if result_text else 0}")
                    
                    mock_res = MockPaddleResult(result_text, os.path.splitext(filename)[0])
                    process_single_result(mock_res, filename, original_path)

                except Exception as e:
                    print(f"  [API调用失败] {e}")
finally:
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)