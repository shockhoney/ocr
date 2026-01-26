import os
import json
import cv2
import numpy as np
import shutil
import unicodedata
import base64
from PIL import Image, ImageDraw, ImageFont 
from openai import OpenAI  

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
OUTPUT_BASE = "output_result"
JSON_DIR = os.path.join(OUTPUT_BASE, "json")
IMG_DIR = os.path.join(OUTPUT_BASE, "image")
TEMP_DIR = "temp_processed" 
FONT_PATH = "simhei.ttf" # 字体路径

# OpenAI / vLLM 配置
VLLM_API_URL = "http://127.0.0.1:8118/v1"
VLLM_API_KEY = "EMPTY"  

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ================= 2. 数据预处理 =================
def preprocess_image_enhanced(img_path, temp_save_path):
    """
    策略：不缩放 + CLAHE(1.6) + USM锐化
    """
    img = cv2.imread(img_path)
    if img is None: return False

    # A: 颜色空间转换与对比度增强
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    img_contrast = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # B: USM 锐化
    gaussian = cv2.GaussianBlur(img_contrast, (0, 0), 3.0)
    img_sharp = cv2.addWeighted(img_contrast, 1.5, gaussian, -0.5, 0, img_contrast)

    cv2.imwrite(temp_save_path, img_sharp)
    return True

# ================= 3. 过滤逻辑 =================
PLACEHOLDER_CHARS = set("口□■▢▣▤▥▦▧▨▩▪▫◻◼◽◾☐☑☒")
def is_meaningful_text(text: str) -> bool:
    if not text: return False
    s = "".join(ch for ch in text if not ch.isspace())
    if not s: return False
    if all(ch in PLACEHOLDER_CHARS for ch in s): return False
    if all(unicodedata.category(ch).startswith(("P", "S")) for ch in s): return False
    if any(unicodedata.category(ch).startswith(("L", "N")) for ch in s): return True
    return False

# ================= 4. 画框逻辑函数 =================
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
                
                # 画框
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=1)
                
                # 画文字
                font_size = max(12, min(40, int(h * 0.6)))
                try:
                    font = ImageFont.truetype(FONT_PATH, font_size)
                except:
                    font = ImageFont.load_default()
                
                text_bbox = draw.textbbox((x1, y1), text, font=font) 
                text_pixel_height = text_bbox[3] - text_bbox[1]
                
                text_y = y1 - text_pixel_height - 5
                if text_y < 0: text_y = y1 + 5

                # 描边 + 文字
                draw.text((x1-1, text_y), text, font=font, fill=(255,255,255))
                draw.text((x1+1, text_y), text, font=font, fill=(255,255,255))
                draw.text((x1, text_y-1), text, font=font, fill=(255,255,255))
                draw.text((x1, text_y+1), text, font=font, fill=(255,255,255))
                draw.text((x1, text_y), text, fill=(255, 0, 0), font=font)

    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imencode('.jpg', img_result)[1].tofile(save_img_path)

# ================= 5.  模拟 Paddle 结果对象 =================
class MockPaddleResult:
    """
    这个类用于欺骗 process_single_result 函数，
    让它以为自己处理的是 PaddleOCRVL 的结果，
    从而复用你现有的保存和清洗逻辑。
    """
    def __init__(self, json_content, base_name):
        self.json_content = json_content
        self.base_name = base_name

    def save_to_json(self, save_dir):
        """模拟 Paddle 的 save_to_json 行为"""
        file_path = os.path.join(save_dir, f"{self.base_name}_res.json")
        try:
            # 清洗 markdown 标记，确保是纯 JSON
            clean_str = self.json_content.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_str)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            # 如果解析失败，强行保存原始内容供调试
            print(f"  [Warning] JSON解析失败，保存原始文本: {e}")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.json_content)

    def save_to_img(self, save_dir):
        """
        因为我们已经有了 draw_ocr_result 自定义画图，
        这里留空即可，避免报错。
        """
        pass

# ================= 6. [修改] OpenAI API 初始化与提示词 =================
client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_URL)

# 获取模型名称 (自动获取列表中的第一个模型)
MODEL_NAME = client.models.list().data[0].id

# 系统提示词：定义角色和严格的 JSON 输出格式
SYSTEM_PROMPT = """你是一个专业的OCR文档分析助手。
你的任务是识别图片中的文字，并进行版面分析。
请严格按照以下 JSON 格式输出，不要输出任何额外的解释：
{
  "parsing_res_list": [
    {
      "block_content": "文字内容",
      "block_bbox": [x1, y1, x2, y2]
    }
  ]
}
注意：
1. block_bbox 必须是 [左上x, 左上y, 右下x, 右下y] 的像素坐标整数。
2. 不同行、不同字号、不同语义的文字块请分开输出。
"""

# 辅助函数：转Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ================= 7. 处理流程 =================
def process_single_result(res, filename, original_path):
    base_name = os.path.splitext(filename)[0]
    
    # 1. 保存 JSON (调用 Mock 对象的 save_to_json)
    res.save_to_json(JSON_DIR)
    
    # 2. 读取并清洗
    json_path = os.path.join(JSON_DIR, f"{base_name}_res.json")
    if not os.path.exists(json_path): json_path = os.path.join(JSON_DIR, f"{base_name}.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        clean_data = {"input_path": original_path, "parsing_res_list": []}
        exclude_keys = {"block_id", "block_order", "group_id"} # block_label 需要保留用来判断image

        if "parsing_res_list" in raw_data:
            for item in raw_data["parsing_res_list"]:
                # 兼容模型可能输出 label 而不是 block_label 的情况
                label = item.get("block_label", item.get("label", "text")).lower()
                content = item.get("block_content", item.get("text", ""))
                
                # 兼容模型输出 box 而不是 block_bbox
                bbox = item.get("block_bbox", item.get("box", []))
                # 确保 item 里有统一的 key 供后续画图使用
                item["block_content"] = content
                item["block_bbox"] = bbox
                item["block_label"] = label

                if "image" in label: continue
                if not is_meaningful_text(content): continue

                clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
                clean_data["parsing_res_list"].append(clean_item)

        # 保存清洗后的 JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
        
        # 3. 画图
        final_img_path = os.path.join(IMG_DIR, f"{base_name}_result.jpg")
        draw_ocr_result(original_path, clean_data, final_img_path)

        print(f"  [完成] {filename} -> IMG: {final_img_path}")

    except Exception as e:
        print(f"  [处理错误] {filename}: {e}")

# ================= 8. 主循环 =================
try:
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            original_path = os.path.join(INPUT_DIR, filename)
            temp_path = os.path.join(TEMP_DIR, filename)
            
            print(f"正在处理: {filename}")
            
            # 1. 预处理
            if preprocess_image_enhanced(original_path, temp_path):
                try:
                    # 2. 准备 OpenAI 请求
                    base64_image = encode_image(temp_path)
                    
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": [
                                {"type": "text", "text": '''请对图片进行版面分析，识别并提取所有可见的文字区域，包括水平、垂直和倾斜排列的文字。
                                       注意文字可能具有不一致的字体大小，需根据内容连续性进行合理合并。
                                输出时应准确标注每个文字区域的文本框坐标（bounding box），并确保语义连续的文字被包含在同一个文本框中,不要输出image块'''},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]}
                        ],
                        temperature=0.2, # 降低随机性
                        max_tokens=2048
                    )
                    
                    # 获取返回的文本
                    result_text = response.choices[0].message.content
                    
                    # 3. 封装成 MockResult，复用原有逻辑
                    base_name = os.path.splitext(filename)[0]
                    mock_res = MockPaddleResult(result_text, base_name)
                    
                    # 4. 进入原有的处理流程
                    process_single_result(mock_res, filename, original_path)

                except Exception as e:
                    print(f"  [API预测失败] {e}")
finally:
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)