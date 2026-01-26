import os
import json
import base64
import requests
import cv2
import numpy as np
import unicodedata

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
JSON_DIR = os.path.join("output_result", "json")
IMG_DIR = os.path.join("output_result", "image")
TEMP_DIR = os.path.join("output_result", "temp_preprocessed")

# VLLM API 配置
API_URL = "http://127.0.0.1:8118/v1/chat/completions"
MODEL_NAME = "PaddleOCR-VL-0.9B" # 请确保与 vllm 启动时的 model 参数一致

# 提示词
PROMPT_TEXT = (
    "请对图片进行版面分析，识别并提取所有可见的文字区域。注意文字可能具有不一致的字体大小，"
    "需根据内容连续性进行合理合并。输出时应准确标注每个文字区域的文本框坐标（bounding box），"
    "并确保语义连续的文字被包含在同一个文本框中。返回结果必须是标准的JSON格式，包含 parsing_res_list 列表。"
)

for d in [JSON_DIR, IMG_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ================= 2. 图像预处理与工具 =================

def preprocess_image_enhanced(img_path, temp_save_path):
    """
    策略：不缩放 + CLAHE对比度增强 + USM锐化
    """
    img = cv2.imread(img_path)
    if img is None: return False

    # A: LAB 空间 CLAHE 增强
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_contrast = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # B: USM 锐化
    gaussian = cv2.GaussianBlur(img_contrast, (0, 0), 3.0)
    img_sharp = cv2.addWeighted(img_contrast, 1.5, gaussian, -0.5, 0, img_contrast)

    cv2.imwrite(temp_save_path, img_sharp)
    return True

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def draw_visualization(img_path, json_data, save_path):
    """手动绘制可视化结果替代 save_to_img"""
    img = cv2.imread(img_path)
    if img is None: return
    
    for item in json_data.get("parsing_res_list", []):
        bbox = item.get("block_bbox", [])
        if len(bbox) == 4:
            # 坐标格式通常是 [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imwrite(save_path, img)

# ================= 3. 文本过滤逻辑 =================
PLACEHOLDER_CHARS = set("口□■▢▣▤▥▦▧▨▩▪▫◻◼◽◾☐☑☒")

def is_meaningful_text(text: str) -> bool:
    if not text: return False
    s = "".join(ch for ch in text if not ch.isspace())
    if not s: return False
    if all(ch in PLACEHOLDER_CHARS for ch in s): return False
    if all(unicodedata.category(ch).startswith(("P", "S")) for ch in s): return False
    if any(unicodedata.category(ch).startswith(("L", "N")) for ch in s): return True
    return False

# ================= 4. API 调用核心函数 =================
def call_openai_vllm(image_path):
    base64_image = encode_image_to_base64(image_path)
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEXT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0  # OCR任务建议低温度
    }

    try:
        response = requests.post(API_URL, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # 简单清洗 Markdown 代码块标记
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        return json.loads(content.strip())
    except Exception as e:
        print(f"  [API错误] {e}")
        return None

# ================= 5. 主循环 =================
if __name__ == "__main__":
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            continue

        base_name = os.path.splitext(filename)[0]
        original_img_path = os.path.join(INPUT_DIR, filename)
        processed_img_path = os.path.join(TEMP_DIR, f"pre_{filename}")

        print(f"正在处理: {filename} ...")

        # 1. 图像增强预处理
        if not preprocess_image_enhanced(original_img_path, processed_img_path):
            print(f"  [跳过] 图像读取失败")
            continue

        # 2. 调用 API 获取结果
        raw_json_data = call_openai_vllm(processed_img_path)
        
        if not raw_json_data or "parsing_res_list" not in raw_json_data:
            print(f"  [失败] 模型未返回有效JSON数据")
            continue

        # 3. 数据清洗与过滤
        clean_res_list = []
        exclude_keys = {"block_id", "block_order", "block_label", "group_id"}
        
        for item in raw_json_data["parsing_res_list"]:
            label = item.get("block_label", "text").lower()
            content = item.get("block_content", "")

            # 过滤 image 类型
            if "image" in label:
                continue
            # 过滤无意义文本
            if not is_meaningful_text(content):
                continue
            
            # 构建保留项
            clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
            clean_res_list.append(clean_item)

        final_data = {
            "input_path": filename,
            "parsing_res_list": clean_res_list
        }

        # 4. 保存 JSON
        json_save_path = os.path.join(JSON_DIR, f"{base_name}.json")
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        # 5. 绘制结果图 (使用增强后的图或原图均可，这里用增强图方便看清楚)
        img_save_path = os.path.join(IMG_DIR, f"{base_name}_vis.jpg")
        draw_visualization(processed_img_path, final_data, img_save_path)

        print(f"  [完成] 结果已保存至 {JSON_DIR}")

    # 清理临时文件夹 (可选)
    # import shutil
    # shutil.rmtree(TEMP_DIR)