import os
import json
import cv2
import numpy as np
import shutil
import unicodedata
import base64
import requests
import re

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
OUTPUT_BASE = "output_result"
JSON_DIR = os.path.join(OUTPUT_BASE, "json")
IMG_DIR = os.path.join(OUTPUT_BASE, "image")
TEMP_DIR = "temp_processed" 

# VLLM 服务配置 (请确认模型名称与启动参数一致)
API_URL = "http://127.0.0.1:8118/v1/chat/completions"
MODEL_NAME = "PaddleOCR-VL-0.9B"

# 确保文件夹存在
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ================= 2. 增强型预处理 (保持不变) =================
def preprocess_image_enhanced(img_path, temp_save_path):
    """
    策略：不缩放 + CLAHE(2.0) + USM锐化
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

# ================= 3. 过滤逻辑 (保持不变) =================
PLACEHOLDER_CHARS = set("口□■▢▣▤▥▦▧▨▩▪▫◻◼◽◾☐☑☒")
def is_meaningful_text(text: str) -> bool:
    if not text: return False
    s = "".join(ch for ch in text if not ch.isspace())
    if not s: return False
    if all(ch in PLACEHOLDER_CHARS for ch in s): return False
    if all(unicodedata.category(ch).startswith(("P", "S")) for ch in s): return False
    if any(unicodedata.category(ch).startswith(("L", "N")) for ch in s): return True
    return False

# ================= 4. API 调用与辅助函数 =================

def encode_image(image_path):
    """将图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_json(content):
    """从模型回复中提取 JSON，兼容 ```json 代码块"""
    content = content.strip()
    # 尝试正则提取 ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        content = match.group(1)
    
    # 尝试寻找首尾的大括号
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1:
        content = content[start:end+1]
        
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("  [解析警告] 无法解析为JSON，原始内容片段:", content[:50])
        return None

def call_openai_api(image_path):
    """调用 VLLM 的 OpenAI 兼容接口"""
    base64_img = encode_image(image_path)
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请对图片进行版面分析，提取所有可见的文字区域，准确输出坐标并合并语义连续的文本行。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }
        ],
        "temperature": 0.0, # OCR 任务不需要随机性
        "max_tokens": 4096
    }
    
    try:
        response = requests.post(API_URL, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        res_json = response.json()
        if 'choices' in res_json and len(res_json['choices']) > 0:
            content = res_json['choices'][0]['message']['content']
            return extract_json(content)
        return None
    except Exception as e:
        print(f"  [API错误] {e}")
        return None

def draw_visualization(img_path, json_data, save_path):
    """手动绘制文本框 (替代 res.save_to_img)"""
    img = cv2.imread(img_path)
    if img is None: return

    if "parsing_res_list" in json_data:
        for item in json_data["parsing_res_list"]:
            bbox = item.get("block_bbox", [])
            # 格式通常是 [xmin, ymin, xmax, ymax]
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                # 画红框，线宽2
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imwrite(save_path, img)

# ================= 5. 主循环 =================
if __name__ == "__main__":
    try:
        for filename in os.listdir(INPUT_DIR):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue

            original_path = os.path.join(INPUT_DIR, filename)
            temp_path = os.path.join(TEMP_DIR, filename)
            base_name = os.path.splitext(filename)[0]

            print(f"正在处理: {filename} ...")
            
            # 1. 预处理 (结果存入 temp)
            if not preprocess_image_enhanced(original_path, temp_path):
                print("  [跳过] 图片读取失败")
                continue

            # 2. 调用 API
            raw_data = call_openai_api(temp_path)
            
            if not raw_data or "parsing_res_list" not in raw_data:
                print("  [失败] 模型未返回有效数据")
                continue

            # 3. 数据清洗与过滤
            clean_res_list = []
            exclude_keys = {"block_id", "block_order", "block_label", "group_id"}

            for item in raw_data["parsing_res_list"]:
                label = item.get("block_label", "text").lower()
                content = item.get("block_content", "")

                # 过滤逻辑
                if "image" in label: continue
                if not is_meaningful_text(content): continue

                # 构建保留项
                clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
                clean_res_list.append(clean_item)

            final_json = {
                "input_path": original_path, # 记录原图路径
                "parsing_res_list": clean_res_list
            }

            # 4. 保存 JSON
            json_save_path = os.path.join(JSON_DIR, f"{base_name}.json")
            with open(json_save_path, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, ensure_ascii=False, indent=4)

            # 5. 生成可视化图片 (基于增强后的图片画框，方便核对)
            img_save_path = os.path.join(IMG_DIR, f"{base_name}_vis.jpg")
            draw_visualization(temp_path, final_json, img_save_path)
            
            print(f"  [完成] 结果已保存")

    finally:
        # 程序结束后清理临时文件
        if os.path.exists(TEMP_DIR):
            try:
                shutil.rmtree(TEMP_DIR)
            except:
                pass