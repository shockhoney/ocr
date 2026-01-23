import os
import json
import cv2
import numpy as np
import shutil
import unicodedata
from paddleocr import PaddleOCRVL

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
OUTPUT_BASE = "output_result"
JSON_DIR = os.path.join(OUTPUT_BASE, "json")
IMG_DIR = os.path.join(OUTPUT_BASE, "image")
TEMP_DIR = "temp_processed"  # 临时文件夹

# 自动创建目录
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ================= 2. 温和的预处理逻辑 (修改重点) =================
def preprocess_image_mild(img_path, temp_save_path):
    """
    温和预处理：
    1. 仅对过小的图片进行轻微放大 (1.5倍)。
    2. 仅做 Gamma 校正 (提亮)，不做锐化和强对比度增强。
    """
    img = cv2.imread(img_path)
    if img is None:
        return False

    h, w = img.shape[:2]

    # --- 策略A: 智能放大 ---
    # 之前是无脑 2倍，现在改为：只有宽度小于 1000 像素的小图才放大
    # 大模型通常在 1000px - 2000px 范围内效果最好
    if w < 1000:
        scale_factor = 1.5  # 降低放大倍数
        # 使用线性插值，比立方插值更平滑，噪点更少
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # --- 策略B: Gamma 校正 (替代 CLAHE) ---
    # 很多海报字体检测不到是因为暗部细节丢失。
    # Gamma < 1.0 会在不增加噪点的情况下提亮暗部。
    # 0.9 是一个非常保守的值，只做轻微提亮，保持原图风味。
    gamma = 0.9
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)
    img = cv2.LUT(img, lookUpTable)

    # --- 移除了 锐化 和 CLAHE 操作 ---
    
    cv2.imwrite(temp_save_path, img)
    return True

# ================= 3. 文本过滤逻辑 (保持不变) =================
PLACEHOLDER_CHARS = set("口□■▢▣▤▥▦▧▨▩▪▫◻◼◽◾☐☑☒")

def is_meaningful_text(text: str) -> bool:
    if not text: return False
    s = "".join(ch for ch in text if not ch.isspace())
    if not s: return False
    # 过滤纯符号/纯占位符
    if all(ch in PLACEHOLDER_CHARS for ch in s): return False
    if all(unicodedata.category(ch).startswith(("P", "S")) for ch in s): return False
    # 必须包含字母或数字(含中文)
    if any(unicodedata.category(ch).startswith(("L", "N")) for ch in s): return True
    return False

# ================= 4. 初始化模型 =================
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

# 提示词微调：去掉过于具体的描述，让模型自由发挥，防止过度关注某类特征
prompt = "请对图片进行版面分析，提取所有可见的文字区域。识别标题、正文及海报中的装饰性文字。准确输出每个文字区域的坐标，并合并语义连续的文本行。"

# ================= 5. 处理函数 =================
def process_single_result(res, filename, original_path):
    base_name = os.path.splitext(filename)[0]
    
    # --- A. 图片处理 ---
    res.save_to_img(IMG_DIR)
    order_img_path = os.path.join(IMG_DIR, f"{base_name}_layout_order_res.png")
    if os.path.exists(order_img_path):
        os.remove(order_img_path)

    # --- B. JSON处理 ---
    res.save_to_json(JSON_DIR)
    
    json_path = os.path.join(JSON_DIR, f"{base_name}_res.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(JSON_DIR, f"{base_name}.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        clean_data = {
            "input_path": original_path, # 记录原图路径
            "parsing_res_list": []
        }

        exclude_keys = {"block_id", "block_order", "block_label", "group_id"}

        if "parsing_res_list" in raw_data:
            for item in raw_data["parsing_res_list"]:
                label = item.get("block_label", "text").lower()
                content = item.get("block_content", "")

                # 过滤逻辑
                if "image" in label: continue
                if not is_meaningful_text(content): continue

                # 数据清洗
                clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
                clean_data["parsing_res_list"].append(clean_item)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
            
        print(f"  [完成] {filename}")

    except Exception as e:
        print(f"  [错误] {e}")

# ================= 6. 主循环 =================
try:
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            original_path = os.path.join(INPUT_DIR, filename)
            temp_path = os.path.join(TEMP_DIR, filename)
            
            # 1. 预处
