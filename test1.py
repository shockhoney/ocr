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
TEMP_DIR = "temp_processed" 

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ================= 2. 优化的预处理逻辑 =================
def preprocess_image_smart(img_path, temp_save_path):
    """
    智能预处理：
    1. 分辨率标准化: 将宽度统一调整到 1280px (大模型的最佳视野范围)，防止大字破碎或小字模糊。
    2. 微弱增强: 使用极低阈值的 CLAHE 增强局部对比度，只提炼轮廓，不破坏原图。
    """
    img = cv2.imread(img_path)
    if img is None: return False

    h, w = img.shape[:2]
    
    # --- 策略A: 分辨率标准化 (Target Width = 1280) ---
    target_width = 1280
    scale = target_width / w
    
    # 只有当尺寸差异较大时(小于0.8倍或大于1.2倍)才缩放
    # 这样可以修复图3文字破碎的问题(原图可能很大，缩小后文字更紧凑)，也能修复小图看不清的问题
    if scale < 0.8 or scale > 1.2:
        new_w = target_width
        new_h = int(h * scale)
        # 缩小用 INTER_AREA (更清晰)，放大用 INTER_LINEAR
        method = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (new_w, new_h), interpolation=method)

    # --- 策略B: 微弱的 CLAHE (局部对比度增强) ---
    # 转换到 LAB 空间，只处理 L (亮度) 通道
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # clipLimit=1.2 是非常保守的值 (默认通常是2.0-4.0)
    # 既能让 "Northern Lights" 这种浅色字显形，又不会让背景产生噪点干扰标准文字
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # 合并回 BGR
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    cv2.imwrite(temp_save_path, img)
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

# ================= 4. 模型初始化与 Prompt =================
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

# 提示词：强调对艺术字体的关注
prompt = "请对图片进行版面分析，提取所有可见的文字区域。特别注意识别海报中的艺术标题（如变形字体）和底部小字。准确输出坐标并合并语义连续的文本行。"

# ================= 5. 处理流程 =================
def process_single_result(res, filename, original_path):
    base_name = os.path.splitext(filename)[0]
    
    # 保存图片并清理
    res.save_to_img(IMG_DIR)
    order_img = os.path.join(IMG_DIR, f"{base_name}_layout_order_res.png")
    if os.path.exists(order_img): os.remove(order_img)

    # 保存 JSON
    res.save_to_json(JSON_DIR)
    json_path = os.path.join(JSON_DIR, f"{base_name}_res.json")
    if not os.path.exists(json_path): json_path = os.path.join(JSON_DIR, f"{base_name}.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        clean_data = {"input_path": original_path, "parsing_res_list": []}
        exclude_keys = {"block_id", "block_order", "block_label", "group_id"}

        if "parsing_res_list" in raw_data:
            for item in raw_data["parsing_res_list"]:
                label = item.get("block_label", "text").lower()
                content = item.get("block_content", "")

                if "image" in label: continue
                if not is_meaningful_text(content): continue

                clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
                clean_data["parsing_res_list"].append(clean_item)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
        print(f"  [完成] {filename}")

    except Exception as e:
        print(f"  [JSON错误] {e}")

# ================= 6. 主循环 =================
try:
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            original_path = os.path.join(INPUT_DIR, filename)
            temp_path = os.path.join(TEMP_DIR, filename)
            
            print(f"正在处理: {filename}")
            
            # 使用智能预处理
            if preprocess_image_smart(original_path, temp_path):
                try:
                    output = pipeline.predict(temp_path, prompt=prompt)
                    for res in output:
                        process_single_result(res, filename, original_path)
                except Exception as e:
                    print(f"  [预测失败] {e}")
finally:
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
