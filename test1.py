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

# ================= 2. 增强型预处理 (无缩放 + 强对比 + USM锐化) =================
def preprocess_image_enhanced(img_path, temp_save_path):
    """
    策略：
    1. 不缩放：保持原图分辨率，确保大字不破碎，小字不模糊。
    2. CLAHE (2.0): 标准强度对比度增强，让隐蔽文字显形。
    3. USM 锐化: 强化边缘，专门针对艺术字和细体字。
    """
    img = cv2.imread(img_path)
    if img is None: return False

    # --- 步骤A: 颜色空间转换与对比度增强 ---
    # 转换到 LAB 空间，仅处理亮度通道 (L)，防止色彩失真
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 提升 CLAHE 阈值到 2.0 (标准强度)
    # 这会比之前的 1.2 更强力地拉开文字和背景的差距
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # 合并回 BGR
    img_contrast = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # --- 步骤B: USM 锐化 (Unsharp Masking) ---
    # 原理: 原图 + (原图 - 模糊图) * amount
    # 这种锐化方式比普通卷积核更自然，适合处理海报上的艺术字
    gaussian = cv2.GaussianBlur(img_contrast, (0, 0), 3.0)
    # 权重 1.5 表示中等强度的锐化
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

# ================= 4. 模型初始化 =================
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

# 提示词：去掉繁琐描述，简明扼要
prompt = "请对图片进行版面分析，提取所有可见的文字区域，准确输出坐标并合并语义连续的文本行。"

# ================= 5. 处理流程 =================
def process_single_result(res, filename, original_path):
    base_name = os.path.splitext(filename)[0]
    
    # 图片处理
    res.save_to_img(IMG_DIR)
    order_img = os.path.join(IMG_DIR, f"{base_name}_layout_order_res.png")
    if os.path.exists(order_img): os.remove(order_img)

    # JSON处理
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
            
            # 使用增强型预处理 (无缩放)
            if preprocess_image_enhanced(original_path, temp_path):
                try:
                    output = pipeline.predict(temp_path, prompt=prompt)
                    for res in output:
                        process_single_result(res, filename, original_path)
                except Exception as e:
                    print(f"  [预测失败] {e}")
finally:
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
