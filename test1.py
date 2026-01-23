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
TEMP_DIR = "temp_processed"  # 临时存放预处理图片的文件夹

# 自动创建目录
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ================= 2. 图像预处理函数 (新增核心功能) =================
def preprocess_image(img_path, temp_save_path):
    """
    预处理步骤：
    1. 放大: 提高小字的分辨率
    2. CLAHE: 增强局部对比度 (让文字从背景浮现)
    3. 锐化: 让艺术字边缘更清晰
    """
    img = cv2.imread(img_path)
    if img is None:
        return False

    # A. 自动放大 (如果图片宽度小于 1600，放大 2 倍)
    # 大模型对高分辨率输入的细节捕捉能力更强
    h, w = img.shape[:2]
    if w < 1600:
        scale_factor = 2.0
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # B. 转换到 LAB 色彩空间进行 CLAHE 增强 (只增强亮度通道 L，保持色彩)
    # 相比直接转灰度，这样能保留海报的颜色信息，对 VLM 理解语义更有帮助
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 应用 CLAHE (限制对比度自适应直方图均衡化)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # 合并通道
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # C. 边缘锐化 (让模糊的文字变清晰)
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    # 保存预处理后的图片
    cv2.imwrite(temp_save_path, img)
    return True

# ================= 3. 定义文本过滤逻辑 =================
PLACEHOLDER_CHARS = set("口□■▢▣▤▥▦▧▨▩▪▫◻◼◽◾☐☑☒")

def is_meaningful_text(text: str) -> bool:
    if not text: return False
    s = "".join(ch for ch in text if not ch.isspace())
    if not s: return False
    if all(ch in PLACEHOLDER_CHARS for ch in s): return False
    if all(unicodedata.category(ch).startswith(("P", "S")) for ch in s): return False
    if any(unicodedata.category(ch).startswith(("L", "N")) for ch in s): return True
    return False

# ================= 4. 初始化模型 =================
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

# 优化提示词：增加对艺术字和标题的强调
prompt = "请对图片进行版面分析。注意识别海报中艺术化、变形或断裂的标题文字（如 'FUTURE' 等设计字体），同时识别底部的小字信息。准确输出每个文字区域的文本框坐标。过滤掉非文字的装饰图案。"

# ================= 5. 核心处理流程 =================
def process_single_result(res, filename, original_path):
    base_name = os.path.splitext(filename)[0]
    
    # --- A. 图片处理 ---
    res.save_to_img(IMG_DIR)
    # 删除多余的排序图
    order_img_path = os.path.join(IMG_DIR, f"{base_name}_layout_order_res.png")
    if os.path.exists(order_img_path):
        os.remove(order_img_path)

    # --- B. JSON处理 ---
    res.save_to_json(JSON_DIR)
    
    # 路径修正
    json_path = os.path.join(JSON_DIR, f"{base_name}_res.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(JSON_DIR, f"{base_name}.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        clean_data = {
            "input_path": original_path, # 重要：写回原图路径，而不是临时处理图的路径
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
        print(f"  [JSON错误] {e}")


# ================= 6. 主循环 =================
try:
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            original_path = os.path.join(INPUT_DIR, filename)
            temp_path = os.path.join(TEMP_DIR, filename) # 临时文件路径
            
            print(f"正在处理: {filename}")

            # 1. 执行预处理
            if preprocess_image(original_path, temp_path):
                # 2. 将预处理后的图片传给模型
                # 注意：这里传的是 temp_path (对比度更高、更清晰的大图)
                try:
                    output = pipeline.predict(temp_path, prompt=prompt)
                    for res in output:
                        # 3. 处理结果 (传入原始路径用于JSON记录)
                        process_single_result(res, filename, original_path)
                except Exception as e:
                    print(f"  [预测失败] {e}")
            else:
                print(f"  [跳过] 图片读取失败")

finally:
    # (可选) 程序结束后清理临时文件夹
    # if os.path.exists(TEMP_DIR):
    #     shutil.rmtree(TEMP_DIR)
    pass
