import os
import json
import shutil
import unicodedata
import cv2  # 需要安装 opencv-python
import numpy as np
from paddleocr import PaddleOCRVL

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
OUTPUT_DIR = "output_result"
JSON_DIR = os.path.join(OUTPUT_DIR, "json")
IMG_DIR = os.path.join(OUTPUT_DIR, "image")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp_processed")  # 新增临时文件夹用于存放预处理图片

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ================= 2. 图像预处理函数 (新增) =================
def preprocess_poster_image(image_path, save_path):
    """
    针对海报文字提取的预处理：
    1. 转换到 LAB 空间对亮度通道做 CLAHE (自适应直方图均衡化)，增强文字与背景的对比度。
    2. 使用 USM (Unsharp Masking) 技术进行锐化，使艺术字体边缘更清晰。
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [警告] 无法读取图片: {image_path}")
        return False

    try:
        # --- 步骤 1: 增强对比度 (CLAHE) ---
        # 将图片转换为 LAB 色彩空间 (L: 亮度, A/B: 颜色通道)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 创建 CLAHE 对象 (Clip Limit 控制对比度强度，TileGridSize 控制局部大小)
        # 海报通常色彩丰富，clipLimit 设为 2.0 既能增强文字显现，又不会引入过多噪点
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # 合并通道并转回 BGR
        limg = cv2.merge((l, a, b))
        img_contrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # --- 步骤 2: 边缘锐化 (USM 锐化) ---
        # 原理：原图 + (原图 - 高斯模糊图) * 强度
        gaussian = cv2.GaussianBlur(img_contrast, (0, 0), 3.0)
        img_sharp = cv2.addWeighted(img_contrast, 1.5, gaussian, -0.5, 0, img_contrast)

        # 保存预处理后的图片
        cv2.imwrite(save_path, img_sharp)
        return True
    except Exception as e:
        print(f"  [预处理失败] {e}")
        return False

# ================= 3. 定义过滤逻辑 =================
PLACEHOLDER_CHARS = set("口□■▢▣▤▥▦▧▨▩▪▫◻◼◽◾☐☑☒")

def is_meaningful_text(text: str) -> bool:
    if not text:
        return False
    
    s = "".join(ch for ch in text if not ch.isspace())
    if not s:
        return False

    if all(ch in PLACEHOLDER_CHARS for ch in s):
        return False
    if all(unicodedata.category(ch).startswith(("P", "S")) for ch in s):
        return False
    if any(unicodedata.category(ch).startswith(("L", "N")) for ch in s):
        return True

    return False

# ================= 4. 初始化模型 =================
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

prompt = '''请对图片进行版面分析，识别并提取所有可见的文字区域，包括水平、垂直和倾斜排列的文字。
注意文字可能具有不一致的字体大小，需根据内容连续性进行合理合并。
输出时应准确标注每个文字区域的文本框坐标（bounding box），并确保语义连续的文字被包含在同一个文本框中。'''

# ================= 5. 核心处理函数 =================
def process_single_result(res, filename, original_input_path):
    base_name = os.path.splitext(filename)[0]
    
    # --- A. 图片处理 ---
    # PaddleOCRVL 会自动保存可视化的结果图
    res.save_to_img(IMG_DIR)
    
    # 清理多余的 layout_order 图片
    order_img_path = os.path.join(IMG_DIR, f"{base_name}_layout_order_res.png")
    if os.path.exists(order_img_path):
        os.remove(order_img_path)

    # --- B. JSON处理 ---
    res.save_to_json(JSON_DIR)
    
    # PaddleOCR 生成的 JSON 文件名可能不固定，通常是 base_name_res.json
    json_path = os.path.join(JSON_DIR, f"{base_name}_res.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(JSON_DIR, f"{base_name}.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 修正：将 JSON 中的 input_path 改回原始文件路径，而不是临时预处理文件的路径
        clean_data = {
            "input_path": original_input_path, 
            "parsing_res_list": []
        }

        exclude_keys = {"block_id", "block_order", "block_label", "group_id"}

        if "parsing_res_list" in raw_data:
            for item in raw_data["parsing_res_list"]:
                label = item.get("block_label", "text").lower()
                content = item.get("block_content", "")

                if "image" in label:
                    continue
                
                if not is_meaningful_text(content):
                    continue

                clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
                clean_data["parsing_res_list"].append(clean_item)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
            
        print(f"  [处理完成] {filename}")

    except Exception as e:
        print(f"  [错误] JSON处理出错: {e}")

# ================= 6. 主循环 =================
try:
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            original_img_path = os.path.join(INPUT_DIR, filename)
            
            # 定义临时预处理文件的路径 (保持文件名一致，方便后续逻辑)
            temp_img_path = os.path.join(TEMP_DIR, filename)
            
            print(f"正在处理: {filename} ...")
            
            # 1. 执行图像预处理
            success = preprocess_poster_image(original_img_path, temp_img_path)
            
            # 如果预处理成功，使用处理后的图片；否则使用原图
            target_path = temp_img_path if success else original_img_path
            
            try:
                # 2. 将(预处理后的)图片送入模型
                output = pipeline.predict(target_path, prompt=prompt)
                
                for res in output:
                    # 传入 original_img_path 是为了在最终 JSON 中记录真实的图片来源
                    process_single_result(res, filename, original_img_path)
                    
            except Exception as e:
                print(f"  [失败] 模型预测出错 {filename}: {e}")

finally:
    # (可选) 程序结束后清理临时文件夹
    # print("正在清理临时文件...")
    # shutil.rmtree(TEMP_DIR)
    pass