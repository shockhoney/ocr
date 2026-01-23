import os
import json
import cv2
import numpy as np
import shutil
import unicodedata
from PIL import Image, ImageDraw, ImageFont # 引入PIL用于绘制中文
from paddleocr import PaddleOCRVL

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
OUTPUT_BASE = "output_result"
JSON_DIR = os.path.join(OUTPUT_BASE, "json")
IMG_DIR = os.path.join(OUTPUT_BASE, "image")
TEMP_DIR = "temp_processed" 

# 字体设置 (非常重要：没有中文字体，中文会显示为方框)
# Windows常见路径: "C:/Windows/Fonts/msyh.ttc" (微软雅黑) 或 "simhei.ttf"
# Linux常见路径: "/usr/share/fonts/..."
# 如果找不到，请把一个 .ttf 文件放到代码同级目录
FONT_PATH = "simhei.ttf" 

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ================= 2. 图像处理与过滤逻辑 =================
def preprocess_image_enhanced(img_path, temp_save_path):
    """保持上一版效果最好的预处理：不缩放 + CLAHE + USM锐化"""
    img = cv2.imread(img_path)
    if img is None: return False

    # 1. CLAHE 增强对比度
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_contrast = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # 2. USM 锐化
    gaussian = cv2.GaussianBlur(img_contrast, (0, 0), 3.0)
    img_sharp = cv2.addWeighted(img_contrast, 1.5, gaussian, -0.5, 0, img_contrast)

    cv2.imwrite(temp_save_path, img_sharp)
    return True

PLACEHOLDER_CHARS = set("口□■▢▣▤▥▦▧▨▩▪▫◻◼◽◾☐☑☒")
def is_meaningful_text(text: str) -> bool:
    if not text: return False
    s = "".join(ch for ch in text if not ch.isspace())
    if not s: return False
    if all(ch in PLACEHOLDER_CHARS for ch in s): return False
    if all(unicodedata.category(ch).startswith(("P", "S")) for ch in s): return False
    if any(unicodedata.category(ch).startswith(("L", "N")) for ch in s): return True
    return False

# ================= 3. 自定义画图函数 (核心修改) =================
def draw_ocr_result(original_img_path, json_data, save_img_path):
    """
    读取原图，根据 JSON 里的坐标和文字，画绿框红字
    """
    # 1. 读取原图 (OpenCV格式)
    img_cv = cv2.imdecode(np.fromfile(original_img_path, dtype=np.uint8), -1)
    if img_cv is None: return

    # 2. 转换为 PIL 格式 (为了写中文)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 3. 遍历数据
    if "parsing_res_list" in json_data:
        for item in json_data["parsing_res_list"]:
            text = item.get("block_content", "")
            bbox = item.get("block_bbox", [])
            
            if len(bbox) == 4 and text:
                x, y, w, h = [int(v) for v in bbox]
                
                # --- A. 画框 (绿色) ---
                # PIL画矩形: [x0, y0, x1, y1]
                draw.rectangle([x, y, w,  h], outline=(0, 255, 0), width=3)
                
                # --- B. 画文字 (红色) ---
                # 动态计算字号: 设为高度的 60%，但不小于 12，不大于 40
                font_size = max(12, min(40, int(h * 0.6)))
                
                try:
                    # 尝试加载字体
                    font = ImageFont.truetype(FONT_PATH, font_size)
                except:
                    # 加载失败则使用默认字体 (可能无法显示中文)
                    font = ImageFont.load_default()
                
                # 计算文字位置 (在框的上方)
                # left, top, right, bottom
                text_bbox = draw.textbbox((x, y), text, font=font) 
                text_height = text_bbox[3] - text_bbox[1]
                
                # 绘制位置：框的上方，如果上方没位置了就画在框里面
                text_y = y - text_height - 5
                if text_y < 0: text_y = y + 5

                # 文字描边 (为了在复杂背景上看清，画一层白色描边)
                draw.text((x-1, text_y), text, font=font, fill=(255,255,255))
                draw.text((x+1, text_y), text, font=font, fill=(255,255,255))
                draw.text((x, text_y-1), text, font=font, fill=(255,255,255))
                draw.text((x, text_y+1), text, font=font, fill=(255,255,255))
                
                # 正文 (红色)
                draw.text((x, text_y), text, fill=(255, 0, 0), font=font)

    # 4. 转回 OpenCV 格式并保存
    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imencode('.jpg', img_result)[1].tofile(save_img_path)

# ================= 4. 主程序 =================
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

prompt = "请对图片进行版面分析，提取所有可见的文字区域，准确输出坐标并合并语义连续的文本行。"

try:
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            print(f"正在处理: {filename}")
            original_path = os.path.join(INPUT_DIR, filename)
            temp_path = os.path.join(TEMP_DIR, filename)
            
            # 1. 预处理
            if preprocess_image_enhanced(original_path, temp_path):
                try:
                    # 2. 预测
                    output = pipeline.predict(temp_path, prompt=prompt)
                    
                    for res in output:
                        base_name = os.path.splitext(filename)[0]
                        
                        # --- 步骤 1: 保存并清洗 JSON ---
                        # 先保存原始数据到临时文件
                        res.save_to_json(JSON_DIR)
                        raw_json_path = os.path.join(JSON_DIR, f"{base_name}_res.json")
                        if not os.path.exists(raw_json_path):
                            raw_json_path = os.path.join(JSON_DIR, f"{base_name}.json")
                        
                        # 读取并过滤
                        with open(raw_json_path, 'r', encoding='utf-8') as f:
                            raw_data = json.load(f)

                        clean_data = {"input_path": original_path, "parsing_res_list": []}
                        exclude_keys = {"block_id", "block_order", "block_label", "group_id"}

                        if "parsing_res_list" in raw_data:
                            for item in raw_data["parsing_res_list"]:
                                label = item.get("block_label", "text").lower()
                                content = item.get("block_content", "")
                                
                                # 过滤
                                if "image" in label: continue
                                if not is_meaningful_text(content): continue
                                
                                # 存入
                                clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
                                clean_data["parsing_res_list"].append(clean_item)

                        # 覆盖保存清洗后的 JSON
                        final_json_path = os.path.join(JSON_DIR, f"{base_name}.json")
                        with open(final_json_path, 'w', encoding='utf-8') as f:
                            json.dump(clean_data, f, ensure_ascii=False, indent=4)
                        
                        # 删除可能存在的 _res.json 冗余文件 (可选)
                        if raw_json_path != final_json_path and os.path.exists(raw_json_path):
                            os.remove(raw_json_path)

                        # --- 步骤 2: 根据清洗后的 JSON 画图 ---
                        # 这步替代了原来的 res.save_to_img()
                        final_img_path = os.path.join(IMG_DIR, f"{base_name}_result.jpg")
                        draw_ocr_result(original_path, clean_data, final_img_path)
                        
                        print(f"  [完成] JSON: {final_json_path}, IMG: {final_img_path}")

                except Exception as e:
                    print(f"  [预测失败] {e}")

finally:
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
