import os
import json
import shutil
import unicodedata
import cv2  # 需要安装 opencv-python
import numpy as np
from paddleocr import PaddleOCRVL
from PIL import Image, ImageDraw, ImageFont

# ================= 0. 绘图工具函数 =================
def draw_simple_result(img_path, data, save_path):
    """简单绘制红框和蓝色文字"""
    if not os.path.exists(img_path): return
    
    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # 加载字体：Windows 常用 simhei.ttf，找不到则用默认
        try:
            # 如果是 Windows，通常有 simhei.ttf；Linux 可能需要指定路径
            font = ImageFont.truetype("simhei.ttf", 30) 
        except:
            font = ImageFont.load_default()

        for item in data.get("parsing_res_list", []):
            bbox = item.get("block_bbox") # [x1, y1, x2, y2]
            text = item.get("block_content")
            
            if bbox and text:
                # 1. 画红框
                draw.rectangle(bbox, outline="red", width=2)
                # 2. 画文字 (画在框的左上角上方，防止遮挡)
                # 如果文字超出了上边界，稍微往下挪一点
                text_y = bbox[1] - 30
                if text_y < 0: text_y = bbox[1]
                
                draw.text((bbox[0], text_y), text, fill="blue", font=font)
                
        img.save(save_path)
    except Exception as e:
        print(f"  [绘图警告] {e}")

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
OUTPUT_DIR = "output_result"
JSON_DIR = os.path.join(OUTPUT_DIR, "json")
IMG_DIR = os.path.join(OUTPUT_DIR, "image")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp_processed")

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ================= 2. 图像预处理函数 =================
def preprocess_poster_image(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [警告] 无法读取图片: {image_path}")
        return False

    try:
        # CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        limg = cv2.merge((l, a, b))
        img_contrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # USM Sharp
        gaussian = cv2.GaussianBlur(img_contrast, (0, 0), 3.0)
        img_sharp = cv2.addWeighted(img_contrast, 1.5, gaussian, -0.5, 0, img_contrast)

        cv2.imwrite(save_path, img_sharp)
        return True
    except Exception as e:
        print(f"  [预处理失败] {e}")
        return False

# ================= 3. 定义过滤逻辑 =================
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

prompt = '''请对图片进行版面分析，识别并提取所有可见的文字区域，包括水平、垂直和倾斜排列的文字。
注意文字可能具有不一致的字体大小，需根据内容连续性进行合理合并。
输出时应准确标注每个文字区域的文本框坐标（bounding box），并确保语义连续的文字被包含在同一个文本框中。'''

# ================= 5. 核心处理函数 (已修改调用绘图) =================
def process_single_result(res, filename, original_input_path):
    base_name = os.path.splitext(filename)[0]
    
    # --- A. 保存原始可视化图 (Paddle自带) ---
    res.save_to_img(IMG_DIR)
    # 删除多余的 layout_order 图片
    order_img_path = os.path.join(IMG_DIR, f"{base_name}_layout_order_res.png")
    if os.path.exists(order_img_path):
        os.remove(order_img_path)

    # --- B. JSON处理 ---
    res.save_to_json(JSON_DIR)
    
    # 获取 JSON 路径
    json_path = os.path.join(JSON_DIR, f"{base_name}_res.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(JSON_DIR, f"{base_name}.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        clean_data = {
            "input_path": original_input_path, 
            "parsing_res_list": []
        }

        exclude_keys = {"block_id", "block_order", "block_label", "group_id"}

        if "parsing_res_list" in raw_data:
            for item in raw_data["parsing_res_list"]:
                label = item.get("block_label", "text").lower()
                content = item.get("block_content", "")

                if "image" in label: continue
                if not is_meaningful_text(content): continue

                clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
                clean_data["parsing_res_list"].append(clean_item)

        # 保存清洗后的 JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
            
        # ------------------------------------------------------------------
        # ★★★ 新增：调用 draw_simple_result 进行可视化绘制 ★★★
        # ------------------------------------------------------------------
        # 1. 设置保存路径
        vis_save_path = os.path.join(IMG_DIR, f"{base_name}_text_vis.jpg")
        
        # 2. 选择底图：优先用 temp_processed 里的增强图 (对比度好)，没有则用原图
        temp_img_source = os.path.join(TEMP_DIR, filename)
        source_img = temp_img_source if os.path.exists(temp_img_source) else original_input_path
        
        # 3. 执行绘制
        draw_simple_result(source_img, clean_data, vis_save_path)
        # ------------------------------------------------------------------

        print(f"  [处理完成] {filename}")

    except Exception as e:
        print(f"  [错误] JSON处理或绘图出错: {e}")

# ================= 6. 主循环 =================
try:
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            original_img_path = os.path.join(INPUT_DIR, filename)
            temp_img_path = os.path.join(TEMP_DIR, filename)
            
            print(f"正在处理: {filename} ...")
            
            # 1. 预处理
            success = preprocess_poster_image(original_img_path, temp_img_path)
            target_path = temp_img_path if success else original_img_path
            
            try:
                # 2. 预测
                output = pipeline.predict(target_path, prompt=prompt)
                
                # 3. 处理结果
                for res in output:
                    process_single_result(res, filename, original_img_path)
                    
            except Exception as e:
                print(f"  [失败] 模型预测出错 {filename}: {e}")

finally:
    pass