import os
import json
import cv2
import numpy as np
import shutil
import unicodedata
from PIL import Image, ImageDraw, ImageFont 

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
OUTPUT_BASE = "output_result"
JSON_DIR = os.path.join(OUTPUT_BASE, "json")
IMG_DIR = os.path.join(OUTPUT_BASE, "image")
TEMP_DIR = "temp_processed" 
FONT_PATH = "simhei.ttf" # [新增] 字体路径，请确保该文件存在

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ================= 2. 增强型预处理 (保持不变) =================
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
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    img_contrast = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # --- 步骤B: USM 锐化 (Unsharp Masking) ---
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

# ================= 4. [新增] 画框逻辑函数 =================
def draw_ocr_result(original_img_path, json_data, save_img_path):
    """
    读取原图，根据 JSON 里的坐标 [x1, y1, x2, y2] 画绿框红字
    """
    img_cv = cv2.imdecode(np.fromfile(original_img_path, dtype=np.uint8), -1)
    if img_cv is None: return

    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    if "parsing_res_list" in json_data:
        for item in json_data["parsing_res_list"]:
            text = item.get("block_content", "")
            bbox = item.get("block_bbox", [])
            
            # 确保 bbox 有4个值
            if len(bbox) == 4 and text:
                # --- 修改点：解析坐标为 x1, y1, x2, y2 ---
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # 计算宽度和高度（用于确定字号）
                w = x2 - x1
                h = y2 - y1
                
                # --- A. 画框 ---
                # PIL 的 rectangle 方法接收 [x1, y1, x2, y2]
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=1)
                
                # --- B. 画文字 ---
                # 动态字号：高度的 60%
                font_size = max(12, min(40, int(h * 0.6)))
                try:
                    font = ImageFont.truetype(FONT_PATH, font_size)
                except:
                    font = ImageFont.load_default()
                
                # 计算文字区域大小
                text_bbox = draw.textbbox((x1, y1), text, font=font) 
                text_pixel_height = text_bbox[3] - text_bbox[1]
                
                # 确定文字绘制位置 (优先画在框上方)
                text_y = y1 - text_pixel_height - 5
                # 如果上方超出图片边缘，改画在框内
                if text_y < 0: 
                    text_y = y1 + 5

                # 文字描边 (白色)
                draw.text((x1-1, text_y), text, font=font, fill=(255,255,255))
                draw.text((x1+1, text_y), text, font=font, fill=(255,255,255))
                draw.text((x1, text_y-1), text, font=font, fill=(255,255,255))
                draw.text((x1, text_y+1), text, font=font, fill=(255,255,255))
                
                # 正文 (红色)
                draw.text((x1, text_y), text, fill=(255, 0, 0), font=font)

    # 保存
    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imencode('.jpg', img_result)[1].tofile(save_img_path)

# ================= 5. 模型初始化 =================
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

prompt = "请对图片进行版面分析，识别并提取所有可见的文字区域，包括水平、垂直和倾斜排列的文字。注意文字可能具有不一致的字体大小，需根据内容连续性进行合理合并。输出时应准确标注每个文字区域的文本框坐标（bounding box），并确保语义连续的文字被包含在同一个文本框中"

# ================= 6. 处理流程 =================
def process_single_result(res, filename, original_path):
    base_name = os.path.splitext(filename)[0]
    
    # 图片处理 (官方方法保存的图片，后续会被我们自定义的覆盖或共存)
    # res.save_to_img(IMG_DIR) 
    # order_img = os.path.join(IMG_DIR, f"{base_name}_layout_order_res.png")
    # if os.path.exists(order_img): os.remove(order_img)

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

        # 保存清洗后的 JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
        
        # [应用] 使用清洗后的数据和原图，绘制最终的可视化图片
        final_img_path = os.path.join(IMG_DIR, f"{base_name}_result.jpg")
        draw_ocr_result(original_path, clean_data, final_img_path)

        print(f"  [完成] {filename} -> IMG: {final_img_path}")

    except Exception as e:
        print(f"  [JSON错误] {e}")

# ================= 7. 主循环 =================
try:
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            original_path = os.path.join(INPUT_DIR, filename)
            temp_path = os.path.join(TEMP_DIR, filename)
            
            print(f"正在处理: {filename}")
            
            # 使用增强型预处理
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
