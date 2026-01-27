# batch_ocr_dashscope.py
import os
import sys
import json
import cv2
import numpy as np
import dashscope
from dashscope import MultiModalConversation

# 配置 API Key
dashscope.api_key = "sk-7f9cfd5f5f154c13b03604128a0b5491"

# 配置输入输出
INPUT_FOLDER = "main_file"
OUTPUT_ROOT = "ocr_outputs"
IMG_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "imgs")
JSON_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "jsons")

def ensure_dirs():
    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

import re

def extract_json_from_content(content):
    """
    从模型返回的 Markdown 文本中提取 JSON 对象
    """
    try:
        # 尝试匹配 ```json ... ```
        match = re.search(r"```json(.*?)```", content, re.DOTALL)
        if match: return json.loads(match.group(1).strip())
        
        # 尝试匹配纯列表 [...]
        match_list = re.search(r"\[.*\]", content, re.DOTALL)
        if match_list: return json.loads(match_list.group(0).strip())
        
        return None
    except Exception as e:
        print(f"JSON Parse Error: {e}")
        return None

def get_ocr_result(local_path):
    """
    调用 DashScope Qwen-VL-OCR 进行文字识别
    """
    file_uri = f"file://{local_path}"
    
    prompt = '''定位所有的文字行，并且以json格式返回旋转矩形([x1, y1, x2, y2])的坐标结果。 返回格式必须是纯 JSON 列表，格式如下：
           [
             {"text": "文本内容", "box": [xmin, ymin, xmax, ymax]},
             ...
           ]其中x1,y1是文字行的左上角坐标，x2,y2是文字行的右下角坐标
           另外，你需要根据提取的语义信息和距离，将文字行进行合理组合，有的语义连续的文本可能不在同一行，需要合并为一个文本行'''

    messages = [{
        "role": "user",
        "content": [
            {
                "image": file_uri,
                "min_pixels": 28 * 28 * 4,
                "max_pixels": 28 * 28 * 8192,
                "enable_rotate": True
            },
            {
                "text": prompt
            }
        ]
    }]

    try:
        # 使用 qwen-vl-max 来支持自定义 Prompt 指令
        response = MultiModalConversation.call(
            model='qwen-vl-max', 
            messages=messages
        )
    except Exception as e:
        print(f"\n[Network Error] API 调用失败: {e}")
        return None
    
    if response.status_code == 200:
        content = response.output.choices[0].message.content
        # 提取 JSON
        return extract_json_from_content(content)
    else:
        print(f"Error calling API for {local_path}: code={response.code}, message={response.message}")
        return None

def draw_and_save(image_path, ocr_result, save_path):
    """
    绘制 OCR 结果并保存
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return

        height, width = img.shape[:2]
        
        # 兼容处理：ocr_result 应该是一个列表
        data_list = ocr_result if isinstance(ocr_result, list) else []
        
        for item in data_list:
            text = item.get("text", "")
            box = item.get("box", [])
            
            # 格式: [xmin, ymin, xmax, ymax] (归一化 0-1000)
            if len(box) == 4:
                x1 = int(box[0] / 1000 * width)
                y1 = int(box[1] / 1000 * height)
                x2 = int(box[2] / 1000 * width)
                y2 = int(box[3] / 1000 * height)
                
                # 绘制矩形框 (红色)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # 绘制简单的文字背景条 (防止文字看不清)
                cv2.rectangle(img, (x1, y1-20), (x1+100, y1), (0, 0, 255), -1)
                
                # 注意：OpenCV putText 不支持中文，这里只绘制 "Text" 占位或使用英文内容
                # 如果需要中文，需使用 PIL
                display_text = "Text" 
                cv2.putText(img, display_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(save_path, img)
        print(f"Saved visualization to {save_path}")
        
    except Exception as e:
        print(f"Exception drawing {image_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    ensure_dirs()
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' does not exist.")
        return

    # 支持的图片扩展名
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(exts)]
    
    print(f"Found {len(files)} images in {INPUT_FOLDER}")
    
    for i, filename in enumerate(files):
        file_path = os.path.abspath(os.path.join(INPUT_FOLDER, filename))
        print(f"\n[{i+1}/{len(files)}] Processing {filename} ...")
        
        # 1. OCR 识别
        ocr_result = get_ocr_result(file_path)
        
        if ocr_result:
            base_name = os.path.splitext(filename)[0]
            
            # 2. 保存 JSON 结果
            json_save_path = os.path.join(JSON_OUTPUT_DIR, f"{base_name}.json")
            with open(json_save_path, 'w', encoding='utf-8') as f:
                json.dump(ocr_result, f, ensure_ascii=False, indent=2)
            
            # 3. 绘制并保存图片
            img_save_path = os.path.join(IMG_OUTPUT_DIR, f"{base_name}_ocr_vis.jpg")
            draw_and_save(file_path, ocr_result, img_save_path)
        else:
            print("Skipping drawing due to failed OCR.")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()
