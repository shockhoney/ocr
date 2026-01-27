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

def get_ocr_result(local_path):
    """
    调用 DashScope Qwen-VL-OCR 进行文字识别
    """
    file_uri = f"file://{local_path}"
    
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
                "text": "定位所有的文字行"
            }
        ]
    }]

    try:
        # 调用模型
        # 注意：model='qwen-vl-ocr-2025-08-28' 和 task='advanced_recognition' 是关键
        response = MultiModalConversation.call(
            model='qwen-vl-ocr-2025-08-28',
            messages=messages,
            ocr_options={"task": "advanced_recognition"}
        )
    except Exception as e:
        print(f"\n[Network Error] API 调用失败: {e}")
        print("请检查网络连接或 DNS 设置 (Unable to resolve dashscope.aliyuncs.com)")
        return None
    
    if response.status_code == 200:
        if "ocr_result" in response.output.choices[0].message.content[0]:
            return response.output.choices[0].message.content[0]["ocr_result"]
        else:
            print(f"Warning: No 'ocr_result' in response for {local_path}")
            return None
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

        if 'words_info' in ocr_result:
            words_info = ocr_result["words_info"]
            
            for line in words_info:
                # 高精版 OCR 返回 location 为 8 个坐标点 (x1, y1, x2, y2, x3, y3, x4, y4)
                # 对应：左上 -> 右上 -> 右下 -> 左下
                loc = line.get('location', [])
                if len(loc) == 8:
                    pts = np.array([
                        [loc[0], loc[1]],
                        [loc[2], loc[3]],
                        [loc[4], loc[5]],
                        [loc[6], loc[7]]
                    ], np.int32)
                    
                    pts = pts.reshape((-1, 1, 2))
                    
                    # 绘制多边形框 (红色)
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                    
                    # 可选：绘制文字 (由于 opencv 不支持中文，这里暂不绘制文字内容，或者需要用 pillow)
        
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
