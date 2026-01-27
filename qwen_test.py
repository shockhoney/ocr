from openai import OpenAI
import os
import json
import base64
import re
import cv2  
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps

# ================= 配置区域 =================
input_folder = "main_file" 
output_root = "ocr_outputs"

client = OpenAI(
    api_key="sk-7f9cfd5f5f154c13b03604128a0b5491", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# ===========================================


def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def pil_to_base64(img):
    """将 PIL 图片转为 base64"""
    buffered = BytesIO()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_json_from_content(content):
    """JSON提取逻辑"""
    try:
        match = re.search(r"```json(.*?)```", content, re.DOTALL)
        if match: return json.loads(match.group(1).strip())
        match_list = re.search(r"\[.*\]", content, re.DOTALL)
        if match_list: return json.loads(match_list.group(0).strip())
        return None
    except:
        return None

def process_single_image(file_path, img_output_dir, json_output_dir):
    filename = os.path.basename(file_path)
    print(f"正在处理: {filename} ...")
    
    try:
        # 1. 加载图片
        original_img = Image.open(file_path)
        original_img = ImageOps.exif_transpose(original_img)
        
        # 转 base64 (用于 API 请求)
        base64_img = pil_to_base64(original_img)
        
        prompt = """
        请对这张海报进行OCR文字检测。注意：传入的图片大小都是626x626大小。
        要求：
        1. 识别所有可见文字，语义连续的文字可能出现在不同行，你需要根据语义将不同行文字合并在同一个box内,文本box要严格贴合检测的文字，防止出现文本框漂移的情况。
        2. 返回格式必须是纯 JSON 列表，格式如下：
           [
             {"text": "文本内容", "box": [xmin, ymin, xmax, ymax]},
             ...
           ]
        3. 不要输出任何分析文字，只输出JSON代码块。
        """
        
        completion = client.chat.completions.create(
            model="qwen-vl-ocr-2025-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            stream=False,
            extra_body={'enable_thinking': True} 
        )
        
        content = completion.choices[0].message.content
        ocr_result = extract_json_from_content(content)
        
        if not ocr_result:
            print(f"  [Error] 解析JSON失败。")
            return

        # --- 2. 使用 OpenCV 进行绘图 ---
        cv_img = pil_to_cv2(original_img)
        height, width = cv_img.shape[:2]
        
        data_list = ocr_result if isinstance(ocr_result, list) else ocr_result.get("data", [])
        
        for item in data_list:
            text_content = item.get("text", "")
            box = item.get("box", [])
            
            if len(box) == 4:
                # 坐标计算 (反归一化)
                x_min = int(box[0] / 1000 * width)
                y_min = int(box[1] / 1000 * height)
                x_max = int(box[2] / 1000 * width)
                y_max = int(box[3] / 1000 * height)
                
                # 定义四个顶点 (模仿 draw_bbox.py 的逻辑)
                # 左上, 右上, 右下, 左下
                p1 = (x_min, y_min)
                p2 = (x_max, y_min)
                p3 = (x_max, y_max)
                p4 = (x_min, y_max)

                # 使用 cv2.line 绘制四条边 (红色, 粗细为2)
                cv2.line(cv_img, p1, p2, (0, 0, 255), 2)
                cv2.line(cv_img, p2, p3, (0, 0, 255), 2)
                cv2.line(cv_img, p3, p4, (0, 0, 255), 2)
                cv2.line(cv_img, p4, p1, (0, 0, 255), 2)

        # 3. 保存结果
        base_name = os.path.splitext(filename)[0]
        
        # 保存图片
        save_img_path = os.path.join(img_output_dir, f"{base_name}_ocr.jpg")
        cv2.imwrite(save_img_path, cv_img)
        
        # 保存 JSON
        save_json_path = os.path.join(json_output_dir, f"{base_name}.json")
        with open(save_json_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=2)
            
        print(f"  -> 完成")
        
    except Exception as e:
        print(f"  -> 处理异常 {filename}: {str(e)}")

def main():
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹不存在 -> {input_folder}")
        return

    img_output_dir = os.path.join(output_root, "imgs")
    json_output_dir = os.path.join(output_root, "jsons")
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)
    
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_ext)]
    
    total = len(image_files)
    print(f"找到 {total} 张图片，开始处理...\n")

    for index, filename in enumerate(image_files):
        print(f"[{index+1}/{total}] ", end="")
        file_path = os.path.join(input_folder, filename)
        process_single_image(file_path, img_output_dir, json_output_dir)

    print("\n" + "="*30)
    print(f"处理完成！\n图片保存在: {img_output_dir}\nJSON保存在: {json_output_dir}")

if __name__ == "__main__":
    main()