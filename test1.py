import os
import json
import cv2
import numpy as np
from paddleocr import PaddleOCRVL

# ================= 配置区域 =================
INPUT_DIR = "main_file"
OUTPUT_DIR = "output_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化 (保持不变)
pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url="http://127.0.0.1:8118/v1")

# 提示词：强制要求返回 bbox，且必须是 [x, y, w, h] 格式
prompt = '''
请分析图片布局。
必须输出纯JSON列表，不要包含任何Markdown标记。
格式：[{"text": "内容", "label": "text/title/figure", "bbox": [x, y, w, h]}]
注意：bbox必须是像素坐标(整数)，格式为[x, y, width, height]。
'''

# ================= 核心画图逻辑 =================
def draw_from_json_file(img_path, json_path, save_img_path):
    # 1. 读取原图
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    if img is None:
        print(f"❌ 错误: 无法读取图片 {img_path}")
        return

    # 2. 读取刚才保存的 JSON 文件
    if not os.path.exists(json_path):
        print(f"❌ 错误: JSON文件未生成 {json_path}")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 兼容性处理：如果 JSON 外面包裹了一层 key（比如有些模型会包在 'content' 里）
        if isinstance(data, dict):
            # 尝试找可能的列表字段，找不到就假设 data 本身就是 dict 形式的单个对象
            data = data.get('content', data.get('result', [data]))
        if not isinstance(data, list):
            data = [data]

        count = 0
        # 3. 遍历并画图
        for item in data:
            # 容错：支持 bounding_box 或 bbox 或 box 字段
            bbox = item.get("bbox", item.get("bounding_box", item.get("box")))
            label = item.get("label", item.get("type", "text"))

            if bbox and isinstance(bbox, list) and len(bbox) == 4:
                x, y, w, h = [int(v) for v in bbox]
                
                # 定义颜色 (Title红色, Text黄色, 其他蓝色)
                colors = {'title': (0,0,255), 'text': (0,255,255), 'header':(0,0,255)}
                color = colors.get(label, (255, 0, 0)) # 默认蓝色

                # A. 画边框
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # B. 画标签背景 (实心)
                text_scale = 0.6
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
                cv2.rectangle(img, (x, y - th - 5), (x + tw, y), color, -1)
                
                # C. 画文字
                cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0,0,0), 1)
                count += 1
        
        # 4. 保存图片
        cv2.imencode('.jpg', img)[1].tofile(save_img_path)
        print(f"✅ 已保存图片: {save_img_path} (绘制了 {count} 个框)")

    except Exception as e:
        print(f"⚠️ 画图失败 ({os.path.basename(img_path)}): {e}")

# ================= 主程序 =================
print(f"🚀 开始批处理，源文件夹: {INPUT_DIR}")

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        img_path = os.path.join(INPUT_DIR, filename)
        # 获取不带后缀的文件名，用于生成 json 和 output 图片名
        base_name = os.path.splitext(filename)[0]
        
        print(f"\n>> 正在处理: {filename}")
        
        # 1. 预测
        output = pipeline.predict(img_path, prompt=prompt)

        for res in output:
            # 2. 保存 JSON (这是 Paddle 自带的，确保一定会生成 JSON)
            # save_to_json 默认会使用 output_dir/文件名.json 保存
            res.save_to_json(save_path=OUTPUT_DIR)
            
            # 3. 计算刚刚保存的 JSON 路径
            # PaddleOCR 的保存规则通常是: save_path/文件名.json
            json_file_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
            
            # 4. 定义处理后的图片保存路径
            result_img_path = os.path.join(OUTPUT_DIR, f"vis_{filename}")

            # 5. 读取刚才生成的 JSON 并画图
            draw_from_json_file(img_path, json_file_path, result_img_path)

print("\n🎉 全部完成！")
