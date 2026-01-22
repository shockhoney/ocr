import os
import json
import cv2
import numpy as np
from paddleocr import PaddleOCRVL

# ================= 配置区域 =================
INPUT_DIR = "main_file"          # 输入图片文件夹
OUTPUT_DIR = "output_result1"     # 结果保存文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化模型
pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url="http://127.0.0.1:8118/v1")

# 核心提示词：明确要求返回坐标 [x, y, w, h] 和 类型 (label)
prompt = '''
请分析图片布局。输出纯JSON数组，格式如下：
[{"text": "内容", "label": "text/title/figure", "box": [x, y, w, h]}]
注意：box必须是像素坐标(整数)。
'''

# ================= 画图函数 (复刻截图风格) =================
def draw_and_save(img_path, json_str, save_name):
    try:
        # 1. 读取图片 (兼容中文路径)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img is None: return

        # 2. 清洗并解析 JSON
        content = json_str.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)

        # 定义颜色 (BGR格式): 黄色, 红色, 蓝色
        colors = {"text": (0, 255, 255), "title": (0, 0, 255), "figure": (255, 0, 0)}

        # 3. 遍历每个识别到的物体
        for item in data:
            # 获取坐标和标签
            x, y, w, h = item.get("box", [0,0,0,0])
            label = item.get("label", "text")
            color = colors.get(label, (0, 255, 0)) # 默认绿色

            # --- 绘制逻辑 (复刻你的截图效果) ---
            # A. 画空心矩形框
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # B. 画标签背景 (在框的左上角上方画一个实心矩形)
            label_txt = f"{label}"
            (txt_w, txt_h), baseline = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x, y - txt_h - 5), (x + txt_w, y), color, -1) 
            
            # C. 写标签文字 (黑色字体)
            cv2.putText(img, label_txt, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 4. 保存图片
        save_path = os.path.join(OUTPUT_DIR, f"vis_{save_name}")
        cv2.imencode('.jpg', img)[1].tofile(save_path)
        print(f"  --> 图片已保存: {save_path}")

    except Exception as e:
        print(f"  [画图跳过] JSON解析或绘图失败: {e}")

# ================= 主程序 =================
print(f"开始批处理 {INPUT_DIR} 下的图片...")

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        image_path = os.path.join(INPUT_DIR, filename)
        print(f"正在处理: {filename}")

        # 1. 预测
        output = pipeline.predict(image_path, prompt=prompt)

        for res in output:
            # 2. 保存原始 JSON (Paddle自带功能)
            res.save_to_json(save_path=OUTPUT_DIR)
            
            # 3. 绘制并保存带框图片 (自定义功能)
            if hasattr(res, 'result'):
                draw_and_save(image_path, res.result, filename)

print("全部完成！")
