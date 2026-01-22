import os
from paddleocr import PaddleOCRVL

INPUT_DIR = "main_file"
OUTPUT_DIR = "output_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 初始化模型
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

# 2. 提示词
prompt = "请对图片进行版面分析，识别并提取所有可见的文字区域，包括海报上水平、垂直和倾斜排列的文字，并注意文字可能具有不一致的字体大小。同时，请准确输出每个文字区域的文本框坐标（bounding box）。"

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(INPUT_DIR, filename)
        print(f"正在处理: {filename}")
        
        try:
            # 预测
            output = pipeline.predict(img_path, prompt=prompt)

            for res in output:
                res.save_to_img(OUTPUT_DIR)
                res.save_to_json(OUTPUT_DIR)
                
                print(f"  已保存可视化结果到: {OUTPUT_DIR}")     
        except Exception as e:
            print(f"  处理失败: {e}")


