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
prompt = "请对图片进行版面分析，提取所有区域的文字(海报上的文字，可能不具备结构化，包含水平，垂直，倾斜，字体大小不一等各种各样的文字）和文本框坐标(bounding box)。"

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


