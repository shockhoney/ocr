import os
from paddleocr import PaddleOCRVL

# ================= 配置 =================
INPUT_DIR = "main_file"
OUTPUT_DIR = "output_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 初始化模型
# 注意：如果要达到你截图中的那种精准版面分析效果（带 text, title, figure 标签和置信度），
# 官方通常会在 pipeline 里集成 PP-DocLayout 模型。
# 但如果你只是想可视化 VLM 返回的坐标，直接用这个 pipeline 即可。
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

# 2. 提示词
# 想要 save_to_img 能画图，关键是让模型返回的数据能被 Paddle 解析。
# 对于 VLM，通常需要提示它返回 JSON 格式的坐标。
prompt = "请对图片进行版面分析，提取所有区域的文字和坐标(bounding box)。"

print("开始批处理...")
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(INPUT_DIR, filename)
        print(f"正在处理: {filename}")
        
        try:
            # 预测
            output = pipeline.predict(img_path, prompt=prompt)

            for res in output:
                # ---------------------------------------------------------
                # 核心代码：这就是你要的官方集成功能
                # save_to_img 会自动读取原图，画上框，保存到指定目录
                # ---------------------------------------------------------
                res.save_to_img(OUTPUT_DIR)
                res.save_to_json(OUTPUT_DIR)
                
                print(f"  已保存可视化结果到: {OUTPUT_DIR}")
                
        except Exception as e:
            print(f"  处理失败: {e}")
            # 如果报错 'AttributeError: ... has no attribute 'save_to_img''
            # 请尝试更新 paddleocr: pip install -U paddleocr

print("全部完成")
