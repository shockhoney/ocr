import os
import json
from paddleocr import PaddleOCRVL

INPUT_DIR = "main_file"
OUTPUT_DIR = "output_result_demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 初始化模型
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

# 2. 提示词
prompt = "请对图片进行版面分析，识别并提取所有可见的文字区域，并输出坐标。"

print("开始处理...")
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(INPUT_DIR, filename)
        print(f"正在处理: {filename}")
        
        try:
            # 预测
            output = pipeline.predict(img_path, prompt=prompt)

            for res in output:
                # --- A. 保留官方的可视化画图功能 ---
                res.save_to_img(OUTPUT_DIR)
                
                # --- B. 自定义精简版 JSON 保存逻辑 ---
                # 1. 获取基础文件名 (如 image.png -> image)
                base_name = os.path.splitext(filename)[0]
                json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
                
                # 2. 提取并过滤数据
                simple_data = []
                
                # 检查是否存在 parsing_res_list (这是PaddleOCRVL存放核心结果的地方)
                # 注意：根据你的原始JSON，数据在 'parsing_res_list' 里
                # 如果 res 是对象，通常用 . 访问；如果是字典，用 [] 访问。这里做兼容处理。
                if hasattr(res, 'parsing_res_list'):
                    raw_list = res.parsing_res_list
                else:
                    # 假如 res 是字典形式
                    raw_list = res.get('parsing_res_list', [])

                for item in raw_list:
                    # 获取文本内容 (block_content) 和 坐标 (block_bbox)
                    # 兼容对象属性访问和字典键访问
                    content = item.get('block_content', "") if isinstance(item, dict) else getattr(item, 'block_content', "")
                    bbox = item.get('block_bbox', []) if isinstance(item, dict) else getattr(item, 'block_bbox', [])

                    # 过滤条件：只保留有实际文字的内容
                    # (原JSON里 'image' 类型的 block_content 是空的，这里直接过滤掉)
                    if content and content.strip() != "":
                        simple_data.append({
                            "text": content,
                            "bbox": bbox
                        })
                
                # 3. 保存为新的 JSON 文件
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(simple_data, f, ensure_ascii=False, indent=4)
                    
                print(f"  已保存精简版JSON: {json_path}")
                print(f"  已保存可视化图片: {OUTPUT_DIR}")
                
        except Exception as e:
            print(f"  处理失败 {filename}: {e}")
