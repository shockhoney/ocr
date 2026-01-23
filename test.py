import os
import json
from paddleocr import PaddleOCRVL

# ================= 配置 =================
INPUT_DIR = "main_file"
OUTPUT_DIR = "output_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 初始化模型
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

# 2. 提示词
prompt = "请对图片进行版面分析，识别并提取所有可见的文字区域。需特别注意文字可能以水平、垂直或倾斜方式排列，并且字体大小可能不一致。请将语义上连续的文字内容合并至同一个文本框中，并准确输出每个文字区域的文本框坐标（包括左上角坐标、宽度和高度）。"

# ================= 核心功能：清洗JSON =================
def clean_json_file(json_path):
    """
    读取全量JSON，提取 text 和 bbox，覆盖原文件
    """
    if not os.path.exists(json_path):
        print(f"  [警告] 未找到JSON文件: {json_path}")
        return

    try:
        # 1. 读取原始全量数据
        with open(json_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        simplified_data = []
        
        # 2. 提取 parsing_res_list 中的数据 (根据你提供的JSON结构)
        if isinstance(full_data, dict) and "parsing_res_list" in full_data:
            for item in full_data["parsing_res_list"]:
                # 获取内容和坐标
                content = item.get("block_content", "")
                bbox = item.get("block_bbox", [])
                
                # 过滤逻辑：只要内容不为空字符串，就保留
                if content and str(content).strip():
                    simplified_data.append({
                        "text": content,
                        "bbox": bbox
                    })
        
        # 3. 覆盖写入精简后的数据
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_data, f, ensure_ascii=False, indent=4)
            
        print(f"  --> JSON已清洗，保留 {len(simplified_data)} 条文本数据")

    except Exception as e:
        print(f"  [清洗失败] JSON格式异常: {e}")

# ================= 主流程 =================
print("开始批处理...")

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(INPUT_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        print(f"\n正在处理: {filename}")
        
        try:
            # 1. 预测
            output = pipeline.predict(img_path, prompt=prompt)

            for res in output:
                # 2. 调用官方方法保存图片 (带框)
                # 这会在 OUTPUT_DIR 生成 {filename}_res.png 或类似的图片
                res.save_to_img(OUTPUT_DIR)
                print(f"  已保存可视化图片")

                # 3. 调用官方方法保存全量 JSON
                # 这会在 OUTPUT_DIR 生成 {base_name}_res.json
                res.save_to_json(OUTPUT_DIR)
                
                # 4. 计算生成的 JSON 路径并进行清洗
                # PaddleOCR 默认保存的文件名通常是: 原文件名_res.json
                # 我们先尝试找 _res.json，如果找不到再找 .json
                expected_json_name = f"{base_name}_res.json"
                json_full_path = os.path.join(OUTPUT_DIR, expected_json_name)
                
                if not os.path.exists(json_full_path):
                    # 备用方案：如果名字里没有_res
                    json_full_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")

                # 执行清洗
                clean_json_file(json_full_path)
                
        except Exception as e:
            print(f"  处理异常: {e}")

print("\n全部完成！")
