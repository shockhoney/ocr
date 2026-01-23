import os
import json
from paddleocr import PaddleOCRVL

INPUT_DIR = "main_file"
JSON_DIR = os.path.join("output_result", "json")
IMG_DIR = os.path.join("output_result", "image")

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1")
prompt = "请对图片进行版面分析，识别并提取所有可见的文字区域，包括水平、垂直和倾斜排列的文字。注意文字可能具有不一致的字体大小，需根据内容连续性进行合理合并。输出时应准确标注每个文字区域的文本框坐标（bounding box），并确保语义连续的文字被包含在同一个文本框中。在可视化结果中，不要显示原始图像块（image block）。最后，如果json文件中block_content为空或者全是特殊符号没有文本也需要剔除"


def process_single_result(res, filename):
    base_name = os.path.splitext(filename)[0]
    res.save_to_img(IMG_DIR)
    order_img_path = os.path.join(IMG_DIR, f"{base_name}_layout_order_res.png")
    if os.path.exists(order_img_path):
        os.remove(order_img_path)

    # ---  JSON处理：保存、读取、过滤、覆盖 ---
    res.save_to_json(JSON_DIR)
    
    # 确定JSON文件路径 (兼容 _res 后缀)
    json_path = os.path.join(JSON_DIR, f"{base_name}_res.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(JSON_DIR, f"{base_name}.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 构建清洗后的数据结构
        clean_data = {
            "input_path": raw_data.get("input_path", ""),
            "parsing_res_list": []
        }

        # 定义需要剔除的键
        exclude_keys = {"block_id", "block_order","block_label" "group_id"}

        if "parsing_res_list" in raw_data:
            for item in raw_data["parsing_res_list"]:
                label = item.get("block_label", "text").lower()
                
                # 过滤1: 剔除 image 类型的块
                if "image" in label:
                    continue
                
                # 过滤2: 剔除 block_id 等指定字段
                # 使用字典推导式，保留不在 exclude_keys 中的字段
                clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
                clean_data["parsing_res_list"].append(clean_item)

        # 覆盖保存
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
            
        print(f"  [完成] {filename}")

    except Exception as e:
        print(f"  [错误] JSON处理出错: {e}")



for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        img_path = os.path.join(INPUT_DIR, filename)
        
        try:
            output = pipeline.predict(img_path, prompt=prompt)
            for res in output:
                process_single_result(res, filename)
        except Exception as e:
            print(f"  [失败] 无法处理 {filename}: {e}")

