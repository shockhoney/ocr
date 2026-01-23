import os
import json
from paddleocr import PaddleOCRVL

INPUT_DIR = "main_file"
JSON_OUT_DIR = os.path.join("output_result", "json")
IMG_OUT_DIR = os.path.join("output_result", "image")
os.makedirs(JSON_OUT_DIR, exist_ok=True)
os.makedirs(IMG_OUT_DIR, exist_ok=True)

# 1. 初始化模型
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

# 提示词：请求版面分析
prompt = "请对图片进行版面分析，识别并提取所有可见的文字区域。需特别注意文字可能以水平、垂直或倾斜方式排列，并且字体大小可能不一致。请将语义上连续的文字内容合并至同一个文本框中，并准确输出每个文字区域的文本框坐标（包括左上角坐标、宽度和高度）。"

def process_and_clean(res, base_name):
    """
    处理单个结果：保存图片 -> 保存原始JSON -> 清洗JSON -> 覆盖保存
    """
    res.save_to_img(IMG_OUT_DIR)
    res.save_to_json(JSON_OUT_DIR)
    
    unwanted_img = os.path.join(IMG_OUT_DIR, f"{base_name}_layout_order_res.png")
    if os.path.exists(unwanted_img):
        os.remove(unwanted_img)
        
    #  读取并清洗 JSON
    json_path = os.path.join(JSON_OUT_DIR, f"{base_name}_res.json")
    if not os.path.exists(json_path):
        # 兼容备用文件名（有些版本可能是原名.json）
        json_path = os.path.join(JSON_OUT_DIR, f"{base_name}.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        #  构建目标数据结构
        final_data = {
            "input_path": raw_data.get("input_path", ""),
            "parsing_res_list": []
        }

        #  过滤逻辑：剔除 block_label 为 "image" 的项
        if "parsing_res_list" in raw_data:
            for item in raw_data["parsing_res_list"]:
                # 获取标签，转小写比较，确保过滤 "image", "figure" 等
                label = item.get("block_label", "text").lower()
                
                # 只有当标签不是 image 时才保留
                if "image" not in label:
                    final_data["parsing_res_list"].append(item)

        # 6. 覆盖写入最终的清洗版 JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
            
        print(f"  [JSON] 已清洗并保存: {json_path}")
        print(f"  [IMG]  已保存可视化图: {IMG_OUT_DIR}")

    except Exception as e:
        print(f"  [错误] JSON处理失败: {e}")


for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        img_path = os.path.join(INPUT_DIR, filename)
        base_name = os.path.splitext(filename)[0]

        try:
            output = pipeline.predict(img_path, prompt=prompt)
            for res in output:
                process_and_clean(res, base_name)
                
        except Exception as e:
            print(f"  预测失败: {e}")
