import os
import json
from paddleocr import PaddleOCRVL

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
JSON_DIR = os.path.join("output_result", "json")
IMG_DIR = os.path.join("output_result", "image")

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ================= 2. 初始化模型 =================
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

# 提示词
prompt = "请对图片进行版面分析，识别并提取所有可见的文字区域，包括水平、垂直和倾斜排列的文字。注意文字可能具有不一致的字体大小，需根据内容连续性进行合理合并。输出时应准确标注每个文字区域的文本框坐标（bounding box），并确保语义连续的文字被包含在同一个文本框中。在可视化结果中，不要显示原始图像块（image block）。"

# ================= 3. 处理函数 =================
def process_single_result(res, filename):
    base_name = os.path.splitext(filename)[0]
    
    # --- A. 图片处理 ---
    res.save_to_img(IMG_DIR)
    order_img_path = os.path.join(IMG_DIR, f"{base_name}_layout_order_res.png")
    if os.path.exists(order_img_path):
        os.remove(order_img_path)

    # --- B. JSON处理 ---
    res.save_to_json(JSON_DIR)
    
    # 获取JSON路径
    json_path = os.path.join(JSON_DIR, f"{base_name}_res.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(JSON_DIR, f"{base_name}.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        clean_data = {
            "input_path": raw_data.get("input_path", ""),
            "parsing_res_list": []
        }

        # 定义要删除的无用键
        exclude_keys = {"block_id", "block_order", "block_label", "group_id"}

        if "parsing_res_list" in raw_data:
            for item in raw_data["parsing_res_list"]:
                label = item.get("block_label", "text").lower()
                content = item.get("block_content", "")

                # ----------------- 新增过滤逻辑开始 -----------------
                
                # 1. 剔除 image 类型
                if "image" in label:
                    continue

                # 2. 剔除内容为空的情况 (去空格后为空)
                if not content or not content.strip():
                    continue

                # 3. 剔除全为特殊符号的情况
                # 逻辑：如果 content 中不存在任何一个 字母、数字或汉字，则视为无效
                # isalnum() 对汉字也会返回 True，非常适合此场景
                has_valid_char = any(char.isalnum() for char in content)
                if not has_valid_char:
                    continue
                    
                # ----------------- 新增过滤逻辑结束 -----------------

                # 构建保留的数据项
                clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
                clean_data["parsing_res_list"].append(clean_item)

        # 覆盖保存
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
            
        print(f"  [完成] {filename} (已过滤空内容及纯符号)")

    except Exception as e:
        print(f"  [错误] JSON处理出错: {e}")


# ================= 4. 主循环 =================
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        img_path = os.path.join(INPUT_DIR, filename)
        
        try:
            output = pipeline.predict(img_path, prompt=prompt)
            for res in output:
                process_single_result(res, filename)
        except Exception as e:
            print(f"  [失败] 无法处理 {filename}: {e}")
