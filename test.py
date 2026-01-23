import os
import json
import unicodedata  # 引入用于判断字符类型的库
from paddleocr import PaddleOCRVL

# ================= 1. 配置区域 =================
INPUT_DIR = "main_file"
JSON_DIR = os.path.join("output_result", "json")
IMG_DIR = os.path.join("output_result", "image")

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ================= 2. 定义过滤逻辑 =================

# 定义常见的OCR占位符或乱码字符集
PLACEHOLDER_CHARS = set("口□■▢▣▤▥▦▧▨▩▪▫◻◼◽◾☐☑☒")

def is_meaningful_text(text: str) -> bool:
    """
    判断文本是否具有实际含义。
    如果是空、纯占位符、或纯符号（无字母/数字/汉字），返回 False。
    """
    if not text:
        return False
    
    # 1. 去除所有空白字符（空格、换行、制表符）
    s = "".join(ch for ch in text if not ch.isspace())
    if not s:
        return False

    # 2. 过滤纯占位符（如 "□ □ □"）
    if all(ch in PLACEHOLDER_CHARS for ch in s):
        return False

    # 3. 过滤纯符号/纯标点
    # unicodedata.category 返回字符类别：
    # 'P' 开头为标点 (Punctuation)，如 , . " '
    # 'S' 开头为符号 (Symbol)，如 + = $ ^ | ~
    if all(unicodedata.category(ch).startswith(("P", "S")) for ch in s):
        return False

    # 4. 只有包含 字母(Letter) 或 数字(Number) 才算有效
    # 注意：汉字在 Unicode 中属于 'Lo' (Letter, other)，所以这里也包含了中文
    if any(unicodedata.category(ch).startswith(("L", "N")) for ch in s):
        return True

    return False

# ================= 3. 初始化模型 =================
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://127.0.0.1:8118/v1"
)

prompt = "请对图片进行版面分析，识别并提取所有可见的文字区域，包括水平、垂直和倾斜排列的文字。注意文字可能具有不一致的字体大小，需根据内容连续性进行合理合并。输出时应准确标注每个文字区域的文本框坐标（bounding box），并确保语义连续的文字被包含在同一个文本框中。在可视化结果中，不要显示原始图像块（image block）。"

# ================= 4. 核心处理函数 =================
def process_single_result(res, filename):
    base_name = os.path.splitext(filename)[0]
    
    # --- A. 图片处理 ---
    res.save_to_img(IMG_DIR)
    order_img_path = os.path.join(IMG_DIR, f"{base_name}_layout_order_res.png")
    if os.path.exists(order_img_path):
        os.remove(order_img_path)

    # --- B. JSON处理 ---
    res.save_to_json(JSON_DIR)
    
    # 确定JSON路径
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

        # 需要剔除的字段键名
        exclude_keys = {"block_id", "block_order", "block_label", "group_id"}

        if "parsing_res_list" in raw_data:
            for item in raw_data["parsing_res_list"]:
                label = item.get("block_label", "text").lower()
                content = item.get("block_content", "")

                # ----------------- 过滤逻辑开始 -----------------
                
                # 1. 过滤 image 类型
                if "image" in label:
                    continue
                
                # 2. 使用增强版逻辑过滤无意义文本 (空/符号/乱码)
                if not is_meaningful_text(content):
                    continue

                # ----------------- 过滤逻辑结束 -----------------

                # 构建保留项 (剔除 block_id 等)
                clean_item = {k: v for k, v in item.items() if k not in exclude_keys}
                clean_data["parsing_res_list"].append(clean_item)

        # 覆盖保存
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
            
        print(f"  [完成] {filename} (已清洗符号和乱码)")

    except Exception as e:
        print(f"  [错误] JSON处理出错: {e}")


# ================= 5. 主循环 =================
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        img_path = os.path.join(INPUT_DIR, filename)
        
        try:
            output = pipeline.predict(img_path, prompt=prompt)
            for res in output:
                process_single_result(res, filename)
        except Exception as e:
            print(f"  [失败] 无法处理 {filename}: {e}")
