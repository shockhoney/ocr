import os
import json
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
prompt = "请对图片进行版面分析，识别并提取所有可见的文字区域。需特别注意文字可能以水平、垂直或倾斜方式排列，并且字体大小可能不一致。请将语义上连续的文字内容合并至同一个文本框中，并准确输出每个文字区域的文本框坐标（包括左上角坐标、宽度和高度）。"

# --- 辅助函数：暴力提取数据 ---
def extract_parsing_list(res_obj):
    """
    尝试所有可能的方式从 res 对象中获取 parsing_res_list
    """
    data_list = []
    
    # 尝试方式 1: 直接属性访问 (res.parsing_res_list)
    if hasattr(res_obj, 'parsing_res_list'):
        data_list = res_obj.parsing_res_list
        
    # 尝试方式 2: 字典键访问 (res['parsing_res_list'])
    elif hasattr(res_obj, 'get'):
        data_list = res_obj.get('parsing_res_list', [])
        
    # 尝试方式 3: 尝试转换为字典 (res.__dict__)
    if not data_list and hasattr(res_obj, '__dict__'):
        data_list = res_obj.__dict__.get('parsing_res_list', [])

    return data_list

print("开始处理...")
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(INPUT_DIR, filename)
        print(f"正在处理: {filename}")
        
        try:
            # 预测
            output = pipeline.predict(img_path, prompt=prompt)

            for res in output:
                # 1. 保存带框图片 (保留官方功能)
                res.save_to_img(OUTPUT_DIR)
                
                # 2. 自定义精简 JSON
                base_name = os.path.splitext(filename)[0]
                json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
                
                # === 核心修改：使用强力提取函数 ===
                raw_list = extract_parsing_list(res)
                
                # 调试信息：如果列表为空，打印出来看看
                if not raw_list:
                    print(f"  [警告] 无法在 res 对象中找到数据，尝试 raw dump: {dir(res)}")
                
                simple_data = []
                for item in raw_list:
                    # 获取内容的兼容写法
                    # 尝试 item['block_content'] 或 item.block_content
                    if isinstance(item, dict):
                        content = item.get('block_content', "")
                        bbox = item.get('block_bbox', [])
                    else:
                        content = getattr(item, 'block_content', "")
                        bbox = getattr(item, 'block_bbox', [])

                    # 过滤空文本
                    if content and str(content).strip():
                        simple_data.append({
                            "text": content,
                            "bbox": bbox
                        })
                
                # 3. 保存
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(simple_data, f, ensure_ascii=False, indent=4)
                    
                print(f"  已保存 JSON (包含 {len(simple_data)} 条文本): {json_path}")
                
        except Exception as e:
            print(f"  处理失败 {filename}: {e}")
