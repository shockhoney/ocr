import os
from paddleocr import PaddleOCRVL

INPUT_FOLDER = "main_file"  
OUTPUT_FOLDER = "output"    


def get_custom_prompt():
    """定义提示词"""
    return '''
# Role
你是一个专业的OCR文档分析助手。

# Task
请识别图片中的所有文字内容，并将其整理为结构化的JSON格式。

# Constraints
1. 忽略图像质量问题，尽最大努力识别所有可见文字。
2. 将语义上属于同一行的文字合并。
3. 仅输出纯JSON字符串。

# Output Format (JSON)
[
  {
    "text": "识别内容",
    "location": "大致位置"
  }
]
'''

def process_single_image(pipeline, image_path, prompt, save_dir):
    """
    封装好的单张图片处理函数
    """
    try:
        print(f"正在处理: {image_path} ...")
        
        # 1. 调用模型预测
        output = pipeline.predict(image_path, prompt=prompt)

        # 2. 保存结果
        for res in output:
            # res.print() # 如果不想刷屏可以注释掉这行
            res.save_to_json(save_path=save_dir)
            
        print(f"完成: {image_path}")
        
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")

def main():
    pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server", 
        vl_rec_server_url="http://127.0.0.1:8118/v1"
    )
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    prompt = get_custom_prompt()

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    file_list = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]
    total_files = len(file_list)
    
    for index, filename in enumerate(file_list):
        full_image_path = os.path.join(INPUT_FOLDER, filename)
        
        process_single_image(pipeline, full_image_path, prompt, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
