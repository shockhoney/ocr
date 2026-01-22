# from paddleocr import PaddleOCRVL
#指向启动的vllm服务
# pipeline = PaddleOCRVL(vl_rec_backend = "vllm-server",vl_rec_server_url = "http://127.0.0.1:8118/v1")

# output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png")

# for res in output:
#     res.print()
#     res.save_to_json(save_path="output")
#     res.save_to_markdown(save_path="output")


import os
from paddleocr import PaddleOCRVL
from PIL import Image

# 指向启动的vllm服务
pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url="http://127.0.0.1:8118/v1")

def process_images_in_folder(folder_path, output_folder):
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"Processing {filename}...")
            
            # 使用PaddleOCRVL进行预测
            output = pipeline.predict(file_path)

            # 保存结果到JSON和Markdown
            output_json_path = os.path.join(output, f"{filename}_result.json")
            output_markdown_path = os.path.join(output, f"{filename}_result.md")
            output.save_to_json(save_path=output_json_path)
            output.save_to_markdown(save_path=output_markdown_path)

            # 可视化图片保存
            image = Image.open(file_path)
            result_image_path = os.path.join(output, f"{filename}_visualized.png")
            output.save_to_image(image, result_image_path) 

input_folder = "main_file"  
output = "ocr_output"   

os.makedirs(output, exist_ok=True)

process_images_in_folder(input_folder, output_folder)
