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

# 初始化OCR管道
pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url="http://127.0.0.1:8118/v1")

# 处理文件夹内所有图片
def process_images_in_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有图片文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing: {filename}")
            output = pipeline.predict(file_path)
            save_results(output, output_folder, filename)

# 保存OCR结果到文件和可视化图片
def save_results(output, output_folder, filename):
    
    output.save_to_json(save_path=os.path.join(output_folder, f"{filename}_result.json"))
    output.save_to_markdown(save_path=os.path.join(output_folder, f"{filename}_result.md"))
    
    image = Image.open(filename) 
    image_with_results = output.visualize() 
    image_with_results.save(os.path.join(output_folder, f"{filename}_visualized.png"))


folder_path = "main_file" 
output_folder = "output_results"
process_images_in_folder(folder_path, output_folder)
