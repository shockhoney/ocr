# from paddleocr import PaddleOCRVL
#指向启动的vllm服务
# pipeline = PaddleOCRVL(vl_rec_backend = "vllm-server",vl_rec_server_url = "http://127.0.0.1:8118/v1")

# output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png")

# for res in output:
#     res.print()
#     res.save_to_json(save_path="output")
#     res.save_to_markdown(save_path="output")


import os
import json
from paddleocr import PaddleOCRVL
from PIL import Image

# 指向启动的vllm服务
pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url="http://127.0.0.1:8118/v1")

# 批量处理文件夹中的图片并保存结果
def process_images_in_folder(folder_path, output_folder):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 获取文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 只处理图片文件
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"Processing {filename}...")

            # 使用PaddleOCRVL进行预测
            output = pipeline.predict(file_path)

            # 确保输出文件夹存在
            os.makedirs(output_folder, exist_ok=True)

            # 保存结果到JSON文件
            output_json_path = os.path.join(output_folder, f"{filename}_result.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=4)

            # 保存结果到Markdown文件
            output_markdown_path = os.path.join(output_folder, f"{filename}_result.md")
            with open(output_markdown_path, 'w', encoding='utf-8') as f:
                f.write(str(output))  # 将结果转换为字符串保存到Markdown

            # 可视化图片保存
            result_image_path = os.path.join(output_folder, f"{filename}_visualized.png")
            image = Image.open(file_path)  # 打开原始图片
            output.save_to_image(image, result_image_path)  # 保存带有识别结果的可视化图片

            print(f"Processed {filename}, results saved to {output_folder}")

# 设定文件夹路径
input_folder = "main_file"  # 这里替换为你的图片文件夹路径
output_folder = "path_to_output_folder"    # 这里替换为你希望保存结果的文件夹路径

# 批量处理文件夹中的图片
process_images_in_folder(input_folder, output_folder)
