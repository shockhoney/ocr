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
from PIL import Image, ImageDraw

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
            save_results(output, output_folder, file_path)  # 传递完整路径

# 保存OCR结果到文件和可视化图片
def save_results(output, output_folder, file_path):
    # output 可能是一个列表，我们需要逐个处理每个结果
    for idx, res in enumerate(output):
        # 保存结果为JSON和Markdown
        res.save_to_json(save_path=os.path.join(output_folder, f"{os.path.basename(file_path)}_result_{idx}.json"))
        res.save_to_markdown(save_path=os.path.join(output_folder, f"{os.path.basename(file_path)}_result_{idx}.md"))
        
        # 手动绘制OCR结果的可视化图片
        try:
            image = Image.open(file_path)  # 使用完整路径打开图片
            draw = ImageDraw.Draw(image)

            # 获取OCR结果并绘制识别框
            for item in res.get("text_boxes", []):
                # item[0] 是一个四个点的坐标，绘制矩形框
                draw.polygon(item[0], outline="red", width=2)
            
            # 保存可视化的图片
            image.save(os.path.join(output_folder, f"{os.path.basename(file_path)}_visualized_{idx}.png"))
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except Exception as e:
            print(f"Error while processing {file_path}: {e}")

# 调用函数处理指定文件夹中的图片
folder_path = "main_file"  # 输入文件夹路径
output_folder = "output_results"  # 输出文件夹路径
process_images_in_folder(folder_path, output_folder)
