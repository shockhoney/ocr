import os
import json
import argparse
from PIL import Image
import cv2
import layoutparser as lp
from paddleocr import PaddleOCR

# =============================
# 1. 加载 Layout 模型
# =============================

# 使用 PubLayNet 模型代替 Detectron2LayoutModel
model = lp.PubLayNetLayoutModel()

# 加载 PaddleOCR 引擎
ocr_agent = PaddleOCR(use_angle_cls=True, lang='en')  # 如果需要其他语言可修改lang

# =============================
# 2. 处理一张图像
# =============================

def process_image(image_path):
    # 读取 image
    image = cv2.imread(image_path)
    image_rgb = image[..., ::-1]  # BGR→RGB for layoutparser

    # 布局检测：得到一组 TextBlock
    layout = model.detect(image_rgb)

    # 只保留 “Text” 类型区域
    text_blocks = layout.filter_by(label="Text")

    results = []

    # 对每个识别到的文本区域做 OCR
    for block in text_blocks:
        x1, y1, x2, y2 = map(int, block.coordinates)
        crop = image_rgb[y1:y2, x1:x2]  # region crop

        # 使用 PaddleOCR 对文本区域进行 OCR 识别
        ocr_result = ocr_agent.ocr(crop, cls=True)

        # 获取 OCR 结果并合并为字符串
        text_str = " ".join([line[1][0] for line in ocr_result[0]])

        results.append({
            "text": text_str,
            "bbox": [x1, y1, x2, y2],
            "block_score": float(block.score)
        })

    return results, layout

# =============================
# 3. 批量处理目录
# =============================

def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for img_name in image_files:
        path = os.path.join(input_dir, img_name)
        results, layout = process_image(path)

        # 输出 JSON
        json_file = os.path.splitext(img_name)[0] + ".json"
        with open(os.path.join(output_dir, json_file), "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)

        # 可视化 Layout 检测 & OCR 文本框
        vis_img = layout.draw(
            image=Image.open(path),
            box_width=3,
            box_color="green",
            show_element_id=False
        )
        vis_img.save(os.path.join(vis_dir, img_name))

        print(f"[OK] {img_name} -> {len(results)} text blocks")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="待处理图片文件夹路径")
    ap.add_argument("--output_dir", default="layout_output", help="输出结果目录")
    args = ap.parse_args()

    batch_process(args.input_dir, args.output_dir)
