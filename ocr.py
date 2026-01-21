import cv2
import json
import os
from paddleocr import PaddleOCR

# 简化后的参数
y_overlap_th = 0.6  # 同一行判定：y重叠比例阈值
x_gap_ratio = 0.6   # 行内拼接：gap阈值 = x_gap_ratio * 行内中位高度
center_y_ratio = 0.55  # 同一行兜底：中心线距离阈值比例
merge_pad = 4        # 合并时外扩像素，减少过度合并
contain_eps = 2      # bbox包含判断容差

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # 选择英文OCR

# 处理文件夹中的所有图片
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取文件夹中所有图片
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'webp'))]
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        print(f"Processing {image_file}...")
        
        vis_img, merged_items = process_image(image_path)

        # 保存可视化图片
        vis_img_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_vis.png")
        cv2.imwrite(vis_img_path, vis_img)

        # 保存识别结果JSON
        json_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_items, f, ensure_ascii=False, indent=2)

# 处理单张图片
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Cannot read {image_path}")
        return

    # 使用正确的OCR方法，不传入cls参数
    result = ocr.ocr(image_path)

    # 解析OCR输出
    items = []
    for line in result[0]:
        text = line[1][0]
        bbox = line[0]
        items.append({"text": text, "bbox": bbox})

    # 行合并
    merged_items = merge_by_lines(items)

    # 可视化
    vis_img = draw_vis(img, merged_items)
    return vis_img, merged_items

# 根据行合并文本框
def merge_by_lines(items):
    items_sorted = sorted(items, key=lambda it: (it["bbox"][1], it["bbox"][0]))  # 按y和x排序
    line_groups = []

    # 遍历每个项，按行分组
    for it in items_sorted:
        placed = False
        for line in line_groups:
            ref_bbox = line[0]["bbox"]
            y_overlap = _y_overlap_ratio(it["bbox"], ref_bbox)
            if y_overlap >= y_overlap_th:
                line.append(it)
                break
        if not placed:
            line_groups.append([it])

    # 将每行的文本合并
    merged_all = []
    for line in line_groups:
        line_sorted = sorted(line, key=lambda it: it["bbox"][0])
        merged_text = " ".join([it["text"] for it in line_sorted])
        merged_all.append({"text": merged_text, "bbox": line_sorted[0]["bbox"]})

    return merged_all

# 计算y轴重叠比例
def _y_overlap_ratio(a, b):
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    inter = max(0, min(ay2, by2) - max(ay1, by1))
    ha = max(1, ay2 - ay1)
    hb = max(1, by2 - by1)
    return inter / min(ha, hb)

# 可视化文本框
def draw_vis(img, items):
    vis = img.copy()
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(vis, it["text"], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return vis

# 批量处理文件夹
if __name__ == "__main__":
    input_folder = 'main_file'  # 输入图片文件夹路径
    output_folder = 'your_output_folder'  # 输出结果文件夹路径
    process_folder(input_folder, output_folder)
