import os
import json
import argparse
from PIL import Image, ImageDraw
import cv2

try:
    import layoutparser as lp
except Exception:
    lp = None

from paddleocr import PaddleOCR


def build_layout_model():
    if lp is None:
        return None

    if hasattr(lp, "PubLayNetLayoutModel"):
        try:
            return lp.PubLayNetLayoutModel()
        except Exception:
            pass

    if hasattr(lp, "Detectron2LayoutModel"):
        try:
            return lp.Detectron2LayoutModel(
                "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={
                    0: "Text",
                    1: "Title",
                    2: "List",
                    3: "Table",
                    4: "Figure",
                },
            )
        except Exception:
            return None

    return None


model = build_layout_model()
ocr_agent = PaddleOCR(use_angle_cls=True, lang="en")


def run_ocr(image):
    if hasattr(ocr_agent, "predict"):
        return ocr_agent.predict(image)
    return ocr_agent.ocr(image)


def normalize_box(box):
    if not box:
        return None

    if isinstance(box, (list, tuple)) and len(box) == 4 and all(
        isinstance(v, (int, float)) for v in box
    ):
        x1, y1, x2, y2 = box
        return [int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))]

    if isinstance(box, (list, tuple)) and box and isinstance(box[0], (list, tuple)):
        points = [p for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
        if not points:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    return None


def parse_ocr_result(ocr_result):
    if not ocr_result:
        return []

    lines = ocr_result
    if isinstance(ocr_result, list) and len(ocr_result) == 1 and isinstance(ocr_result[0], list):
        lines = ocr_result[0]

    parsed = []
    for line in lines:
        box = None
        text = ""
        score = 0.0

        if isinstance(line, dict):
            box = line.get("points") or line.get("bbox") or line.get("box")
            text = line.get("text") or ""
            score = line.get("score", 0.0)
        elif isinstance(line, (list, tuple)):
            if len(line) >= 2:
                box = line[0]
                meta = line[1]
                if isinstance(meta, (list, tuple)):
                    if len(meta) > 0 and meta[0] is not None:
                        text = meta[0]
                    if len(meta) > 1 and meta[1] is not None:
                        score = meta[1]
                elif isinstance(meta, dict):
                    text = meta.get("text") or ""
                    score = meta.get("score", 0.0)
                elif meta is not None:
                    text = str(meta)
            elif len(line) == 1:
                box = line[0]

        rect = normalize_box(box)
        if rect is None:
            continue

        parsed.append(
            {
                "text": text,
                "bbox": rect,
                "score": float(score) if score is not None else 0.0,
            }
        )

    return parsed


def draw_boxes(image_path, boxes):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
    return image


def process_image(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = []
    boxes = []

    if model is not None:
        layout = model.detect(image_rgb)
        try:
            text_blocks = layout.filter_by(label="Text")
        except Exception:
            text_blocks = layout

        for block in text_blocks:
            x1, y1, x2, y2 = map(int, block.coordinates)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_bgr[y1:y2, x1:x2]
            ocr_result = run_ocr(crop)
            ocr_items = parse_ocr_result(ocr_result)
            text_str = " ".join([item["text"] for item in ocr_items if item["text"]])

            results.append(
                {
                    "text": text_str,
                    "bbox": [x1, y1, x2, y2],
                    "block_score": float(getattr(block, "score", 0.0)),
                }
            )
            boxes.append([x1, y1, x2, y2])
    else:
        ocr_result = run_ocr(image_bgr)
        ocr_items = parse_ocr_result(ocr_result)
        for item in ocr_items:
            results.append(
                {
                    "text": item["text"],
                    "bbox": item["bbox"],
                    "block_score": float(item["score"]),
                }
            )
            boxes.append(item["bbox"])

    return results, boxes


def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if model is None:
        print("[WARN] Layout model unavailable; using PaddleOCR detection only.")

    for img_name in image_files:
        path = os.path.join(input_dir, img_name)
        results, boxes = process_image(path)

        json_file = os.path.splitext(img_name)[0] + ".json"
        with open(os.path.join(output_dir, json_file), "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)

        vis_img = draw_boxes(path, boxes)
        vis_img.save(os.path.join(vis_dir, img_name))

        print(f"[OK] {img_name} -> {len(results)} text blocks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Input image directory")
    ap.add_argument("--output_dir", default="layout_output", help="Output directory")
    args = ap.parse_args()

    batch_process(args.input_dir, args.output_dir)
