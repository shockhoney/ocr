import os
import json
import argparse
import unicodedata
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from paddleocr import PaddleOCR

# -----------------------------
# 核心：Unicode 智能字符过滤
# -----------------------------
_PUNCT_SET = set(",.!?;:()[]{}@#$%^&*_-+=~`'\"/\\<>|…“”‘’«»·•，。！？；：（）【】「」『』《》")

def _is_valid_char(c: str) -> bool:
    """
    判断单字符是否属于“有效字符”：字母/数字、空格和常见标点
    """
    if not c:
        return False

    cat = unicodedata.category(c)
    if cat and cat[0] in ("L", "N"):
        return True
    if cat == "Zs":
        return True
    if c in _PUNCT_SET:
        return True

    return False


def filter_symbol_items(items: List[Dict[str, Any]], valid_ratio_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    智能过滤 OCR 识别结果中的无效文本项（Unicode 兼容）
    """
    filtered = []
    for item in items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        valid_char_count = sum(1 for c in text if _is_valid_char(c))
        valid_ratio = valid_char_count / max(len(text), 1)
        if valid_ratio < valid_ratio_threshold:
            continue
        if not any(ch.isalnum() for ch in text):
            continue
        filtered.append(item)
    return filtered

# -----------------------------
# 文本框合并（重叠、近邻）
# -----------------------------
def _bbox_intersection_area(a, b) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    w = ix2 - ix1
    h = iy2 - iy1
    if w <= 0 or h <= 0:
        return 0
    return int(w * h)


def _should_merge_bbox(a, b, merge_pad: int = 0, contain_eps: int = 2) -> bool:
    # 重叠或近邻判断
    if _bbox_intersection_area(a, b) > 0:
        return True
    return False


def merge_overlapping_items(items: List[Dict[str, Any]], merge_pad: int = 0, contain_eps: int = 2) -> List[Dict[str, Any]]:
    if not items:
        return items

    n = len(items)
    bboxes = [it["bbox"][:] for it in items]
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    for i in range(n):
        for j in range(i + 1, n):
            if _should_merge_bbox(bboxes[i], bboxes[j], merge_pad=merge_pad, contain_eps=contain_eps):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    merged_items: List[Dict[str, Any]] = []
    for _, idxs in groups.items():
        mb = bboxes[idxs[0]]
        for k in idxs[1:]:
            mb = _bbox_union(mb, bboxes[k])

        idxs_sorted = sorted(idxs, key=lambda ii: (bboxes[ii][1], bboxes[ii][0]))
        merged_text = " ".join(
            [str(items[ii].get("text", "")).strip() for ii in idxs_sorted if str(items[ii].get("text", "")).strip()]
        ).strip()

        merged_conf = float(max(float(items[ii].get("confidence", 0.0)) for ii in idxs))

        x1, y1, x2, y2 = mb
        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        merged_items.append({
            "text": merged_text,
            "confidence": merged_conf,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "poly": poly,
            "merged_from": idxs_sorted,
        })

    merged_items.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))
    return merged_items

# -----------------------------
# 按行聚类与拼接
# -----------------------------
def merge_by_lines(items: List[Dict[str, Any]], y_overlap_th: float = 0.6, x_gap_ratio: float = 0.6, center_y_ratio: float = 0.55) -> List[Dict[str, Any]]:
    if not items:
        return items

    items_sorted = sorted(items, key=lambda it: (it["bbox"][1], it["bbox"][0]))
    line_groups: List[List[Dict[str, Any]]] = []

    for it in items_sorted:
        placed = False
        for line in line_groups:
            ref_bbox = line[0]["bbox"]
            yov = _y_overlap_ratio(it["bbox"], ref_bbox)
            if yov >= y_overlap_th:
                line.append(it)
                line[0]["bbox"] = _bbox_union(line[0]["bbox"], it["bbox"])
                placed = True
                break
        if not placed:
            new_it = dict(it)
            line_groups.append([new_it])

    merged_all: List[Dict[str, Any]] = []
    for line in line_groups:
        line_elems = sorted(line, key=lambda it: it["bbox"][0])
        cur = dict(line_elems[0])
        cur_bbox = cur["bbox"][:]
        for nxt in line_elems[1:]:
            gap = float(nxt["bbox"][0] - cur_bbox[2])
            if gap <= 0.6:
                cur_bbox = _bbox_union(cur_bbox, nxt["bbox"])
                cur["bbox"] = cur_bbox
                cur["text"] = (cur.get("text", "") + " " + nxt.get("text", "")).strip()
            else:
                cur["text"] = " ".join(str(cur.get("text", "")).split())
                cur["poly"] = [[cur["bbox"][0], cur["bbox"][1]], [cur["bbox"][2], cur["bbox"][1]], [cur["bbox"][2], cur["bbox"][3]], [cur["bbox"][0], cur["bbox"][3]]]
                merged_all.append(cur)
                cur = dict(nxt)
                cur_bbox = cur["bbox"][:]
        cur["text"] = " ".join(str(cur.get("text", "")).split())
        merged_all.append(cur)

    merged_all.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))
    return merged_all

# -----------------------------
# 提取OCR结果的函数
# -----------------------------
def extract_res_objects(raw):
    """
    兼容 PaddleOCR.predict 的不同返回结构，统一抽取 res 字段
    """
    outs = [raw] if isinstance(raw, dict) else raw
    res_list = []
    for out in outs:
        if isinstance(out, dict) and "res" in out:
            res_list.append(out["res"])
        else:
            res_list.append(out)
    return res_list

# -----------------------------
# 保存输出文件
# -----------------------------
def save_output(merged_items, img, vis_path, json_path):
    h, w = img.shape[:2]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": vis_path,
                "shape": [int(h), int(w)],
                "items": merged_items,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    vis = draw_vis(img, merged_items)
    cv2.imwrite(vis_path, vis)

# -----------------------------
# 可视化函数
# -----------------------------
def draw_vis(img, items):
    vis = img.copy()
    h, w = vis.shape[:2]
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

        label = f'{it.get("text","")} ({float(it.get("confidence",0)):.2f})'
        cv2.putText(
            vis, label, (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1
        )
    return vis

# -----------------------------
# 主函数：批量处理文件夹
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="main_file", help="Input directory containing images")
    ap.add_argument("--out_dir", default="output", help="Output directory for visualization and json files")
    ap.add_argument("--lang", default="en", help="OCR language (e.g., en, ch, fr)")
    ap.add_argument("--min_conf", type=float, default=0.6, help="Minimum confidence threshold")
    ap.add_argument("--valid_ratio", type=float, default=0.5, help="有效字符比例阈值，用于文本过滤")
    ap.add_argument("--merge_pad", type=int, default=8, help="bbox 合并时外扩像素")
    ap.add_argument("--contain_eps", type=int, default=2, help="bbox 包含关系判断像素容差")
    ap.add_argument("--enable_line_merge", action="store_true", help="启用按行聚类+行内拼接")
    ap.add_argument("--y_overlap_th", type=float, default=0.6, help="同一行判定：y 重叠比例阈值")
    ap.add_argument("--x_gap_ratio", type=float, default=0.6, help="行内拼接：gap 阈值")
    ap.add_argument("--center_y_ratio", type=float, default=0.55, help="中心线距离阈值比例")

    args = ap.parse_args()

    input_path = args.input_dir
    output_path = args.out_dir
    vis_dir = os.path.join(output_path, "vis")
    json_dir = os.path.join(output_path, "json")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    ocr = PaddleOCR(lang=args.lang)

    image_files = [f for f in os.listdir(input_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"[INFO] Found {len(image_files)} images")

    for fname in image_files:
        in_path = os.path.join(input_path, fname)
        stem = os.path.splitext(fname)[0]

        out_vis_path = os.path.join(vis_dir, f"{stem}_vis.jpg")
        out_json_path = os.path.join(json_dir, f"{stem}.json")

        try:
            img = cv2.imread(in_path)
            if img is None:
                print(f"[WARN] Cannot read {fname}")
                continue

            # OCR识别
            raw = ocr.predict(img)
            res_objects = extract_res_objects(raw)

            items = []
            for res in res_objects:
                items.extend(build_items_from_res(res, img.shape, args.min_conf))

            # 过滤无效字符
            filtered_items = filter_symbol_items(items)

            # 合并重叠文本框
            merged_items = merge_overlapping_items(filtered_items)

            # 按行聚类
            if args.enable_line_merge:
                merged_items = merge_by_lines(merged_items)

            save_output(merged_items, img, out_vis_path, out_json_path)

            print(f"[OK] {fname} -> {len(merged_items)} valid texts")

        except Exception as e:
            print(f"[ERROR] {fname}: {e}\n")

if __name__ == "__main__":
    main()
