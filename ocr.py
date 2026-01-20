# ocr.py
# Batch OCR with outputs split into two folders: vis/ and json/
# 改进点：
# 1) Unicode 智能字符过滤（支持法语重音等非 ASCII 字符，避免误删导致“截断”）
# 2) 新增：按行聚类 + 行内拼接（解决同一行被拆成多个框、字距大导致 merge_pad 合不起来的问题）
# 3) 保留：bbox 重叠/近邻合并（连通域合并）

import os
import json
import argparse
import unicodedata
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

try:
    from paddleocr import PaddleOCR
except Exception as e:
    raise RuntimeError(
        "Cannot import paddleocr. Please install:\n"
        "  pip install paddleocr paddlepaddle\n"
    ) from e


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_images(input_dir: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for fn in os.listdir(input_dir):
        if fn.lower().endswith(exts):
            yield fn


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


# -----------------------------
# 核心：Unicode 智能字符过滤
# -----------------------------
_PUNCT_SET = set(",.!?;:()[]{}@#$%^&*_-+=~`'\"/\\<>|…“”‘’«»·•，。！？；：（）【】「」『』《》")


def _is_valid_char(c: str) -> bool:
    """
    判断单字符是否属于“有效字符”：
    - 字母/数字（Unicode 兼容：含法语重音字母）
    - 空格分隔符
    - 常见标点
    """
    if not c:
        return False

    cat = unicodedata.category(c)
    # L*: Letter, N*: Number
    if cat and cat[0] in ("L", "N"):
        return True

    # Space separators
    if cat == "Zs":
        return True

    # Common punctuation
    if c in _PUNCT_SET:
        return True

    return False


def filter_symbol_items(
    items: List[Dict[str, Any]],
    valid_ratio_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    智能过滤 OCR 识别结果中的无效文本项（Unicode 兼容）

    过滤规则：
    1) 空字符串丢弃
    2) 文本中“有效字符”占比 < valid_ratio_threshold 丢弃
    3) 不包含任何字母/数字（Unicode）的纯符号文本丢弃
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

        # 至少包含一个字母或数字（Unicode）
        if not any(ch.isalnum() for ch in text):
            continue

        filtered.append(item)

    return filtered


# -----------------------------
# Polygon helpers
# -----------------------------
def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    将 4 个点排序为顺时针顺序：
    [top-left, top-right, bottom-right, bottom-left]
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # tl
    rect[2] = pts[np.argmax(s)]      # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # tr
    rect[3] = pts[np.argmax(diff)]   # bl
    return rect


def poly_to_quad(poly: np.ndarray) -> np.ndarray:
    """
    将任意多边形近似为最小外接旋转矩形（四边形）
    """
    rect = cv2.minAreaRect(poly.astype(np.float32))
    box = cv2.boxPoints(rect)
    return order_points_clockwise(box)


def poly_to_aligned_rect(poly: np.ndarray) -> List[int]:
    """
    将 polygon 转换为轴对齐矩形（AABB）: [x1,y1,x2,y2]
    """
    x1 = int(np.min(poly[:, 0]))
    y1 = int(np.min(poly[:, 1]))
    x2 = int(np.max(poly[:, 0]))
    y2 = int(np.max(poly[:, 1]))
    return [x1, y1, x2, y2]


# -----------------------------
# Coordinate remap
# -----------------------------
def _maybe_normalized(poly: np.ndarray) -> bool:
    return (
        np.max(poly[:, 0]) <= 1.5 and
        np.max(poly[:, 1]) <= 1.5 and
        np.min(poly[:, 0]) >= -0.5 and
        np.min(poly[:, 1]) >= -0.5
    )


def remap_poly_to_image(poly: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    将 OCR 输出 polygon 坐标映射回原始图像坐标系（兼容归一化/已是原图/resize 后坐标）
    """
    poly = poly.astype(np.float32)

    # Case 1: normalized coords
    if _maybe_normalized(poly):
        poly[:, 0] *= img_w
        poly[:, 1] *= img_h
        return poly

    max_x = float(np.max(poly[:, 0]))
    max_y = float(np.max(poly[:, 1]))

    # Case 2: already image coords
    if max_x <= img_w * 1.05 and max_y <= img_h * 1.05:
        return poly

    # Case 3: det resized / padded coords
    sx = img_w / max(max_x, 1.0)
    sy = img_h / max(max_y, 1.0)
    sx = float(np.clip(sx, 0.05, 20.0))
    sy = float(np.clip(sy, 0.05, 20.0))
    poly[:, 0] *= sx
    poly[:, 1] *= sy
    return poly


# -----------------------------
# Parse OCR output
# -----------------------------
def extract_res_objects(raw):
    """
    兼容 PaddleOCR.predict 的不同返回结构，统一抽取 res 字段
    """
    outs = _to_list(raw)
    res_list = []
    for out in outs:
        if isinstance(out, dict) and isinstance(out.get("res"), dict):
            res_list.append(out["res"])
        elif isinstance(out, dict):
            res_list.append(out)
    return res_list


def build_items_from_res(res: Dict[str, Any], img_shape, min_conf: float):
    """
    从 PaddleOCR 的 res 中构建统一 OCR item 结构：
    {text, confidence, poly, bbox}
    """
    img_h, img_w = img_shape[:2]

    texts = _to_list(res.get("rec_texts", []))
    scores = _to_list(res.get("rec_scores", []))
    polys = _to_list(res.get("dt_polys", None))
    if not polys:
        polys = _to_list(res.get("rec_polys", None))

    n = min(len(texts), len(scores), len(polys))
    items = []

    for i in range(n):
        score = float(scores[i])
        if score < min_conf:
            continue

        poly = np.array(polys[i], dtype=np.float32)
        if poly.ndim != 2 or poly.shape[1] != 2:
            continue

        poly = remap_poly_to_image(poly, img_w, img_h)
        bbox = poly_to_aligned_rect(poly)

        items.append({
            "text": str(texts[i]).strip(),
            "confidence": score,
            "poly": poly.tolist(),
            "bbox": bbox,
        })

    return items


# -----------------------------
# bbox merge (overlap / contain / near by padding)
# -----------------------------
def _bbox_expand(b, pad: int):
    if pad <= 0:
        return b
    x1, y1, x2, y2 = b
    return [x1 - pad, y1 - pad, x2 + pad, y2 + pad]


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


def _bbox_contains(a, b, eps: int = 0) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (ax1 <= bx1 + eps and ay1 <= by1 + eps and ax2 >= bx2 - eps and ay2 >= by2 - eps)


def _bbox_union(a, b):
    return [
        int(min(a[0], b[0])),
        int(min(a[1], b[1])),
        int(max(a[2], b[2])),
        int(max(a[3], b[3])),
    ]


def _should_merge_bbox(a, b, merge_pad: int = 0, contain_eps: int = 2) -> bool:
    # 1) overlap
    if _bbox_intersection_area(a, b) > 0:
        return True

    # 2) containment
    if _bbox_contains(a, b, eps=contain_eps) or _bbox_contains(b, a, eps=contain_eps):
        return True

    # 3) near / touch via padding
    if merge_pad > 0:
        ea = _bbox_expand(a, merge_pad)
        eb = _bbox_expand(b, merge_pad)
        if _bbox_intersection_area(ea, eb) > 0:
            return True

    return False


def merge_overlapping_items(
    items: List[Dict[str, Any]],
    merge_pad: int = 0,
    contain_eps: int = 2
) -> List[Dict[str, Any]]:
    """
    对 OCR item 的 bbox 进行连通区域合并（重叠/包含/外扩相交）
    """
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
# NEW: line merge (解决“同一行被截断/拆框”的关键后处理)
# -----------------------------
def _bbox_h(b) -> int:
    return max(1, int(b[3] - b[1]))


def _bbox_center_y(b) -> float:
    return (b[1] + b[3]) / 2.0


def _y_overlap_ratio(a, b) -> float:
    """
    y 方向重叠比例：交集高度 / min(height_a, height_b)
    """
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    inter = max(0, min(ay2, by2) - max(ay1, by1))
    ha = max(1, ay2 - ay1)
    hb = max(1, by2 - by1)
    return inter / float(min(ha, hb))


def merge_by_lines(
    items: List[Dict[str, Any]],
    y_overlap_th: float = 0.6,
    x_gap_ratio: float = 0.6,
    center_y_ratio: float = 0.55,
) -> List[Dict[str, Any]]:
    """
    将 items 按“同一行”聚类，然后行内按 x 排序拼接。

    适用场景：
    - 艺术字体/大字距导致检测拆成多个框，merge_pad 仍合不起来
    - 一段话被拆成多个相邻框，输出看起来“截断”

    参数说明：
    - y_overlap_th: y 重叠比例阈值（越大越严格）
    - x_gap_ratio: 允许拼接的水平间隙阈值 = x_gap_ratio * 行内中位高度
    - center_y_ratio: 中心线距离阈值 = center_y_ratio * 行内中位高度（用于 y_overlap 不稳定时兜底）
    """
    if not items:
        return items

    # 按 y, x 排序，便于逐行归并
    items_sorted = sorted(items, key=lambda it: (it["bbox"][1], it["bbox"][0]))
    line_groups: List[List[Dict[str, Any]]] = []

    def line_stats(line: List[Dict[str, Any]]) -> Tuple[List[int], float]:
        hs = [_bbox_h(it["bbox"]) for it in line]
        hs_sorted = sorted(hs)
        med_h = float(hs_sorted[len(hs_sorted) // 2])
        cy = float(sum(_bbox_center_y(it["bbox"]) for it in line) / max(len(line), 1))
        return hs, (med_h, cy)

    for it in items_sorted:
        placed = False
        for line in line_groups:
            # 用该行的“当前 union bbox”作为参考
            ref_bbox = line[0]["bbox"]
            yov = _y_overlap_ratio(it["bbox"], ref_bbox)

            hs, (med_h, line_cy) = line_stats(line)
            cy_dist = abs(_bbox_center_y(it["bbox"]) - line_cy)

            if yov >= y_overlap_th or cy_dist <= center_y_ratio * med_h:
                line.append(it)
                # 更新行 reference：让 line[0] 成为 union bbox 的代表
                ub = line[0]["bbox"]
                line[0]["bbox"] = _bbox_union(ub, it["bbox"])
                placed = True
                break

        if not placed:
            # 新建行组；注意：我们会把 line[0] 当作 union 代表，所以复制一份以免污染原 item
            new_it = dict(it)
            line_groups.append([new_it])

    merged_all: List[Dict[str, Any]] = []

    for line in line_groups:
        # line[0] 是 union 代表，但也可能含真实文本；为了安全起见，仍按 bbox x 排序处理全部元素
        line_elems = sorted(line, key=lambda it: it["bbox"][0])

        heights = [_bbox_h(it["bbox"]) for it in line_elems]
        heights_sorted = sorted(heights)
        med_h = float(heights_sorted[len(heights_sorted) // 2])
        gap_th = float(x_gap_ratio * med_h)

        cur = dict(line_elems[0])
        cur_bbox = cur["bbox"][:]

        for nxt in line_elems[1:]:
            gap = float(nxt["bbox"][0] - cur_bbox[2])

            # gap < 0 说明重叠，必合并；gap 小于阈值也合并
            if gap <= gap_th:
                cur_bbox = _bbox_union(cur_bbox, nxt["bbox"])
                cur["bbox"] = cur_bbox

                # text 拼接：默认加空格，最后统一规整空格
                t1 = str(cur.get("text", "")).strip()
                t2 = str(nxt.get("text", "")).strip()
                if t1 and t2:
                    cur["text"] = (t1 + " " + t2).strip()
                else:
                    cur["text"] = (t1 + t2).strip()

                cur["confidence"] = float(max(float(cur.get("confidence", 0.0)), float(nxt.get("confidence", 0.0))))
            else:
                # finalize cur
                cur["text"] = " ".join(str(cur.get("text", "")).split())
                x1, y1, x2, y2 = cur["bbox"]
                cur["poly"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                merged_all.append(cur)

                cur = dict(nxt)
                cur_bbox = cur["bbox"][:]

        # finalize last
        cur["text"] = " ".join(str(cur.get("text", "")).split())
        x1, y1, x2, y2 = cur["bbox"]
        cur["poly"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        merged_all.append(cur)

    merged_all.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))
    return merged_all


# -----------------------------
# Visualization
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="main_file", help="Input directory containing images")
    ap.add_argument("--out_dir", default="ocr_out", help="Output directory for vis and json")
    ap.add_argument("--lang", default="en", help="OCR language (e.g., en, ch, fr)")
    ap.add_argument("--min_conf", type=float, default=0.6, help="Minimum confidence threshold")
    ap.add_argument("--valid_ratio", type=float, default=0.5, help="有效字符比例阈值，用于文本过滤")

    # overlap/near merge
    ap.add_argument("--merge_pad", type=int, default=8, help="bbox 合并时外扩像素（用于相邻留缝）")
    ap.add_argument("--contain_eps", type=int, default=2, help="bbox 包含关系判断像素容差")

    # NEW: line merge
    ap.add_argument("--enable_line_merge", action="store_true", help="启用按行聚类+行内拼接（强烈建议海报/标题使用）")
    ap.add_argument("--y_overlap_th", type=float, default=0.6, help="同一行判定：y 重叠比例阈值")
    ap.add_argument("--x_gap_ratio", type=float, default=0.6, help="行内拼接：gap 阈值 = x_gap_ratio * 行内中位高度")
    ap.add_argument("--center_y_ratio", type=float, default=0.55, help="同一行兜底：中心线距离阈值比例")

    args = ap.parse_args()

    input_path = args.input_dir
    output_path = args.out_dir

    vis_dir = os.path.join(output_path, "vis")
    json_dir = os.path.join(output_path, "json")
    ensure_dir(vis_dir)
    ensure_dir(json_dir)

    ocr = PaddleOCR(
        lang=args.lang,
        use_textline_orientation=False,
        text_det_unclip_ratio=1.6,
        use_doc_unwarping=False,
    )

    image_files = list(list_images(input_path))
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

            # 1) OCR
            raw = ocr.predict(img)
            res_objects = extract_res_objects(raw)

            # 2) build items
            items = []
            for res in res_objects:
                items.extend(build_items_from_res(res, img.shape, args.min_conf))
            print(f"[INFO] {fname} - Raw OCR items: {len(items)}")

            # 3) Unicode smart filter
            filtered_items = filter_symbol_items(items, valid_ratio_threshold=args.valid_ratio)
            print(f"[INFO] {fname} - Filtered valid items: {len(filtered_items)}")

            # 4) merge by overlap/near
            merged_items = merge_overlapping_items(
                filtered_items, merge_pad=args.merge_pad, contain_eps=args.contain_eps
            )
            print(f"[INFO] {fname} - After overlap/near merge: {len(merged_items)}")

            # 5) NEW: line merge (recommended for poster/title)
            if args.enable_line_merge:
                merged_items = merge_by_lines(
                    merged_items,
                    y_overlap_th=args.y_overlap_th,
                    x_gap_ratio=args.x_gap_ratio,
                    center_y_ratio=args.center_y_ratio,
                )
                print(f"[INFO] {fname} - After line merge: {len(merged_items)}")

            # 6) save JSON
            h, w = img.shape[:2]
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "image": fname,
                        "shape": [int(h), int(w)],
                        "items": merged_items,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # 7) save visualization
            vis = draw_vis(img, merged_items)
            cv2.imwrite(out_vis_path, vis)

            print(f"[OK] {fname} -> {len(merged_items)} valid texts\n")

        except Exception as e:
            print(f"[ERROR] {fname}: {e}\n")


if __name__ == "__main__":
    main()
