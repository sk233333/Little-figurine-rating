#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
玩偶瑕疵检测 第一版
==================
输入：上一步 SAM 抠好的白底图
输出：标记图 + JSON 报告

目录结构：
    Little-figurine-rating/
    ├── Punctuation_selection_of_images/
    │   └── image_assist_ans/      <- 白底图输入（*_white.png）
    └── Defect_detection/
        ├── defect_detect.py
        ├── ref/                   <- OK 标准品参考图（可选，放一张最好的）
        └── out/                   <- 检测结果
            ├── qml_001_defect.png 标记图
            └── report.json        报告

两种模式：
    1) ref 模式（推荐）：有 OK 参考图 → 差分检测，最准
       放一张确认无瑕疵的同机位图到 ref/ 即可
    2) solo 模式：无参考 → 通用规则（色差 + 划痕 + 对称性）
       注意：solo 模式会把玩偶本身的深色部件（黑头、深眼）也算进去，
       属于误报，需配合分区或人工复核。能用 ref 就用 ref。

用法：
    python3 defect_detect.py
    python3 defect_detect.py --only qml_001
    python3 defect_detect.py --height-mm 100        玩偶实际高度，用于mm换算
    python3 defect_detect.py --mm-per-px 0.05       直接指定比例
    python3 defect_detect.py --no-symmetry          关掉对称性检测
"""

import argparse
import glob
import json
import math
import os

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

# 白底图目录（上一步的输出）
DEFAULT_INPUT = os.path.join(
    os.path.dirname(BASE),                    # Little-figurine-rating/
    "Punctuation_selection_of_images",
    "image_assist_ans",
)
REF_DIR = os.path.join(BASE, "ref")
OUT_DIR = os.path.join(BASE, "out")


# ------------------------------------------------------------------
# 分区定义：玩偶站立，按垂直位置切 A/B/C 面
# ratio 是相对主体 bbox 高度的比例
# ------------------------------------------------------------------
ZONES = [
    # name,  起始比例, 结束比例, 等效直径阈值(mm), 允许数量
    ("A", 0.00, 0.40, 0.20, 1),    # 头/脸/胸：最严
    ("B", 0.40, 0.80, 0.30, 2),    # 身/手臂：中等
    ("C", 0.80, 1.01, 0.50, 3),    # 腿/脚/底：宽松
]

# 判级规则
GRADE_RULES = [
    # (等级, A面上限, 总数量上限, 最大单瑕疵直径mm)
    ("全新",     0, 0, 0.0),
    ("微瑕",     0, 2, 0.5),
    ("明显瑕疵", 1, 4, 1.0),
]


# ------------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------------
def body_mask(img):
    """从白底图分离玩偶主体：非白区域即主体"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = (gray < 240).astype(np.uint8) * 255
    # 去噪
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    # 只留最大连通域
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == idx).astype(np.uint8) * 255
    return m


def bbox_of(m):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def equiv_diameter_mm(area_px, mm_per_px):
    """像素面积 -> 等效直径(mm)"""
    area_mm2 = area_px * (mm_per_px ** 2)
    return 2.0 * math.sqrt(area_mm2 / math.pi)


def clean_mask(m, open_ksize=3, close_ksize=3, min_area=4):
    """形态学去噪 + 去掉过小连通域"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255
    return out


def collect(mask, mm_per_px):
    """连通域 -> 瑕疵列表"""
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    items = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x, y = int(cents[i][0]), int(cents[i][1])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        d = equiv_diameter_mm(area, mm_per_px)
        items.append({
            "area_px": area, "cx": x, "cy": y,
            "w": w, "h": h, "diameter_mm": round(d, 3),
        })
    return items


# ------------------------------------------------------------------
# 检测方式一：参考图差分（最准）
# ------------------------------------------------------------------
def detect_by_ref(img, ref, body):
    """
    和 OK 标准品做差分，差异区域即瑕疵
    要求：两张图同机位、同姿态、同光照
    """
    if ref is None or ref.shape != img.shape:
        return np.zeros(img.shape[:2], np.uint8), "ref尺寸不匹配，跳过"

    # 转到 LAB，对颜色差异更敏感
    a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    b = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.int16)
    diff = np.abs(a - b).sum(axis=2)

    # 差异阈值：调小更敏感，调大更保守
    _, m = cv2.threshold(diff.astype(np.uint8), 40, 255, cv2.THRESH_BINARY)

    # 只在主体范围内
    m = cv2.bitwise_and(m, body)
    m = clean_mask(m, open_ksize=3, close_ksize=5, min_area=6)
    return m, "差分"


# ------------------------------------------------------------------
# 检测方式二：无参考，通用规则
# ------------------------------------------------------------------
def detect_solo(img, body, use_symmetry=True):
    """
    无参考图时的通用检测：
      1) 色差/污渍：LAB 空间中偏离主体的色块
      2) 划痕：细长边缘
      3) 对称性：左右不对称处（玩偶正面站立，天然对称）
    """
    h, w = img.shape[:2]

    # --- 1) 深色异常点（黑点、污渍、掉漆露底）---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # 主体内部的亮度分布，取较暗的部分作为候选
    body_px = L[body > 0]
    if len(body_px) > 0:
        # 用分位数自适应：比主体整体亮度明显偏暗的算异常
        thr = int(np.percentile(body_px, 8))
        m_dark = ((L < thr) & (L > 5)).astype(np.uint8) * 255
    else:
        m_dark = np.zeros((h, w), np.uint8)
    m_dark = cv2.bitwise_and(m_dark, body)

    # --- 2) 划痕：细长结构 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    # 只保留细长的：闭运算后仍细的
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    m_scratch_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    m_scratch_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k2)
    m_scratch = cv2.bitwise_or(m_scratch_h, m_scratch_v)
    m_scratch = cv2.bitwise_and(m_scratch, body)

    m = cv2.bitwise_or(m_dark, m_scratch)

    # --- 3) 对称性检测（可选）---
    if use_symmetry:
        flipped = cv2.flip(img, 1)
        d = cv2.absdiff(img, flipped)
        dg = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        _, m_sym = cv2.threshold(dg, 45, 255, cv2.THRESH_BINARY)
        m_sym = cv2.bitwise_and(m_sym, body)
        m_sym = clean_mask(m_sym, 3, 5, 12)
        # 对称性只作为补充，取交集区域的一半避免整片
        m = cv2.bitwise_or(m, cv2.bitwise_and(m_sym, m_dark))

    m = clean_mask(m, open_ksize=3, close_ksize=5, min_area=6)
    return m, "规则"


# ------------------------------------------------------------------
# 分区 + 判级
# ------------------------------------------------------------------
def assign_zone(cy, y1, height, zones):
    r = (cy - y1) / max(1.0, float(height))
    for name, r0, r1, thr, allow in zones:
        if r0 <= r < r1:
            return name
    return zones[-1][0]


def grade(items, zones, bbox_h_mm=None):
    """按区域阈值过滤，再按数量判级"""
    zone_thr = {z[0]: z[3] for z in zones}
    zone_allow = {z[0]: z[4] for z in zones}

    kept = []
    for it in items:
        z = it["zone"]
        if it["diameter_mm"] >= zone_thr.get(z, 0.3):
            kept.append(it)

    n_a = sum(1 for i in kept if i["zone"] == "A")
    total = len(kept)
    max_d = max([i["diameter_mm"] for i in kept], default=0.0)

    if n_a == 0 and total == 0:
        g = "全新"
    elif n_a == 0 and total <= 2 and max_d < 0.5:
        g = "微瑕"
    elif n_a <= 1 and total <= 4 and max_d < 1.0:
        g = "明显瑕疵"
    else:
        g = "残次"

    return kept, g, {"A": n_a, "total": total, "max_diameter_mm": round(max_d, 3)}


# ------------------------------------------------------------------
# 可视化
# ------------------------------------------------------------------
def draw(img, items, bbox, out_path, stem, grade_name, mode):
    vis = img.copy()
    x1, y1, x2, y2 = bbox

    # 画分区线
    H = max(1, y2 - y1)
    for name, r0, r1, thr, allow in ZONES:
        yy = int(y1 + H * r1)
        if yy < img.shape[0]:
            cv2.line(vis, (x1, yy), (x2, yy), (200, 200, 200), 1)
            cv2.putText(vis, f"{name}", (x2 + 6, yy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    color = {"A": (0, 0, 255), "B": (0, 140, 255), "C": (0, 200, 200)}
    for it in items:
        c = color.get(it["zone"], (0, 0, 255))
        cv2.circle(vis, (it["cx"], it["cy"]), max(8, int(it["diameter_mm"] * 20)),
                   c, 2)
        cv2.putText(vis, f"{it['zone']} {it['diameter_mm']}mm",
                    (it["cx"] + 10, it["cy"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)

    cv2.putText(vis, f"{stem}  {grade_name}  [{mode}]  n={len(items)}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(out_path, vis)


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT, help="白底图目录")
    ap.add_argument("--ref", default=REF_DIR, help="OK标准品参考图目录")
    ap.add_argument("--out", default=OUT_DIR, help="输出目录")
    ap.add_argument("--only", default=None)
    ap.add_argument("--height-mm", type=float, default=100.0,
                    help="玩偶实际高度(mm)，用于像素->毫米换算")
    ap.add_argument("--mm-per-px", type=float, default=None,
                    help="直接指定毫米/像素，优先于 --height-mm")
    ap.add_argument("--no-symmetry", action="store_true", help="关掉对称性检测")
    ap.add_argument("--show", action="store_true", help="弹窗显示结果")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not os.path.isdir(args.input):
        print(f"白底图目录不存在: {args.input}")
        print("先跑上一步生成白底图，或用 --input 指定")
        return

    # 找参考图
    ref_path = None
    if os.path.isdir(args.ref):
        cands = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            cands.extend(glob.glob(os.path.join(args.ref, ext)))
        if cands:
            ref_path = sorted(cands)[0]
            print(f"[参考] {ref_path}")

    files = []
    for pat in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(glob.glob(os.path.join(args.input, pat)))
    files = sorted(set(files))
    if args.only:
        files = [f for f in files if args.only in os.path.basename(f)]
    if not files:
        print(f"没找到白底图: {args.input}")
        return

    print(f"[任务] {len(files)} 张   模式={'差分' if ref_path else '规则'}")
    print("-" * 60)

    report = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        stem = stem.replace("_white", "")      # 去掉白底图后缀

        img = cv2.imread(f)
        if img is None:
            print(f"  {stem}  读取失败，跳过")
            continue

        body = body_mask(img)
        bb = bbox_of(body)
        if bb is None:
            print(f"  {stem}  没找到主体，跳过")
            continue
        x1, y1, x2, y2 = bb
        h_px = max(1, y2 - y1)

        # 像素 -> 毫米
        if args.mm_per_px:
            mm_per_px = args.mm_per_px
        else:
            mm_per_px = args.height_mm / h_px

        # 检测
        if ref_path:
            ref = cv2.imread(ref_path)
            ref = cv2.resize(ref, (img.shape[1], img.shape[0])) \
                if ref.shape != img.shape else ref
            mask, mode = detect_by_ref(img, ref, body)
        else:
            mask, mode = detect_solo(img, body, use_symmetry=not args.no_symmetry)

        items = collect(mask, mm_per_px)

        # 分区
        for it in items:
            it["zone"] = assign_zone(it["cy"], y1, h_px, ZONES)

        # 判级
        kept, g, stat = grade(items, ZONES)

        # 输出
        op = os.path.join(args.out, f"{stem}_defect.png")
        draw(img, kept, bb, op, stem, g, mode)

        report.append({
            "file": os.path.basename(f),
            "grade": g,
            "mode": mode,
            "mm_per_px": round(mm_per_px, 5),
            "defects": kept,
            "stat": stat,
        })

        print(f"  {stem}  {g}  A面{stat['A']}处 合计{stat['total']}处 "
              f"最大{stat['max_diameter_mm']}mm")

        if args.show:
            cv2.imshow(f"{stem} {g}", cv2.imread(op))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    rp = os.path.join(args.out, "report.json")
    with open(rp, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    print("-" * 60)
    print(f"[完成] 标记图 + report.json -> {args.out}")
    if not ref_path:
        print("\n[建议] 放一张确认无瑕疵的同机位图到 ref/ 目录，")
        print("       可切换到差分模式，准确率会明显提升。")


if __name__ == "__main__":
    main()
