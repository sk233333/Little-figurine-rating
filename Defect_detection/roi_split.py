#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 ROI 切分
================
输入: canonical/<sku>/<stem>.png + _mask.png + .json
输出: rois/<sku>/<stem>/
        full.png            整图粗检图(512)
        full_valid.png      有效区(物体 mask)
        roi_000.png ...     关键区 ROI(512)
        roi_000_valid.png
        index.json          ROI 位置/尺度映射

两级方案:
    Level1 full   : 检大瑕疵(>2mm)与结构缺陷
    Level2 roi_xxx: 关键区细检，检 0.5mm

用法:
    python3 roi_split.py --sku qml_demo
    python3 roi_split.py --sku qml_demo --mode grid    # 全覆盖(块多)
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from common import load_cfg   # noqa: E402


def split_one(canon_path, cfg, out_root, mode, ref_mm_px=None):
    stem = os.path.splitext(os.path.basename(canon_path))[0]
    meta_path = canon_path.replace(".png", ".json")
    mask_path = canon_path.replace(".png", "_mask.png")
    if not (os.path.exists(meta_path) and os.path.exists(mask_path)):
        print(f"  跳过 {stem}: 缺 meta 或 mask")
        return None
    with open(meta_path) as f:
        meta = json.load(f)

    img = cv2.imread(canon_path, cv2.IMREAD_COLOR)
    msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    msk = (msk > 127).astype(np.uint8)
    C = meta["canonical_size"]
    mm_px = meta["mm_per_px_canon"]

    out_dir = os.path.join(out_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    l1 = int(cfg.get("level1_px", 512))
    cv2.imwrite(os.path.join(out_dir, "full.png"),
                cv2.resize(img, (l1, l1), interpolation=cv2.INTER_AREA))
    cv2.imwrite(os.path.join(out_dir, "full_valid.png"),
                cv2.resize(msk * 255, (l1, l1), interpolation=cv2.INTER_NEAREST))

    # ---- ROI 网格 ----
    # 网格几何必须与"哪张图"无关：参考尺度用 SKU 级固定值 ref_mm_px（见 main），
    # 否则逐图 mm_per_px 有 ±6% 波动 -> 同一 roi_id 的物理范围也会漂。
    roi_mm = float(cfg.get("roi_mm", 85.0))
    roi_px = int(round(roi_mm / float(ref_mm_px or mm_px)))
    roi_px = max(64, min(roi_px, C))
    overlap = float(cfg.get("roi_overlap", 0.12))
    step = max(1, int(roi_px * (1.0 - overlap)))
    out_px = int(cfg.get("roi_out_px", 512))

    if mode == "grid":
        regions = [[0.0, 0.0, 1.0, 1.0]]
    else:
        regions = cfg.get("key_regions") or [[0.0, 0.0, 1.0, 1.0]]

    cands = []
    seen = set()
    for (rx0, ry0, rx1, ry1) in regions:
        x0 = int(rx0 * C); x1 = int(rx1 * C)
        y0 = int(ry0 * C); y1 = int(ry1 * C)
        y = y0
        while y + roi_px <= y1 or y < y1:
            x = x0
            while x + roi_px <= x1 or x < x1:
                xa, ya = x, y
                xb, yb = min(x + roi_px, C), min(y + roi_px, C)
                if xb - xa < roi_px:
                    xa = max(0, xb - roi_px)
                if yb - ya < roi_px:
                    ya = max(0, yb - roi_px)
                key = (xa, ya)
                if key not in seen:
                    seen.add(key)
                    cands.append({"rect": [int(xa), int(ya),
                                           int(xb - xa), int(yb - ya)]})
                if x + roi_px >= x1:
                    break
                x += step
            if y + roi_px >= y1:
                break
            y += step

    # 数量上限 + 固定顺序(行优先)
    # 关键：slot 集合只由几何(regions/roi_px/step)决定，不依赖逐图的 mask 覆盖率，
    # 这样同一 SKU 下 roi_000..roi_N 在每张图里都对应同一物理部位。
    max_rois = int(cfg.get("roi_max_count", 6))
    cands.sort(key=lambda d: (d["rect"][1], d["rect"][0]))
    cands = cands[:max_rois]

    rois = []
    for i, c in enumerate(cands):
        xa, ya, w, h = c["rect"]
        patch = img[ya:ya + h, xa:xa + w]
        pmask = msk[ya:ya + h, xa:xa + w]
        idx = f"roi_{i:03d}"
        cv2.imwrite(os.path.join(out_dir, f"{idx}.png"),
                    cv2.resize(patch, (out_px, out_px),
                               interpolation=cv2.INTER_AREA))
        cv2.imwrite(os.path.join(out_dir, f"{idx}_valid.png"),
                    cv2.resize(pmask * 255, (out_px, out_px),
                               interpolation=cv2.INTER_NEAREST))
        rois.append({
            "id": idx,
            "rect": [xa, ya, w, h],
            "cover": round(float(pmask.mean()), 3),
            "mm_per_px_roi": mm_px * w / float(out_px),
        })

    index = {
        "stem": stem,
        "sku": meta["sku"],
        "canonical_size": C,
        "mm_per_px_canon": mm_px,
        "roi_px": roi_px,
        "roi_count": len(rois),
        "rois": rois,
    }
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", default=None, choices=["key", "grid"])
    args = ap.parse_args()

    cfg, _ = load_cfg(args.sku, args.config)
    mode = args.mode or cfg.get("roi_mode", "key")

    canon_dir = os.path.join(BASE, "canonical", args.sku)
    files = sorted(glob.glob(os.path.join(canon_dir, "*.png")))
    files = [f for f in files if not f.endswith("_mask.png")]
    if not files:
        print(f"没找到 canonical 图: {canon_dir}（先跑 normalize.py）")
        sys.exit(0)

    out_root = os.path.join(BASE, "rois", args.sku)

    # 第一遍：取 SKU 级参考尺度(中位数)，让所有图共用同一套 ROI 网格几何
    refs = []
    for f in files:
        mp = f.replace(".png", ".json")
        if os.path.exists(mp):
            with open(mp) as fh:
                refs.append(float(json.load(fh)["mm_per_px_canon"]))
    ref_mm_px = float(np.median(refs)) if refs else None
    print(f"[ROI] SKU={args.sku}  mode={mode}  共 {len(files)} 张 -> {out_root}")
    print(f"      参考尺度 mm_per_px={ref_mm_px:.5f} (SKU 中位数, 逐图范围 "
          f"{min(refs):.5f}~{max(refs):.5f})")

    tot = 0
    for f in files:
        idx = split_one(f, cfg, out_root, mode, ref_mm_px=ref_mm_px)
        if idx:
            tot += idx["roi_count"]
            print(f"  {idx['stem']}  ROI={idx['roi_count']}  边长={idx['roi_px']}px")
    print(f"\n[完成] 共生成 {tot} 个 ROI")
    print("提示: 数量应控制在 4~6 个/张(关键区)；过多说明 roi_mm 设小了或用了 grid")


if __name__ == "__main__":
    main()
