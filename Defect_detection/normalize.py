#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 归一化
=============
输入: SAM 白底抠图 (image_assist_ans/*_white.png，或 --raw-dir 指定)
      默认目录会自动搜索（白底图在兄弟项目 Punctuation_selection_of_images/ 下）
输出: canonical/<sku>/<stem>.png      正方形、物体居中、统一尺度
      canonical/<sku>/<stem>_mask.png 物体 mask
      canonical/<sku>/<stem>.json     meta(含 mm/px 标定)

用法:
    python3 normalize.py --sku qml_demo
    python3 normalize.py --sku qml_demo --mask-dir ../masks   # 有 mask 时优先用 mask
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

from common import load_cfg, resolve_raw_dir   # noqa: E402


def derive_mask_from_white(img, tol=12):
    """白底图 -> 物体 mask。物体本身有大面积纯白时不可靠，此时请用 --mask-dir"""
    d = 255 - img.min(axis=2)
    return (d > tol).astype(np.uint8)


def find_mask(mask_dir, stem):
    if not mask_dir:
        return None
    for ext in (".png", ".jpg", ".jpeg", ".PNG"):
        for name in (f"{stem}_mask{ext}", f"{stem}{ext}"):
            p = os.path.join(mask_dir, name)
            if os.path.exists(p):
                m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    return (m > 127).astype(np.uint8)
    return None


def biggest_component(m):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return m
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == idx).astype(np.uint8)


def mask_pose(m):
    """由物体 mask 的二阶矩求姿态：质心、相对竖直方向的倾角(度)、伸长比。

    手办在不同照片里摆放角度不同，只按 bbox 裁剪会让各图的尺度/朝向全都不一样，
    后续 ROI 切分和 PatchCore 的 patch 就对不上同一部位。这里用主轴把姿态量出来。
    """
    ys, xs = np.nonzero(m)
    cx, cy = float(xs.mean()), float(ys.mean())
    dx, dy = xs - cx, ys - cy
    mu20 = float((dx * dx).mean())
    mu02 = float((dy * dy).mean())
    mu11 = float((dx * dy).mean())
    tilt = 0.5 * np.degrees(np.arctan2(2.0 * mu11, mu02 - mu20))
    elong = max(mu20, mu02) / max(min(mu20, mu02), 1e-9)
    return cx, cy, tilt, elong


def align_canvas(img, m, pad_ratio=0.06, min_elong=1.10):
    """把物体按 mask 主轴摆正 + 居中 + 统一尺度，返回 (画布, mask, side)。

    绕质心旋转 -tilt 使主轴竖直，再按旋转后 bbox 的较长边撑满正方形画布。
    物体接近圆形时主轴不可靠（伸长比过小），此时不旋转，只做居中+缩放。
    """
    H, W = img.shape[:2]
    cx, cy, tilt, elong = mask_pose(m)
    if elong < min_elong:
        tilt = 0.0
    R = cv2.getRotationMatrix2D((cx, cy), -tilt, 1.0)

    ys, xs = np.nonzero(m)
    qx = R[0, 0] * xs + R[0, 1] * ys + R[0, 2]
    qy = R[1, 0] * xs + R[1, 1] * ys + R[1, 2]
    bw = float(qx.max() - qx.min())
    bh = float(qy.max() - qy.min())
    side = max(4, int(max(bw, bh) * (1.0 + pad_ratio)))

    bcx = (qx.max() + qx.min()) / 2.0
    bcy = (qy.max() + qy.min()) / 2.0
    R[0, 2] += side / 2.0 - bcx
    R[1, 2] += side / 2.0 - bcy

    canvas = cv2.warpAffine(img, R, (side, side), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))
    cmask = cv2.warpAffine(m * 255, R, (side, side),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return canvas, cmask, side, bh


def normalize_one(img_path, cfg, mask_dir, out_dir, pad_ratio=0.06):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  跳过 {stem}: 读图失败")
        return None
    H, W = img.shape[:2]

    # ---- mask ----
    m = find_mask(mask_dir, stem)
    src = "mask_file"
    if m is None:
        m = derive_mask_from_white(img)
        src = "derived(非白)"
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    m = biggest_component(m)
    if m.sum() == 0:
        print(f"  跳过 {stem}: mask 为空")
        return None

    ys, xs = np.where(m > 0)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    obj_h_px = y2 - y1 + 1
    obj_w_px = x2 - x1 + 1

    # ---- canonical 尺寸 ----
    obj_h_mm = float(cfg["object_height_mm"])
    target_px = obj_h_mm * float(cfg.get("canon_target_px_per_mm", 6.67))
    canon = int(min(target_px, float(cfg.get("canon_max_px", 3072))))
    canon = int(max(canon, int(cfg.get("canon_min_px", 512))))

    # ---- 摆正(按 mask 主轴) + 居中 + 统一尺度 ----
    canvas, cmask, side, bh = align_canvas(img, m, pad_ratio)

    # ---- mm/px 标定（原图尺度，用摆正后的竖直高度，倾角不再虚高物体高度）----
    if cfg.get("mm_per_px_orig"):
        mm_px_orig = float(cfg["mm_per_px_orig"])
    else:
        mm_px_orig = float(cfg["object_height_mm"]) / bh

    # ---- resize 到 canonical ----
    canon_img = cv2.resize(canvas, (canon, canon), interpolation=cv2.INTER_AREA)
    canon_mask = cv2.resize(cmask, (canon, canon), interpolation=cv2.INTER_NEAREST)
    canon_mask = (canon_mask > 127).astype(np.uint8)

    mm_px_canon = mm_px_orig * side / float(canon)

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), canon_img)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_mask.png"), canon_mask * 255)

    meta = {
        "sku": cfg.get("sku"),
        "stem": stem,
        "src": img_path,
        "mask_src": src,
        "orig_w": W, "orig_h": H,
        "obj_bbox_orig": [x1, y1, x2, y2],
        "obj_w_px": obj_w_px, "obj_h_px": obj_h_px,
        "mm_per_px_orig": mm_px_orig,
        "canonical_size": canon,
        "mm_per_px_canon": mm_px_canon,
        "object_height_mm": obj_h_mm,
        "aligned": True,
    }
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def realign_one(canon_png, pad_ratio=0.06):
    """对已生成的 canonical 图原地重做摆正（原图已丢失时用）。

    canonical 图只是原图按 bbox 裁的方图、没有旋转，所以可以直接在它上面
    再求一次 mask 主轴并摆正；分辨率损失就是 canonical 本身的分辨率。
    """
    stem = os.path.splitext(os.path.basename(canon_png))[0]
    mask_png = canon_png.replace(".png", "_mask.png")
    meta_png = canon_png.replace(".png", ".json")
    img = cv2.imread(canon_png, cv2.IMREAD_COLOR)
    m = cv2.imread(mask_png, cv2.IMREAD_GRAYSCALE)
    if img is None or m is None:
        print(f"  跳过 {stem}: 缺图或 mask")
        return None
    with open(meta_png) as f:
        meta = json.load(f)
    if meta.get("aligned"):
        print(f"  跳过 {stem}: 已摆正过（避免重复旋转）")
        return None

    m = biggest_component((m > 127).astype(np.uint8))
    canon = int(meta.get("canonical_size", img.shape[0]))
    canvas, cmask, side, _ = align_canvas(img, m, pad_ratio)
    ci = cv2.resize(canvas, (canon, canon), interpolation=cv2.INTER_AREA)
    cm = cv2.resize(cmask, (canon, canon), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(canon_png, ci)
    cv2.imwrite(mask_png, (cm > 127).astype(np.uint8) * 255)
    # 这里 mm/px 的参照系是「旧 canonical」，不是原图
    meta["mm_per_px_canon"] = float(meta["mm_per_px_canon"]) * side / float(canon)
    meta["aligned"] = True
    with open(meta_png, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw-dir", default=None,
                    help="白底图目录，默认自动搜索 "
                         "Punctuation_selection_of_images/image_assist_ans")
    ap.add_argument("--mask-dir", default=None,
                    help="mask 目录(可选)，有则优先使用，避免白色物体误判")
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--realign", action="store_true",
                    help="不读原图，直接对已有的 canonical/<sku>/ 原地重做摆正")
    args = ap.parse_args()

    cfg, _ = load_cfg(args.sku, args.config)

    if args.realign:
        canon_dir = os.path.join(BASE, "canonical", args.sku)
        files = sorted(glob.glob(os.path.join(canon_dir, "*.png")))
        files = [f for f in files if not f.endswith("_mask.png")]
        if not files:
            print(f"没找到 canonical 图: {canon_dir}")
            sys.exit(0)
        print(f"[重对齐] SKU={args.sku}  共 {len(files)} 张 -> {canon_dir}")
        n = 0
        for f in files:
            meta = realign_one(f)
            if meta:
                n += 1
                print(f"  {meta['stem']}  mm/px={meta['mm_per_px_canon']:.4f}")
        print(f"\n[完成] {n}/{len(files)}")
        return

    raw, ok = resolve_raw_dir(args.raw_dir, args.sku)
    if not ok:
        print(f"原图目录不存在: {raw}")
        print("  白底图由 image_assist.py 生成，默认在兄弟项目 "
              "Punctuation_selection_of_images/image_assist_ans/")
        print("  也可用 --raw-dir 指定")
        sys.exit(1)

    out_dir = os.path.join(BASE, "canonical", args.sku)
    files = sorted(glob.glob(os.path.join(raw, args.pattern)))
    files += sorted(glob.glob(os.path.join(raw, "*.jpg")))
    files += sorted(glob.glob(os.path.join(raw, "*.jpeg")))
    files = sorted(set(files))
    if not files:
        print(f"没找到图片: {raw}")
        sys.exit(0)

    print(f"[归一化] SKU={args.sku}  共 {len(files)} 张 -> {out_dir}")
    ok = 0
    for f in files:
        meta = normalize_one(f, cfg, args.mask_dir, out_dir)
        if meta:
            ok += 1
            print(f"  {meta['stem']}  canon={meta['canonical_size']}  "
                  f"mm/px={meta['mm_per_px_canon']:.4f}  (mask={meta['mask_src']})")
    print(f"\n[完成] {ok}/{len(files)} -> {out_dir}")
    print("提示: mm/px 决定 0.5mm 能否被检出，务必确认 0.5/（mm_per_px_canon）>= 3")


if __name__ == "__main__":
    main()
