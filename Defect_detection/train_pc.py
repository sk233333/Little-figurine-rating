#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 训练 PatchCore (per-SKU x per-ROI)
==========================================
1. 收集 ROI 图 -> datasets/<sku>/<roi_id>/good/
2. 姿态增强(磁盘上生成，规避 anomalib 版本差异)
3. Folder datamodule + PatchCore + Engine 训练
4. 阈值标定: 在 OK 品上跑一遍，取分位数作为 map_thresh

用法:
    python3 train_pc.py --sku qml_demo --roi all      # full + 所有 ROI
    python3 train_pc.py --sku qml_demo --roi full     # 只训整图粗检
    python3 train_pc.py --sku qml_demo --roi roi_000
"""
import argparse
import glob
import json
import os
import random
import shutil
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from common import load_cfg   # noqa: E402


def augment(src_path, dst_dir, stem, cfg_aug):
    """磁盘上生成姿态增强副本：小角度旋转/平移/缩放 + 亮度"""
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        return 0
    h, w = img.shape[:2]
    n = int(cfg_aug.get("copies", 8))
    rot = float(cfg_aug.get("rot_deg", 5.0))
    tr = float(cfg_aug.get("translate", 0.03))
    sc = float(cfg_aug.get("scale", 0.05))
    br = float(cfg_aug.get("brightness", 0.08))
    made = 0
    for i in range(n):
        ang = random.uniform(-rot, rot)
        tx = random.uniform(-tr, tr) * w
        ty = random.uniform(-tr, tr) * h
        s = random.uniform(1 - sc, 1 + sc)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
        M[0, 2] += tx
        M[1, 2] += ty
        out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))   # 白底
        if br > 0:
            f = random.uniform(1 - br, 1 + br)
            out = np.clip(out.astype(np.float32) * f, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(dst_dir, f"{stem}_aug{i:02d}.png"), out)
        made += 1
    return made


def build_dataset(sku, roi_id, cfg, split="train"):
    """组装 anomalib Folder 所需目录 datasets/<sku>/<roi_id>/good/"""
    roi_root = os.path.join(BASE, "rois", sku)
    stems = sorted([os.path.basename(p) for p in glob.glob(os.path.join(roi_root, "*"))
                    if os.path.isdir(p)])
    if not stems:
        print(f"没找到 ROI 数据: {roi_root}（先跑 roi_split.py）")
        sys.exit(1)

    # 正常样本白名单：配置了就只取白名单内的 stem，避免瑕疵图被当成正常样本
    allow = [str(s).strip() for s in (cfg.get("normal_stems") or []) if str(s).strip()]
    if allow:
        allow_set = set(allow)
        skipped = [s for s in stems if s not in allow_set]
        stems = [s for s in stems if s in allow_set]
        if skipped:
            head = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
            print(f"  [筛选] 排除 {len(skipped)} 张非正常样本: {head}")
        missing = [s for s in allow if s not in set(os.path.basename(p)
                                                   for p in glob.glob(os.path.join(roi_root, "*")))]
        if missing:
            head = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            print(f"  [警告] normal_stems 里 {len(missing)} 个 stem 在 rois/ 下不存在: {head}")
        if not stems:
            print("normal_stems 过滤后没有可用样本，请检查配置")
            sys.exit(1)
    else:
        print(f"  [警告] 未配置 normal_stems，{len(stems)} 张全部当正常样本；"
              f"若有瑕疵图混在 rois/{sku}/ 会污染训练集")

    # 确定性划分
    random.Random(42).shuffle(stems)
    vr = float(cfg.get("train", {}).get("val_ratio", 0.2))
    n_val = max(1, int(len(stems) * vr))
    val_stems = set(stems[:n_val])
    train_stems = [s for s in stems if s not in val_stems]
    chosen = train_stems if split == "train" else sorted(val_stems)

    split_file = os.path.join(BASE, "datasets", sku, f"{roi_id}_split.json")
    os.makedirs(os.path.dirname(split_file), exist_ok=True)

    good_dir = os.path.join(BASE, "datasets", sku, roi_id, "good")
    if os.path.exists(good_dir):
        shutil.rmtree(good_dir)
    os.makedirs(good_dir, exist_ok=True)

    n = 0
    for s in chosen:
        for p in glob.glob(os.path.join(roi_root, s, f"{roi_id}*.png")):
            if p.endswith("_valid.png"):
                continue
            shutil.copy2(p, os.path.join(good_dir, f"{s}_{os.path.basename(p)}"))
            n += 1

    aug_cfg = cfg.get("aug", {})
    if aug_cfg.get("enable", True) and split == "train":
        base_files = sorted(glob.glob(os.path.join(good_dir, "*.png")))
        for p in base_files:
            st = os.path.splitext(os.path.basename(p))[0]
            n += augment(p, good_dir, st, aug_cfg)

    with open(split_file, "w") as f:
        json.dump({"train": train_stems, "val": sorted(val_stems)}, f, indent=2)
    return good_dir, n


def find_ckpt(out_dir):
    """返回最新的 checkpoint（按 mtime）。

    anomalib 的 Engine 每次实例化都会在 default_root_dir 下新建版本目录
    (v0, v1, v2...)，用 glob(...)[0] 会取到最旧的那个——重训后用的一直是旧模型。
    """
    c = glob.glob(os.path.join(out_dir, "**", "*.ckpt"), recursive=True)
    return max(c, key=os.path.getmtime) if c else None


def predict_map(engine, model, ckpt, img_path):
    """返回异常图 (h,w) float32（anomalib 内部输入尺寸，默认 256）"""
    # anomalib 2.7.0: 必须用 data_path，不能用 datamodule=PredictDataset(...)
    res = engine.predict(model=model, ckpt_path=ckpt,
                         data_path=img_path, return_predictions=True)
    b = res[0] if isinstance(res, list) else res
    am = getattr(b, "anomaly_map", None)
    if am is None:
        return None
    am = np.asarray(am, dtype=np.float32).squeeze()
    if am.ndim == 3:
        am = am[0]
    return am


def val_items(sku, roi_id):
    """留出的 OK 样本（未参与训练）: [(roi图, 有效区mask 或 None), ...]"""
    sf = os.path.join(BASE, "datasets", sku, f"{roi_id}_split.json")
    if not os.path.exists(sf):
        return []
    with open(sf) as f:
        val = json.load(f).get("val", [])
    items = []
    for s in val:
        d = os.path.join(BASE, "rois", sku, s)
        ip = os.path.join(d, f"{roi_id}.png")
        vp = os.path.join(d, f"{roi_id}_valid.png")
        if os.path.exists(ip):
            items.append((ip, vp if os.path.exists(vp) else None))
    return items


def calibrate(engine, model, ckpt, items, out_dir, cfg):
    """像素级阈值标定: 在「留出的 OK 图」异常图上取高分位数

    旧版有两处方法错误，已修:
      1. 拿「图像级 pred_score」当「像素级阈值」——两者尺度不同；
      2. 在含增强副本的**训练集**上标定，阈值偏乐观。
    现在只在留出的正常样本上按像素取分位数，直接控制像素级误报率。

    分位数支持按 ROI 覆盖: calib.quantile_by_roi.<roi_id>。
    同一个高分位数(如 0.9999)对低方差 ROI 合适，但对方差大的 ROI 会把阈值
    顶到接近 OK 峰值，导致该区域的中等瑕疵被吞——此时给该 ROI 单独降分位数。
    roi_id 由 out_dir(	results/<sku>/<roi_id>) 推导。
    """
    cal = cfg.get("calib", {})
    roi_id = os.path.basename(os.path.normpath(out_dir))
    q_by_roi = cal.get("quantile_by_roi") or {}
    q = float(q_by_roi.get(roi_id, cal.get("quantile", 0.9999)))
    vals, n_img = [], 0
    for ip, vp in items:
        try:
            am = predict_map(engine, model, ckpt, ip)
        except Exception as e:
            print(f"    [标定] 跳过 {os.path.basename(ip)}: {type(e).__name__}: {e}")
            continue
        if am is None:
            continue
        n_img += 1
        if vp:
            v = cv2.imread(vp, cv2.IMREAD_GRAYSCALE)
            if v is not None:
                v = cv2.resize(v, (am.shape[1], am.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 127
                if v.any():
                    vals.append(am[v])
                    continue
        vals.append(am.reshape(-1))
    if not vals:
        print("    [标定] 没有可用留出样本，使用配置兜底值")
        return float(cfg.get("map_thresh", 0.5))
    pooled = np.concatenate(vals)
    th = float(np.quantile(pooled, q))
    with open(os.path.join(out_dir, "threshold.json"), "w") as f:
        json.dump({"map_thresh": th, "quantile": q, "roi_id": roi_id,
                   "quantile_by_roi": roi_id in q_by_roi,
                   "n_img": n_img, "n_px": int(pooled.size),
                   "max": float(pooled.max())}, f, indent=2)
    print(f"    [标定] map_thresh={th:.4f} (q={q}{' [按ROI覆盖]' if roi_id in q_by_roi else ''}, "
          f"{n_img} 张留出样本, {pooled.size} 像素, max={pooled.max():.4f})")
    return th


def train_one(sku, roi_id, cfg):
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from pc_import import PatchCore
    except ImportError:
        print("请先安装 anomalib:  pip install \"anomalib[cu124]\"")
        sys.exit(1)

    good_dir, n = build_dataset(sku, roi_id, cfg, split="train")
    print(f"\n[{sku}/{roi_id}] 训练样本 {n} 张 (含增强) -> {good_dir}")
    if n == 0:
        print("  样本为空，跳过")
        return

    pc = cfg.get("patchcore", {})
    tr = cfg.get("train", {})
    model = PatchCore(
        backbone=pc.get("backbone", "wide_resnet50_2"),
        layers=pc.get("layers", ["layer2", "layer3"]),
        coreset_sampling_ratio=pc.get("coreset_sampling_ratio", 0.05),
        num_neighbors=pc.get("num_neighbors", 3),
    )

    root = os.path.join(BASE, "datasets", sku, roi_id)
    dm = Folder(
        name=f"{sku}_{roi_id}",
        root=root,
        normal_dir="good",
        train_batch_size=int(tr.get("batch_size", 8)),
        eval_batch_size=int(tr.get("batch_size", 8)),
        num_workers=int(tr.get("num_workers", 0)),
    )

    out_dir = os.path.join(BASE, "results", sku, roi_id)
    os.makedirs(out_dir, exist_ok=True)
    engine = Engine(accelerator="auto", default_root_dir=out_dir,
                    max_epochs=int(tr.get("epochs", 1)))
    engine.fit(model=model, datamodule=dm)

    ckpt = find_ckpt(out_dir)
    if ckpt:
        print(f"    权重: {ckpt}")
        items = val_items(sku, roi_id)
        if not items:
            print("    [警告] 无留出样本(datasets/<sku>/<roi>_split.json)，跳过标定")
        calibrate(engine, model, ckpt, items, out_dir, cfg)
    else:
        print("    未找到 ckpt")


def recalib_one(sku, roi_id, cfg):
    """只重跑阈值标定，不重新训练（改 calib.quantile 后用）"""
    try:
        from anomalib.engine import Engine
        from pc_import import PatchCore
    except ImportError:
        print("请先安装 anomalib:  pip install \"anomalib[cu124]\"")
        sys.exit(1)
    out_dir = os.path.join(BASE, "results", sku, roi_id)
    ckpt = find_ckpt(out_dir)
    if not ckpt:
        print(f"[{sku}/{roi_id}] 未找到 ckpt，跳过")
        return
    model = PatchCore.load_from_checkpoint(ckpt)
    # 只跑标定、不训练，Engine 不要往 out_dir 写版本目录
    engine = Engine(accelerator="auto",
                    default_root_dir=os.path.join(BASE, "_scratch", "recalib"))
    items = val_items(sku, roi_id)
    print(f"[{sku}/{roi_id}] 重新标定（{len(items)} 张留出样本）")
    calibrate(engine, model, ckpt, items, out_dir, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--roi", default="all", help="all / full / roi_000")
    ap.add_argument("--recalib", action="store_true",
                    help="只重跑阈值标定，不重新训练")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg, _ = load_cfg(args.sku, args.config)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.roi == "all":
        targets = ["full"]
        idx_files = glob.glob(os.path.join(BASE, "rois", args.sku, "*", "index.json"))
        maxn = 0
        for p in idx_files:
            with open(p) as f:
                d = json.load(f)
            maxn = max(maxn, d.get("roi_count", 0))
        targets += [f"roi_{i:03d}" for i in range(maxn)]
        print(f"[训练] 目标: {targets}")
    else:
        targets = [args.roi]

    if args.recalib:
        for t in targets:
            recalib_one(args.sku, t, cfg)
        print("\n[完成] 阈值已更新在 results/<sku>/<roi>/threshold.json")
        return

    for t in targets:
        train_one(args.sku, t, cfg)
    print("\n[完成] 权重与阈值在 results/<sku>/<roi>/")


if __name__ == "__main__":
    main()
