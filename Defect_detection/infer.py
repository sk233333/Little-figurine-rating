#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4 推理 + 判级
==================
Level1(整图) 与 Level2(ROI) 异常图融合(取 max) -> 屏蔽设计图案 ->
连通域 -> px 换算 mm -> 分区阈值过滤 -> 评级

输出: reports/<sku>/<stem>_defect.png   标注图
      reports/<sku>/<stem>.json        结构化结果
      reports/<sku>/summary.json       汇总

用法:
    python3 infer.py --sku qml_demo
    python3 infer.py --sku qml_demo --stem xxx   只跑一张
    python3 infer.py --sku qml_test --model-sku qml_demo   用 A 的模型检测 B 的图
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


def find_ckpt(sku, roi_id):
    """返回最新的 checkpoint（按 mtime）。

    anomalib 的 Engine 每次实例化都会在 default_root_dir 下新建版本目录
    (v0, v1, v2...)，用 glob(...)[0] 会取到最旧的那个——重训后用的一直是旧模型。
    """
    c = glob.glob(os.path.join(BASE, "results", sku, roi_id, "**", "*.ckpt"),
                  recursive=True)
    return max(c, key=os.path.getmtime) if c else None


def load_thresh(sku, roi_id, cfg):
    p = os.path.join(BASE, "results", sku, roi_id, "threshold.json")
    if os.path.exists(p):
        with open(p) as f:
            return float(json.load(f).get("map_thresh", cfg.get("map_thresh", 0.5)))
    return float(cfg.get("map_thresh", 0.5))


def predict_map(engine, model, ckpt, img_path, out_px):
    """返回 (out_px,out_px) float32 原始异常图（不做 min-max 归一化）

    保留原始数值是必要的：每个 roi_id 训的是独立模型，分数尺度不同，
    归一化后再合并会让阈值失去意义。阈值统一由各自 roi 的标定结果决定。
    """
    try:
        # anomalib 2.7.0: PredictDataset 没有 name/category 属性，
        # 传给 datamodule=/dataset= 会在 Engine._setup_workspace 抛 AttributeError。
        # 改用 data_path，由 Engine 内部自行包装成 PredictDataset + DataLoader。
        res = engine.predict(model=model, ckpt_path=ckpt,
                             data_path=img_path,
                             return_predictions=True)
        b = res[0] if isinstance(res, list) else res
        am = getattr(b, "anomaly_map", None)
        if am is None:
            return None
        am = np.asarray(am, dtype=np.float32).squeeze()
    except Exception as e:
        print(f"    [推理] 失败 {os.path.basename(img_path)}: {type(e).__name__}: {e}")
        return None
    if am.ndim == 3:
        am = am[0]
    return cv2.resize(am, (out_px, out_px), interpolation=cv2.INTER_LINEAR)


def zone_of(cx, cy, cfg, C):
    """按配置判定分区，后面的优先"""
    zones = cfg.get("zones", {})
    order = ["hidden", "face", "body"]
    for zname in order:
        rects = zones.get(zname) or []
        for (x0, y0, x1, y1) in rects:
            if x0 * C <= cx <= x1 * C and y0 * C <= cy <= y1 * C:
                return zname
    return "body"


def grade(kept, cfg):
    if not kept:
        return "全新", 0.0
    g = cfg.get("grading", {})
    t1 = float(g.get("t1", 0.4)); t2 = float(g.get("t2", 1.0))
    esc = int(g.get("escalate_count", 3))
    max_mm = max(d["eq_mm"] for d in kept)
    cnt = len(kept)
    if max_mm <= t1:
        lv = "微瑕"
    elif max_mm <= t2:
        lv = "明显瑕疵"
    else:
        lv = "残次"
    if cnt >= esc and lv == "微瑕":
        lv = "明显瑕疵"
    return lv, max_mm


def infer_one(stem, cfg, sku, engine, model_cache, model_sku=None):
    """对单张图推理判级。

    sku       : 测试集目录（读 canonical/<sku>/ 和 rois/<sku>/，写 reports/<sku>/）
    model_sku : 模型/阈值所在目录（读 results/<model_sku>/）。
                与 sku 不同即「用 A 的模型跑 B 的图」，用于拿训练集模型检测测试集。
                不传则默认等于 sku（自测）。
    """
    msku = model_sku or sku
    canon_dir = os.path.join(BASE, "canonical", sku)
    roi_dir = os.path.join(BASE, "rois", sku, stem)
    img_path = os.path.join(canon_dir, f"{stem}.png")
    msk_path = os.path.join(canon_dir, f"{stem}_mask.png")
    meta_path = os.path.join(canon_dir, f"{stem}.json")
    idx_path = os.path.join(roi_dir, "index.json")
    for p in (img_path, msk_path, meta_path, idx_path):
        if not os.path.exists(p):
            print(f"  跳过 {stem}: 缺 {os.path.basename(p)}")
            return None

    with open(meta_path) as f:
        meta = json.load(f)
    with open(idx_path) as f:
        index = json.load(f)
    C = meta["canonical_size"]
    mm_px = meta["mm_per_px_canon"]

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE) > 127

    combined = np.zeros((C, C), dtype=np.float32)
    binary = np.zeros((C, C), dtype=np.uint8)
    th_used = {}

    def run(roi_id, src_path):
        ck = find_ckpt(msku, roi_id)
        if not ck:
            return None
        if roi_id not in model_cache:
            from pc_import import PatchCore
            model_cache[roi_id] = (PatchCore.load_from_checkpoint(ck), ck)
        model, ckpt = model_cache[roi_id]
        return predict_map(engine, model, ckpt, src_path, C)

    def merge(roi_id, m, rect):
        """把某个 roi 的异常图按「该 roi 自己的阈值」并入 combined / binary

        每个 roi_id 是独立训练的模型，分数尺度不同，必须各用各的阈值，
        不能归一化后共用一个阈值。
        """
        if m is None:
            return
        th = load_thresh(msku, roi_id, cfg)
        th_used[roi_id] = th
        if rect is None:
            # full_valid.png 是 level1_px(默认512)，异常图已放大到 C，需先对齐尺寸
            v = cv2.resize(
                cv2.imread(os.path.join(roi_dir, "full_valid.png"),
                           cv2.IMREAD_GRAYSCALE),
                (C, C), interpolation=cv2.INTER_NEAREST) > 127
            m = np.where(v, m, 0.0)
            combined[:] = np.maximum(combined, m)
            binary[:] = np.maximum(binary, ((m > th) & v).astype(np.uint8))
        else:
            x, y, w, h = rect
            mr = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(
                (cv2.imread(os.path.join(roi_dir, f"{roi_id}_valid.png"),
                            cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8),
                (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            mr = np.where(v, mr, 0.0)
            combined[y:y + h, x:x + w] = np.maximum(combined[y:y + h, x:x + w], mr)
            binary[y:y + h, x:x + w] = np.maximum(
                binary[y:y + h, x:x + w], ((mr > th) & v).astype(np.uint8))

    # Level1 整图
    merge("full", run("full", os.path.join(roi_dir, "full.png")), None)

    # Level2 ROI
    for r in index["rois"]:
        rid = r["id"]
        p = os.path.join(roi_dir, f"{rid}.png")
        if not os.path.exists(p):
            continue
        merge(rid, run(rid, p), r["rect"])

    # 只在物体内
    combined = np.where(msk, combined, 0.0)
    binary = np.where(msk, binary, 0).astype(np.uint8)

    # 屏蔽设计图案
    for (x0, y0, x1, y1) in (cfg.get("exclude_regions") or []):
        a, b = int(x0 * C), int(x1 * C)
        c, d = int(y0 * C), int(y1 * C)
        combined[b:d, a:c] = 0.0
        binary[b:d, a:c] = 0

    th = th_used.get("full")
    if th is None:
        th = th_used.get("roi_000", float(cfg.get("map_thresh", 0.5)))

    # 形态学清理
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    n, lab, stats, cent = cv2.connectedComponentsWithStats(binary, 8)
    thr_mm = cfg.get("thresholds_mm", {})
    # 最大尺寸上限: 比物体本身还大的连通域是整片光照/配准偏差，不是瑕疵
    max_mm = float(cfg.get("max_defect_mm", 0) or 0)
    defects = []
    regions = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        cx, cy = float(cent[i][0]), float(cent[i][1])
        eq_px = 2 * np.sqrt(area / np.pi)
        eq_mm = eq_px * mm_px
        z = zone_of(cx, cy, cfg, C)
        lim = float(thr_mm.get(z, thr_mm.get("body", 0.8)))
        if eq_mm < lim:
            continue
        item = {
            "cx": round(cx, 1), "cy": round(cy, 1),
            "area_px": area, "eq_mm": round(float(eq_mm), 3),
            "zone": z, "threshold_mm": lim,
            "bbox": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                     int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
        }
        if max_mm > 0 and eq_mm > max_mm:
            regions.append(item)
            continue
        defects.append(item)

    lv, max_mm_kept = grade(defects, cfg)
    score = float(combined[msk].max()) if msk.any() else 0.0

    # 可视化
    vis = img.copy()
    for d in defects:
        x, y, w, h = d["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(vis, f"{d['eq_mm']:.2f}mm {d['zone']}",
                    (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 1)
    for d in regions:
        x, y, w, h = d["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 128, 0), 1)
    cv2.putText(vis, f"{stem}  {lv}  n={len(defects)} max={max_mm_kept:.2f}mm",
                (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    rep_dir = os.path.join(BASE, "reports", sku)
    os.makedirs(rep_dir, exist_ok=True)
    cv2.imwrite(os.path.join(rep_dir, f"{stem}_defect.png"), vis)
    out = {
        "sku": sku, "stem": stem, "grade": lv,
        "pred_score": round(score, 4), "map_thresh": th,
        "map_thresh_by_roi": {k: round(v, 4) for k, v in th_used.items()},
        "mm_per_px_canon": mm_px, "canonical_size": C,
        "defect_count": len(defects), "max_mm": round(float(max_mm_kept), 3),
        "region_anomalies": regions,
        "defects": defects,
        "accept_list": cfg.get("accept_list", []),
    }
    with open(os.path.join(rep_dir, f"{stem}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stem", default=None)
    ap.add_argument("--model-sku", default=None,
                    help="模型/阈值所在目录(默认同 --sku)。用训练集模型检测测试集时填训练集 sku")
    args = ap.parse_args()

    cfg, _ = load_cfg(args.sku, args.config)

    try:
        from anomalib.engine import Engine
    except ImportError:
        print("请先安装 anomalib:  pip install \"anomalib[cu124]\"")
        sys.exit(1)
    # 纯推理的 Engine 不写权重，给它一个独立的 scratch 目录，
    # 避免在 results/ 下生成 Patchcore/latest/ 之类的垃圾版本目录。
    engine = Engine(accelerator="auto",
                    default_root_dir=os.path.join(BASE, "_scratch", "infer"))

    canon_dir = os.path.join(BASE, "canonical", args.sku)
    if args.stem:
        stems = [args.stem]
    else:
        stems = [os.path.splitext(os.path.basename(p))[0]
                 for p in sorted(glob.glob(os.path.join(canon_dir, "*.png")))
                 if not p.endswith("_mask.png")]
    if not stems:
        print(f"没找到 canonical 图: {canon_dir}")
        sys.exit(0)

    print(f"[推理] SKU={args.sku}  共 {len(stems)} 张")
    if args.model_sku and args.model_sku != args.sku:
        print(f"[推理] 使用模型 SKU={args.model_sku}  (跨 SKU 检测)")
    cache = {}
    results = []
    for s in stems:
        r = infer_one(s, cfg, args.sku, engine, cache, args.model_sku)
        if r:
            results.append(r)
            print(f"  {s}  {r['grade']}  n={r['defect_count']}  "
                  f"max={r['max_mm']}mm  score={r['pred_score']}")

    rep_dir = os.path.join(BASE, "reports", args.sku)
    with open(os.path.join(rep_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    from collections import Counter
    print("\n[汇总]", dict(Counter(r["grade"] for r in results)))
    print(f"[完成] -> {rep_dir}")


if __name__ == "__main__":
    main()
