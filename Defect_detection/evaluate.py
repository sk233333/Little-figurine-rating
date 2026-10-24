#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5 评估
============
分档 + 分类型报 AUROC，避免小瑕疵漏检被大缺陷的好成绩掩盖。

标签 CSV 格式(stem,label,size_mm,type):
    stem,label,size_mm,type
    ok_001,0,0,none
    ng_001,1,0.3,dot
    ng_002,1,1.2,scratch
    ng_003,1,0,missing      # 结构缺陷 size 填 0

分档: <0.5mm / 0.5~2mm / 结构缺陷(type=missing|deform)
用法:
    python3 evaluate.py --sku qml_demo --labels labels.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))


def auroc(scores, labels):
    """rank-based AUROC，无需 sklearn"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos, neg = labels.sum(), (1 - labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    sorted_s = scores[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def band_of(size, typ):
    if typ in ("missing", "deform", "structure"):
        return "结构缺陷"
    if size < 0.5:
        return "<0.5mm"
    if size <= 2.0:
        return "0.5~2mm"
    return ">2mm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--score-key", default="pred_score")
    args = ap.parse_args()

    rep_dir = os.path.join(BASE, "reports", args.sku)
    rows = []
    with open(args.labels, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    scores, labels, bands, types = [], [], [], []
    miss = 0
    for r in rows:
        stem = r["stem"].strip()
        p = os.path.join(rep_dir, f"{stem}.json")
        if not os.path.exists(p):
            miss += 1
            continue
        with open(p) as f:
            d = json.load(f)
        s = d.get(args.score_key, None)
        if s is None:
            # 退化：用最大瑕疵尺寸当分数
            s = d.get("max_mm", 0.0)
        scores.append(float(s))
        labels.append(int(r["label"]))
        size = float(r.get("size_mm", 0) or 0)
        typ = (r.get("type") or "none").strip()
        bands.append(band_of(size, typ))
        types.append(typ)

    if miss:
        print(f"[警告] {miss} 个样本没有推理结果（先跑 infer.py）")
    if not scores:
        print("没有可评估的样本")
        sys.exit(0)

    print(f"\n[评估] {args.sku}  n={len(scores)}  正例={sum(labels)}")
    print("-" * 46)
    print(f"  总体 AUROC        : {auroc(scores, labels):.4f}")

    print("\n  按瑕疵大小分档:")
    for b in ["<0.5mm", "0.5~2mm", ">2mm", "结构缺陷"]:
        idx = [i for i, x in enumerate(bands) if x == b]
        if not idx:
            continue
        sub_l = [labels[i] for i in idx]
        if sum(sub_l) == 0 or sum(sub_l) == len(sub_l):
            print(f"    {b:<10} n={len(idx):<3} (需同时有正负样本)")
            continue
        print(f"    {b:<10} n={len(idx):<3} AUROC={auroc([scores[i] for i in idx], sub_l):.4f}")

    print("\n  按瑕疵类型:")
    for t in sorted(set(types)):
        idx = [i for i, x in enumerate(types) if x == t]
        sub_l = [labels[i] for i in idx]
        if sum(sub_l) == 0 or sum(sub_l) == len(sub_l):
            continue
        print(f"    {t:<10} n={len(idx):<3} AUROC={auroc([scores[i] for i in idx], sub_l):.4f}")

    print("\n提示: 每档至少 10~15 个样本才有统计意义；")
    print("      重点看 <0.5mm 档——它最容易被大缺陷的好成绩掩盖。")


if __name__ == "__main__":
    main()
