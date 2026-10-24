#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标记设计图案屏蔽区（眼睛 / 腮红 / 斑点等）
==========================================
在 canonical 图上拖拽画矩形，这些区域不参与瑕疵判级。

操作:
    左键拖拽 = 画矩形
    u = 撤销上一个
    s = 保存并退出（写回 configs/<sku>.yaml 的 exclude_regions）
    q = 直接退出不保存

用法:
    python3 mark_exclude.py --sku qml_demo --stem ok_001
"""
import argparse
import os
import sys

import cv2
import yaml

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from common import ensure_config   # noqa: E402

rects = []
drawing = False
x0 = y0 = -1
canvas = None
base = None


def on_mouse(event, x, y, flags, param):
    global drawing, x0, y0, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x0, y0 = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        canvas = base.copy()
        _draw_all(canvas)
        cv2.rectangle(canvas, (x0, y0), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, x2 = sorted([x0, x])
        y1, y2 = sorted([y0, y])
        if x2 - x1 > 4 and y2 - y1 > 4:
            rects.append((x1, y1, x2, y2))
            print(f"  + 矩形 {len(rects)}: {(x1, y1, x2, y2)}")
        canvas = base.copy()
        _draw_all(canvas)


def _draw_all(img):
    for i, (a, b, c, d) in enumerate(rects):
        cv2.rectangle(img, (a, b), (c, d), (0, 0, 255), 2)
        cv2.putText(img, str(i + 1), (a + 4, b + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def main():
    global canvas, base
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--stem", required=True)
    args = ap.parse_args()

    p = os.path.join(BASE, "canonical", args.sku, f"{args.stem}.png")
    if not os.path.exists(p):
        print(f"没找到: {p}（先跑 normalize.py）")
        sys.exit(1)
    img = cv2.imread(p)
    H, W = img.shape[:2]
    disp = 900
    sc = disp / float(max(H, W))
    base = cv2.resize(img, (int(W * sc), int(H * sc)))
    canvas = base.copy()

    cv2.namedWindow("mark exclude")
    cv2.setMouseCallback("mark exclude", on_mouse)
    print("拖拽画矩形(设计图案区)  u=撤销  s=保存  q=退出")
    while True:
        cv2.imshow("mark exclude", canvas)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and rects:
            rects.pop()
            canvas = base.copy()
            _draw_all(canvas)
            print(f"  撤销，剩 {len(rects)} 个")
        elif k == ord("s"):
            break
        elif k == ord("q"):
            cv2.destroyAllWindows()
            print("未保存")
            return
    cv2.destroyAllWindows()

    norm = [[round(a / float(base.shape[1]), 4), round(b / float(base.shape[0]), 4),
             round(c / float(base.shape[1]), 4), round(d / float(base.shape[0]), 4)]
            for (a, b, c, d) in rects]
    cfg_path = ensure_config(args.sku)
    if not os.path.exists(cfg_path):
        print(f"配置不存在: {cfg_path}")
        sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["exclude_regions"] = norm
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"[完成] 已写入 {cfg_path}  exclude_regions={len(norm)} 个")


if __name__ == "__main__":
    main()
