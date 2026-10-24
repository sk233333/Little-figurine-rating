#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共工具：路径定位 + 配置解析
============================
集中解决两类「找不到」：

1) 白底图目录不在本项目下
   image_assist.py 在兄弟项目 Punctuation_selection_of_images/ 里运行，
   白底图输出到 ../Punctuation_selection_of_images/image_assist_ans，
   而不是 Defect_detection/../image_assist_ans。

2) 换新 SKU / 新机器时 configs/<sku>.yaml 不存在
   从 configs/template.yaml 自动生成一份。

其他脚本统一用法：

    from common import BASE, load_cfg, resolve_raw_dir
"""
import glob
import os
import shutil
import sys

import yaml

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE)

CONFIGS_DIR = os.path.join(BASE, "configs")
TEMPLATE_CFG = os.path.join(CONFIGS_DIR, "template.yaml")

IMG_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")

# 白底图目录候选（按优先级）。前三个是兄弟项目位置。
RAW_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "Punctuation_selection_of_images", "image_assist_ans"),
    os.path.join(PROJECT_ROOT, "Punctuation_selection_of_images", "image_assist_xia_ans"),
    os.path.join(PROJECT_ROOT, "image_assist_ans"),
    os.path.join(BASE, "image_assist_ans"),
]


def _count_images(d):
    return sum(len(glob.glob(os.path.join(d, p))) for p in IMG_PATTERNS)


def _has_images(d):
    return _count_images(d) > 0


def _ask_index(prompt, n, default=1):
    """读取 1~n 的编号（返回 0 基下标）；非交互环境返回 default。"""
    if not sys.stdin.isatty():
        print(f"[非交互环境] 使用默认 {default}")
        return default - 1
    while True:
        s = input(f"{prompt} [默认 {default}]: ").strip()
        if s == "":
            return default - 1
        if s.isdigit() and 1 <= int(s) <= n:
            return int(s) - 1
        print(f"  无效输入，请输入 1~{n}")


def resolve_raw_dir(raw_dir=None, sku=None, interactive=True):
    """定位白底图目录，返回 (绝对路径, 是否可用)。

    优先级：显式参数 > data/<sku>/raw > 兄弟项目 image_assist_ans / image_assist_xia_ans
            > 本项目 image_assist_ans
    多个候选目录都有图时，interactive=True 会提示用户选择（运行后再选）。
    返回的第二个值表示该目录是否存在且里面有图片，供调用方报错用。
    """
    if raw_dir:
        p = os.path.abspath(raw_dir)
        return p, _has_images(p)

    cands = []
    if sku:
        cands.append(os.path.join(BASE, "data", sku, "raw"))
    cands += RAW_CANDIDATES

    usable = [c for c in cands if os.path.isdir(c) and _has_images(c)]
    if not usable:
        return os.path.abspath(cands[0]), False
    if len(usable) == 1 or not interactive:
        return os.path.abspath(usable[0]), True

    print("\n发现多个白底图目录，请选择本次使用的：")
    for i, c in enumerate(usable):
        print(f"  {i + 1}) {c}  ({_count_images(c)} 张)")
    idx = _ask_index("请输入编号", len(usable))
    return os.path.abspath(usable[idx]), True


def ensure_config(sku, config=None):
    """返回配置文件路径；configs/<sku>.yaml 缺失时从 template.yaml 自动生成。"""
    path = os.path.abspath(config) if config else os.path.join(CONFIGS_DIR, f"{sku}.yaml")
    if os.path.exists(path) or config:
        return path
    if os.path.exists(TEMPLATE_CFG):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copy2(TEMPLATE_CFG, path)
        print(f"[配置] 不存在，已从模板生成: {path}")
    return path


def load_cfg(sku, config=None):
    """读取配置，返回 (cfg, path)。缺失则报错退出。"""
    path = ensure_config(sku, config)
    if not os.path.exists(path):
        print(f"配置不存在: {path}")
        print(f"  生成方式:  cp {TEMPLATE_CFG} {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), path
