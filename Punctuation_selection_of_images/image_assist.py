#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 2 交互标注切割工具（独立项目）
================================
默认处理 OK 图（--type ok）：
    读图目录: image/
    输出：
        image_assist_ans/       白底抠图结果图（每张一张）
                                qml_001_white.png
        image_assist_process/   四图对比（每张一张）
                                qml_001_vis.png
        （其余中间图不再保存；需要训练时加 --save-mask 才额外存 *_mask.png）

处理瑕图时加 --type xia：
    读图目录: image_xia/
    输出：
        image_assist_xia_ans/   白底抠图结果图
        image_assist_xia_process/ 四图对比

用法：
    python3 image_assist.py                 先问你是 OK 图还是瑕图，再批量标注
    python3 image_assist.py --type xia      直接指定瑕图，跳过询问
    python3 image_assist.py --type ok       直接指定 OK 图
    python3 image_assist.py qml_001         只标这一张
    python3 image_assist.py --skip-done     跳过已标注的
    python3 image_assist.py --start 10      从第 11 张开始
    python3 image_assist.py --device cpu    没显卡时用
    python3 image_assist.py --save-mask     额外保存 *_mask.png（训练用）

STEP 1 画框   : 左键点两个角            ENTER 确认
STEP 2 标点   : 左键前景(红) 右键背景(蓝)  ENTER 分割
                 r = 全部重来   u = 撤销上一个点   ESC = 取消这张
STEP 3 修结果 : 结果窗口按键
                 1/2/3  切换候选 mask
                 ]      补洞档位 +1   (OFF -> L1 -> L2 -> L3)
                 [      补洞档位 -1   (L3 -> L2 -> L1 -> OFF)
                 f      循环档位      o 直接回 OFF
                 a      补点迭代精修
                 ENTER  保存并进入下一张
                 n      跳过这张       p 回到上一张       q 退出
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "image")
ANS_DIR = os.path.join(BASE_DIR, "image_assist_ans")
PROCESS_DIR = os.path.join(BASE_DIR, "image_assist_process")

# 瑕图：读 image_xia/ -> 写 image_assist_xia_ans/ + image_assist_xia_process/
XIA_IMAGE_DIR = os.path.join(BASE_DIR, "image_xia")
XIA_ANS_DIR = os.path.join(BASE_DIR, "image_assist_xia_ans")
XIA_PROCESS_DIR = os.path.join(BASE_DIR, "image_assist_xia_process")

CKPT = os.path.join(BASE_DIR, "sam2.1_hiera_small.pt")

# --type 到默认目录的映射
TYPE_DIRS = {
    "ok": (IMAGE_DIR, ANS_DIR, PROCESS_DIR),
    "xia": (XIA_IMAGE_DIR, XIA_ANS_DIR, XIA_PROCESS_DIR),
}

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
           "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP")

FILL_NAMES = {0: "OFF", 1: "L1", 2: "L2", 3: "L3"}

_points = []
_box = []
_last_level = [0]      # 补洞档位跨图记忆


# ------------------------------------------------------------------
# 形态学补洞
# ------------------------------------------------------------------
def postprocess(mask, level):
    """level: 0=不处理 1=补洞 2=补洞+闭运算+最大域 3=最强"""
    if level == 0:
        return (mask > 0).astype(np.uint8)
    if not HAS_SCIPY:
        return (mask > 0).astype(np.uint8)

    m = mask.astype(bool)

    # 闭运算：弥合边缘缺口与锯齿（L2 轻，L3 强）
    if level >= 2:
        size = 5 if level == 2 else 11
        iters = 1 if level == 2 else 2
        for _ in range(iters):
            m = ndimage.binary_closing(m, structure=np.ones((size, size)))

    # 补内部孔洞（眼睛、腮红、文字挖空等）
    m = ndimage.binary_fill_holes(m)

    # 只保留最大连通域，清掉周边碎片
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = (lab == (int(np.argmax(sizes)) + 1))

    # L3 额外平滑
    if level == 3:
        m = ndimage.binary_closing(m, structure=np.ones((5, 5)))
        m = ndimage.binary_fill_holes(m)

    return m.astype(np.uint8)


# ------------------------------------------------------------------
# 出图
# ------------------------------------------------------------------
def make_white_bg(image, mask):
    """白底抠图：背景纯白，物体保留原色"""
    out = np.full_like(image, 255)
    out[mask > 0] = image[mask > 0]
    return out


def make_overlay(image, mask, alpha=0.45):
    """红色半透明叠加（只在窗口里看，不存盘）"""
    ov = image.copy().astype(np.float32)
    m = mask > 0
    ov[m] = (image[m].astype(np.float32) * (1 - alpha)
             + np.array([255, 60, 60], np.float32) * alpha)
    return np.clip(ov, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# 第 1 步：画框（r 重画 / u 撤销）
# ------------------------------------------------------------------
def draw_box(image, prefix=""):
    global _box
    _box = []
    fig, ax = plt.subplots(figsize=(10, 12))
    rp = [None]

    def reset_view(tip=""):
        ax.clear()
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"{prefix}第1步 画框：左键点两个角 -> 回车确认\n"
            f"r = 全部重画    u = 撤销上一个点    ESC = 取消这张"
            + (f"\n{tip}" if tip else ""), fontsize=11)

    reset_view()

    def on_click(e):
        global _box
        if e.inaxes != ax or e.button != 1 or len(_box) >= 2:
            return
        _box.append([int(e.xdata), int(e.ydata)])
        ax.plot(e.xdata, e.ydata, "go", markersize=8)
        if len(_box) == 2:
            x1, y1 = _box[0]
            x2, y2 = _box[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            _box = [[x1, y1], [x2, y2]]
            rp[0] = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.5,
                              edgecolor="green", facecolor="none")
            ax.add_patch(rp[0])
            print("    框已画出，回车确认 / r 重画")
        fig.canvas.draw()

    def on_key(e):
        global _box
        k = (e.key or "").lower()
        if k == "enter" and len(_box) == 2:
            plt.close()
        elif k == "r":
            _box = []
            rp[0] = None
            reset_view("已清空，请重新点两个角")
            fig.canvas.draw()
            print("    → 框已清除，重新画")
        elif k == "u":
            if _box:
                _box.pop()
                rp[0] = None
                reset_view(f"已撤销，还剩 {len(_box)} 个点")
                for p in _box:
                    ax.plot(p[0], p[1], "go", markersize=8)
                fig.canvas.draw()
                print(f"    → 撤销一个点，剩 {len(_box)} 个")
        elif k == "escape":
            _box = []
            plt.close()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    if len(_box) != 2:
        return None
    (x1, y1), (x2, y2) = _box
    print(f"    框 [{x1},{y1},{x2},{y2}]  宽={x2-x1} 高={y2-y1}")
    return np.array([x1, y1, x2, y2])


# ------------------------------------------------------------------
# 第 2 步：标点（r 重标 / u 撤销）
# ------------------------------------------------------------------
def pick_points(image, box, prev_points=None, prefix=""):
    global _points
    _points = list(prev_points) if prev_points else []
    fig, ax = plt.subplots(figsize=(10, 12))

    def reset_view(tip=""):
        ax.clear()
        ax.imshow(image)
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.5,
                               edgecolor="green", facecolor="none",
                               linestyle="--"))
        for p in _points:
            c = "red" if p[2] == 1 else "blue"
            ax.plot(p[0], p[1], "x", color=c, markersize=13,
                    markeredgewidth=2.5)
        ax.axis("off")
        ax.set_title(
            f"{prefix}第2步 标点：左键=前景(红) 右键=背景(蓝) -> 回车分割\n"
            f"r = 全部重标    u = 撤销上一个点    ESC = 取消这张"
            + (f"\n{tip}" if tip else ""), fontsize=11)

    reset_view()

    def on_click(e):
        global _points
        if e.inaxes != ax:
            return
        x, y = int(e.xdata), int(e.ydata)
        if e.button == 1:
            _points.append([x, y, 1])
            ax.plot(x, y, "rx", markersize=14, markeredgewidth=2.5)
            ax.text(x + 8, y - 8, f"F{len(_points)}", color="red",
                    fontsize=10, weight="bold")
            print(f"      +前景 ({x},{y})")
        elif e.button == 3:
            _points.append([x, y, 0])
            ax.plot(x, y, "bx", markersize=14, markeredgewidth=2.5)
            ax.text(x + 8, y - 8, f"B{len(_points)}", color="blue",
                    fontsize=10, weight="bold")
            print(f"      -背景 ({x},{y})")
        fig.canvas.draw()

    def on_key(e):
        global _points
        k = (e.key or "").lower()
        if k == "enter" and len(_points) > 0:
            plt.close()
        elif k == "r":
            _points = []
            reset_view("已清空，请重新标点")
            fig.canvas.draw()
            print("    → 点已清除，重新标")
        elif k == "u":
            if _points:
                _points.pop()
                reset_view(f"已撤销，还剩 {len(_points)} 个点")
                fig.canvas.draw()
                print(f"    → 撤销一个点，剩 {len(_points)} 个")
        elif k == "escape":
            _points = []
            plt.close()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    if not _points:
        return None, None
    pts = np.array([[p[0], p[1]] for p in _points], dtype=np.float32)
    lbl = np.array([p[2] for p in _points], dtype=np.int32)
    print(f"    点 前景={int((lbl==1).sum())} 背景={int((lbl==0).sum())}")
    return pts, lbl


# ------------------------------------------------------------------
# 第 3 步：结果窗口（4 张：原图 / Mask / 叠加 / 白底）
# ------------------------------------------------------------------
def review(image, box, masks, scores, stem, prefix,
           ans_dir, process_dir, single=False, save_mask=False):
    idx = [int(np.argmax(scores))]
    ml, sl = list(masks), list(scores)
    state = {"act": "skip", "redo": False, "level": _last_level[0]}

    fig, axes = plt.subplots(1, 4, figsize=(24, 9))

    def set_level(v, tag=""):
        global _last_level
        v = max(0, min(3, v))
        if v != state["level"]:
            print(f"    补洞 {FILL_NAMES[state['level']]} -> {FILL_NAMES[v]} {tag}")
        state["level"] = v
        _last_level[0] = v
        redraw()

    def redraw():
        i = idx[0]
        m = postprocess(ml[i], state["level"])
        for ax in axes:
            ax.clear()
            ax.axis("off")

        # 第 1 格：原图 + 框 + 点
        axes[0].imshow(image)
        x1, y1, x2, y2 = box
        axes[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.5,
                                    edgecolor="green", facecolor="none"))
        for p in _points:
            c = "red" if p[2] == 1 else "blue"
            axes[0].plot(p[0], p[1], "x", color=c, markersize=10,
                         markeredgewidth=2)
        axes[0].set_title(f"输入（框 + {len(_points)} 点）")

        # 第 2 格：Mask
        axes[1].imshow(m, cmap="gray")
        axes[1].set_title(f"候选#{i} 分数={sl[i]:.3f}\n补洞={FILL_NAMES[state['level']]}")

        # 第 3 格：红色叠加
        axes[2].imshow(make_overlay(image, m))
        axes[2].set_title("红色叠加")

        # 第 4 格：白底抠图
        axes[3].imshow(make_white_bg(image, m))
        axes[3].set_title("白底抠图（结果）")

        nav = "回车=保存退出" if single else "回车=保存并下一张  n=跳过  p=上一张"
        fig.suptitle(
            f"{prefix}   1/2/3=候选   ] =补洞+1   [ =补洞-1   f=循环   o=OFF   "
            f"a=补点   {nav}   q=退出", fontsize=12)
        fig.canvas.draw()

    def on_key(e):
        k = (e.key or "").lower()
        if k in ("1", "2", "3"):
            n = int(k) - 1
            if n < len(ml):
                idx[0] = n
                print(f"    → 候选 #{n} (分数 {sl[n]:.3f})")
                redraw()
        elif k in ("]", "=", "+"):
            set_level(state["level"] + 1, "(升)")
        elif k in ("[", "-", "_"):
            set_level(state["level"] - 1, "(降)")
        elif k == "f":
            set_level((state["level"] + 1) % 4, "(循环)")
        elif k == "o":
            set_level(0, "(还原)")
        elif k == "a":
            state["redo"] = True
            plt.close()
        elif k == "enter":
            state["act"] = "save"
            plt.close()
        elif k == "n" and not single:
            state["act"] = "skip"
            plt.close()
        elif k == "p" and not single:
            state["act"] = "prev"
            plt.close()
        elif k in ("q", "escape"):
            state["act"] = "quit"
            plt.close()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.tight_layout()
    plt.show()

    if state["redo"]:
        return "redo", None
    if state["act"] != "save":
        return state["act"], None

    i = idx[0]
    mask = postprocess(ml[i], state["level"]).astype(np.uint8)
    white = make_white_bg(image, mask)

    # ---- 只存两张 ----
    # 1) 白底结果图 -> image_assist_ans/
    pw = os.path.join(ans_dir, f"{stem}_white.png")
    Image.fromarray(white).save(pw)

    # 2) 四图对比 -> image_assist_process/
    pv = os.path.join(process_dir, f"{stem}_vis.png")
    fig.savefig(pv, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) 可选：训练用的灰度 mask（默认不存）
    if save_mask:
        pm = os.path.join(process_dir, f"{stem}_mask.png")
        Image.fromarray(mask * 255).save(pm)

    print(f"    ✓ 已保存  补洞={FILL_NAMES[state['level']]}  "
          f"占比={mask.mean()*100:.1f}%")
    return "save", mask


# ------------------------------------------------------------------
def collect_images(image_dir, only=None, skip_done=False, check_dir=None):
    files = []
    for pat in IMG_EXT:
        files.extend(glob.glob(os.path.join(image_dir, pat)))
    files = sorted(set(files))
    if only:
        files = [f for f in files if only in os.path.basename(f)]
    if skip_done and check_dir:
        files = [f for f in files
                 if not os.path.exists(os.path.join(
                     check_dir,
                     os.path.splitext(os.path.basename(f))[0] + "_white.png"))]
    return files


def load_model(checkpoint, device):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"\n找不到权重: {checkpoint}\n"
            f"下载: wget https://dl.fbaipublicfiles.com/segment_anything_2/"
            f"092824/sam2.1_hiera_small.pt -O {checkpoint}")
    print(f"[模型] 加载 SAM 2 ({device}) ...")
    model = build_sam2(config_file="configs/sam2.1/sam2.1_hiera_s.yaml",
                       ckpt_path=checkpoint, device=device)
    print("[模型] 就绪 ✓")
    return SAM2ImagePredictor(model)


# ------------------------------------------------------------------
def process_one(predictor, path, prefix, ans_dir, process_dir,
                single=False, save_mask=False):
    stem = os.path.splitext(os.path.basename(path))[0]
    image = np.array(Image.open(path).convert("RGB"))
    H, W = image.shape[:2]
    predictor.set_image(image)
    print(f"    尺寸 {W}×{H}")

    box = draw_box(image, prefix)
    if box is None:
        return "skip"

    pts, lbl = pick_points(image, box, prefix=prefix)
    if pts is None:
        return "skip"

    mask_input = None
    while True:
        kw = {"box": box, "multimask_output": True,
              "point_coords": pts, "point_labels": lbl}
        if mask_input is not None:
            kw["mask_input"] = mask_input
        masks, scores, logits = predictor.predict(**kw)
        print(f"    候选分数 {[round(float(s), 3) for s in scores]}")

        act, _ = review(image, box, masks, scores, stem, prefix,
                        ans_dir, process_dir, single, save_mask)
        if act in ("save", "skip", "prev", "quit"):
            return act
        if act == "redo":
            print("    [补点] 缺失处左键，多余处右键，回车重算")
            pts, lbl = pick_points(image, box, prev_points=_points, prefix=prefix)
            if pts is None:
                return "skip"
            try:
                lo = logits[int(np.argmax(scores))]
                if lo.ndim == 2:
                    lo = lo[None, ...]
                mask_input = lo
            except Exception:
                mask_input = None
            continue
    return "skip"


# ------------------------------------------------------------------
def choose_type():
    """运行时选择要处理的图片类型。"""
    print("\n请选择要处理的图片类型：")
    print(f"  1) OK 图   读 {IMAGE_DIR}")
    print(f"             写 {ANS_DIR}")
    print(f"  2) 瑕图    读 {XIA_IMAGE_DIR}")
    print(f"             写 {XIA_ANS_DIR}")
    if not sys.stdin.isatty():
        print("[非交互环境] 使用默认 1) OK 图")
        return "ok"
    while True:
        s = input("请输入 1 或 2 [默认 1]: ").strip()
        if s in ("", "1"):
            return "ok"
        if s == "2":
            return "xia"
        print("  无效输入，请输入 1 或 2")


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="SAM 2 交互标注切割（运行时选择 OK 图 / 瑕图）")
    ap.add_argument("name", nargs="?", default=None,
                    help="只标某一张，可写文件名或关键字")
    ap.add_argument("--type", default=None, choices=["ok", "xia"],
                    help="ok=OK图(image/)  xia=瑕图(image_xia/)；不填则运行时询问")
    ap.add_argument("--image-dir", default=None,
                    help="覆盖 --type 的读图目录")
    ap.add_argument("--ans-dir", default=None,
                    help="覆盖 --type 的白底结果输出目录")
    ap.add_argument("--process-dir", default=None,
                    help="覆盖 --type 的对比图输出目录")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--start", type=int, default=0, help="从第几张开始(0基)")
    ap.add_argument("--skip-done", action="store_true", help="跳过已标注的")
    ap.add_argument("--save-mask", action="store_true",
                    help="额外保存灰度 mask（训练时才需要）")
    args = ap.parse_args()

    if args.type is None:
        args.type = choose_type()

    d_image, d_ans, d_process = TYPE_DIRS[args.type]
    args.image_dir = args.image_dir or d_image
    args.ans_dir = args.ans_dir or d_ans
    args.process_dir = args.process_dir or d_process

    os.makedirs(args.ans_dir, exist_ok=True)
    os.makedirs(args.process_dir, exist_ok=True)

    print(f"[类型] {args.type}"
          + ("（OK图）" if args.type == "ok" else "（瑕图）"))

    if not os.path.isdir(args.image_dir):
        print(f"图片目录不存在: {args.image_dir}")
        print("请先整理： mkdir -p image && mv *.jpg *.png image/")
        sys.exit(1)
    if not HAS_SCIPY:
        print("[提示] 未装 scipy，补洞不可用。安装: pip install scipy")

    only = args.name
    files = collect_images(args.image_dir, only=only,
                           skip_done=args.skip_done,
                           check_dir=args.ans_dir)
    if not files:
        print(f"没有待处理图片（目录: {args.image_dir}）")
        sys.exit(0)

    single = bool(only) and len(files) == 1
    print(f"[任务] 共 {len(files)} 张")
    print(f"  白底结果 -> {args.ans_dir}")
    print(f"  对比图   -> {args.process_dir}")
    print("=" * 60)

    predictor = load_model(args.checkpoint, args.device)

    i = max(0, min(args.start, len(files) - 1))
    done = 0
    while 0 <= i < len(files):
        path = files[i]
        stem = os.path.splitext(os.path.basename(path))[0]
        prefix = f"[{i+1}/{len(files)}] {stem}"
        print(f"\n{prefix}")

        act = process_one(predictor, path, prefix, args.ans_dir,
                          args.process_dir, single, args.save_mask)

        if act == "save":
            done += 1
            if single:
                break
            i += 1
        elif act == "skip":
            print("    → 跳过")
            if single:
                break
            i += 1
        elif act == "prev":
            i = max(0, i - 1)
        elif act == "quit":
            print(f"\n[结束] 已保存 {done} 张")
            print("下次继续： python3 image_assist.py --skip-done")
            return

    print(f"\n[完成] 共保存 {done} 张")
    print(f"  {args.ans_dir}      *_white.png（白底结果）")
    print(f"  {args.process_dir}  *_vis.png（四图对比）")


if __name__ == "__main__":
    main()
