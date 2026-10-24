#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4.5 单图无参考表面异常检测（辅助复核）
==========================================
用途：检出「细长 / 低对比」的表面瑕疵（污渍、划痕、色斑）、「大黑点」，
      以及「细划痕 / 毛发」（衣领白毛绒上被划出的深色细线）。

跨图比对(PatchCore)对这类瑕疵天生失效——同一只玩偶换一张照片，配准残差
就落在轮廓高光上，量级(0.76~0.91)反而高过污渍本身(0.67)，改阈值只是
在「漏检污渍」和「误报轮廓」之间来回换。本模块换范式：**不依赖参考图、
不依赖跨图配准**，只用单张图自身的低频背景做差分。

--------------------------------------------------------------
数学形式
--------------------------------------------------------------
记灰度图 I，物体掩膜 M_obj（canonical 的 *_mask.png）。

1) 表面掩膜（只保留光面塑料区，排除设计图案与轮廓）
   chroma(p) = sqrt((a*-128)^2 + (b*-128)^2)          LAB 色度
   grad(p)   = |∇I(p)|                                Sobel 梯度幅值
   D         = dilate( (I<g0) ∨ (chroma>c0), k_d )     设计图案带
   S         = erode(M_obj, k_e) ∧ ¬D ∧ (chroma≤c0) ∧ (grad<g0)

2) 低频背景差分（正 = 比局部背景暗）
   R(p) = G_σb * I (p) − I(p)                          G_σb = 高斯核(σb)

3) 定向线状匹配滤波（归一化卷积，沿瑕疵走向积分）
   对 θ ∈ {0, 180/n, ..., 180−180/n}：
       num_θ = G_σl ⊛ (R · S)      沿 θ 方向的一维高斯积分
       den_θ = G_σl ⊛ S
   线状响应
       R̂(p) = max_θ  num_θ(p) / max(den_θ(p), ε)
   归一化卷积的关键：分子分母同掩膜滤波，故 R̂ 是「窗口内表面像素的
   加权均值」，不会因掩膜稀疏而被除小数放大；den < s0（窗口内表面
   占比过低）处直接置 0，不外推。

4) 噪声尺度（不用 MAD！）
   原图平坦区被压缩量化，实测 59% 的表面像素 R 恰好为 0，MAD 退化为 0。
   改用高分位估计全局尺度 σ_global：
       σ_global = percentile(|R̂|_S, q_n) / 1.2816
       （1.2816 = 标准正态的 90% 分位，故 q_n=90 时即高斯噪声估计）

5) 局部对比度归一化（关键：全局 σ 会让 z 跨图不可比）
   实测同一参数下 OK 件 top z 可达 55.4 / 34.5 / 30.7，而污渍只有 8.89——
   因为 σ_global 受该图局部对比度/清晰度调制，模糊的图整片 z 抬高。
   改用逐点局部尺度（滑窗 std，与像素位置无关的图内相对量）：
       μ(p)   = boxFilter(R̂, w)(p)
       s(p)   = sqrt( max(boxFilter(R̂², w)(p) − μ(p)², 0) )
       den(p) = max( s(p), f·σ_global )        下限防除零、防放大平坦区
       Z(p)   = R̂(p) / den(p) ,  p ∈ S          （默认 w=81, f=0.5）
   local_win=0 时退回全局：den(p) = σ_global。

6) 异常 z 图与阈值（默认跨图固定阈值，不用本图分位）
       若 z_thresh > 0：  τ = z_thresh                  （跨图可比，默认 6.0）
       若 thresh_quantile > 0：τ = percentile(Z_S, q_t)  （随图自适应，会漏掉低对比件）
   候选 = (Z > τ) ∧ S 的连通域，面积 ≥ A_min 者输出，按 Z_max 降序编号。
   实测（despeckle + A_min=20，z_thresh=6）：OK 件 19/30 张被标记(30 个候选，
   均为毛绒细纹噪声)；11-47-25 命中 5 个(鼻/双臂/胸口/双腿)。A_min=8 时 OK 误报
   升到 26/30 张(94 个候选)。改用本图分位 q_t=99.5 则 OK 件 26/30 张被标记
   (63 个候选)，误报不可用。

7) 暗斑（大黑点 / 脸眼细瑕）检测 —— 与 (1)~(6) 并行的第二条支路
   污渍是低对比(实测 12~16 灰阶)的细长/弥散瑕疵，信噪比低、只能辅助复核；
   而「大黑点」对比度高达 ~100 灰阶（实测斑块核心 158 vs 周围环 235），信噪比
   高一个量级，可以做到接近零误报，故单独一条支路、用更直接的判据。
   与 (1)~(6) 的三点关键区别：
     a) 用物体掩膜 M_obj，**不用表面掩膜 S**。因为 S 会排除 gray<g0 的像素，
        而黑点本身灰度就低（~158 < 180），用 S 会把黑点直接排除掉。
     b) 不做定向线状滤波。黑点是面状暗区，不需要沿走向积分（那是给细长划痕的）。
     c) 局部尺度用 **MAD**（中值绝对偏差）+ 小窗，而非滑窗 std：
        环(p) = dilate(连通域, rr) − 连通域，仅取 M_obj 内像素
        dark_core(p) = percentile(gray[连通域], 10)     暗斑核灰度(最暗 10%)
        mad_w(p) = median_w( |R − median_w(R)| )         逐点局部 MAD
        Z_dark(p) = R(p) / max(1.4826·mad_w(p), f·σ_global)
        候选 = (Z_dark > dark_z) ∧ M_obj 的连通域，面积 ∈ [dark_min_px, dark_max_px]
               且域内 Z_dark,max ≥ dark_z_peak_min（滤低对比噪声块）
               且紧贴环(见下)满足全部色相/灰度门限
    **紧贴环是关键**：ring = dilate(连通域, rr) − 连通域 ∧ M_obj，rr = dark_ring_mm
    换算成像素。必须取窄环(默认 0.2mm ≈ 4px)：宽环(1.0mm ≈ 22px)会探入相邻的
    深蓝虹膜，使眼区细暗弧的环灰度降到 116.7、对比度降到 22.9，反而被环门限
    整条拒绝。窄环下 4 个目标瑕疵的环灰度为 145~193、对比度 37.7~75.2。
    环门限（全部为「>0 才启用」的软门）：
        ring_gray ≥ dark_ring_gray(140)         周围是亮塑料，排除深色设计图案
        ring_gray − dark_core ≥ dark_contrast(30) 确实足够暗，排除柔和阴影
        dark_core ≥ dark_core_gray_min(60)       核不能是纯黑设计区(鼻/虹膜 core≈15~34)
        ring 色相中位数 ≥ dark_ring_hue_min(60)  **核心判别量**，见下
        ring 饱和度中位数 ≥ dark_ring_sat_min(60) 排除灰白毛绒/塑料高光
        ring 灰度 std ≤ dark_ring_std_max(25)     排除跨边界的混合环
        ring 环形色相 std ≤ dark_ring_hue_std_max(2.5) 排除色相散乱的环
    为什么色相中位数是唯一完美判别量：玩偶的彩色塑料（teal 眼睑、粉色腮红）在
    OpenCV HSV 下 H≈95/176，而 OK 件上被门限误命中的区域是米色/棕色毛绒，H≈14~20，
    两者完全分开；灰度类判别量(ring_sat_p10、ring_gray_p90)则在目标与误报间重叠。
    实测（bgy 30 张 OK + bgy_xia 6 张瑕，σb=6, dark_z=8.0, w=21, f=0.5, ring 0.2mm）：
    OK 件 0/30 误报；11-38-53 的脸/眼 4 个瑕疵全部命中——
      眼上细暗弧 a=9  z=14.8 core=93.8 hue=95.0 hstd=1.18
      眼下细暗弧 a=58 z=33.8 core=75.1 hue=95.0 hstd=2.15
      左眼单点   a=1  z=13.0 core=95.0 hue=95.0 hstd=2.07
      腮红黑点   a=10 z=20.9 core=155.1 hue=176.0 hstd=1.01
    其中 ring_std≤25 是**必需**的：去掉它会在 bgy_016 的 (320,401) 合模线上误报
    (a=15 z=14.2 core=139.0 hue=103.0 hstd=1.35，只有 ring_std=40.1 超标)。
    注意 dark_z 不能贪高：细暗弧的 z 随阈值上升面积会骤塌（实测 dz=8→16px、
    dz=9→4px），一旦低于 dark_min_px 就被面积门限整条丢掉。dark_min_px 已由 8 降到 1
    （紧贴环色相门限才是真正判别器，面积门限只作兜底），dz=8.0 时 arc 面积 16px。

8) 连通域去噪：为什么不用形态学开运算
   二值图上去噪的顺手写法是 3×3 MORPH_OPEN，但它会**抹掉宽 2~3px 的细线**。
   而本项目的瑕疵（合模缝发黑、细划痕）恰恰就是这种细线：实测 canonical 1026
   下左毛球交界缝线 z=8.20 / area=11px，开运算后候选数 6→1，直接漏检。
   改用 despeckle()：只保留「自身 + ≥2 个 8 邻域像素」的点，即删孤立单点与
   线头、保留细线主体。同参数对照（30 张 OK + 6 张瑕，dark_z=6.5）：
   开运算 0/1，despeckle 0/6 —— 零误报且全部命中。
   三条支路现统一用 despeckle()；污渍支路因不再被开运算抹掉细线，须把 A_min 由 8
   提到 20，否则 OK 件误报升到 26/30 张、94 个候选（A_min=8 的开运算基线是
   8/30 张、10 个候选，非 0）。

9) 细划痕 / 毛发检测 —— 第三条并行支路（衣领白毛绒上的深色细线）
   与 (1)~(6) 和 (7) 的又一区别：
     a) **不走表面掩膜 S**。划痕灰度 ~128 < design_gray(180)，会被 (1) 的
        设计图案带整片排除（划痕处 surf 比例 = 0、z = 0），所以只能另起支路。
     b) 用**中值背景**做差分（核 scratch_bg_k=31）：划痕只有 2~4px 宽，中值
        背景不会被它拉低，故 res = medianBlur(gray, 31) − gray 在划痕处显著为正。
     c) 物体掩膜先做**闭运算补缝**（scratch_mask_close=15）：canonical 掩膜在
        白毛绒区常出现小缺口，划痕恰好落在缺口上（实测 m=0 于 (714,390)）。
   判据（全部满足才命中）：
        b = despeckle( (res > scratch_contrast) ∧ close(M_obj, kc) ) 的连通域
        面积 ∈ [scratch_min_px(20), scratch_max_px(600)]     排除单点噪声与成片暗区
        长宽比 max(w,h)/min(w,h) ≥ scratch_aspect_min(2.0)   细长线状
        紧贴环(rr=8px) 灰度均值 ≥ scratch_ring_gray(215)      周围是亮面(白毛绒/白塑料)
        紧贴环 饱和度均值 ≤ scratch_ring_sat(30)              周围是中性色(非彩色)
        核灰度(10 分位) ∈ [scratch_core_gray_min(80),
                          scratch_core_gray_max(210)]          非纯黑设计图案(眼/鼻 core≈15~34)
        **到物体轮廓距离 ≥ scratch_edge_mm(0.7mm)**           核心判别量，见下
   「到物体轮廓距离」silhouette_dist()：物体轮廓 = 掩膜最大连通域**填洞后**的
   边界（填洞是为了不被掩膜内部小缺口带偏）。实测真实划痕离轮廓 140.8px，而
   所有误报（头部顶边/身体右侧边/身体底部/脸颊边缘/彩色区边界的轮廓暗边与
   剪影阴影）都紧贴轮廓（≤4.2px）。仅此一条即可把 23 个误报全部清除。
   实测（bgy 30 张 OK + bgy_xia 6 张瑕，contrast=40, close=15, rr=8）：
   OK 件 0/30 误报；11-38-53 的衣领划痕命中 1 个——
      area=83 w=22 h=7 aspect=3.14 ring=244.8 ring_sat=11.5 core=128.8
      contrast=116.0 edge=6.54mm @(390,714)
   鲁棒性：close=0/9/15/21、contrast=35/40/45/50、rr=5/8/12 均稳定 1 命中 0 误报。

10) 低对比擦痕 / 弥散污渍检测 —— 第四条并行支路（白色塑料大平面上的淡灰擦痕）
    为什么前三条支路都会漏掉它：以 11-47-25 下巴擦痕为例（人工可见、宽约 60px、
    只有 5~10 灰阶），实测
      - ridge 原始响应 R̂ ≈ 7.9 灰阶，但周围 sd_local ≈ 1.4、σ_global ≈ 0.75，
        归一化后 Z 仅 4.91 < z_thresh(6.0) → 污渍支路整片漏检；
      - 擦痕灰度 ~213 > design_gray(180)，不是「大黑点」→ 暗斑支路不适用；
      - 擦痕宽 60px、长宽比 ~2.8，但对比度 10 灰阶 << scratch_contrast(40)
        → 细划痕支路不适用。
    本支路改用「平坦白塑区 + 原始幅度 + 成片面积」三重判据（不做局部归一化，
    因为局部尺度会被擦痕自己抬高，正是前三条失效的根因）：
      a) 平坦白塑区 F：
           F = S ∧ (sd15 < scuff_sd_max(8)) ∧ (I > scuff_gray_min(195))
               ∧ (chroma_σ3 ≤ scuff_chroma_max(8)) ∧ (dist_design > scuff_design_px(20))
               ∧ (dist_silhouette > scuff_silh_min(50))
         sd15 = 15×15 滑窗 std（纹理量）。chroma_max 收紧到 8（全局是 25）是关键：
         实测白塑 chroma≈5.4，而 OK 件误报的青色/米色高光 chroma 达 12.6~14.0，
         收紧后 bgy_018/bgy_022 的高光整片被排除。dist_silhouette 排除肢体交界处的
         结构阴影阶跃（bgy_010 的最大误报就在白色躯体与米色阴影的交界上）。
      b) b = (G_1.6 * R̂ > scuff_ridge(5.0)) ∧ F 的连通域
         轻度平滑(σ=1.6)抹掉单像素尖峰，保留成片擦痕。
      c) 面积 ∈ [scuff_min_px(250), scuff_max_px(3000)] —— **核心判别量**。
         擦痕是成片区域，OK 件的高光/纹理尖峰面积小。
    实测（bgy 30 张 OK + bgy_xia 6 张瑕，r=5.0, A_min=250px, chroma≤8）：
      OK 件 **0/30 误报、0 候选**；11-47-25 下巴擦痕命中——
        R1 r=6.7 a=299px bbox=[476,456,58,21]（人工框选位置完全一致）。
    裕度说明（务必注意）：这是**临界可辨**，不是高置信度。同一判据下 OK 件最大的
    平坦区响应为 bgy_003 的 227px、bgy_018 的 202px，瑕图 299px 仅高出约 1.3 倍。
    取 A_min=200 会变成 2/30 误报。因此本支路输出仍必须人工复核。

--------------------------------------------------------------
定位：输出的是**候选框**，供人工复核，不是自动判级结论。
      低对比瑕疵(实测 12~16 灰阶)在这个信噪比下无法做到零误报零漏报。
      实测裕度：污渍 z(8.46) / OK 最大 p99.99 ≈ 0.50~0.85 (<1)，即污渍信号
      不超过 OK 件噪声底，本模块只能做「辅助复核」，不能替代自动判级。
      大黑点(对比度 ~100 灰阶)不在此列，信噪比高一个量级，误报率显著更低。
      第 (10) 条支路同理：0/30 误报是**筛出来的**，裕度仅 1.3 倍，换一批样件或
      换打光条件就可能失效；对这类亚 10 灰阶擦痕，根治办法是掠射光/偏振光成像。

输出: reports/<sku>/<stem>_refree.png         候选框 + z 热力（并排）
      reports/<sku>/overlay/<stem>_overlay.png 全分辨率原图 + 红/蓝/品红/黄标注（单独文件夹，每张图一个）
      reports/<sku>/<stem>_overlay_face.png   暗斑候选放大（若有）
      reports/<sku>/<stem>_overlay_scratch.png 细划痕候选放大（若有）
      reports/<sku>/<stem>_overlay_scuff.png  低对比擦痕候选放大（若有）
      reports/<sku>/<stem>_overlay_speck.png  灰面斑点候选放大（若有）
      reports/<sku>/<stem>_refree.json        候选列表
      reports/<sku>/refree_summary.json       汇总

用法:
    python3 ref_free.py --sku bgy_xia
    python3 ref_free.py --sku bgy_xia --stem "Screenshot from 2026-10-22 11-47-25_white"
    python3 ref_free.py --sku bgy_xia --z-thresh 7.0      临时收/放固定阈值(污渍支路)
    python3 ref_free.py --sku bgy_xia --dark-z 8.0        临时收/放暗斑阈值
    python3 ref_free.py --sku bgy_xia --no-dark           关闭暗斑(大黑点)支路
    python3 ref_free.py --sku bgy_xia --no-scratch        关闭细划痕/毛发支路
    python3 ref_free.py --sku bgy_xia --no-scuff          关闭低对比擦痕/弥散污渍支路
    python3 ref_free.py --sku bgy_xia --no-speck          关闭灰面斑点支路
    python3 ref_free.py --sku bgy_xia --quantile 99.0     改用本图分位阈值(不推荐)
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

# 参数默认值；configs/<sku>.yaml 的 ref_free: 段可逐项覆盖
DEFAULTS = {
    "bg_sigma": 6.0,          # σb  低频背景高斯尺度(px)
    "line_sigma": 12.0,       # σl  定向积分核长度尺度(px)
    "angles": 12,             # n   方向数（180° 均分）
    "min_support": 0.3,       # s0  窗口内表面占比下限，低于此不外推
    "chroma_max": 25.0,       # c0  表面色度上限(LAB)，超过视为彩色图案
    "design_gray": 180,       # g0  灰阶下限，低于视为深色图案
    "design_dilate": 9,       # k_d 设计图案掩膜膨胀(px)
    "grad_max": 40.0,         # 表面梯度上限(Sobel 幅值)
    "obj_erode": 9,           # k_e 物体掩膜内缩(px)，去掉轮廓带
    "noise_percentile": 90.0,  # q_n 全局噪声尺度分位（也是局部尺度的下限基准）
    "local_win": 81,          # 局部尺度窗口(px)，0=退回全局 σ
    "local_floor": 0.5,       # 局部尺度下限 = f·σ_global
    "z_thresh": 6.0,          # 固定阈值（局部归一化后跨图可比）；<=0 则改用 thresh_quantile
    "thresh_quantile": 0.0,   # >0 时改用本图分位阈值（会随图自适应，跨图不可比）
    "min_area": 20,           # A_min 最小连通面积(px)。去噪改用 despeckle 后细线不再被
                              # 抹掉，须用较大面积门限抑制毛绒细纹噪声（见模块头 (8)）
    "min_mm": 0.0,            # 最小等效直径(mm)，0=不启用
    "max_mm": 15.0,           # 最大等效直径(mm)，超过视为整片光照
    "colormap_scale": 12.0,   # 热力图色标满量程 z
    # ---- 暗斑(大黑点/脸眼细瑕)支路 ----
    "dark_enable": True,      # 是否启用暗斑检测
    "dark_z": 8.0,            # 暗斑 z 阈值(局部归一化后跨图可比)；细暗弧在高阈值下面积会塌到
                              # dark_min_px 以下(实测 dz=9 时 arc 只剩 4px)，故取 8.0
    "dark_min_px": 1,         # 暗斑最小连通面积(px)。细暗弧经 despeckle 后可只剩 1px，
                              # 真正的判别由紧贴环色相门限承担(见 dark_ring_hue_min)
    "dark_max_px": 400,       # 暗斑最大连通面积(px)，超过视为大片深色设计图案
    "dark_z_peak_min": 12.0,  # 连通域内最大 z 下限；>0 才启用。滤掉低对比噪声块
    "dark_ring_mm": 0.2,      # 紧贴环宽度(mm)。**不能取宽环**：1.0mm 会探入相邻深蓝虹膜，
                              # 使眼区细暗弧的环灰度降到 116.7、对比度降到 22.9，反被环门限拒绝
    "dark_ring_gray": 140.0,  # 紧贴环灰度下限，低于视为深色设计图案(眼睛/花纹)
    "dark_contrast": 30.0,    # 环-核灰度差下限(绝对暗度)，防把柔和阴影当黑点
    "dark_core_gray_min": 60.0,   # 核灰度下限(>0 才启用)：低于此视为纯黑设计区(鼻/虹膜 core≈15~34)
    "dark_ring_hue_min": 60.0,    # 紧贴环色相中位数下限(OpenCV HSV H∈[0,180])。核心判别量：
                                  # 瑕疵所在的彩色塑料环 teal/粉 为 95/176，OK 误报的米色/棕毛绒为 14~20
    "dark_ring_sat_min": 60.0,    # 紧贴环饱和度中位数下限；排除灰白毛绒/塑料高光
    "dark_ring_std_max": 25.0,    # 紧贴环灰度标准差上限(必需)：无此门会误报 bgy_016 的
                                  # (320,401) 合模线(a=15 z=14.2 core=139 rs=40.1，其余门限全过)
    "dark_ring_hue_std_max": 2.5, # 紧贴环环形色相标准差上限(度)；>2.5 视为色相散乱
    # 局部尺度估计：默认 MAD(中值绝对偏差)+小窗，见模块头公式 (7)
    "dark_scale_mad": True,   # True=MAD 尺度(抗设计边界/瑕疵自身抬爆)，False=滑窗 std
    "dark_local_win": 21,     # 局部尺度窗口(px)，0=退回全局 σ
    "dark_scale_floor": 0.5,  # 局部尺度下限 = f·σ_global，防平坦区除零放大
    # 背景估计：median=中值背景(默认)，gauss=高斯背景
    "dark_bg_mode": "median",  # 中值背景不被相邻深色设计区(深蓝虹膜)下拉
    "dark_bg_median_k": 21,    # 中值背景核(px)，须为奇数
    # ---- 细划痕/毛发支路 ----
    "scratch_enable": True,    # 是否启用细划痕检测
    "scratch_bg_k": 31,        # 中值背景核(px，奇数)。划痕只有 2~4px 宽，中值背景不把它糊掉
    "scratch_contrast": 40.0,  # 暗于中值背景的灰阶下限 (bg − gray > c)
    "scratch_mask_close": 15,  # 物体掩膜闭运算核(px)：划痕常落在掩膜缝隙处，需先补缝
    "scratch_min_px": 20,      # 最小连通面积(px)
    "scratch_max_px": 600,     # 最大连通面积(px)，超过视为成片暗区(交给其它支路)
    "scratch_aspect_min": 2.0, # 长宽比下限：细长线状
    "scratch_ring_px": 8,      # 紧贴环宽度(px)
    "scratch_ring_gray": 215.0,      # 紧贴环灰度下限：周围须是亮面(白毛绒/白塑料)
    "scratch_ring_sat": 30.0,        # 紧贴环饱和度上限：周围须是中性(非彩色)
    "scratch_core_gray_min": 80.0,   # 核灰度下限：排除纯黑设计图案(眼/鼻 core≈15~34)
    "scratch_core_gray_max": 210.0,  # 核灰度上限：须明显暗于环(中性灰划痕，非亮纹理)
    "scratch_edge_mm": 0.7,    # 到物体轮廓(最大连通域填洞后)的最小距离(mm)。**核心判别量**：
                               # 轮廓暗边/剪影阴影紧贴轮廓(实测 ≤4px)，细划痕在物体内部(实测 140px)
    # ---- 低对比擦痕/弥散污渍支路(第4支路) ----
    # 针对"白色塑料大平面上的淡灰擦痕"：对比度仅 5~10 灰阶、宽 30~60px，
    # 前三条支路全部漏检(实测下巴擦痕 ridge 响应 7.9，但 z 仅 4.91 < z_thresh=6.0)。
    "scuff_enable": True,       # 是否启用低对比擦痕支路
    "scuff_ridge": 5.0,         # 平坦区内 ridge_response 原始幅度下限(灰阶)。
                                # 实测：瑕图下巴擦痕 7.9；OK 件平坦区 p99.9 中位 5.4、最大 9.3
    "scuff_sd_max": 8.0,        # 局部纹理 sd(15×15) 上限：擦痕所在的白色塑料应是低纹理平面
    "scuff_chroma_max": 8.0,    # 分支局部色度上限(LAB，比全局 chroma_max=25 更严)：
                                # 实测白塑 chroma≈5.4，而 OK 误报的青色/米色高光 12.6~14.0，
                                # 收紧到 8 可把 bgy_018/bgy_022 的高光整片排除
    "scuff_gray_min": 195.0,    # 灰度下限：只在亮白塑料上找，排除深色图案/阴影
    "scuff_design_px": 20.0,    # 距设计区(暗/彩色)的最小距离(px)，避开设计边缘的 halo
    "scuff_silh_min": 50.0,     # 到剪影的最小距离(px)，排除肢体交界处的结构阴影阶跃
    "scuff_smooth_sigma": 1.6,  # ridge 图轻度平滑(px)：抹掉单像素尖峰，保留成片擦痕
    "scuff_min_px": 250,        # 最小连通面积(px)。**核心判别量**：擦痕是成片区域，
                                # OK 件的高光/纹理尖峰面积小。实测 OK 最大 227px，
                                # 瑕图下巴 299px → 取 250 可做到 OK 0/30
    "scuff_max_px": 3000,       # 最大连通面积(px)，超过视为成片光照/整体色差
    # ---- 灰面密集微斑点支路(第5支路，头部/灰色塑料) ----
    # 灰色塑料件上的"成片深色麻点/污点"：单点仅 2~5px、对比度 8~25 灰阶。
    # 前四条支路全部漏检，且不是阈值问题：
    #   污渍支路：surface_mask 用 gray<design_gray(180) 判"设计图案"，灰色头身
    #             (实测中位 182、下四分位 163) 被整片当成设计图案排除；
    #   暗斑支路：紧贴环要求饱和度高(>=60)且色相>=60，灰面环 sat≈29 → 全被拒；
    #   细划痕支路：要求环灰度>=215(亮面)，灰头 ≈180 → 被拒；
    #   擦痕支路：要求灰度>195 且成片面积>=250px，微斑点面积仅 2~120px → 被拒。
    # 本支路改为「光滑低饱和塑料面板 + 深色微点计数」的区域级判据：
    #   面板 F：gray∈[speck_gray_min, speck_gray_max] ∧ chroma<=speck_chroma_max
    #           ∧ sd15<=speck_sd_max ∧ 到剪影>=15px，取面积>=speck_panel_min_px 的连通域
    #   头带：面板只取物体 bbox 顶部 speck_band 比例以内，避开腿/腹部的纹理
    #   微点：res = median15(gray) − gray > speck_contrast，面积∈[2,120]px，
    #         核灰度>=speck_core_min，环-核对比度>=speck_ring_contrast
    #   判据：同一面板内微点数 >= speck_min_count 且 Σ(对比度×面积) >= speck_min_strength
    # 实测（30 张 OK + 6 张瑕）：11-49-47 头部面板 n=5、强度 1698；
    #   OK 件最大 n=3（n>=4 时 OK 0/30）。裕度约 1.7 倍(计数)，仍需人工复核。
    "speck_enable": True,
    "speck_panel_min_px": 40000,  # 参与统计的面板最小面积(px)，排除小块高光/耳朵
    "speck_gray_min": 140.0,      # 灰面灰度下限
    "speck_gray_max": 210.0,      # 灰面灰度上限（>210 是白色件，交给擦痕/污渍支路）
    "speck_chroma_max": 18.0,     # 灰面色度上限(LAB)，排除彩色设计
    "speck_sd_max": 16.0,         # 面板局部纹理 sd(15×15) 上限
    "speck_band": 0.40,           # 只统计物体 bbox 顶部这一比例内（头部）
    "speck_edge_px": 18,          # 面板腐蚀宽度(px)，排除角/接缝/剪影处的结构暗边
    "speck_bg_k": 15,             # 中值背景核(px，奇数)，须远大于斑点直径
    "speck_contrast": 10.0,       # 暗于中值背景的灰阶下限
    "speck_ring_contrast": 8.0,   # 环-核灰度差下限
    "speck_core_min": 55.0,       # 核灰度下限，排除纯黑设计
    "speck_min_px": 2,            # 单点最小面积(px)
    "speck_max_px": 120,          # 单点最大面积(px)，超过不算"微点"
    "speck_ring_px": 9,           # 紧贴环宽度(px)
    "speck_min_count": 4,         # 面板内微点数量下限。**核心判别量**：
                                  # 实测瑕图 5、OK 最大 3 → 取 4 可做到 OK 0/30
    "speck_min_strength": 1200.0, # 面板内 Σ(对比度×面积) 下限，滤掉弱纹理点
}


def params(cfg, overrides=None):
    p = dict(DEFAULTS)
    p.update(cfg.get("ref_free") or {})
    if overrides:
        p.update({k: v for k, v in overrides.items() if v is not None})
    return p


def surface_mask(img, msk, p):
    """返回 (表面掩膜 S, 灰度图 I)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    c0, g0 = float(p["chroma_max"]), float(p["design_gray"])
    kd = int(p["design_dilate"]) | 1
    design = cv2.dilate(((gray < g0) | (chroma > c0)).astype(np.uint8),
                        np.ones((kd, kd), np.uint8))
    ke = int(p["obj_erode"]) | 1
    obj = cv2.erode(msk.astype(np.uint8), np.ones((ke, ke), np.uint8)) > 0

    surf = obj & (design == 0) & (chroma <= c0) & (grad < float(p["grad_max"]))
    return surf, gray


def ridge_response(res, mask, angles, sig, min_support):
    """定向线状匹配滤波 R̂ = max_θ num_θ / max(den_θ, ε)，见模块头公式 (3)。"""
    C = res.shape[0]
    r = (res * mask).astype(np.float32)
    m = mask.astype(np.float32)

    k = int(sig * 3) | 1
    x = np.arange(k) - k // 2
    g1 = np.exp(-x ** 2 / (2.0 * sig ** 2))
    g1 /= g1.sum()

    num = np.zeros_like(res)
    den = np.zeros_like(res)
    n = max(1, int(angles))
    for a in range(n):
        ang = 180.0 * a / float(n)
        ctr = (C / 2.0, C / 2.0)
        M = cv2.getRotationMatrix2D(ctr, ang, 1.0)
        Mi = cv2.getRotationMatrix2D(ctr, -ang, 1.0)
        # 转到位 -> 沿行方向一维高斯积分 -> 转回来
        rn = cv2.filter2D(cv2.warpAffine(r, M, (C, C), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE),
                          -1, g1.reshape(1, -1))
        dn = cv2.filter2D(cv2.warpAffine(m, M, (C, C), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE),
                          -1, g1.reshape(1, -1))
        rn = cv2.warpAffine(rn, Mi, (C, C), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        dn = cv2.warpAffine(dn, Mi, (C, C), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        np.maximum(num, rn, out=num)
        np.maximum(den, dn, out=den)

    out = num / np.maximum(den, 1e-6)
    out[den < float(min_support)] = 0.0
    return out


def anomaly_z(img, msk, p):
    """返回 (表面掩膜, z 图, 尺度信息 dict)。

    z 图按模块头公式 (5)(6) 生成：local_win>1 时用逐点局部尺度归一化，
    否则退回全局 σ_global。
    """
    surf, gray = surface_mask(img, msk, p)
    res = cv2.GaussianBlur(gray, (0, 0), float(p["bg_sigma"])) - gray
    r = ridge_response(res, surf, int(p["angles"]), float(p["line_sigma"]),
                       float(p["min_support"]))
    info = {"surface_ratio": round(float(surf.mean()), 4)}
    sv = r[surf]
    if sv.size == 0:
        return surf, np.zeros_like(r, dtype=np.float32), info
    sig_g = max(1e-6, float(np.percentile(np.abs(sv), float(p["noise_percentile"]))) / 1.2816)

    win = int(p.get("local_win", 0) or 0)
    if win > 1:
        w = win | 1
        m1 = cv2.boxFilter(r, -1, (w, w), normalize=True)
        m2 = cv2.boxFilter(r * r, -1, (w, w), normalize=True)
        sd = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
        den = np.maximum(sd, float(p["local_floor"]) * sig_g)
        info["scale_mode"] = f"local{win}"
    else:
        den = np.full_like(r, sig_g)
        info["scale_mode"] = "global"

    info["noise_sigma"] = round(sig_g, 4)
    return surf, np.where(surf, r / den, 0.0).astype(np.float32), info


def object_mask(msk, p):
    """物体掩膜内缩(px)，去掉轮廓带。暗斑检测用它而非表面掩膜 S。"""
    ke = int(p["obj_erode"]) | 1
    return cv2.erode(msk.astype(np.uint8), np.ones((ke, ke), np.uint8)) > 0


def silhouette_dist(msk):
    """到「物体轮廓」的距离(px)。

    物体轮廓 = 掩膜**最大连通域填洞后**的边界（见模块头公式 (9)）。
    细划痕落在物体内部、离轮廓很远（实测 140px）；而轮廓暗边/剪影阴影紧贴
    轮廓（实测 ≤4px）。取最大连通域并填洞，是为了不被掩膜内部的小缺口/噪点
    带偏——本项目 canonical 掩膜在衣领等白毛绒区常出现小缺口。
    """
    m = (np.asarray(msk).astype(np.uint8) > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros(m.shape, np.float32)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    obj = (lab == big).astype(np.uint8)
    ff = obj.copy()
    h, w = ff.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    filled = obj | (ff == 0).astype(np.uint8)
    return cv2.distanceTransform(filled, cv2.DIST_L2, 5)


def local_mad(x, w):
    """逐点局部 MAD（中值绝对偏差），抗离群点。

    cv2.medianBlur 仅支持 CV_8U，故把残差平移量化到 uint8 再取中值：
        q = clip(x + 128, 0, 255)   (步长 1 灰阶)
        mad = median_w( |q − median_w(q)| )
    量化步长 1 灰阶对 MAD≈5 的量级误差约 25%，实测足够（见模块头公式 (7)）。
    """
    q = np.clip(x + 128.0, 0.0, 255.0).astype(np.uint8)
    med = cv2.medianBlur(q, int(w) | 1)
    dev = np.abs(q.astype(np.int16) - med.astype(np.int16)).astype(np.uint8)
    return cv2.medianBlur(dev, int(w) | 1).astype(np.float32)


def dark_bg(gray, p):
    """低频背景估计。median 模式取局部中值，gauss 模式取高斯模糊。

    为什么默认用中值：玩偶是大块平色塑料，瑕疵紧贴设计图案边界（眼区细暗弧
    紧贴 teal/深蓝虹膜）。高斯背景会跨边界把相邻深色设计区平均进来，导致
    res=G*I−I 在瑕疵处被下拉、且局部 MAD 被污染抬高，两者叠加把 z 压死。
    中值背景取窗口内多数色（teal 占多数），背景≈teal，瑕疵的 res 不再被
    相邻深色区拉低；且 σ_global 从 12.8 降到 4.7，floor 上限随之放宽。
    """
    mode = str(p.get("dark_bg_mode", "median") or "median").lower()
    if mode == "gauss":
        return cv2.GaussianBlur(gray, (0, 0), float(p["bg_sigma"]))
    k = int(p.get("dark_bg_median_k", 21) or 21) | 1
    return cv2.medianBlur(gray.astype(np.uint8), k).astype(np.float32)


def dark_z_map(img, msk, p):
    """暗斑 z 图。返回 (物体掩膜, z 图, 尺度信息 dict)。

    与 anomaly_z 的区别（见模块头公式 (7)）：
      - 黑点是面状暗区，不做定向线状滤波；
      - 用物体掩膜 M_obj 而非表面掩膜 S（S 会排除 gray<g0 的像素，而黑点
        本身灰度就低，用 S 会把黑点直接排除掉）。
    局部尺度默认用 MAD + 小窗（dark_local_win/dark_scale_floor）：滑窗 std
    会被「设计图案边界」与「瑕疵自身」两类离群点抬爆，把眼区细暗弧的 z 压到
    2.1；改 MAD 后同一位置 z 提升到 11~24，而平坦区由 floor 项压住不放大。
    背景默认用中值（dark_bg_mode）：高斯背景跨设计边界把相邻深色区平均进来，
    使眼区细暗弧的 res 被下拉、局部 MAD 被污染到 30，z 仅 2.16，整条支路漏检。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    obj = object_mask(msk, p)
    res = dark_bg(gray, p) - gray
    info = {"surface_ratio": round(float(obj.mean()), 4)}
    info["bg_mode"] = str(p.get("dark_bg_mode", "median") or "median")
    ov = res[obj]
    if ov.size == 0:
        return obj, np.zeros_like(res, dtype=np.float32), info
    sig_g = max(1e-6, float(np.percentile(np.abs(ov), float(p["noise_percentile"]))) / 1.2816)

    win = int(p.get("dark_local_win", p.get("local_win", 0)) or 0)
    floor = float(p.get("dark_scale_floor", p["local_floor"]))
    if win > 1 and bool(p.get("dark_scale_mad", True)):
        den = np.maximum(local_mad(res, win) * 1.4826, floor * sig_g)
        info["scale_mode"] = f"mad{win}"
    elif win > 1:
        w = win | 1
        m1 = cv2.boxFilter(res, -1, (w, w), normalize=True)
        m2 = cv2.boxFilter(res * res, -1, (w, w), normalize=True)
        sd = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
        den = np.maximum(sd, floor * sig_g)
        info["scale_mode"] = f"local{win}"
    else:
        den = np.full_like(res, sig_g)
        info["scale_mode"] = "global"

    info["noise_sigma"] = round(sig_g, 4)
    return obj, np.where(obj, res / den, 0.0).astype(np.float32), info


def circ_std(h):
    """环形色相标准差(度)。h 为 OpenCV HSV 的 H 通道(0~180，无向角)。

    用单位向量平均避免 0/180 环绕：R = |mean(e^{i·2h·π/180})|，
    std = sqrt(-2·ln R)。R→1 表示色相集中(std→0)，R→0 表示色相散乱(std→大)。
    """
    if h.size == 0:
        return 0.0
    a = h.astype(np.float64) * (2.0 * np.pi / 180.0)
    r = min(1.0, float(np.hypot(np.sin(a).mean(), np.cos(a).mean())))
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(r, 1e-9)))))


def despeckle(b):
    """去孤立噪点，但保留细线/细缝结构。

    这里**不能**用 3×3 形态学开运算：开运算会把宽 2~3px 的细线整片抹掉。
    实测 canonical 1026 下，左毛球与身体交界的那条深色缝线（z=8.2，
    area=11~17px）过完开运算后候选数从 6 掉到 1，用户指的瑕疵直接漏检。
    改为「自身 + ≥2 个 8 邻域像素」才保留，只删孤立单点与线头。
    实测（30 张 OK + 6 张瑕，dark_z=6.5）：开运算 0/1，本函数 0/6。
    """
    nb = cv2.filter2D(b.astype(np.float32), -1, np.ones((3, 3), np.float32),
                      borderType=cv2.BORDER_CONSTANT)
    return ((nb >= 3.0) & (b > 0)).astype(np.uint8)


def detect_dark(img, msk, p, mm_px):
    """检出暗斑(大黑点/脸眼细瑕)候选，返回 (候选 list（按 z_max 降序）, 尺度信息 dict)。

    判据见模块头公式 (7)：
      Z_dark > dark_z 的连通域，面积 ∈ [dark_min_px, dark_max_px]，且域内 z_max ≥
      dark_z_peak_min；对紧贴环 ring = dilate(域, rr) − 域 ∧ M_obj 统计：
        ring_gray ≥ dark_ring_gray       （周围是亮塑料，排除眼睛/花纹等深色设计）
        ring_gray − dark_core ≥ dark_contrast（确实足够暗，排除柔和阴影）
        dark_core ≥ dark_core_gray_min   （核不能是纯黑设计区，鼻/虹膜 core≈15~34）
        ring 色相中位数 ≥ dark_ring_hue_min（**核心判别量**：彩色塑料环才可能是瑕疵背景）
        ring 饱和度中位数 ≥ dark_ring_sat_min（排除灰白毛绒/高光）
        ring 灰度 std ≤ dark_ring_std_max    （排除跨边界的混合环，否则 bgy_016 合模线误报）
        ring 环形色相 std ≤ dark_ring_hue_std_max（排除色相散乱的环）
    rr 由 dark_ring_mm 换算，且**必须用紧贴环(0.2mm)**：宽环(1.0mm)会探入相邻
    深蓝虹膜，把眼区细暗弧的环灰度拉到 116.7、对比度拉到 22.9，反被环门限拒绝。
    """
    dark_z = float(p.get("dark_z", 0) or 0)
    if not p.get("dark_enable", True) or dark_z <= 0:
        return [], {}
    obj, z, dinfo = dark_z_map(img, msk, p)
    if not obj.any():
        return [], dinfo

    b = despeckle(((z > dark_z) & obj).astype(np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(b, 8)
    if n <= 1:
        return []

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)

    min_px = int(p["dark_min_px"])
    max_px = int(p["dark_max_px"])
    z_min = float(p.get("dark_z_peak_min", 0) or 0)
    rr = max(2, int(round(float(p["dark_ring_mm"]) / max(float(mm_px), 1e-6))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    pad = rr + 2
    ring_gray_min = float(p["dark_ring_gray"])
    contrast = float(p["dark_contrast"])
    core_min = float(p.get("dark_core_gray_min", 0) or 0)
    hue_min = float(p.get("dark_ring_hue_min", 0) or 0)
    sat_min = float(p.get("dark_ring_sat_min", 0) or 0)
    ring_std_max = float(p.get("dark_ring_std_max", 0) or 0)
    ring_hstd_max = float(p.get("dark_ring_hue_std_max", 0) or 0)

    cands = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px or area > max_px:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        # 按 bbox 裁剪后再做环膨胀，避免对每个候选做一次全图 dilate
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        m = (lab[y0:y1, x0:x1] == i)
        g = gray[y0:y1, x0:x1]
        core = float(np.percentile(g[m], 10.0))
        ring_m = cv2.dilate(m.astype(np.uint8), k).astype(bool) & ~m & obj[y0:y1, x0:x1]
        if not ring_m.any():
            continue
        rg = g[ring_m]
        ring = float(rg.mean())
        if ring < ring_gray_min or (ring - core) < contrast:
            continue
        if core_min > 0 and core < core_min:
            continue
        if z_min > 0 and float(z[y0:y1, x0:x1][m].max()) < z_min:
            continue
        rh = hue[y0:y1, x0:x1][ring_m]
        ring_hue = float(np.median(rh))
        if hue_min > 0 and ring_hue < hue_min:
            continue
        if sat_min > 0 and float(np.median(sat[y0:y1, x0:x1][ring_m])) < sat_min:
            continue
        ring_sd = float(rg.std())
        if ring_std_max > 0 and ring_sd > ring_std_max:
            continue
        ring_hsd = circ_std(rh)
        if ring_hstd_max > 0 and ring_hsd > ring_hstd_max:
            continue
        cands.append({
            "cx": round(float(cent[i][0]), 1), "cy": round(float(cent[i][1]), 1),
            "area_px": area, "eq_mm": round(2.0 * np.sqrt(area / np.pi) * mm_px, 3),
            "z_max": round(float(z[y0:y1, x0:x1][m].max()), 2),
            "ring_gray": round(ring, 1), "core_gray": round(core, 1),
            "contrast": round(ring - core, 1),
            "ring_hue": round(ring_hue, 1),
            "ring_hue_std": round(ring_hsd, 2),
            "ring_std": round(ring_sd, 1),
            "bbox": [x, y, w, h],
        })
    cands.sort(key=lambda d: -d["z_max"])
    return cands, dinfo


def detect_scratch(img, msk, p, mm_px):
    """检出细划痕/毛发候选，返回 (候选 list（按对比度降序）, 信息 dict)。

    判据见模块头公式 (9)：在中值背景差分图上取「暗于背景」的连通域，要求
      - 面积 ∈ [scratch_min_px, scratch_max_px]
      - 长宽比 ≥ scratch_aspect_min（细长线状）
      - 紧贴环灰度 ≥ scratch_ring_gray（周围是亮面）
      - 紧贴环饱和度 ≤ scratch_ring_sat（周围是中性色，排除彩色区）
      - 核灰度 ∈ [scratch_core_gray_min, scratch_core_gray_max]（中性灰，排除纯黑图案/亮纹理）
      - 到物体轮廓距离 ≥ scratch_edge_mm（**核心判别量**，排除轮廓暗边/剪影阴影）
    物体掩膜先做闭运算补缝：实测衣领划痕恰好落在掩膜的小缺口上，不补缝则
    划痕被切成碎片、面积掉到门限以下。
    """
    if not p.get("scratch_enable", True):
        return [], {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    k = int(p.get("scratch_bg_k", 31)) | 1
    bg = cv2.medianBlur(gray.astype(np.uint8), k).astype(np.float32)
    res = bg - gray

    kc = int(p.get("scratch_mask_close", 15)) | 1
    mu = cv2.morphologyEx(msk.astype(np.uint8), cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))) > 0
    b = despeckle(((res > float(p["scratch_contrast"])) & mu).astype(np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(b, 8)
    info = {"bg_k": k, "mask_close_px": kc}
    if n <= 1:
        return [], info

    sd = silhouette_dist(msk)
    edge_px = float(p.get("scratch_edge_mm", 0.7)) / max(float(mm_px), 1e-6)
    H, W = gray.shape
    rr = max(2, int(p.get("scratch_ring_px", 8)))
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    pad = rr + 2
    min_px, max_px = int(p["scratch_min_px"]), int(p["scratch_max_px"])
    aspect_min = float(p["scratch_aspect_min"])
    ring_min = float(p["scratch_ring_gray"])
    sat_max = float(p["scratch_ring_sat"])
    core_lo = float(p["scratch_core_gray_min"])
    core_hi = float(p["scratch_core_gray_max"])

    cands = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px or area > max_px:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(w, h) / max(1, min(w, h)) < aspect_min:
            continue
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        m = (lab[y0:y1, x0:x1] == i)
        g = gray[y0:y1, x0:x1]
        core = float(np.percentile(g[m], 10.0))
        if core < core_lo or core > core_hi:
            continue
        ring_m = cv2.dilate(m.astype(np.uint8), kk).astype(bool) & ~m & mu[y0:y1, x0:x1]
        if not ring_m.any():
            continue
        ring = float(g[ring_m].mean())
        if ring < ring_min:
            continue
        ring_sat = float(sat[y0:y1, x0:x1][ring_m].mean())
        if ring_sat > sat_max:
            continue
        sil = float(sd[y0:y1, x0:x1][m].mean())
        if sil < edge_px:
            continue
        cands.append({
            "cx": round(float(cent[i][0]), 1), "cy": round(float(cent[i][1]), 1),
            "area_px": area, "eq_mm": round(2.0 * np.sqrt(area / np.pi) * mm_px, 3),
            "len_mm": round(max(w, h) * mm_px, 3),
            "contrast": round(ring - core, 1),
            "ring_gray": round(ring, 1), "core_gray": round(core, 1),
            "ring_sat": round(ring_sat, 1),
            "edge_mm": round(sil * mm_px, 3),
            "bbox": [x, y, w, h],
        })
    cands.sort(key=lambda d: -d["contrast"])
    return cands, info


def detect_scuff(img, msk, p, mm_px):
    """检出「白色塑料平面上的低对比擦痕/弥散污渍」，返回 (候选 list, 信息 dict)。

    为什么需要第4支路：实测 11-47-25 下巴擦痕的 ridge 响应原始幅度约 7.9 灰阶，
    但它周围的局部尺度 sd_local≈1.4、σ_global≈0.75，归一化后 z 仅 4.91，
    低于 z_thresh=6.0 而被前三支路全部漏检。同时 OK 件平坦白塑区的 p99.9 可达
    5.4~9.3，靠 z 阈值本身无法把两者分开。

    本支路改用「平坦白塑区 + 原始幅度 + 成片面积」三重判据：
      1) 平坦白塑区：surface 掩膜 ∩ 局部纹理 sd(15×15) < scuff_sd_max
         ∩ 灰度 > scuff_gray_min ∩ 分支局部色度 ≤ scuff_chroma_max
         ∩ 距设计区 > scuff_design_px ∩ 距剪影 > scuff_silh_min
      2) 幅度：ridge_response 轻度平滑后 > scuff_ridge（不做局部归一化）
      3) 成片：连通面积 ∈ [scuff_min_px, scuff_max_px]
         —— 擦痕是成片区域，OK 件的高光/纹理尖峰面积小，这一条是主要判别量。

    实测（30 张 OK + 6 张瑕图）：OK 0/30 误报；11-47-25 下巴擦痕命中
    (bbox≈[476,456,58,21]，299px)。裕度很小（次大的 OK 响应 227px），属临界可辨。
    """
    if not p.get("scuff_enable", True):
        return [], {}
    surf, gray = surface_mask(img, msk, p)
    res = cv2.GaussianBlur(gray, (0, 0), float(p["bg_sigma"])) - gray
    ridge = ridge_response(res, surf, int(p["angles"]), float(p["line_sigma"]),
                           float(p["min_support"]))
    ridge = cv2.GaussianBlur(ridge, (0, 0), float(p.get("scuff_smooth_sigma", 1.6)))

    # 局部纹理(15×15 标准差)。必须先转 float32：gray 为 uint8，
    # gray*gray 会溢出回绕，导致 sd 变成伪随机量。
    gf = gray.astype(np.float32)
    m1 = cv2.boxFilter(gf, -1, (15, 15), normalize=True)
    m2 = cv2.boxFilter(gf * gf, -1, (15, 15), normalize=True)
    sd = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))

    # 分支局部色度（比全局 chroma_max 更严，用于排除彩色高光）
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
    chroma_s = cv2.GaussianBlur(chroma, (0, 0), 3.0)

    design = ((gray < float(p["design_gray"])) | (chroma > float(p["chroma_max"]))).astype(np.uint8)
    ddist = cv2.distanceTransform(1 - design, cv2.DIST_L2, 5)
    sil = silhouette_dist(msk)

    flat = (surf & (sd < float(p["scuff_sd_max"]))
            & (gray > float(p["scuff_gray_min"]))
            & (chroma_s <= float(p["scuff_chroma_max"]))
            & (ddist > float(p["scuff_design_px"]))
            & (sil > float(p["scuff_silh_min"])))

    b = ((ridge > float(p["scuff_ridge"])) & flat).astype(np.uint8)
    n, lb, stats, cent = cv2.connectedComponentsWithStats(b, 8)
    info = {"flat_ratio": round(float(flat.mean()), 4),
            "ridge_px": int(b.sum())}
    if n <= 1:
        return [], info

    min_px = int(p["scuff_min_px"])
    max_px = int(p["scuff_max_px"])
    cands = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px or area > max_px:
            continue
        m = (lb == i)
        cands.append({
            "cx": round(float(cent[i][0]), 1), "cy": round(float(cent[i][1]), 1),
            "area_px": area, "eq_mm": round(2.0 * np.sqrt(area / np.pi) * mm_px, 3),
            "ridge_max": round(float(ridge[m].max()), 2),
            "bbox": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                     int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
        })
    cands.sort(key=lambda d: -d["ridge_max"])
    return cands, info


def detect_speckle(img, msk, p):
    """检出「灰色/低饱和塑料平面上成片的深色微斑点」(模具麻点、污点)。

    为什么需要第5支路：以 11-49-47 头部为例，灰头上有几十个 2~5px、对比度
    8~25 灰阶的深色小点，单点都达不到前四条支路的判据（见 DEFAULTS 注释）。
    根因不是阈值，而是"灰色塑料"这个材质本身被前四条支路当成背景/设计排除。

    做法（区域级计数，而不是逐点判定）：
      1) 光滑灰面板 F：gray∈[gmin,gmax] ∧ chroma<=cmax ∧ sd15<=sd_max
         ∧ 到剪影>=15px，取面积 >= speck_panel_min_px 的连通域；
      2) 只保留物体 bbox 顶部 speck_band 比例内的部分（头部），并腐蚀
         speck_edge_px 去掉角/接缝/剪影暗边——这些结构暗边在 OK 件上同样存在；
      3) 面板内找微暗点：res = median_k(gray) − gray > speck_contrast，
         面积∈[min_px,max_px]，核灰度 >= speck_core_min，
         环-核对比度 >= speck_ring_contrast；
      4) 一个面板内微点数 >= speck_min_count 且 Σ(对比度×面积) >=
         speck_min_strength 时，输出一个"成片斑点"候选（框住这些微点）。

    实测（30 张 OK + 6 张瑕）：11-49-47 头部 n=5、强度 1698；
    OK 件最大 n=3、n>=4 时 OK 0/30。裕度约 1.7 倍，属临界可辨，须人工复核。
    """
    if not p.get("speck_enable", True):
        return [], {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)

    gf = gray.astype(np.float32)
    m1 = cv2.boxFilter(gf, -1, (15, 15), normalize=True)
    m2 = cv2.boxFilter(gf * gf, -1, (15, 15), normalize=True)
    sd = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    sil = silhouette_dist(msk)

    G = (msk & (gray >= float(p["speck_gray_min"]))
         & (gray <= float(p["speck_gray_max"]))
         & (chroma <= float(p["speck_chroma_max"]))
         & (sd <= float(p["speck_sd_max"]))
         & (sil >= 15.0)).astype(np.uint8)

    ys, _ = np.where(msk)
    yh = int(ys.min() + (ys.max() - ys.min()) * float(p["speck_band"]))

    bg_k = int(p["speck_bg_k"]) | 1
    res = (cv2.medianBlur(gray, bg_k).astype(np.int16)
           - gray.astype(np.int16)).astype(np.float32)

    ring_k = np.ones((2 * int(p["speck_ring_px"]) + 1, 2 * int(p["speck_ring_px"]) + 1), np.uint8)
    er = int(p["speck_edge_px"])
    er_k = np.ones((2 * er + 1, 2 * er + 1), np.uint8)

    nG, lbG, stG, _ = cv2.connectedComponentsWithStats(G, 8)
    cands, info = [], {"panels": 0, "max_count": 0, "max_strength": 0.0}
    for gi in range(1, nG):
        if stG[gi, cv2.CC_STAT_AREA] < int(p["speck_panel_min_px"]):
            continue
        panel = cv2.morphologyEx((lbG == gi).astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((31, 31), np.uint8))
        pin = cv2.erode(panel, er_k) > 0
        pin[yh:, :] = False
        if pin.sum() < 4000:
            continue
        info["panels"] += 1

        b = despeckle(((res > float(p["speck_contrast"])) & pin).astype(np.uint8))
        n, lb, stats, cent = cv2.connectedComponentsWithStats(b, 8)
        specks, strength = [], 0.0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < int(p["speck_min_px"]) or area > int(p["speck_max_px"]):
                continue
            core = float(np.percentile(gray[lb == i], 10))
            if core < float(p["speck_core_min"]):
                continue
            ring = ((cv2.dilate((lb == i).astype(np.uint8), ring_k) > 0)
                    & pin & (lb != i))
            if ring.sum() < 12:
                continue
            contrast = float(gray[ring].mean()) - core
            if contrast < float(p["speck_ring_contrast"]):
                continue
            specks.append({
                "cx": round(float(cent[i][0]), 1), "cy": round(float(cent[i][1]), 1),
                "area_px": area, "core_gray": round(core, 1),
                "contrast": round(contrast, 1),
            })
            strength += contrast * area

        info["max_count"] = max(info["max_count"], len(specks))
        info["max_strength"] = round(max(info["max_strength"], strength), 1)
        if len(specks) < int(p["speck_min_count"]):
            continue
        if strength < float(p["speck_min_strength"]):
            continue
        xs0 = min(s["cx"] for s in specks)
        es0 = min(s["cy"] for s in specks)
        xs1 = max(s["cx"] for s in specks)
        es1 = max(s["cy"] for s in specks)
        # 框用"光滑灰面板本身"的范围（而不是仅微点包络）：麻点稀疏时框也要
        # 圈住整片待复核区域，人工才能看出这是头部成片麻点而不是零散几粒。
        px, py, pw, ph = cv2.boundingRect(pin.astype(np.uint8))
        cands.append({
            "cx": round(sum(s["cx"] for s in specks) / len(specks), 1),
            "cy": round(sum(s["cy"] for s in specks) / len(specks), 1),
            "count": len(specks),
            "strength": round(strength, 1),
            "bbox": [int(px), int(py), int(pw), int(ph)],
            "speck_bbox": [int(xs0) - 6, int(es0) - 6,
                           int(xs1 - xs0) + 12, int(es1 - es0) + 12],
            "specks": specks,
        })
    cands.sort(key=lambda d: -d["strength"])
    return cands, info


def draw_detections(vis, cands, dark_cands, regions=(), scratch_cands=(),
                    scuff_cands=(), speck_cands=(), thick=2, font_scale=0.5):
    """在 vis 上叠加标注：污渍候选=红框(#k)，暗斑候选=蓝框(Dk)，
    细划痕候选=品红框(Sk)，低对比擦痕=黄框(Rk)，灰面斑点=绿框(Mk)，
    区域异常=橙框。"""
    for k, d in enumerate(cands, 1):
        x, y, w, h = d["bbox"]
        x0, y0 = max(0, x - 12), max(0, y - 12)
        cv2.rectangle(vis, (x0, y0), (x + w + 12, y + h + 12), (0, 0, 255), thick)
        cv2.putText(vis, f"#{k} z={d['z_max']:.1f}", (x0, max(14, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                    max(1, thick - 1), cv2.LINE_AA)
    # 暗斑(大黑点/脸眼细瑕)支路：蓝框，标注环-核对比度
    for k, d in enumerate(dark_cands, 1):
        x, y, w, h = d["bbox"]
        x0, y0 = max(0, x - 12), max(0, y - 12)
        cv2.rectangle(vis, (x0, y0), (x + w + 12, y + h + 12), (255, 0, 0), thick)
        cv2.putText(vis, f"D{k} z={d['z_max']:.1f} c={d['contrast']:.0f} h={d['ring_hue']:.0f}",
                    (x0, max(14, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0),
                    max(1, thick - 1), cv2.LINE_AA)
    # 细划痕/毛发支路：品红框，标注环-核对比度
    for k, d in enumerate(scratch_cands, 1):
        x, y, w, h = d["bbox"]
        x0, y0 = max(0, x - 12), max(0, y - 12)
        cv2.rectangle(vis, (x0, y0), (x + w + 12, y + h + 12), (255, 0, 255), thick)
        cv2.putText(vis, f"S{k} c={d['contrast']:.0f} L={d['len_mm']:.2f}mm",
                    (x0, max(14, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255),
                    max(1, thick - 1), cv2.LINE_AA)
    # 低对比擦痕/弥散污渍支路：黄框，标注 ridge 幅度与面积
    for k, d in enumerate(scuff_cands, 1):
        x, y, w, h = d["bbox"]
        x0, y0 = max(0, x - 12), max(0, y - 12)
        cv2.rectangle(vis, (x0, y0), (x + w + 12, y + h + 12), (0, 255, 255), thick)
        cv2.putText(vis, f"R{k} r={d['ridge_max']:.1f} a={d['area_px']}",
                    (x0, max(14, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255),
                    max(1, thick - 1), cv2.LINE_AA)
    # 灰面斑点支路：绿框，标注微点计数与成片强度
    for k, d in enumerate(speck_cands, 1):
        x, y, w, h = d["bbox"]
        x0, y0 = max(0, x - 12), max(0, y - 12)
        cv2.rectangle(vis, (x0, y0), (x + w + 12, y + h + 12), (0, 255, 0), thick)
        cv2.putText(vis, f"M{k} n={d['count']} s={d['strength']:.0f}",
                    (x0, max(14, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                    max(1, thick - 1), cv2.LINE_AA)
    for d in regions:
        x, y, w, h = d["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 128, 0), 1)


def detect_one(stem, cfg, sku, p, write_img=True):
    canon_dir = os.path.join(BASE, "canonical", sku)
    img_path = os.path.join(canon_dir, f"{stem}.png")
    msk_path = os.path.join(canon_dir, f"{stem}_mask.png")
    meta_path = os.path.join(canon_dir, f"{stem}.json")
    for q in (img_path, msk_path):
        if not os.path.exists(q):
            print(f"  跳过 {stem}: 缺 {os.path.basename(q)}")
            return None

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE) > 127
    mm_px = 1.0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            mm_px = float(json.load(f).get("mm_per_px_canon", 1.0))

    surf, z, info = anomaly_z(img, msk, p)

    q_t = float(p.get("thresh_quantile", 0) or 0)
    if float(p.get("z_thresh", 0) or 0) > 0:
        zt = float(p["z_thresh"])
        thr_mode = "fixed"
    elif q_t > 0:
        zt = float(np.percentile(z[surf], q_t)) if surf.any() else 0.0
        thr_mode = f"quantile{q_t}"
    else:
        zt = float(p["z_thresh"])
        thr_mode = "fixed"

    b = despeckle(((z > zt) & surf).astype(np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(b, 8)

    min_area = int(p["min_area"])
    min_mm = float(p.get("min_mm", 0) or 0)
    max_mm = float(p.get("max_mm", 0) or 0)
    cands, regions = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        eq_mm = 2.0 * np.sqrt(area / np.pi) * mm_px
        if min_mm > 0 and eq_mm < min_mm:
            continue
        item = {
            "cx": round(float(cent[i][0]), 1), "cy": round(float(cent[i][1]), 1),
            "area_px": area, "eq_mm": round(float(eq_mm), 3),
            "z_max": round(float(z[lab == i].max()), 2),
            "bbox": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                     int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
        }
        (regions if (max_mm > 0 and eq_mm > max_mm) else cands).append(item)
    cands.sort(key=lambda d: -d["z_max"])
    dark_cands, dark_info = detect_dark(img, msk, p, mm_px)
    scratch_cands, scratch_info = detect_scratch(img, msk, p, mm_px)
    scuff_cands, scuff_info = detect_scuff(img, msk, p, mm_px)
    speck_cands, speck_info = detect_speckle(img, msk, p)

    if write_img:
        vis = img.copy()
        draw_detections(vis, cands, dark_cands, regions, scratch_cands, scuff_cands,
                        speck_cands)

        scale = max(1.0, float(p["colormap_scale"]))
        heat = cv2.applyColorMap(np.clip(z / scale * 255.0, 0, 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
        heat[~surf] = (heat[~surf] * 0.2).astype(np.uint8)

        C = img.shape[0]
        side = np.hstack([cv2.resize(vis, (C, C)), cv2.resize(heat, (C, C))])
        rep_dir = os.path.join(BASE, "reports", sku)
        ov_dir = os.path.join(rep_dir, "overlay")
        os.makedirs(rep_dir, exist_ok=True)
        os.makedirs(ov_dir, exist_ok=True)
        cv2.imwrite(os.path.join(rep_dir, f"{stem}_refree.png"), side)

        # 叠加图：全分辨率原图 + 清晰标注(红=污渍, 蓝=暗斑, 品红=细划痕)，便于直接查看脸/眼瑕疵
        # 统一输出到 reports/<sku>/overlay/ 单独文件夹（每张图一个效果图）
        ov = img.copy()
        draw_detections(ov, cands, dark_cands, regions, scratch_cands, scuff_cands,
                        speck_cands, thick=3, font_scale=0.7)
        tag1 = f"{stem}  n={len(cands)}"
        tag2 = (f"dark={len(dark_cands)}  scratch={len(scratch_cands)}  "
                f"scuff={len(scuff_cands)}  speck={len(speck_cands)}")
        fs = 0.9
        w1 = cv2.getTextSize(tag1, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0]
        w2 = cv2.getTextSize(tag2, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0]
        maxw = max(w1, w2)
        if maxw > img.shape[1] - 16:  # 5 条支路的计数放到第二行，仍须保证不越界
            fs = max(0.4, fs * (img.shape[1] - 16) / maxw)
            w1 = cv2.getTextSize(tag1, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0]
            w2 = cv2.getTextSize(tag2, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0]
            maxw = max(w1, w2)
        (_, th), _ = cv2.getTextSize(tag1, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.rectangle(ov, (4, 6), (10 + maxw, 12 + 2 * th + 8), (0, 0, 0), -1)
        cv2.putText(ov, tag1, (8, 8 + th), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(ov, tag2, (8, 8 + 2 * th + 8), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(ov_dir, f"{stem}_overlay.png"), ov)
        legacy = os.path.join(rep_dir, f"{stem}_overlay.png")
        if os.path.exists(legacy):  # 清理旧版遗留在主目录的叠加图
            os.remove(legacy)

        # 脸/眼区放大叠加：以暗斑候选并集为中心自动裁剪并放大
        if dark_cands:
            xs0 = min(d["bbox"][0] for d in dark_cands)
            ys0 = min(d["bbox"][1] for d in dark_cands)
            xs1 = max(d["bbox"][0] + d["bbox"][2] for d in dark_cands)
            ys1 = max(d["bbox"][1] + d["bbox"][3] for d in dark_cands)
            mx = max(30, int(0.5 * max(xs1 - xs0, ys1 - ys0)))
            x0, y0 = max(0, xs0 - mx), max(0, ys0 - mx)
            x1, y1 = min(img.shape[1], xs1 + mx), min(img.shape[0], ys1 + mx)
            face = img[y0:y1, x0:x1].copy()
            face = cv2.resize(face, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            fz = lambda v, o: (v - o) * 2.0  # 原图坐标 -> 放大图坐标
            for k, d in enumerate(dark_cands, 1):
                x, y, w, h = d["bbox"]
                cv2.rectangle(face, (int(fz(x - 12, x0)), int(fz(y - 12, y0))),
                              (int(fz(x + w + 12, x0)), int(fz(y + h + 12, y0))),
                              (255, 0, 0), 3)
                cv2.putText(face, f"D{k} z={d['z_max']:.1f} c={d['contrast']:.0f} h={d['ring_hue']:.0f}",
                            (int(fz(x - 12, x0)), max(20, int(fz(y - 12, y0)) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(rep_dir, f"{stem}_overlay_face.png"), face)

        # 细划痕/毛发放大叠加：以划痕候选并集为中心自动裁剪并放大
        if scratch_cands:
            xs0 = min(d["bbox"][0] for d in scratch_cands)
            ys0 = min(d["bbox"][1] for d in scratch_cands)
            xs1 = max(d["bbox"][0] + d["bbox"][2] for d in scratch_cands)
            ys1 = max(d["bbox"][1] + d["bbox"][3] for d in scratch_cands)
            mx = max(40, int(1.0 * max(xs1 - xs0, ys1 - ys0)))
            x0, y0 = max(0, xs0 - mx), max(0, ys0 - mx)
            x1, y1 = min(img.shape[1], xs1 + mx), min(img.shape[0], ys1 + mx)
            crop = img[y0:y1, x0:x1].copy()
            crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            fz = lambda v, o: (v - o) * 3.0  # 原图坐标 -> 放大图坐标
            for k, d in enumerate(scratch_cands, 1):
                x, y, w, h = d["bbox"]
                cv2.rectangle(crop, (int(fz(x - 12, x0)), int(fz(y - 12, y0))),
                              (int(fz(x + w + 12, x0)), int(fz(y + h + 12, y0))),
                              (255, 0, 255), 3)
                lab = f"S{k} c={d['contrast']:.0f} L={d['len_mm']:.2f}mm"
                ly = max(28, int(fz(y - 12, y0)) - 8)
                (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                lx = min(int(fz(x - 12, x0)),
                         max(6, crop.shape[1] - tw - 6))
                cv2.rectangle(crop, (lx - 4, ly - th - 6), (lx + tw + 4, ly + 6),
                              (0, 0, 0), -1)
                cv2.putText(crop, lab, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(rep_dir, f"{stem}_overlay_scratch.png"), crop)

        # 低对比擦痕放大叠加：擦痕只有 5~10 灰阶，主图上看不清，必须放大复核
        if scuff_cands:
            xs0 = min(d["bbox"][0] for d in scuff_cands)
            ys0 = min(d["bbox"][1] for d in scuff_cands)
            xs1 = max(d["bbox"][0] + d["bbox"][2] for d in scuff_cands)
            ys1 = max(d["bbox"][1] + d["bbox"][3] for d in scuff_cands)
            mx = max(40, int(1.0 * max(xs1 - xs0, ys1 - ys0)))
            x0, y0 = max(0, xs0 - mx), max(0, ys0 - mx)
            x1, y1 = min(img.shape[1], xs1 + mx), min(img.shape[0], ys1 + mx)
            crop = img[y0:y1, x0:x1].copy()
            crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            fz = lambda v, o: (v - o) * 3.0  # 原图坐标 -> 放大图坐标
            for k, d in enumerate(scuff_cands, 1):
                x, y, w, h = d["bbox"]
                cv2.rectangle(crop, (int(fz(x - 12, x0)), int(fz(y - 12, y0))),
                              (int(fz(x + w + 12, x0)), int(fz(y + h + 12, y0))),
                              (0, 255, 255), 3)
                lab = f"R{k} r={d['ridge_max']:.1f} a={d['area_px']}px"
                ly = max(28, int(fz(y - 12, y0)) - 8)
                (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                lx = min(int(fz(x - 12, x0)), max(6, crop.shape[1] - tw - 6))
                cv2.rectangle(crop, (lx - 4, ly - th - 6), (lx + tw + 4, ly + 6),
                              (0, 0, 0), -1)
                cv2.putText(crop, lab, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(rep_dir, f"{stem}_overlay_scuff.png"), crop)

        # 灰面斑点放大叠加：微点只有 2~5px，主图上几乎看不见，必须放大并逐点圈出
        if speck_cands:
            xs0 = min(d["bbox"][0] for d in speck_cands)
            ys0 = min(d["bbox"][1] for d in speck_cands)
            xs1 = max(d["bbox"][0] + d["bbox"][2] for d in speck_cands)
            ys1 = max(d["bbox"][1] + d["bbox"][3] for d in speck_cands)
            mx = max(30, int(0.6 * max(xs1 - xs0, ys1 - ys0)))
            x0, y0 = max(0, xs0 - mx), max(0, ys0 - mx)
            x1, y1 = min(img.shape[1], xs1 + mx), min(img.shape[0], ys1 + mx)
            crop = img[y0:y1, x0:x1].copy()
            crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            fz = lambda v, o: (v - o) * 3.0  # 原图坐标 -> 放大图坐标
            for k, d in enumerate(speck_cands, 1):
                x, y, w, h = d["bbox"]
                cv2.rectangle(crop, (int(fz(x - 12, x0)), int(fz(y - 12, y0))),
                              (int(fz(x + w + 12, x0)), int(fz(y + h + 12, y0))),
                              (0, 255, 0), 3)
                for s in d["specks"]:  # 逐个微点圈出，作为"成片"的直接证据
                    cv2.circle(crop, (int(fz(s["cx"], x0)), int(fz(s["cy"], y0))),
                               12, (0, 255, 0), 2)
                lab = f"M{k} n={d['count']} s={d['strength']:.0f}"
                ly = max(28, int(fz(y - 12, y0)) - 8)
                (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                lx = min(int(fz(x - 12, x0)), max(6, crop.shape[1] - tw - 6))
                cv2.rectangle(crop, (lx - 4, ly - th - 6), (lx + tw + 4, ly + 6),
                              (0, 0, 0), -1)
                cv2.putText(crop, lab, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(rep_dir, f"{stem}_overlay_speck.png"), crop)

    out = {
        "sku": sku, "stem": stem,
        "candidate_count": len(cands),
        "dark_candidate_count": len(dark_cands),
        "scratch_candidate_count": len(scratch_cands),
        "scuff_candidate_count": len(scuff_cands),
        "speck_candidate_count": len(speck_cands),
        "z_thresh": round(zt, 2), "thr_mode": thr_mode,
        "dark_z": round(float(p.get("dark_z", 0) or 0), 2),
        "scale_mode": info.get("scale_mode", "global"),
        "dark_scale_mode": dark_info.get("scale_mode", "global"),
        "dark_bg_mode": dark_info.get("bg_mode", "median"),
        "scratch_bg_k": scratch_info.get("bg_k", None),
        "scratch_mask_close_px": scratch_info.get("mask_close_px", None),
        "scuff_flat_ratio": scuff_info.get("flat_ratio", None),
        "speck_panels": speck_info.get("panels", None),
        "speck_max_count": speck_info.get("max_count", None),
        "speck_max_strength": speck_info.get("max_strength", None),
        "noise_sigma": round(info.get("noise_sigma", 0.0), 4),
        "surface_ratio": round(float(surf.mean()), 4),
        "review": ("需复核" if (cands or dark_cands or scratch_cands or scuff_cands
                             or speck_cands)
                   else "未见表面异常"),
        "mm_per_px_canon": mm_px,
        "candidates": cands, "region_anomalies": regions,
        "dark_candidates": dark_cands,
        "scratch_candidates": scratch_cands,
        "scuff_candidates": scuff_cands,
        "speck_candidates": speck_cands,
        "params": {k: p[k] for k in sorted(DEFAULTS)},
    }
    rep_dir = os.path.join(BASE, "reports", sku)
    os.makedirs(rep_dir, exist_ok=True)
    with open(os.path.join(rep_dir, f"{stem}_refree.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stem", default=None)
    ap.add_argument("--quantile", type=float, default=None, help="覆盖 thresh_quantile(>0 时启用本图分位阈值)")
    ap.add_argument("--z-thresh", type=float, default=None, help="覆盖 z_thresh(固定阈值,跨图可比)")
    ap.add_argument("--min-area", type=int, default=None, help="覆盖 min_area")
    ap.add_argument("--dark-z", type=float, default=None, help="覆盖 dark_z(暗斑阈值)")
    ap.add_argument("--no-dark", action="store_true", help="关闭暗斑(大黑点)支路")
    ap.add_argument("--no-scratch", action="store_true", help="关闭细划痕/毛发支路")
    ap.add_argument("--no-scuff", action="store_true", help="关闭低对比擦痕/弥散污渍支路")
    ap.add_argument("--no-speck", action="store_true", help="关闭灰面斑点支路")
    ap.add_argument("--no-img", action="store_true", help="不写标注图，只出 JSON")
    args = ap.parse_args()

    cfg, _ = load_cfg(args.sku, args.config)
    p = params(cfg, {"thresh_quantile": args.quantile, "z_thresh": args.z_thresh,
                     "min_area": args.min_area, "dark_z": args.dark_z})
    if args.no_dark:
        p["dark_enable"] = False
    if args.no_scratch:
        p["scratch_enable"] = False
    if args.no_scuff:
        p["scuff_enable"] = False
    if args.no_speck:
        p["speck_enable"] = False
    if args.quantile is not None and args.quantile > 0:
        p["z_thresh"] = 0.0   # 分位模式优先

    canon_dir = os.path.join(BASE, "canonical", args.sku)
    if args.stem:
        stems = [args.stem]
    else:
        stems = [os.path.splitext(os.path.basename(q))[0]
                 for q in sorted(glob.glob(os.path.join(canon_dir, "*.png")))
                 if not q.endswith("_mask.png")]
    if not stems:
        print(f"没找到 canonical 图: {canon_dir}")
        sys.exit(0)

    scale_mode = f"local{p['local_win']}" if p["local_win"] > 1 else "global"
    thr = f"分位{p['thresh_quantile']}" if p["thresh_quantile"] > 0 else f"固定{p['z_thresh']}"
    print(f"[单图无参考] SKU={args.sku}  共 {len(stems)} 张  "
          f"(σb={p['bg_sigma']} σl={p['line_sigma']} n={p['angles']} "
          f"尺度={scale_mode} τ={thr} A_min={p['min_area']})")
    dark_tag = "关" if not p["dark_enable"] else "dark_z=%.2f" % p["dark_z"]
    print(f"            暗斑支路: {dark_tag}")
    scratch_tag = "关" if not p["scratch_enable"] else (
        "c=%.0f edge=%.2fmm" % (p["scratch_contrast"], p["scratch_edge_mm"]))
    print(f"            细划痕支路: {scratch_tag}")
    scuff_tag = "关" if not p["scuff_enable"] else (
        "r=%.1f A_min=%dpx" % (p["scuff_ridge"], p["scuff_min_px"]))
    print(f"            低对比擦痕支路: {scuff_tag}")
    speck_tag = "关" if not p["speck_enable"] else (
        "n>=%d s>=%.0f" % (p["speck_min_count"], p["speck_min_strength"]))
    print(f"            灰面斑点支路: {speck_tag}")
    results = []
    for s in stems:
        r = detect_one(s, cfg, args.sku, p, write_img=not args.no_img)
        if r:
            results.append(r)
            top = r["candidates"][0] if r["candidates"] else None
            extra = (f"  top z={top['z_max']:.1f}@({top['cx']:.0f},{top['cy']:.0f})"
                     if top else "")
            dtop = r["dark_candidates"][0] if r["dark_candidates"] else None
            dextra = (f"  dark z={dtop['z_max']:.1f}@({dtop['cx']:.0f},{dtop['cy']:.0f}) "
                      f"c={dtop['contrast']:.0f}" if dtop else "")
            stop = r["scratch_candidates"][0] if r["scratch_candidates"] else None
            sextra = (f"  scratch c={stop['contrast']:.0f} "
                      f"L={stop['len_mm']:.2f}mm@({stop['cx']:.0f},{stop['cy']:.0f})"
                      if stop else "")
            rtop = r["scuff_candidates"][0] if r["scuff_candidates"] else None
            rextra = (f"  scuff r={rtop['ridge_max']:.1f} "
                      f"a={rtop['area_px']}px@({rtop['cx']:.0f},{rtop['cy']:.0f})"
                      if rtop else "")
            mtop = r["speck_candidates"][0] if r["speck_candidates"] else None
            mextra = (f"  speck n={mtop['count']} s={mtop['strength']:.0f}"
                      f"@({mtop['cx']:.0f},{mtop['cy']:.0f})" if mtop else "")
            print(f"  {s}  {r['review']}  n={r['candidate_count']}  "
                  f"τ={r['z_thresh']:.1f}{extra}{dextra}{sextra}{rextra}{mextra}")

    rep_dir = os.path.join(BASE, "reports", args.sku)
    os.makedirs(rep_dir, exist_ok=True)
    with open(os.path.join(rep_dir, "refree_summary.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    n_dark = sum(r["dark_candidate_count"] for r in results)
    n_scratch = sum(r["scratch_candidate_count"] for r in results)
    n_scuff = sum(r["scuff_candidate_count"] for r in results)
    n_speck = sum(r["speck_candidate_count"] for r in results)
    print(f"\n[完成] -> {rep_dir}  (污渍候选 {sum(r['candidate_count'] for r in results)} 个, "
          f"暗斑候选 {n_dark} 个, 细划痕候选 {n_scratch} 个, 低对比擦痕 {n_scuff} 个, "
          f"灰面斑点 {n_speck} 个)")
    print(f"        叠加效果图 -> {os.path.join(rep_dir, 'overlay')}  "
          f"({len(results)} 张 <stem>_overlay.png)")


if __name__ == "__main__":
    main()
