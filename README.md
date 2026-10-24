# Little-figurine-rating — 小玩偶/手办瑕疵检测与判级

用「SAM 归一化 + per-SKU PatchCore + 两级 ROI + 单图无参考规则支路」把小件哑光手办的
外观瑕疵（污渍、细划痕、脸/眼暗斑、头部灰面麻点、低对比擦痕）从拍摄图里找出来，
输出**判级结果**和**可人工复核的候选框/放大叠加图**。

---

## 一、这个项目解决了什么问题

| 问题 | 传统做法为什么不行 | 本项目怎么解决 |
|---|---|---|
| 人工目检标准不一、漏检率高 | 0.3~1mm 瑕疵在 1026px 图上只有几个像素，肉眼疲劳即漏 | 归一化到固定尺度 + 两级 ROI，关键区按 0.5mm 阈值细检 |
| 跨图比对抓不到细长/低对比瑕疵 | PatchCore 依赖跨图配准，配准残差落在轮廓高光上，量级高过污渍本身；调阈值只是在「漏检污渍」和「误报轮廓」之间换 | `ref_free.py` 换范式：**单图自身低频背景差分**，不依赖参考图与跨图配准 |
| 不同瑕疵信噪比差异巨大 | 单一阈值必然顾此失彼（大黑点对比度 ~100 灰阶，擦痕只有 5~10 灰阶） | 拆成 **5 条独立支路**（污渍、暗斑、细划痕、低对比擦痕、灰面斑点），各用各的判据 |
| 灰色塑料头壳上的密集麻点全部漏检 | 灰头被 `design_gray=180` 判成设计图案、被环饱和度/环亮度门限整片拒绝 | 第 5 支路改为「光滑灰面板 + 头带 + 边界腐蚀 + 面板级微点计数」 |
| 判级结果无法追溯、 seller 不认 | 只给一个分数/热力图，看不出到底哪里有问题 | 每个候选都有 bbox、面积、对比度等字段，并输出 `overlay/` 标注图 |

> 定位：`ref_free.py` 输出的是**候选框**，供人工复核，**不参与自动判级**；
> 自动判级由 PatchCore 两级检测（`infer.py`）给出。

---

## 二、整体逻辑（数据流）

```
原始白底图
  data/<sku>/raw/  或  兄弟项目 ../Punctuation_selection_of_images/image_assist_ans/
        │
        │  normalize.py   SAM 抠图 + 统一高度/居中 + 缩放到 canonical 1026px
        ▼
canonical/<sku>/<stem>.png + _mask.png + .json（含 mm_per_px_canon）
        │
        │  roi_split.py   按配置的 key_regions 切 ROI（正常 4~6 块/张）
        ▼
rois/<sku>/<stem>/roi_xxx.png
        │
        │  train_pc.py    只用正常件 + 8× 姿态增强 → PatchCore memory bank + 阈值标定
        ▼
results/<sku>/<roi>/  模型权重 + threshold.json
        │
        │  infer.py       full(512) 粗检 + roi_xxx(512) 关键区细检，融合取 max
        ▼
reports/<sku>/<stem>.json + <stem>_defect.png + summary.json
        │                 → 判级：全新 / 微瑕 / 明显瑕疵 / 残次
        │
        │  ref_free.py    （可选）单图无参考检测，5 条支路输出候选框
        ▼
reports/<sku>/<stem>_refree.json + overlay/<stem>_overlay.png
reports/<sku>/<stem>_overlay_{face,scratch,scuff,speck}.png（按支路生成）
```

两级检测的分工：`full` 看整图的大瑕疵/结构缺陷（>2mm）；`roi_xxx` 只在关键区
（脸/眼/正面/标志）按 0.5mm 精度细检；融合规则取 `max`（简单可解释）。
非关键区的小瑕疵本来就不判不合格，所以不需要全图 0.5mm 精度。

### 五条无参考支路一览（`ref_free.py`）

| 支路 | 颜色 | 针对瑕疵 | 核心判据（默认值） | 30 张 OK 件误报 |
|---|---|---|---|---|
| 污渍/划痕主支路 | 红 `#k` | 表面污渍、划痕 | 设计图案带外 + 局部尺度归一化 `z > 6.0` | 19/30 张（毛绒细纹，已知噪声，需人工复核） |
| 暗斑支路 | 蓝 `Dk` | 大黑点、脸/眼细暗弧 | 局部 `Z_d > 8.0` + 紧贴环色相 `≥60`/饱和 `≥60` | **0** |
| 细划痕支路 | 品红 `Sk` | 白毛绒上的深色细线 | 中值背景差分 `c > 40` + 轮廓距离 `≥0.7mm` | **0** |
| 低对比擦痕支路 | 黄 `Rk` | 白塑平面淡灰擦痕 | 平坦区 ridge `>5` 灰阶 + 面积 `≥250px` | **0** |
| 灰面斑点支路 | 绿 `Mk` | 灰头成片微麻点 | 光滑灰面板内微点 `n≥4` 且 `Σ(对比度×面积)≥1200` | **0** |

---

## 三、如何操作

### 1) 环境

```bash
pip install "anomalib[cu124]" opencv-python numpy pyyaml
# 依赖：Python 3.9+、CUDA（训练 PatchCore 需要；ref_free.py 只用 CPU/OpenCV 也能跑）
```

### 2) 建配置

```bash
cd Defect_detection
cp configs/template.yaml configs/<sku>.yaml
# 按需修改：sku / object_height_mm / roi_mm / thresholds_mm / key_regions / normal_stems
```

### 3) 放图

白底图默认从兄弟项目 `../Punctuation_selection_of_images/image_assist_ans/` 自动读取；
也可以放到 `data/<sku>/raw/`（目录不存在时先 `mkdir -p`），或用 `--raw-dir` 指定。
stem 即样本 ID；OK 图与瑕疵图都可放入，但**训练白名单 `normal_stems` 必须只列正常件**，
否则瑕疵图会污染 memory bank。

### 4) 一键跑全流程

```bash
bash run_all.sh <sku>
```

或分步：

```bash
python3 normalize.py  --sku <sku>                  # Stage 1 归一化
python3 roi_split.py  --sku <sku>                  # Stage 2 ROI 切分
python3 train_pc.py   --sku <sku> --roi all        # Stage 3 训练 + 阈值标定
python3 infer.py      --sku <sku>                  # Stage 4 推理判级
python3 ref_free.py   --sku <sku>                  # Stage 4.5 候选复核（可选）
python3 evaluate.py   --sku <sku> --labels labels_<sku>.csv   # Stage 5 评估
```

> ⚠️ 用 A 款训练的模型跑 B 款的图，必须显式传 `--model-sku A`，否则找不到权重，
> 表现为「每张都判全新、n=0、max=0.0mm」。

### 5) 只做表面瑕疵复核（最常用）

```bash
cd Defect_detection
python3 ref_free.py --sku bgy_xia
python3 ref_free.py --sku bgy_xia --stem "Screenshot from 2026-10-22 11-49-47_white"
python3 ref_free.py --sku bgy_xia --no-dark --no-scratch --no-scuff --no-speck
python3 ref_free.py --sku bgy_xia --z-thresh 7.0      # 临时调污渍支路阈值
python3 ref_free.py --sku bgy_xia --dark-z 9.0        # 临时调暗斑阈值
python3 ref_free.py --sku bgy_xia --no-img            # 只出 JSON，不写图
```

### 6) 看结果

| 路径 | 内容 |
|---|---|
| `reports/<sku>/<stem>.json` / `summary.json` | PatchCore 判级结果与分数 |
| `reports/<sku>/<stem>_defect.png` | 判级标注图 |
| `reports/<sku>/<stem>_refree.json` | 五条支路的候选列表（bbox、面积、对比度等） |
| `reports/<sku>/overlay/<stem>_overlay.png` | **全分辨率叠加效果图**（红/蓝/品红/黄/绿框 + 计数标题） |
| `reports/<sku>/<stem>_overlay_face.png` | 脸/眼暗斑放大图（2×） |
| `reports/<sku>/<stem>_overlay_scratch.png` | 细划痕放大图（3×） |
| `reports/<sku>/<stem>_overlay_scuff.png` | 低对比擦痕放大图（3×） |
| `reports/<sku>/<stem>_overlay_speck.png` | 灰面斑点放大图（3×，微点逐个圈出） |

### 7) 判级规则

```
全新      ：无缺陷
微瑕      ：max_mm ≤ 0.4，或缺陷数 < 3
明显瑕疵  ：0.4 < max_mm ≤ 1.0，或缺陷数 ≥ 3
残次      ：max_mm > 1.0
```

500mm 玩偶在 1~1.5m 观察距离下，人眼实用阈值约 **0.5mm**；用 0.2mm 会严重过杀。

---

## 四、已知限制（务必如实对外说明）

- **成像决定上限**：污渍实测只有 12~16 灰阶，信号不超过 OK 件噪声底；
  亚 10 灰阶擦痕（裕度约 1.3 倍）与灰面麻点（裕度约 1.7 倍）都属**临界可辨**，
  必须人工复核。根治手段是加 10~15° 掠射光/偏振光，而不是继续压阈值。
- **污渍主支路仍有噪声**：30 张 OK 件中 19 张被标记（毛绒细纹），这是已知状况，
  不代表真有污渍。
- **几何先验与 SKU 绑定**：`speck_band=0.40`（物体顶部 40% 视为头部）按当前姿态设定，
  换姿态或换 SKU 需要复核；所有 px 参数在改 canonical 分辨率后都要按 `mm_per_px_canon` 换算。
- **ref_free 不做自动判级**：它只提供候选框，最终判级以 PatchCore 结果为准。

---

## 五、技术实现细节（完整版）

以下为各阶段的实现原理、参数表、实测边界与排障方法。

---

### 0. 开工前必须先确认的三件事

**不做这三件，后面全是垃圾进垃圾出。**

#### ① 成像规范（最高优先级）

| 项 | 要求 | 说明 |
|---|---|---|
| 匀光 | 双侧柔光箱 45° 对称 + 底部白色反光板 | 单侧光会产生大面积阴影，模型会把阴影判成瑕疵 |
| 掠射光 | 追加一组 10~15° 侧光 | 哑光表面划痕正面打光看不出，靠掠射光的细长阴影显现 |
| 光圈 | f/8~f/11 | 500mm 大件近距离拍摄景深极浅 |
| 夹具定位 | 泡沫凹槽或定位块 | 保证每次摆放位置一致，降低归一化压力 |
| 清洁 | 棉手套 + 气吹 | 油脂在哑光表面形成暗斑，是误报第一来源 |
| 背景 | 纯白无缝纸（或继续用 SAM 抠图） | — |
| 分辨率 | 玩偶占画面 ≥2500px 高 | 保证 ROI 裁出来后原始像素充足 |

**验收**：同一玩偶取下再放上去连拍 5 次，画面差异肉眼几乎看不出 → 合格。

#### ② 哑光特有的两个坑

- **批次色差**：不同批次哑光程度/色温有细微差异 → 训练集必须**跨批次采样**，每批留 2~3 张做跨批次验证
- **工艺痕迹要写进可接受清单**：合模线、顶针痕、轻微套色偏移 → 在 `configs/<sku>.yaml` 的 `accept_list` 里写明，避免反复争论

#### ③ 人工瑕疵样本（评估的唯一依据）

没有瑕疵样本 = 无法判断模型好坏。花 20 分钟：

| 档位 | 做法 | 数量 |
|---|---|---|
| <0.5mm | 0.3mm 针管笔点 | ≥10 |
| 0.5~2mm | 马克笔小点 / 贴纸剪片 | ≥10 |
| 结构缺陷 | 拆件 / 掰歪 | ≥10 |

---

### 1. 目录结构

```
Defect_detection/
├── common.py                 配置解析 + 目录自动定位（路径找不到的统一出口）
├── pc_import.py              PatchCore 兼容导入（anomalib 版本差异）
├── configs/template.yaml      配置模板（复制改名；缺失时脚本会自动生成）
├── normalize.py               Stage 1 归一化
├── roi_split.py               Stage 2 ROI 切分
├── train_pc.py                Stage 3 训练 + 阈值标定
├── infer.py                   Stage 4 推理 + 判级
├── ref_free.py                Stage 4.5 单图无参考表面异常检测（污渍/划痕 + 大黑点 + 细划痕 + 低对比擦痕 + 灰面斑点）
├── evaluate.py                Stage 5 分档评估
├── mark_exclude.py            标记设计图案屏蔽区（交互）
├── run_all.sh                 一键全流程
├── data/<sku>/raw/            原始白底图（或指向 ../image_assist_ans）
├── canonical/<sku>/           归一化输出 + meta
├── rois/<sku>/<stem>/         ROI 切块
├── datasets/<sku>/<roi>/good/ 训练图（含增强）
├── results/<sku>/<roi>/       模型权重 + threshold.json
└── reports/<sku>/             判级结果 + 标注图
    └── overlay/               叠加效果图（每张图一个 <stem>_overlay.png）
```

---

### 2. 执行步骤

```bash
cd Defect_detection

# 1) 建配置（可跳过，跑脚本时会自动从 template.yaml 生成一份）
cp configs/template.yaml configs/<你的款名>.yaml
#    修改: sku / object_height_mm / roi_mm / thresholds_mm / key_regions

# 2) 放图：白底图由兄弟项目 image_assist.py 生成，默认会去这里自动找
#    ../Punctuation_selection_of_images/image_assist_ans/
#    也可以把白底图放到 data/<sku>/raw/，或用 --raw-dir 指定
#    OK 图和瑕疵图都可以放进来（stem 即样本 ID）

# 2.1) 训练只学"正常"，务必在配置里列白名单，否则瑕疵图会被当成正常样本
#      normal_stems:
#        - ok_001_white
#        - ok_002_white
#      留空 = rois/<sku>/ 下全部当正常（有瑕疵图混入就会污染 memory bank）

# 3) 一键跑
bash run_all.sh <你的款名>

# 或分步
python3 normalize.py  --sku <sku>
python3 roi_split.py  --sku <sku>
python3 train_pc.py   --sku <sku> --roi all
python3 infer.py      --sku <sku>
python3 ref_free.py   --sku <sku>     # 可选：污渍/划痕 + 大黑点 + 细划痕 + 低对比擦痕 + 灰面斑点候选（辅助复核）
python3 evaluate.py   --sku <sku> --labels labels_<sku>.csv
```

> ⚠️ **用 A 款训练的模型跑 B 款的图，必须显式传 `--model-sku A`：**
> ```bash
> python3 infer.py --sku bgy_xia --model-sku bgy
> ```
> 模型与阈值在 `results/<model-sku>/`（默认等于 `--sku`）。漏传时 `find_ckpt()` 返回
> None → 所有 ROI 被跳过 → `map_thresh` 退回默认 0.5、`map_thresh_by_roi={}`、
> `pred_score=0.0`，表现为**每张都判「全新」、n=0、max=0.0mm**（看起来像模型失效，
> 实际是没加载到模型）。

---

### 3. 关键参数怎么定

#### 阈值（不是统一 0.2mm！）

500mm 玩偶观察距离约 1~1.5m，人眼实用阈值约 **0.5mm**。
用 0.2mm 会严重过杀——肉眼都看不见的点被判瑕疵，卖家不认。

#### ROI 边长

```
roi_mm ≤ 171 × 阈值(mm)
```
阈值 0.5mm → `roi_mm ≤ 85`（配置默认 85）

#### 归一化分辨率

`canon_target_px_per_mm = 22.8`，即 **0.044 mm/px**，与实拍原图尺度（~20.8 px/mm）相当。
取这么高的原因：深色瑕疵（合模缝发黑、细划痕）在 canonical 图里只有 **2~3px 宽**，
一旦降采样就会被平均掉——实测同一块左毛球区域，512 时代 `dark_z` 峰值 **8.78~9.47**
（>6.5 可检出），降到 512 后只剩 **3.41~5.43**（<6.5 漏检），根因是 934px 高的物体
被缩到 512（≈0.52×）。改到 1026 后 `mm_per_px_canon ≈ 0.0465`（21.5 px/mm），
与原图尺度基本一致，不再有降采样损失。

`canon = clamp(object_height_mm × canon_target_px_per_mm, canon_min_px, canon_max_px)`：
- 45mm 手办 → 45×22.8 = 1026，被 `canon_min_px=1024` 兜住 → `canon=1026`
- 500mm 玩偶 → 11400，被 `canon_max_px=3072` 截断 → 0.163mm/px（6.1 px/mm，仍满足 0.5mm 占 ≥3px）

跑完 `normalize.py` 会打印 `mm/px`，**务必确认 `mm_per_px_canon ≈ 原图尺度`**
（读 `<stem>.json` 里的 `mm_per_px_orig` / `mm_per_px_canon` 对比）；
若 `mm_per_px_canon` 明显大于 `mm_per_px_orig`（即发生过降采样），说明小瑕疵已被平均掉。

#### 姿态增强

配置里 `aug.enable: true`，默认 8 倍（±5° 旋转 / ±3% 平移 / ±5% 缩放 / ±8% 亮度）。
增强在**磁盘上生成**，规避 anomalib 版本差异。
⚠️ 增强后 patch 数量翻 8 倍，`coreset_sampling_ratio` 已相应设为 0.05；若 bank 仍过大，降到 0.01。

---

### 4. 两级检测策略

```
Level1  full(512)   整图粗检 → 大瑕疵(>2mm) / 结构缺陷
Level2  roi_xxx(512) 关键区细检 → 0.5mm 小瑕疵
        融合: 取 max(OR 逻辑，简单可解释)
```

**为什么不全覆盖**：小瑕疵在侧面/底部本来就不判不合格（分区权重），只有**关键区**需要 0.5mm 精度 → 4~6 个 ROI 足够。

`roi_split.py` 跑完会打印 ROI 数量，**应控制在 4~6 个/张**。过多说明 `roi_mm` 设小了或误用了 `grid` 模式。

---

### 5. 判级规则

```
全新      : 无缺陷
微瑕      : max_mm ≤ 0.4，或缺陷数 < 3
明显瑕疵  : 0.4 < max_mm ≤ 1.0，或缺陷数 ≥ 3
残次      : max_mm > 1.0
```

分区阈值可在配置里分别设 `face / body / hidden`。

---

### 6. 常见故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 热力图整片红 | 阴影 / 光照不均 | 回 Stage 0 做匀光 |
| 边缘一圈误报 | ROI padding 白边 / 景深不足 | 已用 valid mask 屏蔽 padding；仍有问题则收光圈或 focus stacking |
| 眼睛腮红被判瑕疵 | 未屏蔽设计图案 | 跑 `mark_exclude.py` 画屏蔽区 |
| 换一批货大面积报警 | 批次色差 | 训练集跨批次采样 |
| 0.5mm 检不出 | mm/px 不够 | 检查 `normalize.py` 打印的 mm/px，提高拍摄分辨率 |
| anomalib 报 API 错 | 版本差异 | 本项目按 anomalib **v2.x** 编写；贴出报错再调整 |

---

### 7. 最小验证路径（强烈建议先做）

```
1 个 SKU × 1~2 个 ROI × 30 张 OK 品 × 15 个人工瑕疵
```

两天能跑完，跑完你就知道：
- 0.5mm 到底检不检得出
- 误报主要来自哪（阴影？批次色差？合模线？）
- 这套方案值不值得继续投

**跑通再扩展 ROI 数和 SKU 数。**

---

### 8. 单图无参考表面异常检测（ref_free.py）

#### 为什么需要它

PatchCore 是**跨图比对**：拿同一款的一堆正常件建 memory bank，测试图和它比。
这对「细长 / 低对比」的表面瑕疵（污渍、划痕、色斑）天生失效——换一张照片，
配准残差就落在轮廓高光上，量级反而高过污渍本身，改阈值只是在「漏检污渍」和
「误报轮廓」之间来回换。

`ref_free.py` 换范式：**不依赖参考图、不依赖跨图配准**，只用单张图自身的
低频背景做差分。检出的是**候选框**，供人工复核，不参与自动判级。

#### 数学形式

记灰度图 `I`，物体掩膜 `M_obj`（canonical 的 `*_mask.png`）。

```
(1) 表面掩膜 S —— 只保留光面塑料区，排除设计图案与轮廓
    chroma = sqrt((a*-128)^2 + (b*-128)^2)          LAB 色度
    grad   = |∇I|                                   Sobel 梯度幅值
    D      = dilate( (I<g0) ∨ (chroma>c0), k_d )     设计图案带
    S      = erode(M_obj, k_e) ∧ ¬D ∧ (chroma≤c0) ∧ (grad<g0)

(2) 低频背景差分（正 = 比局部背景暗）
    R(p) = G_σb * I (p) − I(p)

(3) 定向线状匹配滤波（归一化卷积，沿瑕疵走向积分）
    对 θ ∈ {0, 180/n, …, 180−180/n}：
        num_θ = G_σl ⊛ (R · S)        沿 θ 方向的一维高斯积分
        den_θ = G_σl ⊛ S
    R̂(p) = max_θ  num_θ(p) / max(den_θ(p), ε)
    den < s0 处置 0 —— 窗口内表面占比过低时不做外推，防除小数放大噪声。

(4) 全局噪声尺度（不用 MAD！）
    原图平坦区被压缩量化，实测 59% 的表面像素 R 恰好为 0，MAD 退化为 0。
    σ_global = percentile(|R̂|_S, q_n) / 1.2816   （1.2816 = 标准正态 90% 分位）

(5) 局部对比度归一化（关键：全局 σ 会让 z 跨图不可比）
    实测同一参数下 OK 件 top z 可达 55.4 / 34.5 / 30.7，而污渍只有 8.89——
    因为 σ_global 受该图局部对比度/清晰度调制，模糊的图整片 z 抬高。
    改用逐点局部尺度（滑窗 std，图内相对量）：
        μ(p)   = boxFilter(R̂, w)(p)
        s(p)   = sqrt( max(boxFilter(R̂², w)(p) − μ(p)², 0) )
        den(p) = max( s(p), f·σ_global )      下限防除零、防放大平坦区
        Z(p)   = R̂(p) / den(p) ,  p ∈ S        （默认 w=81, f=0.5）
    local_win=0 时退回全局：den(p) = σ_global。

(6) 阈值（默认跨图固定阈值，不用本图分位）
    若 z_thresh > 0：        τ = z_thresh                  （跨图可比，默认 6.0）
    若 thresh_quantile > 0：τ = percentile(Z_S, q_t)      （随图自适应，会漏掉低对比件）
    候选 = (Z > τ) ∧ S 的连通域，面积 ≥ A_min 者，按 Z_max 降序编号
```

上面 (1)~(6) 是为「细长 / 低对比」瑕疵设计的：污渍对比度只有 12~16 灰阶，
信号贴着 OK 件噪声底，只能靠定向线状积分把能量攒起来。**大黑点不是这一类**——
它对比度约 77~111 灰阶，信噪比高一个量级，走另一条支路，三点关键区别：

```
(7) 暗斑（大黑点）支路 —— 并行，独立于 (1)~(6)
    a. 用物体掩膜 M_obj（erode(M_obj, k_e)），不用表面掩膜 S。
       S 会排除 gray<g0=180 的像素，而黑点本身灰度就低（~158），
       用 S 会把黑点直接排除掉。
    b. 不做定向线状滤波（那是给细长划痕攒能量的），直接用残差 R。
       R(p) = G_σb * gray(p) − gray(p)          （同一套低频背景）
    c. 同一套局部对比度归一化：
       Z_d(p) = R(p) / max( s(p), f·σ_global )   p ∈ M_obj
       候选 = (Z_d > dark_z) ∧ M_obj 的连通域
    d. 再叠加多重稳健过滤（排除深色设计图案、柔和阴影、灰白毛绒）：
       面积     dark_min_px ≤ area ≤ dark_max_px   （大片深色花纹会被上限挡掉）
       峰值     Z_d,max ≥ dark_z_peak_min          （滤掉低对比噪声块）
       紧贴环   ring = dilate(连通域, rr) − 连通域 ∩ M_obj
       ⚠️ rr 由 dark_ring_mm 换算，**必须取窄环(0.2mm≈4px)**：宽环(1.0mm≈22px)
       会探入相邻深蓝虹膜，使眼区细暗弧的环灰度降到 116.7、对比度降到 22.9，
       反被环门限整条拒绝。窄环下 4 个目标瑕疵的环灰度 145~193、对比度 37.7~75.2。
       环亮度   mean(gray[ring]) ≥ dark_ring_gray        （周围是亮塑料）
       绝对暗度 dark_core = percentile(gray[连通域], 10)
                mean(gray[ring]) − dark_core ≥ dark_contrast
       核暗度   dark_core ≥ dark_core_gray_min       （低于此=纯黑设计区 鼻/虹膜）
       环色相   median(H[ring]) ≥ dark_ring_hue_min  ← **核心判别量**，见下
       环饱和度 median(S[ring]) ≥ dark_ring_sat_min  （排除灰白毛绒/高光）
       环散乱度 std(gray[ring]) ≤ dark_ring_std_max  （排除跨边界混合环）
                circ_std(H[ring]) ≤ dark_ring_hue_std_max（排除色相散乱环）
    e. 按 Z_d,max 降序编号，输出字段含 ring_gray / core_gray / contrast /
       ring_hue / ring_hue_std / ring_std

    为什么「环色相中位数」是唯一完美判别量：玩偶的彩色塑料（teal 眼睑 H≈95、
    粉色腮红 H≈176）与 OK 件上被误命中的米色/棕色毛绒（H≈14~20）完全分开；
    而灰度类判别量（ring 饱和度的低分位、环灰度高分位）在目标与误报间是重叠的。
    其中 `std(gray[ring]) ≤ dark_ring_std_max` 是**必需**的：去掉它会在 bgy_016 的
    (320,401) 合模线上误报（a=15 z=14.2 core=139.0 hue=103.0 hstd=1.35，
    其余门限全部通过，只有环灰度 std=40.1 超标）。
```

注意 `dark_core` 取**连通域自身**的 10% 分位（不是 centroid 小框），
`ring` 取**连通域膨胀环**在 `M_obj` 内的均值（不是 centroid 邻域）——
用 centroid 小框时阈值放宽后连通域变大、centroid 偏向亮边缘，
会让 `ring − dark_core` 被误拒（表现为「阈值越低候选越少」的非单调）。

```
(8) 连通域去噪：为什么不用 3×3 形态学开运算
    二值图上去噪的顺手写法是 b = MORPH_OPEN(b, 3×3 ELLIPSE)，
    但它会**整片抹掉宽 2~3px 的细线**——而本项目的瑕疵恰恰就是这种细线。
    实测 canonical 1026 下，左毛球与身体交界的深色缝线 z=8.20 / area=11px，
    过完开运算后候选数从 6 掉到 1，用户指的那处瑕疵直接漏检。
    改用 despeckle(b)：只保留「自身 + ≥2 个 8 邻域像素」的点，
    即删孤立单点与线头、保留细线主体：
        nb = filter2D(b, ones(3,3))          8 邻域（含自身）计数
        保留  ⟺  b>0 ∧ nb≥3
    同参数对照（30 张 bgy OK + 6 张 bgy_xia 瑕，dark_z=6.5）：
        开运算  0 误报 / 1 命中（漏 5）
        不去噪  1 误报 / 6 命中
        despeckle 0 误报 / 6 命中   ← 采用
    完整扫描见「实测边界」节的对照表。
```

```
(9) 细划痕 / 毛发支路 —— 并行，独立于 (1)~(8)
    衣领白毛绒上被划出的深色细线，灰度 ~128，与 (1)~(6) 和 (7) 都不同：
    a. **不走表面掩膜 S**。划痕灰度 ~128 < design_gray(180)，会被 (1) 的
       设计图案带整片排除（划痕处 surf 比例 = 0、z = 0），只能另起支路。
    b. 用**中值背景**差分（核 scratch_bg_k=31）：
       res(p) = medianBlur(gray, 31)(p) − gray(p)
       划痕只有 2~4px 宽，中值背景不会被它拉低，故 res 在划痕处显著为正。
    c. 物体掩膜先做**闭运算补缝**（scratch_mask_close=15）：canonical 掩膜在
       白毛绒区常出现小缺口，划痕恰好落在缺口上（实测 m=0 于 (714,390)）。
    候选 = despeckle( (res > scratch_contrast) ∧ close(M_obj, kc) ) 的连通域，
    再叠加多重稳健过滤（全部满足才命中）：
       面积     scratch_min_px(20) ≤ area ≤ scratch_max_px(600)  （排除单点/成片暗区）
       长宽比   max(w,h)/min(w,h) ≥ scratch_aspect_min(2.0)      （细长线状）
       紧贴环   ring = dilate(连通域, rr=8px) − 连通域 ∩ close(M_obj, kc)
       环亮度   mean(gray[ring]) ≥ scratch_ring_gray(215)        （周围是亮面：白毛绒/白塑料）
       环饱和度 mean(S[ring]) ≤ scratch_ring_sat(30)             （周围是中性色：非彩色）
       核灰度   core = percentile(gray[连通域], 10)
                scratch_core_gray_min(80) ≤ core ≤ scratch_core_gray_max(210)
                （排除纯黑设计图案：眼/鼻 core≈15~34；也排除亮纹理）
       ★轮廓距离 silhouette_dist()：物体轮廓 = 掩膜**最大连通域填洞后**的边界，
       候选须满足 mean(dist[连通域]) ≥ scratch_edge_mm(0.7mm)。这是**核心判别量**：
       真实划痕离轮廓 140.8px，而全部误报（头部顶边/身体右侧边/身体底部/脸颊
       边缘/彩色区边界的轮廓暗边与剪影阴影）都紧贴轮廓（≤4.2px）。仅此一条
       即可把 23 个误报全部清除。
    按环-核对比度降序编号，输出字段含 contrast / len_mm / ring_gray / core_gray /
    ring_sat / edge_mm。
```

```
(10) 低对比擦痕 / 弥散污渍支路 —— 并行，独立于 (1)~(9)
    白色塑料大平面上的淡灰擦痕（宽 30~60px、对比度仅 5~10 灰阶）。
    以 11-47-25 下巴擦痕为例，前三条支路为何全漏：
      - 污渍支路：ridge 原始响应 7.91 灰阶，但 sd_local≈1.4、σ_global≈0.75，
        归一化后 z 仅 4.91 < z_thresh(6.0)，整片漏检；
      - 暗斑支路：擦痕灰度 ~213 > design_gray(180)，不是"大黑点"；
      - 细划痕支路：擦痕宽 60px、对比度 10 灰阶 << scratch_contrast(40)。
    本支路用「平坦白塑区 + 原始幅度 + 成片面积」三重判据，**不做局部归一化**
    （局部尺度会被擦痕自己抬高，正是前三条失效的根因）：
      a. 平坦白塑区 F：
           F = S ∧ (sd15 < scuff_sd_max) ∧ (I > scuff_gray_min)
               ∧ (chroma_σ3 ≤ scuff_chroma_max) ∧ (dist_design > scuff_design_px)
               ∧ (dist_silhouette > scuff_silh_min)
         sd15 = 15×15 滑窗 std（纹理量）。chroma 收紧到 8（全局是 25）是关键：
         实测白塑 chroma≈5.4，OK 件误报的青色/米色高光 chroma 达 12.6~14.0，
         收紧后 bgy_018/bgy_022 的高光整片被排除；dist_silhouette 排除肢体交界处
         的结构阴影阶跃（bgy_010 的最大误报就在白色躯体与米色阴影的交界上）。
      b. b = (G_1.6 * ridge > scuff_ridge) ∧ F 的连通域
         轻度平滑(σ=1.6)抹掉单像素尖峰，保留成片擦痕。
      c. 面积 ∈ [scuff_min_px, scuff_max_px] —— **核心判别量**：擦痕成片，
         OK 件的高光/纹理尖峰面积小。
    实测（30 张 OK + 6 张瑕，r=5.0, A_min=250px, chroma≤8）：
      OK 件 **0/30 误报、0 候选**；11-47-25 下巴擦痕命中
        R1 r=6.7 a=299px bbox=[476,456,58,21]（与人工框选位置一致）。
    ⚠️ 裕度仅约 1.3 倍（瑕 299px vs OK 最大 227px），属**临界可辨**，
    取 A_min=200 会变成 2/30 误报。本支路输出必须人工复核；
    根治这类亚 10 灰阶擦痕要靠掠射光/偏振光成像，而不是继续调阈值。
```

```
(11) 灰面密集微斑点支路 —— 并行，独立于 (1)~(10)
    灰色塑料头壳上成片的深色小麻点（2~5px、对比度 8~26 灰阶，模具麻点/污点）。
    以 11-49-47 头部为例，前四条支路为何全漏——根因是"灰面"被判成背景：
      - 污渍支路：design_gray=180 把灰头（中位 182、下四分位 163）整片判为设计图案；
      - 暗斑支路：紧贴环要求 ring_sat>=60 / ring_hue>=60，灰面环 sat≈29 → 全被拒；
      - 细划痕支路：要求 ring_gray>=215，灰头≈180 → 被拒；
      - 擦痕支路：要求 gray>195 且面积>=250px，麻点只有 2~120px → 被拒。
    本支路**不逐点判定，而是面板级计数**（单点信噪比太低，逐点必误报）：
      a. 光滑灰面板 G：
           G = M_obj ∧ gray∈[speck_gray_min, speck_gray_max] ∧ chroma ≤ speck_chroma_max
               ∧ sd15 ≤ speck_sd_max ∧ dist_silhouette ≥ 15px
         取面积 ≥ speck_panel_min_px(40000) 的连通域，闭运算 31 补缝；
      b. 只保留物体 bbox 顶部 speck_band(40%) 内的部分（本 SKU 的头部姿态先验），
         并腐蚀 speck_edge_px(18) —— **必需**：角/头接缝/剪影暗边在 OK 件上同样存在，
         实测不腐蚀时 bgy_022/bgy_030/bgy_025 会出现对比度 40~70 的强响应；
      c. 微点：res = median15(gray) − gray > speck_contrast(10)，
         面积 ∈ [speck_min_px(2), speck_max_px(120)]，
         核灰度 ≥ speck_core_min(55)，环(9px)-核对比 ≥ speck_ring_contrast(8)；
      d. 面板内 微点数 ≥ speck_min_count(4) 且 Σ(对比度×面积) ≥ speck_min_strength(1200)
         → 输出一个"成片斑点"候选（绿框圈住整片面板，微点逐个圈出）。
    实测（30 张 OK + 6 张瑕）：11-49-47 头部 n=5、强度 1929；
      OK 件最大 n=3（bgy_006）、0 候选通过 → **0/30 误报**。
    ⚠️ 裕度仅约 1.7 倍（5 vs 3），属**临界可辨**；speck_min_count 已是 OK 边界，
    再放宽即误报。这属材质级低对比瑕疵，根治仍建议掠射光成像。
```


#### 参数（`configs/<sku>.yaml` 的 `ref_free:` 段）

| 参数 | 默认 | 含义 |
|---|---|---|
| `bg_sigma` σb | 6.0 | 低频背景高斯尺度(px)。越大越只看大尺度明暗 |
| `line_sigma` σl | 12.0 | 定向积分核长度尺度(px)，≈ 期望瑕疵长度量级 |
| `angles` n | 12 | 方向数（180° 均分） |
| `min_support` s0 | 0.3 | 窗口内表面占比下限 |
| `chroma_max` c0 | 25.0 | 表面色度上限(LAB)，超过视为彩色图案 |
| `design_gray` g0 | 180 | 灰阶下限，低于视为深色图案 |
| `design_dilate` k_d | 9 | 设计图案掩膜膨胀(px) |
| `grad_max` | 40.0 | 表面梯度上限(Sobel)，超过视为轮廓/花纹边 |
| `obj_erode` k_e | 9 | 物体掩膜内缩(px)，去掉轮廓带 |
| `noise_percentile` q_n | 90.0 | 全局噪声尺度分位（也是局部尺度下限基准） |
| `local_win` w | 81 | 局部尺度窗口(px)；0=退回全局 σ（此时 z 跨图不可比） |
| `local_floor` f | 0.5 | 局部尺度下限 = f·σ_global，防除零/防放大平坦区 |
| `z_thresh` | 6.0 | 固定阈值（局部归一化后跨图可比）；≤0 则改用 `thresh_quantile` |
| `thresh_quantile` q_t | 0.0 | >0 时改用本图分位阈值（随图自适应，实测误报 26/30 张，不可用） |
| `min_area` A_min | 20 | 最小连通面积(px)；去噪改用 despeckle 后由 8 提到 20，抑制毛绒细纹噪声 |
| `min_mm` / `max_mm` | 0 / 15.0 | 等效直径(mm)下限 / 上限 |
| `colormap_scale` | 12.0 | 热力图色标满量程 z |
| `dark_enable` | true | 暗斑支路开关（`--no-dark` 置 false） |
| `dark_z` | 8.0 | 暗斑 z 阈值（局部归一化后跨图可比） |
| `dark_min_px` | 1 | 暗斑最小连通面积(px)。细暗弧经 despeckle 后可只剩 1px，真正的判别由环色相门限承担 |
| `dark_max_px` | 400 | 暗斑最大连通面积(px)，超过视为大片深色设计图案 |
| `dark_z_peak_min` | 12.0 | 连通域内 Z_d,max 下限；0=不启用 |
| `dark_ring_mm` | 0.2 | **紧贴环**宽度(mm)。宽环(1.0)会探入相邻深蓝虹膜，反被环门限拒绝 |
| `dark_ring_gray` | 140.0 | 紧贴环灰度下限，低于视为深色设计图案(眼睛/花纹) |
| `dark_contrast` | 30.0 | 环-核灰度差下限(绝对暗度)，防把柔和阴影当黑点 |
| `dark_core_gray_min` | 60.0 | 核灰度下限；低于此视为纯黑设计区(鼻/虹膜 core≈15~34) |
| `dark_ring_hue_min` | 60.0 | 紧贴环色相中位数下限(H∈[0,180])，**核心判别量** |
| `dark_ring_sat_min` | 60.0 | 紧贴环饱和度中位数下限 |
| `dark_ring_std_max` | 25.0 | 紧贴环灰度标准差上限；**必需**，否则 bgy_016 合模线误报 |
| `dark_ring_hue_std_max` | 2.5 | 紧贴环环形色相标准差上限(度) |
| `scratch_enable` | true | 细划痕/毛发支路开关（`--no-scratch` 置 false） |
| `scratch_bg_k` | 31 | 中值背景核(px，奇数)；划痕仅 2~4px 宽，中值背景不糊掉它 |
| `scratch_contrast` | 40.0 | 暗于中值背景的灰阶下限 (bg − gray > c) |
| `scratch_mask_close` | 15 | 物体掩膜闭运算核(px)；划痕常落在掩膜缝隙处，需先补缝 |
| `scratch_min_px` / `scratch_max_px` | 20 / 600 | 连通面积下限 / 上限(px) |
| `scratch_aspect_min` | 2.0 | 长宽比下限：细长线状 |
| `scratch_ring_px` | 8 | 紧贴环宽度(px) |
| `scratch_ring_gray` | 215.0 | 紧贴环灰度下限：周围须是亮面(白毛绒/白塑料) |
| `scratch_ring_sat` | 30.0 | 紧贴环饱和度上限：周围须是中性(非彩色) |
| `scratch_core_gray_min` / `_max` | 80.0 / 210.0 | 核灰度(10 分位)下限 / 上限：排除纯黑设计图案与亮纹理 |
| `scratch_edge_mm` | 0.7 | 到物体轮廓(最大连通域填洞后)的最小距离(mm)，**核心判别量** |
| `scuff_enable` | true | 低对比擦痕/弥散污渍支路开关（`--no-scuff` 置 false） |
| `scuff_ridge` | 5.0 | 平坦区内 ridge 原始幅度下限(灰阶)，不做局部归一化 |
| `scuff_sd_max` | 8.0 | 15×15 滑窗 std 上限（平坦纹理量） |
| `scuff_chroma_max` | 8.0 | 局部色度上限(LAB，比全局 25 更严)；排除青/米色高光 |
| `scuff_gray_min` | 195.0 | 灰阶下限，只在白塑区找 |
| `scuff_design_px` | 20.0 | 到设计图案带的最小距离(px) |
| `scuff_silh_min` | 50.0 | 到物体轮廓的最小距离(px)，排除肢体交界阴影阶跃 |
| `scuff_smooth_sigma` | 1.6 | ridge 平滑尺度(px)，抹单像素尖峰、留成片擦痕 |
| `scuff_min_px` / `scuff_max_px` | 250 / 3000 | 连通面积下限 / 上限(px)，**核心判别量**；下限取 200 会变 2/30 误报 |
| `speck_enable` | true | 灰面斑点支路开关（`--no-speck` 置 false） |
| `speck_panel_min_px` | 40000 | 参与统计的灰面板最小面积(px)，排除小块高光/耳朵 |
| `speck_gray_min` / `_max` | 140.0 / 210.0 | 灰面板灰度区间；>210 是白色件，交给擦痕/污渍支路 |
| `speck_chroma_max` | 18.0 | 面板色度上限(LAB)，排除彩色设计 |
| `speck_sd_max` | 16.0 | 面板 15×15 局部纹理 sd 上限 |
| `speck_band` | 0.40 | 只统计物体 bbox 顶部这一比例（头部姿态先验，换姿态需复核） |
| `speck_edge_px` | 18 | 面板腐蚀宽度(px)，**必需**：排除角/接缝/剪影暗边 |
| `speck_bg_k` | 15 | 中值背景核(px，奇数)，须远大于斑点直径 |
| `speck_contrast` | 10.0 | 暗于中值背景的灰阶下限 |
| `speck_min_px` / `speck_max_px` | 2 / 120 | 单个微点面积区间(px)，超过不算"微点" |
| `speck_core_min` | 55.0 | 微点核灰度下限，排除纯黑设计 |
| `speck_ring_contrast` | 8.0 | 环-核灰度差下限 |
| `speck_ring_px` | 9 | 紧贴环宽度(px) |
| `speck_min_count` | 4 | **核心判别量**：面板内微点数下限（OK 件最大 3） |
| `speck_min_strength` | 1200.0 | 面板内 Σ(对比度×面积) 下限，滤掉弱纹理点 |

> ⚠️ **这些参数都是按 px 计的，分辨率改了要跟着改。**
> canonical 从 512 提到 1026 后（像素尺寸翻倍），`bg_sigma` / `line_sigma` /
> `design_dilate` / `obj_erode` / `local_win` 现在只覆盖原来**一半的物理尺度**
> （`local_win=81` 从 7.6mm 变成 3.8mm），`dark_min_px` / `dark_max_px` 对应的物理
> 尺寸也减半（`dark_max_px=400` 现在只允许 ≤1.05mm 的暗斑）。
> 换 SKU 或改 `canon_target_px_per_mm` 时，这些值需按新的 `mm_per_px_canon` 重新换算。

#### 输出与调参

```bash
python3 ref_free.py --sku bgy_xia
python3 ref_free.py --sku bgy_xia --stem "xxx_white"
python3 ref_free.py --sku bgy_xia --z-thresh 7.0    # 临时收/放固定阈值
python3 ref_free.py --sku bgy_xia --quantile 99.0   # 改用本图分位阈值（不推荐）
python3 ref_free.py --sku bgy_xia --no-img          # 只出 JSON
python3 ref_free.py --sku bgy_xia --dark-z 6.0      # 临时收/放暗斑阈值
python3 ref_free.py --sku bgy_xia --no-dark         # 关掉暗斑支路
python3 ref_free.py --sku bgy_xia --no-scratch      # 关掉细划痕/毛发支路
python3 ref_free.py --sku bgy_xia --no-scuff        # 关掉低对比擦痕/弥散污渍支路
python3 ref_free.py --sku bgy_xia --no-speck        # 关掉灰面斑点支路
```

- `reports/<sku>/<stem>_refree.png`：左=候选框（红=污渍/划痕，蓝=暗斑，品红=细划痕，黄=低对比擦痕，绿=灰面斑点），右=z 热力
- `reports/<sku>/overlay/<stem>_overlay.png`：**叠加效果图**——全分辨率原图上直接叠加清晰标注
  （红=污渍/划痕，蓝=暗斑，品红=细划痕，黄=低对比擦痕，绿=灰面斑点，
  含标题条 `n=… dark=… scratch=… scuff=… speck=…`）。
  **单独放在 `overlay/` 子文件夹里，每张图一个效果图**，便于直接翻阅。
- `reports/<sku>/<stem>_overlay_face.png`：**脸/眼放大叠加图**——以暗斑候选并集为中心
  自动裁剪并 2× 放大，仅在有暗斑候选时生成（脸/眼细瑕复核用）
- `reports/<sku>/<stem>_overlay_scratch.png`：**细划痕放大叠加图**——以划痕候选并集为中心
  自动裁剪并 3× 放大，仅在有划痕候选时生成（衣领白毛绒划痕复核用）
- `reports/<sku>/<stem>_overlay_scuff.png`：**低对比擦痕放大叠加图**——以擦痕候选并集为中心
  自动裁剪并 3× 放大，仅在有擦痕候选时生成（白塑平面淡灰擦痕复核用）
- `reports/<sku>/<stem>_overlay_speck.png`：**灰面斑点放大叠加图**——以斑点面板为中心
  自动裁剪并 3× 放大，绿框圈出整片面板、每个微点用小绿圈标出，仅在有斑点候选时生成
- `reports/<sku>/<stem>_refree.json`：候选列表（污渍含 `z_max` / `eq_mm` / `bbox`；
  暗斑另含 `ring_gray` / `core_gray` / `contrast` / `ring_hue` / `ring_hue_std` / `ring_std`；
  细划痕另含 `contrast` / `len_mm` / `ring_gray` / `core_gray` / `ring_sat` / `edge_mm`；
  低对比擦痕另含 `ridge_max` / `area_px` / `eq_mm` / `bbox`；
  灰面斑点另含 `count` / `strength` / `bbox` / `speck_bbox` / `specks`）
- `reports/<sku>/refree_summary.json`：汇总（含 `dark_candidate_count` / `scratch_candidate_count` /
  `scuff_candidate_count` / `scuff_flat_ratio` / `speck_candidate_count` / `speck_panels`）

**调参方向**：候选太多 → 抬 `z_thresh` 或加 `min_area`；
轮廓带误报 → 加大 `obj_erode` / 降 `grad_max`；长划痕抓不到 → 加大 `line_sigma`；
暗斑候选太多 → 抬 `dark_z_peak_min` 或抬 `dark_contrast`；深色设计图案被当黑点 →
抬 `dark_ring_hue_min`（核心判别量）或抬 `dark_ring_gray`；合模线误报 → 抬 `dark_ring_std_max`；
细划痕误报（多在轮廓边）→ 加大 `scratch_edge_mm`；彩色区被当划痕 → 降 `scratch_ring_sat`；
划痕抓不到 → 降 `scratch_contrast` 或加大 `scratch_mask_close`；
低对比擦痕误报（多为彩色高光/肢体交界）→ 降 `scuff_chroma_max` 或加大 `scuff_min_px`；
擦痕抓不到 → 略降 `scuff_ridge`，但**不建议**再降 `scuff_min_px`（200 时已 2/30 误报）；
灰面斑点误报 → 抬 `speck_min_count` / `speck_min_strength` 或加大 `speck_edge_px`；
灰头麻点抓不到 → 先确认 `speck_band`/`speck_gray_*` 是否覆盖该姿态的头壳，再略降 `speck_contrast`。

#### 实测边界（当前默认参数）

> 分辨率：**canonical 1026**（45mm × 22.8 px/mm，`mm_per_px_canon ≈ 0.0465`）。
> 以下均为 canon=1026 下重跑的结果（512 时代的数据见 git 历史）。

**污渍支路**（`z_thresh=6.0`，细长/低对比瑕疵）：

| 集合 | 被标记张数 | 候选数 |
|---|---|---|
| bgy 30 张（已知 OK 件） | 19 / 30 | 30 |
| bgy_xia 6 张 | 5 / 6 | 14 |

已知污渍（对比度仅 12~16 灰阶）在跨图比对下漏检、排 85 百分位；
本模块把它排到**本图候选 #1**（z=8.46，bbox 14×6 px）。

对比：旧版「本图分位 q_t=99.5」在同样 30 张 OK 件上 26/30 张被标记、63 个候选，
误报不可用；换成「局部归一化 + 固定 z=6」并保留开运算时是 8/30 张、10 个候选
（非 0，见模块头 (8)）；改用 `despeckle` 保住 2~3px 细线后为 19/30 张、30 个候选。

**暗斑支路**（`dark_z=8.0`，紧贴环 0.2mm + 环色相门限，大黑点/脸眼细瑕）：

| 集合 | 暗斑候选数 |
|---|---|
| bgy 30 张（已知 OK 件） | **0**（误报 0） |
| bgy_xia 6 张 | **4**，全部落在 `11-38-53` 的脸/眼区域 |

`11-38-53` 的 4 个命中（即用户指认的脸/眼瑕疵，蓝框）：

| 位置 | area | z_max | core | ring | contrast | ring_hue |
|---|---|---|---|---|---|---|
| 右眼上细暗弧 (627,511) | 9 | 14.84 | 93.8 | 145.1 | 51.3 | 95.0 |
| 右眼下细暗弧 (708,534) | 58 | 33.75 | 75.1 | 150.3 | 75.2 | 95.0 |
| 左眼单点 (380,560) | 1 | 13.04 | 95.0 | 147.2 | 52.2 | 95.0 |
| 右腮红黑点 (743,582) | 10 | 20.93 | 155.1 | 192.8 | 37.7 | 176.0 |

`dark_z` 稳定性（30 张 OK + 6 张瑕，其余参数同默认）：

| `dark_z` | 6.0 | 7.0 | **8.0（默认）** | 9.0 | 10.0 |
|---|---|---|---|---|---|
| OK 误报 | 0 | 0 | **0** | 0 | 0 |
| 瑕命中 | 4 | 4 | **4** | 4 | 4 |

改造前 `dark_z` 必须在 8.0 附近取（dz=9 时细暗弧面积塌到 4px < `dark_min_px=8` 而整条
漏检），现在因为真正的判别量换成了紧贴环色相、面积门限降为 1 兜底，阈值在 6.0~10.0
整段都稳定。

阈值裕度：把 `dark_z` 一路降到 1.0、并关掉 `dark_z_peak_min`，30 张 OK 件**仍然 0 个**
候选通过环门限；同一极端设置下 bgy_xia 依然只有那 4 个真瑕疵。也就是说环色相门限
本身就把 OK 件全部挡住了，`dark_z` 只影响连通域怎么切块，不再是漏检/误报的来源。
注意 `dark_ring_mm` 换算出的环半径须用真实尺度（本流程 `mm_per_px_canon≈0.0465`，
0.2mm → rr=4px）；若上传图按 `mm=1.0` 量级缩放，`rr` 会被 `max(2, …)` 兜底到 2px，
环被暗斑自身污染、`contrast` / `ring_hue` 计算不可靠。

**细划痕支路**（`scratch_contrast=40`，中值背景差分 + 轮廓距离门限，衣领白毛绒细线）：

| 集合 | 细划痕候选数 |
|---|---|
| bgy 30 张（已知 OK 件） | **0**（误报 0） |
| bgy_xia 6 张 | **1**，落在 `11-38-53` 的衣领白毛绒区 |

`11-38-53` 的 1 个命中（用户指认的衣领细划痕，品红框）：

| 位置 | area | w×h | aspect | ring | ring_sat | core | contrast | edge |
|---|---|---|---|---|---|---|---|---|
| 衣领白毛绒 (390,714) | 83 | 22×7 | 3.14 | 244.8 | 11.5 | 128.8 | 116.0 | 6.54mm |

参数鲁棒性（30 张 OK + 6 张瑕）：`scratch_mask_close`=0/9/15/21、`scratch_contrast`=35/40/45/50、
`scratch_ring_px`=5/8/12 均稳定 **1 命中 0 误报**；`scratch_contrast=30` 会多出 1 个
`edge=0.98mm` 的边缘候选（仍被 `scratch_edge_mm=0.7` 挡住，说明轮廓距离门限有裕度）。

但**它做不到全自动判级**：实测裕度 = 污渍 z / OK 最大 p99.99 ≈ 0.50~0.85（<1），
即污渍信号不超过 OK 件噪声底。真正的决定性升级是**成像**——按第 0 节加一组
10~15° 掠射光，同一污渍对比度 12~16 → 40+ 灰阶，z 8.46 → 20+，误报可清零。

**低对比擦痕支路**（`scuff_ridge=5.0` / `scuff_min_px=250`，白塑平面淡灰擦痕）：

| 集合 | 低对比擦痕候选数 |
|---|---|
| bgy 30 张（已知 OK 件） | **0**（0 候选，零误报） |
| bgy_xia 6 张 | **1**，落在 `11-47-25` 的嘴周/下巴 |

`11-47-25` 的 1 个命中（用户指认的下巴擦痕，黄框）：

| 位置 | area | bbox | ridge_max | eq_mm |
|---|---|---|---|---|
| 下巴擦痕 (502,465) | 299 | 58×21 | 6.71 | 0.907 |

⚠️ **裕度只有约 1.3 倍**：同一判据下 OK 件最大平坦区响应为 `bgy_003` 227px、
`bgy_018` 202px，瑕图 299px；`scuff_min_px=200` 时为 2/30 误报。这是**临界可辨**，
换样件或换打光都可能失效，输出必须人工复核。根治这类亚 10 灰阶擦痕要靠
掠射光/偏振光成像（同一擦痕对比度可从 ~8 提到 40+ 灰阶），而不是继续压阈值。

**灰面斑点支路**（`speck_min_count=4` / `speck_min_strength=1200`，灰头密集成片麻点）：

| 集合 | 灰面斑点候选数 | 面板内最大微点数 |
|---|---|---|
| bgy 30 张（已知 OK 件） | **0**（零误报） | 3（`bgy_006`） |
| bgy_xia 6 张 | **1**，落在 `11-49-47` 的头部 |

`11-49-47` 的 1 个命中（用户指认的头部麻点，绿框）：

| 位置 | 微点数 | 强度 Σ(对比度×面积) | 面板框 |
|---|---|---|---|
| 头部麻点 (417,250) | 5 | 1928.7 | 637×298 |

⚠️ **裕度只有约 1.7 倍**（5 vs OK 最大 3），`speck_min_count=4` 已是 OK 件边界，
再放宽即误报。这属材质级低对比瑕疵，输出必须人工复核；根治建议掠射光成像。
另外 `speck_band=0.40`（物体 bbox 顶部 40% 视为头部）是按当前 SKU 姿态设定的
几何先验，**换姿态或换 SKU 必须重新确认**。

