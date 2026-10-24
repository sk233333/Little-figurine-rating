#!/bin/bash
# ============================================================
# 全流程一键运行
#   bash run_all.sh qml_demo
# ============================================================
set -e
SKU=${1:-qml_demo}
cd "$(dirname "$0")"

echo "=========== Stage 0 检查 ==========="
python3 -c "import anomalib, cv2, yaml; print('anomalib', anomalib.__version__)" || {
  echo "请先安装:  pip install \"anomalib[cu124]\""
  exit 1
}
test -f "configs/${SKU}.yaml" || {
  test -f "configs/template.yaml" || {
    echo "缺少配置模板: configs/template.yaml"
    exit 1
  }
  cp "configs/template.yaml" "configs/${SKU}.yaml"
  echo "已生成配置: configs/${SKU}.yaml"
  echo "  请按需修改: sku / object_height_mm / roi_mm / thresholds_mm / key_regions"
}

echo "=========== Stage 1 归一化 ==========="
python3 normalize.py --sku "${SKU}"

echo "=========== Stage 2 ROI 切分 ==========="
python3 roi_split.py --sku "${SKU}"

echo "=========== Stage 3 训练 PatchCore ==========="
python3 train_pc.py --sku "${SKU}" --roi all

echo "=========== (可选) 标记屏蔽区 ==========="
# python3 mark_exclude.py --sku "${SKU}" --stem ok_001

echo "=========== Stage 4 推理判级 ==========="
python3 infer.py --sku "${SKU}"

echo "=========== Stage 5 评估 ==========="
if [ -f "labels_${SKU}.csv" ]; then
  python3 evaluate.py --sku "${SKU}" --labels "labels_${SKU}.csv"
else
  echo "没有 labels_${SKU}.csv，跳过评估"
  echo "格式: stem,label,size_mm,type"
fi

echo "=========== 完成 ==========="
echo "结果: reports/${SKU}/"
