#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchCore 兼容导入
==================
anomalib 不同版本的类路径不一样：

    2.7.0 : from anomalib.models.image import Patchcore    (小写 c)
    旧版  : from anomalib.models import PatchCore

各脚本统一从这里导入，只写一行：

    from pc_import import PatchCore
"""
try:
    from anomalib.models.image import Patchcore as PatchCore
except ImportError:
    try:
        from anomalib.models import PatchCore          # noqa: F811
    except ImportError:
        raise ImportError(
            '未找到 PatchCore。请安装 anomalib:  pip install "anomalib[cu124]"'
        )
