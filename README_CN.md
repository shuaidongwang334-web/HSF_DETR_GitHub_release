# HSF-DETR 精简复现代码包

该压缩包用于论文 **“HSF-DETR: A Hierarchical Semantic Fusion Transformer for Vehicle Detection under Foggy Conditions”** 的代码复现与开源说明。

本仓库保留 RT-DETR 基础实现，并仅保留论文中使用的三个自定义模块：

- **ETB**：用于替换 RT-DETR 编码器中的 AIFI。
- **HCFM**：用于替换 Neck 中的简单 Concat 融合。
- **GCConv**：用于替换 PAN 路径中的 stride-2 Conv 下采样。

自定义模块位于：

```text
ultralytics/nn/extra_modules/hsf_detr.py
```

最终模型配置文件位于：

```text
ultralytics/cfg/models/rt-detr/rtdetr-HSF-DETR.yaml
```

## 仓库中包含什么

```text
ultralytics/                  # RT-DETR 代码主体
ultralytics/nn/extra_modules/ # ETB、HCFM、GCConv
ultralytics/cfg/models/rt-detr/ # 模型 YAML 和消融 YAML
dataset/yolo2coco.py          # YOLO 标签转 COCO 标注脚本
get_COCO_metrice.py           # COCOeval/TIDE 评估脚本
get_FPS.py                    # FPS/复杂度测试脚本
train.py                      # 训练脚本
val.py                        # 验证脚本
weights/                      # best.pt 和 last.pt
datasets/rtts_labels/         # 上传的 RTTS 标签文件
datasets/splits/              # 根据标签导出的 RTTS 划分文件
configs/training_args_rtts.yaml # 训练参数记录
```

## 需要给审稿人说明的重点

RTTS 数据集是自己按序列/已有组织方式划分的，没有在划分时设置随机种子。因此不能写“RTTS was randomly split with a fixed seed”。更准确的写法是：

```text
RTTS was divided according to the image sequence organization, and the exact split files are released for reproducibility. No random seed was used during RTTS partitioning.
```

训练时设置了随机种子，上传的训练参数中 seed 为 `3405`，这个随机种子用于训练过程，而不是用于 RTTS 数据集划分。

Foggy Cityscapes 使用官方划分，不进行额外随机划分。

## 训练

先修改 `dataset/data.yaml` 或 `configs/datasets/rtts.yaml` 中的数据集路径，然后执行：

```bash
python train.py
```

## 验证并生成 predictions.json

在 `val.py` 中设置：

```python
save_json=True
```

然后执行：

```bash
python val.py
```

生成的预测文件一般位于：

```text
runs/val/实验名/predictions.json
```

## 大中小目标 AP 评估

先将 YOLO 标签转成 COCO 格式，例如 RTTS 验证集：

```bash
python dataset/yolo2coco.py \
  --image_path "/path/to/rtts_data/images/val" \
  --label_path "/path/to/rtts_data/labels/val" \
  --save_path "/path/to/rtts_data/rtts_val_coco.json"
```

再执行：

```bash
python get_COCO_metrice.py \
  --anno_json "/path/to/rtts_data/rtts_val_coco.json" \
  --pred_json "runs/val/HSF-DETR_val/predictions.json"
```

输出中的 `AP_small`、`AP_medium`、`AP_large` 就是小、中、大目标 AP。
