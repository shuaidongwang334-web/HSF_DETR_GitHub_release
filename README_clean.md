# HSF-DETR Clean RT-DETR Package

This cleaned package keeps the basic RT-DETR implementation and only three custom modules used in the paper:

- `ETB`: Entanglement Transformer Block, used to replace AIFI in the RT-DETR encoder.
- `HCFM`: Hierarchical Context Fusion Module and used to replace Concat in neck fusion.
- `GCConv`: Re-parameterized downsampling convolution, used to replace stride-2 Conv in the PAN path.

The custom modules are located in:

```text
ultralytics/nn/extra_modules/hsf_detr.py
```

The final model YAML is:

```text
ultralytics/cfg/models/rt-detr/rtdetr-HSF-DETR.yaml
```

Ablation YAML files are retained:

```text
rtdetr-r18.yaml
rtdetr-ETB.yaml
rtdetr-HCFM.yaml
rtdetr-GCConv.yaml
rtdetr-HSF-DETR.yaml

