# HSF-DETR: Clean RT-DETR Package for Foggy Traffic Object Detection

This repository is a cleaned reproducibility package for the paper **“HSF-DETR: A Hierarchical Semantic Fusion Transformer for Vehicle Detection under Foggy Conditions”**.

The package keeps the basic RT-DETR implementation and only the three custom modules used in the paper:

- **ETB**: Entanglement Transformer Block, used to replace AIFI in the RT-DETR encoder.
- **HCFM**: Hierarchical Context Fusion Module, used to replace simple Concat operations in neck fusion.
- **GCConv**: Re-parameterized downsampling convolution, used to replace stride-2 Conv in the PAN path.

The custom modules are implemented in:

```text
ultralytics/nn/extra_modules/hsf_detr.py
```

The final HSF-DETR model configuration is:

```text
ultralytics/cfg/models/rt-detr/rtdetr-HSF-DETR.yaml
```

Ablation YAML files are also retained:

```text
ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml
ultralytics/cfg/models/rt-detr/rtdetr-ETB.yaml
ultralytics/cfg/models/rt-detr/rtdetr-HCFM.yaml
ultralytics/cfg/models/rt-detr/rtdetr-GCConv.yaml
ultralytics/cfg/models/rt-detr/rtdetr-HSF-DETR.yaml
```

All unrelated experimental modules, unrelated backbones, SAM/FastSAM/NAS-related files, and unrelated YAML files were removed from this release package.

## 1. Repository structure

```text
HSF_DETR_GitHub_release/
├── ultralytics/                         # RT-DETR/Ultralytics implementation
│   ├── nn/extra_modules/hsf_detr.py      # ETB, HCFM and GCConv implementation
│   └── cfg/models/rt-detr/               # RT-DETR and HSF-DETR YAML files
├── dataset/
│   ├── data.yaml                         # RTTS dataset template
│   └── yolo2coco.py                      # YOLO-to-COCO conversion script
├── configs/
│   ├── training_args_rtts.yaml           # training arguments used in the experiment
│   └── datasets/                         # dataset YAML templates
├── datasets/
│   ├── rtts_labels/                      # RTTS label files provided with this package
│   └── splits/                           # RTTS split files generated from labels
├── weights/
│   ├── best.pt                           # trained best checkpoint
│   └── last.pt                           # trained last checkpoint
├── train.py
├── val.py
├── get_COCO_metrice.py
├── get_FPS.py
├── requirements.txt
└── README.md
```

## 2. Method boundary and novelty statement

The novelty of HSF-DETR lies in a task-oriented RT-DETR enhancement framework for foggy traffic perception. It combines frequency-spatial high-level semantic modeling, hierarchical cross-scale feature fusion, and weak-feature-preserving downsampling.

### ETB

ETB is used as a fog-oriented replacement for the original AIFI encoder in RT-DETR. The contribution should not be stated as inventing frequency-spatial entanglement from scratch. It should be described as adapting frequency-domain self-attention and spatial-domain self-attention to the high-level RT-DETR encoder for foggy traffic detection. The motivation is to compensate for global contrast degradation, blurred boundaries, and weak small-target responses under fog.

### HCFM

HCFM is the Hierarchical Context Fusion Module. It replaces simple Concat operations in the RT-DETR neck. The module aligns two input features by 1×1 convolution, extracts local-global attention features at different receptive-field scales, adds a bypass complementary branch, and refines the aggregated feature by RepConv. This is the main cross-scale fusion design in HSF-DETR.

### GCConv

GCConv is used as a weak-feature-preserving downsampling layer in the RT-DETR neck. The novelty should be described as its task-oriented insertion into the downsampling bottlenecks of RT-DETR, not as inventing re-parameterized convolution itself. In foggy scenes, stride-2 downsampling can discard weak contours of distant vehicles and pedestrians; GCConv mitigates this by using multi-branch compensation during training and single-path deployment during inference.

## 3. Installation

Create an environment and install dependencies:

```bash
conda create -n hsf_detr python=3.8 -y
conda activate hsf_detr
pip install -r requirements.txt

**Note:** PyTorch 2.3.0 + CUDA 12.1 was used in our experiments. The equirements.txt uses a wide range (	orch>=1.8.0); for exact reproducibility, install the matching version:
`ash
pip install torch==2.3.0 torchvision==0.18.0
`${nl}pip install pycocotools prettytable tidecv
```

Install the local package in editable mode:

```bash
pip install -e .
```

## 4. Dataset preparation

The original images of RTTS and Foggy Cityscapes are not included in this repository. Please download them from their official sources and organize them in YOLO format.

**Official download links:**
- **RTTS**: [RESIDE benchmark (RESIDE-beta)](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) -- RTTS is included in the RESIDE-beta subset.
- **Foggy Cityscapes**: [Cityscapes Dataset](https://www.cityscapes-dataset.com/) (registration required) -- Download "Foggy Cityscapes" from the downloads page. Also available at [SFSU Synthetic Fog](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/).

### RTTS

RTTS was divided according to image sequence organization. No random seed was used for dataset partitioning. To ensure reproducibility, this package provides the final split files and label files used in the experiment.

Recommended RTTS structure:

```text
/path/to/rtts_data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Dataset YAML template:

```yaml
path: /path/to/rtts_data
train: images/train
val: images/val
test: images/test

nc: 5
names: ['bicycle', 'bus', 'car', 'motorbike', 'person']
```

The provided RTTS split files are located in:

```text
datasets/splits/
```

The training random seed is recorded in:

```text
configs/training_args_rtts.yaml
```

In the uploaded training setting, the training seed is `3405`. This seed controls training-related randomness, not the RTTS split generation.

### Foggy Cityscapes

Foggy Cityscapes follows the official dataset split without additional random partitioning. The dataset YAML should point to the official train/val/test directories. A template is provided in:

```text
configs/datasets/city_foggy.yaml
```

Please keep the class names and class order consistent with your converted labels.

## 5. Training

Before training, edit `dataset/data.yaml` or `configs/datasets/rtts.yaml` to your local dataset path.

Train HSF-DETR:

```bash
python train.py
```

The training arguments used in one experiment are saved in:

```text
configs/training_args_rtts.yaml
```

Key settings from the uploaded training configuration include:

```text
epochs: 250
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.0001
seed: 3405
deterministic: true
split: val
```

Please report the final manuscript settings according to the actual experiment used for the paper.

## 6. Validation and prediction JSON generation

To generate a COCO-style `predictions.json`, set `save_json=True` in `val.py` and use the correct dataset YAML.

Example:

```python
result = model.val(
    data='/path/to/rtts_data.yaml',
    split='val',
    imgsz=640,
    batch=1,
    save_json=True,
    project='runs/val',
    name='HSF-DETR_val',
    exist_ok=True
)
```

Run:

```bash
python val.py
```

The output should contain:

```text
runs/val/HSF-DETR_val/predictions.json
```

## 7. COCO-style AP_small / AP_medium / AP_large evaluation

First convert YOLO labels to COCO format. For RTTS validation set:

```bash
python dataset/yolo2coco.py \
  --image_path "/path/to/rtts_data/images/val" \
  --label_path "/path/to/rtts_data/labels/val" \
  --save_path "/path/to/rtts_data/rtts_val_coco.json"
```

Then run COCO evaluation:

```bash
python get_COCO_metrice.py \
  --anno_json "/path/to/rtts_data/rtts_val_coco.json" \
  --pred_json "runs/val/HSF-DETR_val/predictions.json"
```

The output includes:

```text
AP_small
AP_medium
AP_large
```

COCO scale definition:

```text
small:  area < 32^2
medium: 32^2 <= area < 96^2
large:  area >= 96^2
```

## 8. Model complexity and FPS

Use `val.py` and `get_FPS.py` to report parameters, FLOPs, model size, preprocessing time, inference time, post-processing time, and FPS. Report all models under the same input size, batch size, device, software environment, and testing protocol.

Recommended reporting protocol:

```text
input size: 640 × 640
batch size: 1
same GPU for all models
same PyTorch/Ultralytics environment
same split and same evaluation scripts
```

## 9. Weights

The uploaded package includes:

```text
weights/best.pt
weights/last.pt
```

If GitHub file size limits prevent direct upload, place the weights in a GitHub Release, Google Drive, OneDrive, Baidu Netdisk, or another public storage service, and provide the download link in this section.

## 10. Reproducibility notes

- RTTS split: sequence-based/manual split; no random seed was used during partitioning.
- RTTS reproducibility: the exact label files and split files are provided.
- Training seed: recorded in `configs/training_args_rtts.yaml`.
- Foggy Cityscapes: official split, no additional random partitioning.
- Evaluation: all models should use the same split, same image size, same confidence/IoU settings, and same evaluation scripts.

## 11. Suggested data availability statement

The datasets used in this study are publicly available. RTTS is from the RESIDE benchmark, and Foggy Cityscapes is derived from the Cityscapes dataset with synthetic fog generation. For RTTS, the exact training, validation, and test split files used in this study are released in this repository to ensure reproducibility. For Foggy Cityscapes, the official dataset split is used without additional random partitioning. The implementation code, model configuration files, training scripts, evaluation scripts, training random seed setting, and trained weights are provided in this repository or through the associated download link.
