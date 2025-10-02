# YOLO-NAS Training & Export Guide

This guide explains how to train a YOLO-NAS object detector from scratch on your dataset using **SuperGradients**, and then export the trained model to **ONNX** format for deployment.

---

## 1. Prepare Your Dataset

* Download your dataset directly from **Roboflow** in **YOLOv5 format**, or
* Use an existing dataset on disk that follows the same directory structure:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each `.txt` file inside `labels/` must follow YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

---

## 2. Train the Model

Run `train.py` to train YOLO-NAS from scratch.

**Example usage:**

```bash
python train.py \
    --data-dir /your_dataset_base_dir \
    --classes "weapon,person,hand" \
    --model-arch yolo_nas_s \
    --batch-size 32 \
    --num-epochs 100
```

### Arguments

* `--data-dir` : Path to dataset base directory
* `--classes` : Comma-separated list of class names
* `--model-arch` : Model architecture (e.g., `yolo_nas_s`, `yolo_nas_m`, `yolo_nas_l`)
* `--batch-size` : Training batch size
* `--num-epochs` : Number of training epochs

Checkpoints are saved under a `checkpoints/` directory in your project folder.

---

## 3. Export to ONNX

Once training is complete, export the model using `export_onnx.py`.

**Example usage:**

```bash
python export_onnx.py \
    --checkpoint /path/to/ckpt.pth \
    --num-classes 5 \
    --model-arch YOLO_NAS_S \
    --output model_fp16.onnx \
    --input-shape 1 3 640 640 \
    --fp16
```

### Arguments

* `--checkpoint` : Path to the trained checkpoint `.pth`
* `--num-classes` : Number of dataset classes
* `--model-arch` : Must match the architecture used in training
* `--output` : Output filename for ONNX model
* `--input-shape` : Input tensor shape (default: `1 3 640 640`)
* `--fp16` : (Optional) Export in half precision (requires CUDA)
* `--opset` : ONNX opset version (default: 12)

---

## 4. Notes & Best Practices

* The `model-arch` and `num-classes` **must match** between training and export.
* Use **FP16 export** for optimized deployment on GPUs. If running on CPU or hardware without FP16 support, skip the `--fp16` flag.
* Always validate the exported ONNX model using `onnxruntime` or your target inference engine before deployment.

---
