# YOLOX Training for object detection

This document provides a concise reference for training a YOLOX model starting from a given checkpoint with a custom dataset.

---

## 1. Dataset Preparation

Download your dataset in COCO format (via Roboflow or other tools) and structure it as:

```
data_dir/
    annotations/
        _annotations_train.json
        _annotations_val.json
    train2017/
        *.jpg
    val2017/
        *.jpg
```

⚠️ Sometimes COCO JSONs may contain corrupted or duplicate categories when exported from tools like Roboflow. If training fails, inspect and clean the categories before proceeding.

---

## 2. Experiment File

YOLOX requires an **exp file** (e.g., `yolox_s.py`, `yolox_m.py`) that defines the training recipe. Use the one matching the model variant you plan to train.

---

## 3. Pretrained Weights

Download pretrained weights for faster convergence:

```bash
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

---

## 4. Setup Environment

Clone YOLOX and install dependencies:

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
python3 -m venv .venv
source .venv/bin/activate
cd YOLOX
pip install -r requirements.txt
pip install -v -e .
```

---

## 5. Training

Start training:

```bash
python tools/train.py -f path/to/exp.py -d 1 -b 16 --fp16 -o -c path/to/checkpoint.pth
```

* `-f`: training recipe file (exp)
* `-d`: number of GPUs
* `-b`: batch size
* `-c`: pretrained checkpoint (optional)

---

⚠️ You can use the yolox_s.py as a starting point for your exp file. Notice that the name of the .py file you use as the training recipe will be used to determine the model to be trained. Yolox will usually fallback to yolox_l if you don't name your file a valid model name.



## 6. Prepare Weights for Inference

Strip unnecessary data and (optionally) save in FP16 for faster inference:

```bash
python3 utils/prepare_for_conversion.py --model_ckpt your_model_ckpt.pth --fp16 --output_name weights_only.pth
```

---

## 7. Export to ONNX

Convert the model for deployment:

```bash
python3 tools/export_onnx.py --output-name weapons_only.onnx -f ./exps/yolox_s.py -c ./YOLOX_outputs/yolox_s/weights_only.pth
```

---
