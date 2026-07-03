# SmartPPE AI: Real-Time Workplace Safety Detection Using YOLOv8



## Project Overview

**SmartPPE AI** is a computer vision project that uses YOLOv8 to detect personal protective equipment (PPE) compliance in real-time workplace environments.

It automatically detects:
- Helmets
- Safety vests
- Gloves (optional)
- Masks (if applicable)

Designed for industrial and construction sites, this system monitors PPE compliance automatically and logs non-compliance events.

---

## Key Features

- Real-time PPE detection using YOLOv8
- Multi-class object detection with bounding boxes and confidence scores
- Event logging for safety violations
- Lightweight and suitable for edge deployment

---

## System Architecture

```
[Camera Feed] --> [YOLOv8 Detection] --> [Bounding Boxes & Confidence] --> [Logging / Dashboard]
```

- **Camera Feed**: Live video input
- **YOLOv8 Detection**: Performs PPE object detection
- **Output**: Bounding boxes, confidence, and event logs

---

## Dataset

- **Classes**: Helmet, Vest, Gloves, Mask
- **Format**: YOLO annotation (`.txt`)
- **Size**: ~2,500 images (augmented)
- **Note**: Dataset is excluded from GitHub due to size constraints.

---

## Environment Setup

> `.venv` is excluded. Create your own virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies:** Python 3.10+, PyTorch, OpenCV, YOLOv8 (ultralytics), NumPy, Pandas, Matplotlib

---

## Usage

### Run on Images

```bash
python detect.py --source "data/images/test.jpg" --weights "weights/best.pt" --conf 0.5
```

### Run on Video / Webcam

```bash
python detect.py --source "data/videos/factory.mp4" --weights "weights/best.pt" --conf 0.5

# webcam
python detect.py --source 0 --weights "weights/best.pt" --conf 0.5
```

### Training

```bash
python train.py --img 640 --batch 16 --epochs 50 --data "dataset.yaml" --weights "yolov8m.pt"
```

- `dataset.yaml`: paths and class names
- `yolov8m.pt`: pretrained model
- Adjust `img` and `batch` for GPU capacity

---

## Evaluation Metrics

- **Precision** = Correct Detections / Total Detections
- **Recall** = Correct Detections / Total Ground Truths
- **F1-Score** = Harmonic mean of Precision & Recall
- **mAP@0.5** = Mean Average Precision at IoU 0.5

---

## Results

| Class  | Precision | Recall | F1-Score | mAP@0.5 |
|--------|-----------|--------|----------|---------|
| Helmet | 0.98      | 0.97   | 0.975    | 0.98    |
| Vest   | 0.97      | 0.96   | 0.965    | 0.97    |
| Gloves | 0.95      | 0.94   | 0.945    | 0.95    |
| Mask   | 0.96      | 0.95   | 0.955    | 0.96    |

**Real-time FPS:** 20–25 on GTX 3090

---

## Future Work

- Add more PPE classes (safety shoes, goggles)
- Integrate alert system for non-compliance
- Deploy on edge devices (TensorRT, Jetson)
- Experiment with YOLOv9 / transformer-based models

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

---

## License

MIT License. See `LICENSE` file for details.

---

## Notes for Collaborators

- Do not commit `.venv`, datasets, or large model files.
- Use `requirements.txt` to recreate the environment.
- Large files should be shared via Google Drive / S3.
