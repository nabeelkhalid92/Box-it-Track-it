<div align="center">

# Box-it-Track-it
Accurate cell tracking in microscopy is essential for studying biological dynamics like proliferation and migration. Traditional fully supervised methods demand dense pixel-wise masks for every frame, making them impractical for large-scale use. Recent methods like SAT reduce annotation effort by using sparse point-based supervision, but still require multiple positive and negative points per cell, which remains labor-intensive. BoxTrack offers a lightweight and annotation-efficient alternative, requiring only a single bounding box per cell in the first frame. Without relying on any point-level annotations, it performs end-to-end instance segmentation and tracking over entire sequences. This simplification leads to a substantial reduction in annotation cost while improving performance over SAT. On the CTMC dataset, BoxTrack improves Multiple Object Tracking Accuracy (MOTA) by +15.96% over SAT. For the CTC dataset, it yields a +8.86% MOTA gain.
</div>

## Getting Started

### Box-it-Track-it Installation 

SAM 2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`. Please follow the instructions [here](https://github.com/facebookresearch/sam2?tab=readme-ov-file) to install both PyTorch and TorchVision dependencies. You can install **the BoxTrack version** of SAM 2 on a GPU machine using:
```
cd sam2
pip install -e .
pip install -e ".[notebooks]"
```

Please see [INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) from the original SAM 2 repository for FAQs on potential issues and solutions.

Install other requirements:
```
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru
```

#### SAM 2.1 Checkpoint Download

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

### 📦 Directory Structure

```
.
├── scripts/
│   └── demo.py                # Main Python script
├── sam2/
│   └── ...                    # SAM2 module and configs
├── demo/
│   └── demo_1/
│       ├── images/           # Input image frames OR .mp4 video
│       ├── bbox.txt          # Ground truth bounding boxes
│       ├── tracking_results.txt  # Output tracking result (MOT format)
│       ├── output/           # Annotated visualization frames
├── run_demo.sh               # Shell script to run demo
└── README.md                 # This file
```

## 🚀 How to Run

### 🔧 1. Set Up

Make sure your `sam2` module is available (e.g., `sam2/checkpoints/sam2.1_hiera_large.pt`) and dependencies are installed, including:
- `torch`
- `opencv-python`
- `numpy`

### 📄 2. Prepare Inputs

- **Input Video**: Either a folder of frames or a `.mp4` file.
- **Bounding Boxes**: A `bbox.txt` file in the format:

  ```
  frame_id, object_id, x, y, w, h
  ```

---

### 🖥 3. Run with Shell Script

Edit `run_demo.sh` if needed and run:

```bash
bash run_demo.sh
```

This script runs the pipeline using the following parameters:

- `--video_path`: Path to video or image folder.
- `--txt_path`: Input bounding box file.
- `--model_path`: Path to SAM2 checkpoint.
- `--output_tracking_txt`: Output tracking results.
- `--output_dir`: Directory to save annotated frames.
- `--save_to_video`: Enables saving annotated frames.
- `--plot_option`: Choose from `bbox`, `mask`, or `both`.
- `--min_bbox_side`: Skip boxes smaller than this.
- `--interval_mode`: Choose how to divide the video:
  - `interval`: Fixed-length chunks (e.g., every 300 frames).
  - `first_app`: Use a list of key frame indices.
  - `combined`: Use both above (split long intervals again).
- `--interval_param`: Int (for interval) or comma-separated boundaries (for first_app).
- `--man_track_file`: (Required for first_app/combined) File containing frame split info.

---
## 📦 Output

- **Tracking Results**: Written to `tracking_results.txt` in MOT format.
- **Visualized Frames**: Annotated frames saved in `output/` (if `--save_to_video` is used).
  - Includes segmentation masks, bounding boxes, and object IDs.

---

## 📝 Notes

- This demo always runs SAM2 with `obj_id=0` to avoid ID-related errors.
- Intermediate files and folders are auto-deleted after processing.
- Use GPU (`cuda:0`) for faster inference.

---

## 📋 Example

```bash
python scripts/demo.py \
    --video_path demo/demo_1/images \
    --txt_path demo/demo_1/bbox.txt \
    --model_path sam2/checkpoints/sam2.1_hiera_large.pt \
    --output_tracking_txt demo/demo_1/tracking_results.txt \
    --output_dir demo/demo_1/output \
    --plot_option both \
    --save_to_video \
    --min_bbox_side 5 \
    --interval_mode interval \
    --interval_param 300
```

---
# Minor update
