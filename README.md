# AI Label Studio

A lightweight, locally-deployed image labeling tool for training YOLO-based AI models.  
No cloud dependencies — runs entirely on your machine via a simple web UI in your browser.

## Features

- **Detection labeling** — draw bounding boxes, assign classes, move/resize with mouse
- **Classification labeling** — tick one or more class labels per image (multi-label)
- **Data augmentation on export** — brightness, hue, rotation, and tilt with randomised ranges
- **Ultralytics-compatible export**
  - Detection → `data.yaml` + `train/valid/test/images` + `train/valid/test/labels`
  - Classification → `train/valid/test/` flat images + `_classes.csv` per split (multi-hot)
- **Configurable train/valid/test split ratio** at export time (images shuffled randomly)
- **Import existing datasets** — supports both split structure and flat folders
- Cross-platform: Windows, macOS, Ubuntu

## Requirements

- Python 3.8 or newer
- pip (comes with Python)

No other installation needed — dependencies are installed automatically on first run.

## Quick Start

### Windows

Double-click `run.bat`, or run in a terminal:

```bat
run.bat
```

### macOS / Ubuntu

```bash
chmod +x run.sh
./run.sh
```

Then open **http://localhost:5000** in your browser.

On first run the script creates a virtual environment (`.venv/`) and installs dependencies automatically.

## Manual Setup (any OS)

```bash
python3 -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

## Usage

### 1. Create a project

Click **+ Project**, enter a name, choose the task type:

| Task | Use when |
|---|---|
| **Detection** | You need bounding boxes (YOLOv9 / YOLOv11) |
| **Classification** | Each image gets one or more class tags |

Enter your class names (comma-separated), then click **Create**.

### 2. Add images

Click **+ Upload Images** in the sidebar to upload images from your computer.  
Or click **Import** to load an existing dataset from a local folder path on the server.

### 3. Label

**Detection**

| Action | How |
|---|---|
| Draw box | Click and drag on the canvas (`b` key) |
| Select / move | Click **Select/Move** or press `s`, then drag a box |
| Resize | Select a box, drag a corner handle |
| Change class | Use the dropdown in the right panel |
| Delete box | Press `Delete` or click ✕ |
| Navigate images | `←` / `→` arrow keys, or `a` / `d` |

**Classification**

Click the checkboxes in the right panel to assign classes to the current image.  
Press keys `1`–`9` to quickly toggle the first 9 classes.  
Navigate with `←` / `→`. Labels are saved automatically when you move to another image.

### 4. Export

Click **Export**, configure the split ratios and optional augmentation, then click **Download ZIP**.

**Augmentation options (training split only):**

| Parameter | Effect |
|---|---|
| Brightness ± | Randomly adjusts brightness within the given range |
| Hue ± | Randomly adjusts colour saturation |
| Rotate ± ° | Randomly rotates the image (bounding boxes rotate with it) |
| Tilt ± ° | Randomly applies a horizontal shear |
| Copies / image | How many augmented variants to generate per original image |

Each copy independently draws a random value within the configured range.

## Project structure

```
ai_labeling_system/
├── app.py              # Flask backend
├── requirements.txt    # Python dependencies
├── run.bat             # Windows launcher
├── run.sh              # macOS / Linux launcher
├── static/
│   └── index.html      # Single-page frontend
└── projects/           # Created automatically, stores your labeled data
    └── <project-name>/
        ├── images/     # All uploaded images
        ├── labels/     # Per-image label files
        └── meta.json   # Project metadata (classes, task type)
```

## Dependencies

| Package | Purpose |
|---|---|
| Flask | Web server |
| Pillow | Image loading and augmentation |
