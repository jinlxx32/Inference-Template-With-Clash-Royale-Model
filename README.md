# ⚔️ Clash Royale Model Studio

A modern desktop application built with **CustomTkinter** and **Ultralytics YOLO** for object detection in Clash Royale gameplay. Supports dynamic model loading, photo, video, and whole-folder batch inference, hardware compute scanning (NVIDIA CUDA / CPU), live FPS metrics, and direct Windows File Explorer integration.

---

## 🚀 Key Features

* **Instant Pop-up UI (<0.5s):** Asynchronous background engine warmup avoids window launch freezes.
* **⚡ Preload to VRAM (Zero-Delay):** Uploads models to GPU VRAM with a 1-frame dummy pass to eliminate initial video cold-start pauses.
* **🔍 Inspect / Test Model:** View model architecture, exact file size, total class count, and scroll through all labeled classes (e.g. 71 Clash Royale units).
* **📋 Input Media Queue Manager:** Add files from multiple different folders across your PC without replacing previous selections.
* **➕ Multi-Folder Batching:** Hold `Ctrl` or `Shift` to add multiple files at once. Click `➕ Add File(s)` repeatedly to queue files from different locations.
* **✖ Individual Deselect / Remove:** Review queued files in an interactive scrollable list with file size and parent folder details, and remove any file with a single click.
* **⚠️ Smart Duplicate & Path Notifications:** Automatically alerts if duplicate files are selected or if an invalid/missing file path is encountered.
* **Flat Output with Folder Tags:** All annotated media files are saved directly into `result/` as flat files (e.g. `match_Arena15_annotated.mp4`), never hidden inside nested subdirectories. If files in different folders share the same name, folder names are automatically appended to preserve both.
* **Automatic Duplicate Protection:** Subsequent runs on the same file automatically append an incremental counter (`_1.mp4`, `_2.mp4`) so previous results are never lost.
* **Real-time Performance Metrics:** Live **FPS (speed)**, progress bar, elapsed time, dynamic **ETA**, and detection counters.
* **Hardware Compute Selector:** Automatic scanning for NVIDIA CUDA GPUs and CPU mode with one-click toggling in Settings.
* **Adjustable Video Stride (Frame Skipping):**
  * `Stride 1` (Default): Processes every single frame for maximum smoothness.
  * `Stride 2`: Processes every 2nd frame (2x faster processing).
  * `Stride 3–10`: Quick preview mode for lengthy match recordings.
* **Direct File Explorer Access:** Open the `result/` folder with one click directly from the app header.

---

## 📁 Repository Structure

```
CR_Model/
├── models/                     # Place your trained YOLO models (.pt) here
│   └── best_cr.pt
├── samples/                    # Sample gameplay videos for testing
│   ├── sample1.mp4             # Clash Royale gameplay match (74.8 MB)
│   ├── sample2.mp4             # Clash Royale gameplay match (62.0 MB)
│   └── sample3.mp4             # Clash Royale gameplay match (40.1 MB)
├── result/                     # Annotated output media saved here
│   └── .gitkeep
├── app.py                      # Main desktop application (CustomTkinter)
├── run_app.bat                 # Clickable launcher (starts app & closes terminal)
├── install_requirements.bat    # 1-click dependency installer for Windows
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Quick Start

### 1. Prerequisites
* **Python 3.9+** (Tested on Python 3.11 & 3.13)
* *(Optional for GPU acceleration)*: NVIDIA GPU with CUDA drivers installed.

### 2. Install Dependencies
You can double-click **`install_requirements.bat`**, or run in your terminal:

```bash
pip install -r requirements.txt
```

*(Note for CUDA users)*: If PyTorch defaults to CPU, install PyTorch with CUDA using:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Launch the Application
Double-click **`run_app.bat`** (or execute `python app.py`).

---

## 🎮 How to Use

1. **Select & Inspect Model:**
   * Choose your model from the `models/` dropdown (e.g. `best_cr.pt`).
   * *(Optional)* Click **`🔍 Inspect / Test`** to view all labels and classes.
   * *(Recommended)* Click **`⚡ Preload to VRAM`** to eliminate video cold-start pauses.
2. **Build Your Media Queue:**
   * Click **`➕ Add File(s)`** to select one or multiple videos/photos (hold `Ctrl` or `Shift`).
   * Click **`📁 Add Folder`** to add an entire folder of videos/photos (all media inside are appended to the queue).
   * Click either button anytime to add more files or folders from anywhere on your PC without losing your current queue.
   * Click **`➕ Add Sample`** to add test clips from `samples/`.
   * Review queued files in the list; click **`✖`** on any row to deselect/remove it, or click **`🗑️ Clear All`** to reset.
   * If you add a folder and then add files already in that folder (or vice versa), duplicates are automatically detected and reported in the notification banner.
3. **Configure Parameters:**
   * **Confidence:** Threshold for object detection (default `0.25`).
   * **Video Stride:** Adjust frame-skipping multiplier (default `1`). Click the **`ℹ`** button for guidance.
4. **Run:** Click **`▶ Start Inference`**.
5. **View Results:** Click **`📂 Open in File Explorer`** to open the `result/` folder and inspect your annotated media.

---

## 📄 License
MIT License. Feel free to use and adapt for your own computer vision workflows!
