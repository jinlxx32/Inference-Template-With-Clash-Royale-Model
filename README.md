# ⚔️ Clash Royale Vision Suite

An all-in-one computer vision and real-time game intelligence suite for **Clash Royale**, powered by **CustomTkinter** and **Ultralytics YOLO**.

This repository contains **two complete, independent desktop applications**:
1. 📷 🎥 **Batch Model Studio (`app.py`)**: For testing, benchmarking, and processing recorded match replays, clips, and images with full batch queues and performance telemetry.
2. 📱 🃏 **Live Studio & Card Cycle Tracker (`live_app.py`)**: For live real-time gameplay streaming directly from your phone (via headless ADB/scrcpy) with an active **4-Red / 4-Green Opponent Deck Cycle Tracker**.

---

## ⚠️ Important Note Regarding the Included Model (`best_cr.pt`)

> [!WARNING]
> **Sample Model Accuracy & Known Errors:**
> The included weights (`models/best_cr.pt`) are provided as a demonstration model trained specifically on unit **deployment** animations (`_D` labels and spell casts).
>
> **Known Classification Quirks:**
> * **Accurate Classes:** Units like **Knight**, **Furnace**, **Goblin Barrel**, and **Fireball** generally detect with high accuracy.
> * **Known Misclassifications:** Due to training dataset class imbalance and visual overlap, certain units exhibit known labeling errors (e.g. heavy ground tanks such as **P.E.K.K.A**, **Mega Knight**, or **Royal Giant** may misclassify as **Royal Ghost**, and **Dart Goblin** may register as **Bandit**).
> * **Tuning Confidence:** You can raise the **Confidence slider** (e.g. to `40% - 50%`) to filter out weak false positives.
>
> **100% Modular — Drop In Your Own Model:**
> Both applications are completely decoupled from this specific model file. If you train a new or improved YOLO model, simply drop your `.pt` file into the `models/` directory. It will appear immediately in the **Model dropdown** in both apps with zero code changes needed.

---

## 📱 The Two Applications

### 1. 📷 🎥 Batch Model Studio (`app.py` / `run_app.bat`)
Built for running YOLO inference over saved gameplay footage and dataset folders.
* **Instant Window Pop-up (<0.5s):** Asynchronous background engine warmup avoids window launch freezes.
* **⚡ Preload to VRAM:** Eliminates cold-start pauses when starting video processing.
* **🔍 Inspect Model:** Displays architecture, parameter counts, and all labeled classes.
* **📋 Multi-Folder Queue:** Add files from multiple directories; hold `Ctrl`/`Shift` for multi-select.
* **Adjustable Video Stride (Frame Skipping):** Stride 1 (every frame), Stride 2 (2x speed), or Stride 3–10 (preview mode).
* **Flat Output Protection:** Saves annotated files to `result/` with automated incremental counters (`_1.mp4`, `_2.mp4`).

### 2. 📱 🃏 Live Studio & Deck Tracker (`live_app.py` / `run_live_app.bat`)
Built for live match companion tracking while playing on your Android phone.
* **100% Headless Direct Stream:** Streams your phone screen directly into the UI over an internal kernel pipe. **No floating `scrcpy` window**, and other open windows placed on top **never cover or disrupt the stream**.
* **Zero-Lag Decoupled Frame Sync:** Even if YOLO runs at 15–30 FPS on your hardware, the stream consumer empties the pipe at full speed and always processes the newest frame, completely eliminating delay buildup.
* **Official 4-Card Cycle Tracker:**
  * **🔴 TOP ROW (4 Red Slots):** Shows the 4 cards currently *out of hand* in the opponent's cycle queue, with countdown badges (`1/4`, `2/4`, `3/4`, `Next!`).
  * **🟢 BOTTOM ROW (4 Green Slots):** Shows the 4 cards the opponent currently holds *in hand* ready to deploy against you right now.
  * **Mystery `?` Slots:** Starts with unrevealed slots and tracks `Deck Revealed: X / 8 Cards`.
  * **Anti-Spam Debounce:** Enforces a 3.5s cooldown per card so a single deployment animation won't trigger duplicate cycles.
* **📱 Multi-Device & Emulator Selector:** If you have multiple phones, tablets, or Android emulators (e.g., LDPlayer, BlueStacks, Nox, WSA) connected, select the exact target device from the **`📱 Phone`** dropdown menu and click **`🔄 Scan`** anytime to refresh.
* **⚡ Compute Hardware Selector (GPU / CPU):** Seamlessly switch the active YOLO inference engine between your **NVIDIA CUDA GPU** (shows GPU name and VRAM) and **CPU** on the fly without restarting the stream.
* **Official Card Portraits:** Includes pre-cached official Clash Royale card art (`assets/cards/`) with automatic self-healing download fallback.
* **Match Deployment Log:** Live scrollable match history recording timestamps and unit cards played.

---

## 📁 Repository Structure

```
CR_Model/
├── models/                     # Place your trained YOLO models (.pt) here
│   └── best_cr.pt              # Included sample deployment model
├── assets/
│   └── cards/                  # 204 official Clash Royale card portrait PNGs
├── samples/                    # Sample gameplay videos for testing
│   ├── sample1.mp4
│   ├── sample2.mp4
│   └── sample3.mp4
├── result/                     # Annotated output media saved here
├── tools/
│   └── scrcpy/                 # Bundled scrcpy & ADB binaries for Windows
├── app.py                      # App 1: Batch & video testing desktop app
├── live_app.py                 # App 2: Live phone stream & deck tracker
├── run_app.bat                 # Clickable launcher for App 1
├── run_live_app.bat            # Clickable launcher for App 2
├── install_requirements.bat    # 1-click dependency installer for Windows
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Quick Start

### 1. Prerequisites
* **Windows 10 / 11**
* **Python 3.9+** (Tested on Python 3.11 & 3.13)
* *(Optional for GPU acceleration)*: NVIDIA GPU with CUDA drivers.

### 2. Install Dependencies
Double-click **`install_requirements.bat`**, or run in terminal:

```bash
pip install -r requirements.txt
```

*(Note for NVIDIA CUDA users)*: If PyTorch defaults to CPU, install CUDA-accelerated PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## 🚀 How to Launch

### To use App 1 (Batch Video / Image Testing):
Double-click **`run_app.bat`** (or run `python app.py`).
1. Select your model from the dropdown.
2. Click **`➕ Add File(s)`** or **`➕ Add Sample`** to queue test videos.
3. Click **`▶ Start Inference`** and view results in `result/`.

### To use App 2 (Live Phone Screen & Deck Tracker):
1. **Connect your Android phone** to your PC with a USB cable (or start an emulator like LDPlayer/BlueStacks).
2. **Enable USB Debugging** on your phone (*Settings -> Developer Options -> Turn on USB Debugging*). Tap **Allow** when prompted on your phone.
3. Double-click **`run_live_app.bat`** (or run `python live_app.py`).
4. **Choose your hardware & device:**
   * Select your preferred **⚡ Compute** device (e.g. `GPU: NVIDIA GeForce RTX...` or `CPU`).
   * If multiple phones/emulators are connected, pick your target phone from the **`📱 Phone`** dropdown (click **`🔄 Scan`** if newly plugged in).
5. Click **`▶ Connect & Stream Phone`**.
6. Your screen will stream live in the center monitor, and cards will automatically cycle between Red (in cycle) and Green (in hand) on the right panel as you play!
7. Click **`🔄 Clear / Reset`** between games to reset the tracker.

---

## 📄 License
MIT License. Feel free to use and adapt for your own computer vision workflows!
