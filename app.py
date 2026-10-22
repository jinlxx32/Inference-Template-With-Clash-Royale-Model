"""
Clash Royale Model Studio - CustomTkinter Desktop Application
Features:
- Instant Window Pop-up (<0.5s)
- Model Caching & "⚡ Preload to VRAM" (eliminates 15s cold-start delay)
- "🔍 Inspect / Test Model" displaying labels, classes, model size & architecture
- Image, Video & Whole-Folder Batch YOLO Inference
- Real-time FPS, progress bar, time elapsed, and ETA tracking
- Automatic duplicate output counter (_1, _2, etc.)
- Direct File Explorer integration
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict

import customtkinter as ctk
from tkinter import filedialog, messagebox

# Global lazy references
cv2 = None
torch = None
YOLO = None

# ----------------- Configuration & Paths -----------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
RESULT_DIR = BASE_DIR / "result"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def open_in_explorer(target_path: Path):
    path_str = str(target_path.resolve())
    if not os.path.exists(path_str):
        path_str = str(RESULT_DIR.resolve())
    try:
        if sys.platform == "win32":
            if os.path.isfile(path_str):
                subprocess.run(["explorer.exe", f"/select,{path_str}"], check=False)
            else:
                os.startfile(path_str)
        elif sys.platform == "darwin":
            subprocess.run(["open", path_str], check=False)
        else:
            # Linux / Ubuntu
            subprocess.run(["xdg-open", path_str], check=False)
    except Exception as e:
        messagebox.showerror("Error Opening Explorer", f"Could not open explorer:\n{e}")


def get_unique_output_path(result_dir: Path, stem: str, ext: str) -> Path:
    """
    Generate unique non-colliding output filename:
    - 1st inference: {stem}_annotated{ext}
    - 2nd inference: {stem}_annotated_1{ext}
    - 3rd inference: {stem}_annotated_2{ext}, etc.
    """
    base_file = result_dir / f"{stem}_annotated{ext}"
    if not base_file.exists():
        return base_file

    counter = 1
    while True:
        candidate = result_dir / f"{stem}_annotated_{counter}{ext}"
        if not candidate.exists():
            return candidate
        counter += 1


def get_output_stem(file_path: Path, root_folder: Optional[Path] = None, all_queue_files: Optional[List[Path]] = None) -> str:
    """
    If file comes from a subfolder, appends folder names to make filename unique:
    - Root file: 'match.mp4' -> 'match'
    - In subfolder: 'Arena15/match.mp4' -> 'match_Arena15'
    - Multiple files in queue with same filename: 'recordings/match.mp4' -> 'match_recordings'
    """
    if root_folder is not None:
        try:
            rel = file_path.relative_to(root_folder)
            sub_parts = [p.replace(" ", "_") for p in rel.parent.parts if p and p != "."]
            if sub_parts:
                folder_tag = "_".join(sub_parts)
                return f"{file_path.stem}_{folder_tag}"
        except ValueError:
            pass

    if all_queue_files:
        stems = [f.stem for f in all_queue_files]
        if stems.count(file_path.stem) > 1:
            parent_tag = file_path.parent.name.replace(" ", "_")
            if parent_tag:
                return f"{file_path.stem}_{parent_tag}"

    return file_path.stem


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Clash Royale Model Studio")
        self.geometry("1020x760")
        self.minsize(900, 660)

        self._center_window(1020, 760)

        # ----------------- INSTANT WINDOW DISPLAY -----------------
        self.splash_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.splash_frame.pack(expand=True, fill="both")

        ctk.CTkLabel(
            self.splash_frame,
            text="⚔️ Clash Royale Model Studio",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(expand=True, pady=(180, 5))

        self.lbl_splash_sub = ctk.CTkLabel(
            self.splash_frame,
            text="Initializing interface...",
            font=ctk.CTkFont(size=13),
            text_color="#90CAF9"
        )
        self.lbl_splash_sub.pack(expand=True, pady=(0, 200))

        self.update()

        # State Variables
        self.active_device = ctk.StringVar(value="auto")
        self.selected_model_name = ctk.StringVar()
        self.selected_source_path = ctk.StringVar()
        self.selected_files: List[Path] = []
        self.selected_root_folder: Optional[Path] = None
        self.confidence_var = ctk.DoubleVar(value=0.25)
        self.stride_var = ctk.IntVar(value=1)
        self.ai_engine_ready = False

        # Model Memory Cache (Eliminates repeated cold-start delays)
        self.cached_model = None
        self.cached_model_path: Optional[Path] = None
        self.is_preloading = False

        self.view_inference: Optional[ctk.CTkScrollableFrame] = None
        self.view_settings: Optional[ctk.CTkScrollableFrame] = None

        self.inference_thread: Optional[threading.Thread] = None
        self.stop_requested = threading.Event()
        self.is_inferencing = False
        self.last_output_file: Optional[Path] = None

        self.after(15, self._load_full_ui)

    def _center_window(self, width: int, height: int):
        try:
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            x = max((screen_width - width) // 2, 0)
            y = max((screen_height - height) // 2, 0)
            self.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            pass

    def _load_full_ui(self):
        """Populate sidebar and views smoothly while the window is already visible."""
        self.splash_frame.destroy()

        self._setup_layout()
        self._build_sidebar()
        self.show_view("inference")
        self.refresh_models()
        self.refresh_samples()

        # Warm up AI Engine in background
        threading.Thread(target=self._async_warmup_engine, daemon=True).start()

    def _async_warmup_engine(self):
        """Warm up PyTorch, CUDA, and OpenCV in background."""
        global torch, YOLO, cv2
        try:
            import cv2 as _cv2
            import torch as _torch
            from ultralytics import YOLO as _YOLO
            cv2 = _cv2
            torch = _torch
            YOLO = _YOLO

            devices = self._scan_devices()
            default_dev = devices[0] if devices else "cpu"

            def on_ready():
                self.ai_engine_ready = True
                if hasattr(self, "device_dropdown"):
                    self.device_dropdown.configure(values=devices)
                self.active_device.set(default_dev)
                cuda_ok = torch.cuda.is_available()
                gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
                self.lbl_sidebar_status.configure(
                    text=f"PyTorch {torch.__version__}\n{'⚡ CUDA: ' + gpu_name if cuda_ok else '🖥 CPU Mode'}",
                    text_color="#81C784" if cuda_ok else "gray70"
                )
                self._log(f"AI Engine initialized ({'CUDA GPU' if cuda_ok else 'CPU'}).")

            self.after(0, on_ready)

        except Exception as e:
            def on_fail(err=str(e)):
                self.lbl_sidebar_status.configure(text=f"AI Error:\n{err[:20]}...", text_color="#EF5350")
                self._log(f"Engine init failed: {err}")
            self.after(0, on_fail)

    def _scan_devices(self) -> List[str]:
        global torch
        devices = []
        if torch is not None and torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                devices.append(f"cuda:{i} ({name})")
        devices.append("cpu")
        return devices

    def _setup_layout(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

    def _build_sidebar(self):
        title_label = ctk.CTkLabel(self.sidebar_frame, text="⚔️ CR Studio", font=ctk.CTkFont(size=22, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(25, 5))

        subtitle_label = ctk.CTkLabel(
            self.sidebar_frame, text="YOLO Detection Hub",
            font=ctk.CTkFont(size=12, slant="italic"), text_color="gray60"
        )
        subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 25))

        # Nav Buttons (Live Stream removed as requested)
        self.btn_nav_inference = ctk.CTkButton(
            self.sidebar_frame, text="📷 🎥  Photo / Video", anchor="w", height=44,
            font=ctk.CTkFont(size=14, weight="bold"), command=lambda: self.show_view("inference")
        )
        self.btn_nav_inference.grid(row=2, column=0, padx=15, pady=8, sticky="ew")

        self.btn_nav_settings = ctk.CTkButton(
            self.sidebar_frame, text="⚙️  Settings", anchor="w", height=44,
            font=ctk.CTkFont(size=14), fg_color="transparent", text_color=("gray10", "gray80"),
            hover_color=("gray70", "gray30"), command=lambda: self.show_view("settings")
        )
        self.btn_nav_settings.grid(row=3, column=0, padx=15, pady=8, sticky="ew")

        self.lbl_sidebar_status = ctk.CTkLabel(
            self.sidebar_frame, text="🔄 Warming AI Engine...",
            font=ctk.CTkFont(size=11), text_color="#FFB74D", justify="center"
        )
        self.lbl_sidebar_status.grid(row=5, column=0, padx=15, pady=15)

    def show_view(self, view_name: str):
        default_fg = "transparent"
        active_fg = ["#3B8ED0", "#1F6AA5"]

        self.btn_nav_inference.configure(fg_color=active_fg if view_name == "inference" else default_fg)
        self.btn_nav_settings.configure(fg_color=active_fg if view_name == "settings" else default_fg)

        for v in (self.view_inference, self.view_settings):
            if v is not None:
                v.grid_forget()

        if view_name == "inference":
            if self.view_inference is None:
                self._build_inference_view()
            self.view_inference.grid(row=0, column=0, sticky="nsew")

        elif view_name == "settings":
            if self.view_settings is None:
                self._build_settings_view()
            self.refresh_settings_view()
            self.view_settings.grid(row=0, column=0, sticky="nsew")

    def _build_inference_view(self):
        self.view_inference = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        self.view_inference.grid_columnconfigure(0, weight=1)

        # Top Result folder bar
        top_result_bar = ctk.CTkFrame(self.view_inference, corner_radius=10, fg_color=("#E0E0E0", "#212121"))
        top_result_bar.grid(row=0, column=0, sticky="ew", pady=(0, 15), padx=5)
        top_result_bar.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            top_result_bar, text="📂 Result Folder:", font=ctk.CTkFont(size=14, weight="bold"), text_color="#90CAF9"
        ).grid(row=0, column=0, sticky="w", padx=(15, 10), pady=12)

        self.entry_result_path = ctk.CTkEntry(top_result_bar, font=ctk.CTkFont(family="Consolas", size=12))
        self.entry_result_path.insert(0, str(RESULT_DIR.resolve()))
        self.entry_result_path.configure(state="readonly")
        self.entry_result_path.grid(row=0, column=1, sticky="ew", padx=5, pady=12)

        btn_open_top = ctk.CTkButton(
            top_result_bar, text="📂 Open in File Explorer", font=ctk.CTkFont(size=13, weight="bold"),
            height=36, fg_color="#37474F", hover_color="#263238",
            command=lambda: open_in_explorer(self.last_output_file if self.last_output_file else RESULT_DIR)
        )
        btn_open_top.grid(row=0, column=2, padx=(10, 15), pady=12)

        # ========================================================
        # 1. Model Selection Card with Inspect & Preload
        # ========================================================
        model_card = ctk.CTkFrame(self.view_inference, corner_radius=10)
        model_card.grid(row=1, column=0, sticky="ew", pady=(0, 15), padx=5)
        model_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            model_card, text="1. Select YOLO Model (from models/)", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, columnspan=5, sticky="w", padx=15, pady=(12, 8))

        ctk.CTkLabel(model_card, text="Model File:").grid(row=1, column=0, sticky="w", padx=15, pady=8)
        self.model_dropdown = ctk.CTkOptionMenu(
            model_card, variable=self.selected_model_name, values=["(No models found)"],
            command=self._on_model_selected
        )
        self.model_dropdown.grid(row=1, column=1, sticky="ew", padx=10, pady=8)

        btn_refresh_models = ctk.CTkButton(model_card, text="🔄 Refresh", width=85, command=self.refresh_models)
        btn_refresh_models.grid(row=1, column=2, padx=(5, 5), pady=8)

        self.btn_inspect_model = ctk.CTkButton(
            model_card, text="🔍 Inspect / Test", width=125, fg_color="#455A64", hover_color="#37474F",
            command=self.inspect_selected_model
        )
        self.btn_inspect_model.grid(row=1, column=3, padx=(5, 5), pady=8)

        self.btn_preload_model = ctk.CTkButton(
            model_card, text="⚡ Preload to VRAM", width=140, fg_color="#6A1B9A", hover_color="#4A148C",
            command=self.preload_model_to_vram
        )
        self.btn_preload_model.grid(row=1, column=4, padx=(5, 15), pady=8)

        self.lbl_model_info = ctk.CTkLabel(
            model_card, text="Select a model. Tip: Click '⚡ Preload to VRAM' to eliminate the initial 15s delay!",
            font=ctk.CTkFont(size=12), text_color="gray60"
        )
        self.lbl_model_info.grid(row=2, column=0, columnspan=5, sticky="w", padx=15, pady=(0, 12))

        # ========================================================
        # 2. Input Media Queue Card
        # ========================================================
        source_card = ctk.CTkFrame(self.view_inference, corner_radius=10)
        source_card.grid(row=2, column=0, sticky="ew", pady=(0, 15), padx=5)
        source_card.grid_columnconfigure(0, weight=1)

        # Header with title and count badge
        header_frame = ctk.CTkFrame(source_card, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(12, 6))
        header_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header_frame, text="2. Input Media Queue", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, sticky="w")

        self.lbl_queue_count = ctk.CTkLabel(
            header_frame, text="Queued: 0 files", font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#90CAF9"
        )
        self.lbl_queue_count.grid(row=0, column=1, sticky="e")

        # Top Control Bar (Samples dropdown + Add Sample + Add File(s) + Add Folder + Clear All)
        controls_frame = ctk.CTkFrame(source_card, fg_color="transparent")
        controls_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 8))
        controls_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(controls_frame, text="Sample:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.samples_dropdown = ctk.CTkOptionMenu(controls_frame, values=["(No samples found)"])
        self.samples_dropdown.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        btn_add_sample = ctk.CTkButton(
            controls_frame, text="➕ Add Sample", width=110, fg_color="#1976D2", hover_color="#1565C0",
            command=self.add_sample_to_queue
        )
        btn_add_sample.grid(row=0, column=2, padx=(0, 8))

        btn_browse_add = ctk.CTkButton(
            controls_frame, text="➕ Add File(s)", width=125, fg_color="#2E7D32", hover_color="#1B5E20",
            font=ctk.CTkFont(weight="bold"),
            command=self.browse_add_files
        )
        btn_browse_add.grid(row=0, column=3, padx=(0, 8))

        btn_browse_folder = ctk.CTkButton(
            controls_frame, text="📁 Add Folder", width=125, fg_color="#E65100", hover_color="#BF360C",
            font=ctk.CTkFont(weight="bold"),
            command=self.browse_add_folder
        )
        btn_browse_folder.grid(row=0, column=4, padx=(0, 8))

        btn_clear_queue = ctk.CTkButton(
            controls_frame, text="🗑️ Clear All", width=90, fg_color="#546E7A", hover_color="#37474F",
            command=self.clear_queue
        )
        btn_clear_queue.grid(row=0, column=5, padx=(0, 0))

        # Dynamic Notification & Tip Banner
        self.notification_frame = ctk.CTkFrame(source_card, fg_color=("#E3F2FD", "#1A2733"), corner_radius=6)
        self.notification_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 8))
        self.lbl_queue_notification = ctk.CTkLabel(
            self.notification_frame,
            text="💡 Tip: Click '➕ Add File(s)' or '📁 Add Folder' to queue media. You can add from multiple locations without losing items!",
            font=ctk.CTkFont(size=12),
            text_color="#90CAF9",
            justify="left"
        )
        self.lbl_queue_notification.pack(anchor="w", padx=10, pady=6)

        # Scrollable list displaying queued files
        self.queue_scroll_frame = ctk.CTkScrollableFrame(source_card, height=150, corner_radius=8, fg_color=("#F5F5F5", "#1E1E1E"))
        self.queue_scroll_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 8))
        self.queue_scroll_frame.grid_columnconfigure(0, weight=1)

        # Output rule notice footer
        ctk.CTkLabel(
            source_card,
            text="ℹ️ Every result is saved flat into 'result/' (e.g. filename_annotated.mp4). Files with identical names from different folders will be safely distinguished by their folder name.",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
            justify="left"
        ).grid(row=4, column=0, sticky="w", padx=15, pady=(0, 12))

        # ========================================================
        # 3. Parameters & Action Card
        # ========================================================
        action_card = ctk.CTkFrame(self.view_inference, corner_radius=10)
        action_card.grid(row=3, column=0, sticky="ew", pady=(0, 15), padx=5)
        action_card.grid_columnconfigure(1, weight=1)
        action_card.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(
            action_card, text="3. Inference Parameters & Actions", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=15, pady=(12, 8))

        self.lbl_conf = ctk.CTkLabel(action_card, text=f"Confidence: {self.confidence_var.get():.2f}")
        self.lbl_conf.grid(row=1, column=0, sticky="w", padx=15, pady=5)
        self.slider_conf = ctk.CTkSlider(
            action_card, from_=0.05, to=0.95, number_of_steps=18, variable=self.confidence_var,
            command=lambda val: self.lbl_conf.configure(text=f"Confidence: {val:.2f}")
        )
        self.slider_conf.grid(row=1, column=1, sticky="ew", padx=10, pady=5)

        # Video Stride row with interactive [i] Information Button
        stride_header_frame = ctk.CTkFrame(action_card, fg_color="transparent")
        stride_header_frame.grid(row=1, column=2, sticky="w", padx=(15, 5), pady=5)

        self.lbl_stride = ctk.CTkLabel(
            stride_header_frame,
            text=f"Video Stride: {self.stride_var.get()}",
            font=ctk.CTkFont(weight="bold")
        )
        self.lbl_stride.pack(side="left")

        btn_stride_info = ctk.CTkButton(
            stride_header_frame,
            text="ℹ",
            width=24,
            height=24,
            corner_radius=12,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E88E5",
            hover_color="#1565C0",
            command=self._show_stride_info_dialog
        )
        btn_stride_info.pack(side="left", padx=(8, 0))

        stride_slider_frame = ctk.CTkFrame(action_card, fg_color="transparent")
        stride_slider_frame.grid(row=1, column=3, sticky="ew", padx=10, pady=5)
        stride_slider_frame.grid_columnconfigure(0, weight=1)

        self.slider_stride = ctk.CTkSlider(
            stride_slider_frame, from_=1, to=10, number_of_steps=9, variable=self.stride_var,
            command=self._on_stride_changed
        )
        self.slider_stride.grid(row=0, column=0, sticky="ew")

        self.lbl_stride_desc = ctk.CTkLabel(
            stride_slider_frame,
            text="Frame Skip: 1x (every frame, normal & smoothest)",
            font=ctk.CTkFont(size=11),
            text_color="#90CAF9"
        )
        self.lbl_stride_desc.grid(row=1, column=0, sticky="w", pady=(2, 0))

        buttons_frame = ctk.CTkFrame(action_card, fg_color="transparent")
        buttons_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=15, pady=(15, 15))
        buttons_frame.grid_columnconfigure((0, 1), weight=1)

        self.btn_run_inference = ctk.CTkButton(
            buttons_frame, text="▶  Start Inference", height=42, font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#1565C0", hover_color="#0D47A1", command=self.start_inference
        )
        self.btn_run_inference.grid(row=0, column=0, padx=8, sticky="ew")

        self.btn_stop_inference = ctk.CTkButton(
            buttons_frame, text="⏹  Stop", height=42, font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#C62828", hover_color="#8E0000", state="disabled", command=self.stop_inference
        )
        self.btn_stop_inference.grid(row=0, column=1, padx=8, sticky="ew")

        # ========================================================
        # 4. Progress & Speed Metrics Card
        # ========================================================
        metrics_card = ctk.CTkFrame(self.view_inference, corner_radius=10)
        metrics_card.grid(row=4, column=0, sticky="ew", pady=(0, 15), padx=5)
        metrics_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            metrics_card, text="4. Inference Progress & Live Speed", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(12, 6))

        self.progress_bar = ctk.CTkProgressBar(metrics_card, height=18)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=15, pady=(5, 10))
        self.progress_bar.set(0.0)

        metrics_row = ctk.CTkFrame(metrics_card, fg_color="transparent")
        metrics_row.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 12))
        metrics_row.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.lbl_stat_fps = ctk.CTkLabel(metrics_row, text="⚡ Speed:\n-- FPS", font=ctk.CTkFont(size=13, weight="bold"), text_color="#64B5F6")
        self.lbl_stat_fps.grid(row=0, column=0, sticky="ew", padx=5)

        self.lbl_stat_progress = ctk.CTkLabel(metrics_row, text="📊 Progress:\n0 / 0 frames (0%)", font=ctk.CTkFont(size=13), text_color="gray85")
        self.lbl_stat_progress.grid(row=0, column=1, sticky="ew", padx=5)

        self.lbl_stat_time = ctk.CTkLabel(metrics_row, text="⏱ Time:\nElapsed: 00:00 | ETA: --", font=ctk.CTkFont(size=13), text_color="gray85")
        self.lbl_stat_time.grid(row=0, column=2, sticky="ew", padx=5)

        self.lbl_stat_detections = ctk.CTkLabel(metrics_row, text="🎯 Detections:\n0 objects", font=ctk.CTkFont(size=13), text_color="#81C784")
        self.lbl_stat_detections.grid(row=0, column=3, sticky="ew", padx=5)

        self.log_box = ctk.CTkTextbox(metrics_card, height=120, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_box.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 15))
        self.log_box.insert("end", "[Ready] App initialized. Select a model and sample/folder.\n")

    def _build_settings_view(self):
        self.view_settings = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        self.view_settings.grid_columnconfigure(0, weight=1)

        card = ctk.CTkFrame(self.view_settings, corner_radius=10)
        card.grid(row=0, column=0, sticky="ew", padx=5, pady=10)
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(card, text="Hardware Compute Device & Settings", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(15, 15))

        ctk.CTkLabel(card, text="Active Device:").grid(row=1, column=0, sticky="w", padx=15, pady=10)
        self.device_dropdown = ctk.CTkOptionMenu(card, variable=self.active_device, values=["(Scanning...)"])
        self.device_dropdown.grid(row=1, column=1, sticky="ew", padx=15, pady=10)

        btn_scan_devices = ctk.CTkButton(card, text="🔍 Re-scan Compute Hardware", command=self.rescan_hardware)
        btn_scan_devices.grid(row=2, column=1, sticky="w", padx=15, pady=5)

        ctk.CTkLabel(card, text="System Hardware Info:").grid(row=3, column=0, sticky="nw", padx=15, pady=15)
        self.hardware_info_box = ctk.CTkTextbox(card, height=160, font=ctk.CTkFont(family="Consolas", size=12))
        self.hardware_info_box.grid(row=3, column=1, sticky="ew", padx=15, pady=15)

    def refresh_settings_view(self):
        if not hasattr(self, "device_dropdown"):
            return
        devices = self._scan_devices()
        self.device_dropdown.configure(values=devices if devices else ["cpu"])

        global torch
        if torch is not None:
            info_lines = [
                f"PyTorch Version: {torch.__version__}",
                f"CUDA Available: {torch.cuda.is_available()}",
            ]
            if torch.cuda.is_available():
                info_lines.append(f"Device Count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    info_lines.append(f"  [GPU {i}]: {torch.cuda.get_device_name(i)}")
            else:
                info_lines.append("Running on CPU mode.")
        else:
            info_lines = ["AI Engine still warming up in background..."]

        self.hardware_info_box.delete("1.0", "end")
        self.hardware_info_box.insert("end", "\n".join(info_lines))

    def rescan_hardware(self):
        devices = self._scan_devices()
        self.device_dropdown.configure(values=devices)
        if devices and self.active_device.get() not in devices:
            self.active_device.set(devices[0])
        self.refresh_settings_view()
        messagebox.showinfo("Hardware Scan", f"Detected {len(devices)} device option(s).")

    # ----------------- Model & Sample Logic -----------------
    def refresh_models(self):
        model_files = sorted([f.name for f in MODELS_DIR.glob("*.pt")])
        if not model_files:
            root_models = sorted([f.name for f in BASE_DIR.glob("*.pt")])
            model_files.extend(root_models)

        if hasattr(self, "model_dropdown"):
            if model_files:
                self.model_dropdown.configure(values=model_files)
                if self.selected_model_name.get() not in model_files:
                    self.selected_model_name.set(model_files[0])
                self._update_model_info_text()
            else:
                self.model_dropdown.configure(values=["(No models in models/)"])
                self.selected_model_name.set("")
                self.lbl_model_info.configure(text="⚠️ Please put your YOLO .pt model inside 'models/' folder.")

    def _on_model_selected(self, choice: str):
        # Reset cache if a different model is picked
        if self.cached_model_path and self.cached_model_path.name != choice:
            self.cached_model = None
            self.cached_model_path = None
        self._update_model_info_text()

    def _update_model_info_text(self):
        name = self.selected_model_name.get()
        if not name or name.startswith("("):
            return

        model_path = MODELS_DIR / name if (MODELS_DIR / name).exists() else BASE_DIR / name
        size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0

        is_cached = (self.cached_model is not None and self.cached_model_path == model_path)
        if is_cached:
            status = "⚡ [Ready in VRAM - Zero Delay]"
            color = "#81C784"
        else:
            status = "💤 [Not Preloaded - 1st run will take ~10-15s to warm up]"
            color = "#FFB74D"

        self.lbl_model_info.configure(
            text=f"{name} ({size_mb:.1f} MB) | {status}",
            text_color=color
        )

    def _locate_model_file(self) -> Optional[Path]:
        name = self.selected_model_name.get()
        if not name or name.startswith("("):
            return None
        model_path = MODELS_DIR / name
        if not model_path.exists():
            model_path = BASE_DIR / name
        return model_path if model_path.exists() else None

    # ----------------- Feature 1: Inspect / Test Model -----------------
    def inspect_selected_model(self):
        """Loads and inspects the model architecture, labels count, and class names."""
        global YOLO
        model_path = self._locate_model_file()
        if not model_path:
            messagebox.showwarning("Select Model", "Please select a valid model from the dropdown first.")
            return

        if not self.ai_engine_ready or YOLO is None:
            messagebox.showinfo("Please Wait", "AI Engine is warming up in background. Please wait 1-2 seconds.")
            return

        self._log(f"Inspecting model '{model_path.name}'...")

        try:
            # Use cached model if available to make inspection instantaneous
            if self.cached_model is not None and self.cached_model_path == model_path:
                model = self.cached_model
            else:
                model = YOLO(str(model_path.resolve()))

            names: Dict[int, str] = model.names if hasattr(model, "names") else {}
            num_classes = len(names)
            size_mb = model_path.stat().st_size / (1024 * 1024)
            task_type = getattr(model, "task", "detect")

            # Create Inspection Modal Window
            modal = ctk.CTkToplevel(self)
            modal.title(f"Model Inspector: {model_path.name}")
            modal.geometry("540x520")
            modal.minsize(480, 420)
            modal.grab_set()

            # Center modal relative to app
            x = self.winfo_x() + 100
            y = self.winfo_y() + 80
            modal.geometry(f"+{x}+{y}")

            header_frame = ctk.CTkFrame(modal, corner_radius=10)
            header_frame.pack(fill="x", padx=15, pady=(15, 10))

            ctk.CTkLabel(
                header_frame,
                text=f"🔬 {model_path.name}",
                font=ctk.CTkFont(size=18, weight="bold")
            ).pack(anchor="w", padx=15, pady=(12, 2))

            ctk.CTkLabel(
                header_frame,
                text=f"File Size: {size_mb:.2f} MB  •  Task: {task_type.capitalize()}  •  Classes: {num_classes}",
                font=ctk.CTkFont(size=12),
                text_color="#90CAF9"
            ).pack(anchor="w", padx=15, pady=(0, 12))

            # Class list box
            ctk.CTkLabel(
                modal,
                text=f"Detected Labels ({num_classes} classes):",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=15, pady=(5, 5))

            textbox = ctk.CTkTextbox(modal, font=ctk.CTkFont(family="Consolas", size=12))
            textbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))

            lines = []
            for class_id, label_name in sorted(names.items()):
                lines.append(f"[{class_id:02d}] : {label_name}")

            if not lines:
                lines.append("(No class names metadata found in model weights)")

            textbox.insert("end", "\n".join(lines))
            textbox.configure(state="disabled")

            btn_close = ctk.CTkButton(modal, text="Close", width=100, command=modal.destroy)
            btn_close.pack(pady=(0, 15))

            self._log(f"Model '{model_path.name}' inspected: {num_classes} classes found.")

        except Exception as e:
            messagebox.showerror("Inspection Error", f"Failed to inspect model:\n{e}")
            self._log(f"Inspection failed: {e}")

    # ----------------- Feature 2: Preload to VRAM -----------------
    def preload_model_to_vram(self):
        """Loads model into GPU VRAM and performs a 1-frame warmup pass."""
        global YOLO, torch
        model_path = self._locate_model_file()
        if not model_path:
            messagebox.showwarning("Select Model", "Please select a valid model from the dropdown first.")
            return

        if not self.ai_engine_ready or YOLO is None:
            messagebox.showinfo("Please Wait", "AI Engine is warming up in background. Please wait 1-2 seconds.")
            return

        if self.is_preloading or self.is_inferencing:
            messagebox.showinfo("Busy", "Model operation or inference currently in progress.")
            return

        self.is_preloading = True
        self.btn_preload_model.configure(state="disabled", text="⏳ Preloading...")
        self.lbl_model_info.configure(text="⚡ Preloading weights to VRAM and warming up CUDA kernels...", text_color="#FFB74D")
        self._log(f"Preloading '{model_path.name}' into VRAM with warmup pass...")

        def worker():
            device = self._get_device_identifier()
            try:
                t0 = time.time()
                model = YOLO(str(model_path.resolve()))

                # Run a fast 1-frame dummy warmup pass (eliminates the 15-second cold-start delay)
                import numpy as np
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                model.predict(source=dummy_frame, device=device, verbose=False)

                load_time = time.time() - t0

                def on_success():
                    self.cached_model = model
                    self.cached_model_path = model_path
                    self.is_preloading = False
                    self.btn_preload_model.configure(state="normal", text="✓ Ready in VRAM")
                    self._update_model_info_text()
                    self._log(f"Model preloaded & warmed up in {load_time:.2f}s! Videos will start instantly.")
                    messagebox.showinfo(
                        "Preload Complete",
                        f"Model '{model_path.name}' is now preloaded into VRAM.\n\n"
                        f"Target Device: {device}\nWarmup Time: {load_time:.2f}s\n\n"
                        "Video inferences will now start immediately with zero delay!"
                    )

                self.after(0, on_success)

            except Exception as e:
                def on_error(err=str(e)):
                    self.is_preloading = False
                    self.btn_preload_model.configure(state="normal", text="⚡ Preload to VRAM")
                    self._update_model_info_text()
                    self._log(f"Preload error: {err}")
                    messagebox.showerror("Preload Error", f"Failed to preload model:\n{err}")

                self.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    def refresh_samples(self):
        sample_files = []
        for f in SAMPLES_DIR.iterdir():
            if f.suffix.lower() in IMAGE_EXTENSIONS or f.suffix.lower() in VIDEO_EXTENSIONS:
                sample_files.append(f.name)
        sample_files.sort()

        if hasattr(self, "samples_dropdown"):
            if sample_files:
                self.samples_dropdown.configure(values=sample_files)
                if not self.selected_files:
                    first_sample = SAMPLES_DIR / sample_files[0]
                    self.selected_files = [first_sample]
                    self.samples_dropdown.set(sample_files[0])
                    self._render_queue_list()
            else:
                self.samples_dropdown.configure(values=["(No files in samples/)"])

    def _notify_queue(self, message: str, level: str = "info"):
        """Displays notifications, warnings, and confirmations for the media queue."""
        if not hasattr(self, "lbl_queue_notification"):
            return

        color_map = {
            "info": ("#1565C0", "#90CAF9"),
            "success": ("#2E7D32", "#81C784"),
            "warning": ("#E65100", "#FFB74D"),
            "error": ("#C62828", "#EF5350")
        }
        bg_map = {
            "info": ("#E3F2FD", "#1A2733"),
            "success": ("#E8F5E9", "#1B2E1D"),
            "warning": ("#FFF8E1", "#332711"),
            "error": ("#FFEBEE", "#381717")
        }
        text_color = color_map.get(level, color_map["info"])
        bg_color = bg_map.get(level, bg_map["info"])

        self.notification_frame.configure(fg_color=bg_color)
        self.lbl_queue_notification.configure(text=message, text_color=text_color)
        self._log(f"[Queue {level.upper()}] {message}")

    def browse_add_files(self):
        filetypes = [
            ("Supported Media", "*.mp4 *.avi *.mov *.mkv *.webm *.jpg *.jpeg *.png *.bmp *.webp"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All Files", "*.*")
        ]
        chosen = filedialog.askopenfilenames(
            title="Add Media File(s) to Queue (Hold Ctrl / Shift to Multi-Select)",
            initialdir=str(SAMPLES_DIR.resolve()),
            filetypes=filetypes
        )
        if not chosen:
            return

        added_count = 0
        duplicate_count = 0
        invalid_count = 0
        existing_resolved = {f.resolve() for f in self.selected_files}

        for path_str in chosen:
            p = Path(path_str)
            if not p.exists() or not p.is_file():
                invalid_count += 1
                continue

            if p.suffix.lower() not in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS):
                invalid_count += 1
                continue

            if p.resolve() in existing_resolved:
                duplicate_count += 1
                continue

            self.selected_files.append(p)
            existing_resolved.add(p.resolve())
            added_count += 1

        self._render_queue_list()

        # Build feedback message
        if added_count > 0 and duplicate_count == 0 and invalid_count == 0:
            self._notify_queue(f"✓ Added {added_count} file(s) to queue. Total: {len(self.selected_files)}", "success")
        elif duplicate_count > 0 and added_count > 0:
            self._notify_queue(f"✓ Added {added_count} file(s). ⚠️ Skipped {duplicate_count} duplicate file(s) already in queue.", "warning")
        elif duplicate_count > 0 and added_count == 0:
            self._notify_queue("⚠️ Skipped: Selected file(s) are already in the queue!", "warning")
        elif invalid_count > 0 and added_count == 0:
            self._notify_queue("⚠️ Cannot add: Selected file(s) have wrong path or unsupported format.", "error")
        elif invalid_count > 0 and added_count > 0:
            self._notify_queue(f"✓ Added {added_count} file(s). ⚠️ Skipped {invalid_count} invalid or unsupported file(s).", "warning")

    def browse_add_folder(self):
        chosen = filedialog.askdirectory(
            title="Select Folder to Add All Media Inside (Recursively into Queue)",
            initialdir=str(SAMPLES_DIR.resolve())
        )
        if not chosen:
            return

        folder = Path(chosen)
        if not folder.exists() or not folder.is_dir():
            self._notify_queue(f"⚠️ Cannot add folder: '{folder.name}' does not exist or is not a valid folder.", "error")
            return

        # Scan folder AND all subfolders recursively for supported media
        media_files = sorted([
            f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
        ])

        if not media_files:
            self._notify_queue(f"⚠️ Folder '{folder.name}' contains no supported video or image files.", "warning")
            return

        added_count = 0
        duplicate_count = 0
        existing_resolved = {f.resolve() for f in self.selected_files}

        for p in media_files:
            if p.resolve() in existing_resolved:
                duplicate_count += 1
                continue
            self.selected_files.append(p)
            existing_resolved.add(p.resolve())
            added_count += 1

        self._render_queue_list()

        if added_count > 0 and duplicate_count == 0:
            self._notify_queue(f"✓ Added {added_count} file(s) from folder '{folder.name}'. Total: {len(self.selected_files)}", "success")
        elif added_count > 0 and duplicate_count > 0:
            self._notify_queue(f"✓ Added {added_count} file(s) from '{folder.name}'. ⚠️ Skipped {duplicate_count} duplicate(s) already in queue.", "warning")
        elif added_count == 0 and duplicate_count > 0:
            self._notify_queue(f"⚠️ All {duplicate_count} file(s) in folder '{folder.name}' are already in the queue!", "warning")

    def add_sample_to_queue(self):
        choice = self.samples_dropdown.get()
        if not choice or choice.startswith("("):
            self._notify_queue("⚠️ No valid sample selected in dropdown.", "warning")
            return

        full_path = SAMPLES_DIR / choice
        if not full_path.exists():
            self._notify_queue(f"⚠️ Wrong path: Sample '{choice}' does not exist on disk.", "error")
            return

        if any(f.resolve() == full_path.resolve() for f in self.selected_files):
            self._notify_queue(f"⚠️ Sample '{choice}' is already in the queue!", "warning")
            return

        self.selected_files.append(full_path)
        self._notify_queue(f"✓ Added sample '{choice}' to queue. Total: {len(self.selected_files)}", "success")
        self._render_queue_list()

    def remove_file_from_queue(self, file_path: Path):
        target_res = file_path.resolve()
        initial_len = len(self.selected_files)
        self.selected_files = [f for f in self.selected_files if f.resolve() != target_res]
        if len(self.selected_files) < initial_len:
            self._notify_queue(f"Removed '{file_path.name}' from queue.", "info")
        self._render_queue_list()

    def clear_queue(self):
        if not self.selected_files:
            self._notify_queue("Queue is already empty.", "info")
            return
        count = len(self.selected_files)
        self.selected_files.clear()
        self._notify_queue(f"Cleared {count} file(s) from queue.", "info")
        self._render_queue_list()

    def _render_queue_list(self):
        """Rebuilds the scrollable queue visual items."""
        if not hasattr(self, "queue_scroll_frame"):
            return

        # Clear existing items
        for child in self.queue_scroll_frame.winfo_children():
            child.destroy()

        video_count = sum(1 for f in self.selected_files if f.suffix.lower() in VIDEO_EXTENSIONS)
        image_count = sum(1 for f in self.selected_files if f.suffix.lower() in IMAGE_EXTENSIONS)
        total_count = len(self.selected_files)

        self.lbl_queue_count.configure(
            text=f"Queued: {total_count} file(s) ({video_count} 🎬, {image_count} 🖼️)"
        )

        if not self.selected_files:
            placeholder = ctk.CTkLabel(
                self.queue_scroll_frame,
                text="📭 Queue is empty.\nClick [➕ Add File(s)] or [➕ Add Sample] above to add media for inference.",
                font=ctk.CTkFont(size=12),
                text_color="gray50",
                justify="center"
            )
            placeholder.pack(pady=35, fill="both", expand=True)
            return

        # Render each file row
        for idx, file_item in enumerate(self.selected_files):
            row_frame = ctk.CTkFrame(self.queue_scroll_frame, corner_radius=6, fg_color=("#E0E0E0", "#262626"))
            row_frame.pack(fill="x", pady=2, padx=4)
            row_frame.grid_columnconfigure(1, weight=1)

            is_vid = file_item.suffix.lower() in VIDEO_EXTENSIONS
            icon = "🎬" if is_vid else "🖼️"

            # Icon & index
            lbl_icon = ctk.CTkLabel(
                row_frame, text=f"{idx + 1}. {icon}", font=ctk.CTkFont(size=13, weight="bold"), width=48
            )
            lbl_icon.grid(row=0, column=0, rowspan=2, padx=(8, 4), pady=4)

            # File name
            lbl_name = ctk.CTkLabel(
                row_frame, text=file_item.name, font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
            )
            lbl_name.grid(row=0, column=1, sticky="w", padx=2, pady=(4, 0))

            # Folder & Size detail
            size_str = ""
            try:
                if file_item.exists():
                    size_mb = file_item.stat().st_size / (1024 * 1024)
                    size_str = f" • {size_mb:.1f} MB"
                else:
                    size_str = " • ⚠️ (File missing on disk / wrong path)"
            except Exception:
                pass

            parent_folder = file_item.parent.name or str(file_item.parent)
            res_str = str(file_item.resolve())
            if len(res_str) > 70:
                res_str = res_str[:35] + "..." + res_str[-32:]

            lbl_details = ctk.CTkLabel(
                row_frame,
                text=f"📂 {parent_folder}{size_str}   |   {res_str}",
                font=ctk.CTkFont(size=11),
                text_color="gray60",
                anchor="w"
            )
            lbl_details.grid(row=1, column=1, sticky="w", padx=2, pady=(0, 4))

            # Remove (Deselect) Button
            btn_remove = ctk.CTkButton(
                row_frame,
                text="✖",
                width=28,
                height=28,
                corner_radius=14,
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color="#C62828",
                hover_color="#8E0000",
                command=lambda target=file_item: self.remove_file_from_queue(target)
            )
            btn_remove.grid(row=0, column=2, rowspan=2, padx=(6, 8), pady=4)

    def _on_stride_changed(self, val):
        stride = int(val)
        self.lbl_stride.configure(text=f"Video Stride: {stride}")
        if stride == 1:
            desc = "Frame Skip: 1x (every frame, normal & smoothest)"
        elif stride == 2:
            desc = "Frame Skip: 2x faster (processes every 2nd frame)"
        elif stride <= 4:
            desc = f"Frame Skip: {stride}x faster (processes every {stride}th frame)"
        else:
            desc = f"Frame Skip: {stride}x speed boost (quick preview mode)"
        self.lbl_stride_desc.configure(text=desc)

    def _show_stride_info_dialog(self):
        messagebox.showinfo(
            "What is Video Stride?",
            "🎬 VIDEO STRIDE = FRAME SKIPPING / SPEED MULTIPLIER\n\n"
            "Gameplay videos have 30 or 60 frames every second (thousands of images).\n\n"
            "• Stride = 1 (Default / Recommended for final videos):\n"
            "  Processes EVERY single frame. 100% smooth detection output.\n\n"
            "• Stride = 2 (2x Faster):\n"
            "  Processes every 2nd frame (Frames 1, 3, 5...). Cuts video processing\n"
            "  time in half with barely noticeable difference in tracking.\n\n"
            "• Stride = 3 to 10 (Fast Preview):\n"
            "  Skips more frames to finish long recordings in a fraction of the time.\n\n"
            "💡 Recommendation: Leave at 1 for final results, or 2 for faster testing."
        )

    def _log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        if hasattr(self, "log_box"):
            self.log_box.insert("end", f"[{timestamp}] {message}\n")
            self.log_box.see("end")

    def _get_device_identifier(self) -> str:
        dev_str = self.active_device.get()
        if "cuda" in dev_str.lower():
            if "cuda:" in dev_str:
                return dev_str.split()[0]
            return "0"
        return "cpu"

    def start_inference(self):
        global YOLO
        if not self.ai_engine_ready or YOLO is None:
            messagebox.showinfo("Please Wait", "AI Engine is warming up in background. Please wait 1-2 seconds.")
            return

        model_path = self._locate_model_file()
        if not model_path:
            messagebox.showwarning("Model Required", "Please select a valid YOLO model from the dropdown.")
            return

        if not self.selected_files:
            messagebox.showwarning(
                "Queue Empty",
                "Please add at least one media file to the queue before starting inference.\n\nClick [➕ Add File(s)] or [➕ Add Sample]."
            )
            return

        valid_files = [f for f in self.selected_files if f.exists() and f.is_file()]
        if not valid_files:
            messagebox.showwarning(
                "Wrong Path / Missing Files",
                "None of the files in the queue could be found on disk.\nPlease check your file paths and add valid media files."
            )
            return

        if len(valid_files) < len(self.selected_files):
            missing_count = len(self.selected_files) - len(valid_files)
            self._notify_queue(f"⚠️ Skipped {missing_count} missing file(s) (wrong path or removed from disk).", "warning")

        self.is_inferencing = True
        self.stop_requested.clear()
        self.btn_run_inference.configure(state="disabled")
        self.btn_stop_inference.configure(state="normal")
        self.progress_bar.set(0.0)

        self.inference_thread = threading.Thread(
            target=self._run_inference_worker,
            args=(model_path, valid_files, self.selected_root_folder),
            daemon=True
        )
        self.inference_thread.start()

    def stop_inference(self):
        if self.is_inferencing:
            self._log("Stopping inference... please wait for current step to release.")
            self.stop_requested.set()
            self.btn_stop_inference.configure(state="disabled")

    def _run_inference_worker(self, model_path: Path, media_files: List[Path], root_folder: Optional[Path] = None):
        conf = float(self.confidence_var.get())
        stride = int(self.stride_var.get())
        device = self._get_device_identifier()

        # Check if we can reuse the preloaded/cached model instance
        if self.cached_model is not None and self.cached_model_path == model_path:
            model = self.cached_model
            self.after(0, lambda: self._log(f"Using preloaded VRAM model: {model_path.name}"))
        else:
            self.after(0, lambda: self._log(f"Loading YOLO model: {model_path.name}..."))
            try:
                model = YOLO(str(model_path.resolve()))
                self.cached_model = model
                self.cached_model_path = model_path
                self.after(0, self._update_model_info_text)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Model Load Error", f"Failed to load YOLO model:\n{e}"))
                self.after(0, self._on_inference_finished, False, None)
                return

        total_files = len(media_files)
        self.after(0, lambda: self._log(f"Device: {device} | Conf: {conf:.2f} | Stride: {stride}"))
        self.after(0, lambda: self._log(f"Starting inference on {total_files} file(s)... All outputs -> result/"))

        last_out = None
        if total_files > 1:
            for idx, file_item in enumerate(media_files, start=1):
                if self.stop_requested.is_set():
                    self.after(0, lambda: self._log("Batch inference cancelled by user."))
                    break

                self.after(0, lambda idx=idx, name=file_item.name: self._log(f"[{idx}/{total_files}] Processing '{name}'..."))
                ext = file_item.suffix.lower()
                is_video = ext in VIDEO_EXTENSIONS
                out_ext = ".mp4" if is_video else ext
                # Include folder name if in subfolder or if identical names from different folders exist
                file_stem = get_output_stem(file_item, root_folder, all_queue_files=media_files)
                out_file = get_unique_output_path(RESULT_DIR, file_stem, out_ext)
                last_out = out_file

                if is_video:
                    self._process_single_video(model, file_item, out_file, conf, stride, device, batch_info=(idx, total_files))
                else:
                    self._process_single_image(model, file_item, out_file, conf, device, batch_info=(idx, total_files))

            completed = not self.stop_requested.is_set()
            self.last_output_file = last_out
            self.after(0, self._on_inference_finished, completed, last_out if completed else None)

        else:
            file_item = media_files[0]
            ext = file_item.suffix.lower()
            is_video = ext in VIDEO_EXTENSIONS
            out_ext = ".mp4" if is_video else ext
            file_stem = get_output_stem(file_item, root_folder, all_queue_files=media_files)
            out_file = get_unique_output_path(RESULT_DIR, file_stem, out_ext)
            self.last_output_file = out_file

            if is_video:
                self._process_single_video(model, file_item, out_file, conf, stride, device)
            else:
                self._process_single_image(model, file_item, out_file, conf, device)

            completed = not self.stop_requested.is_set()
            self.after(0, self._on_inference_finished, completed, out_file if completed else None)

    def _process_single_image(self, model, source_path: Path, out_file: Path, conf: float, device: str, batch_info=None):
        global cv2
        start_time = time.time()
        try:
            results = model.predict(source=str(source_path.resolve()), conf=conf, device=device, verbose=False)
            elapsed = time.time() - start_time
            fps = 1.0 / max(elapsed, 0.001)

            if results and len(results) > 0:
                res = results[0]
                num_detections = len(res.boxes) if res.boxes is not None else 0

                annotated_img = res.plot()
                cv2.imwrite(str(out_file.resolve()), annotated_img)

                if batch_info:
                    idx, total = batch_info
                    progress_pct = idx / total
                    self.after(0, lambda: self.progress_bar.set(progress_pct))
                    self.after(0, lambda: self.lbl_stat_progress.configure(
                        text=f"📊 Batch Progress:\n{idx} / {total} files ({int(progress_pct * 100)}%)"
                    ))
                else:
                    self.after(0, lambda: self.progress_bar.set(1.0))
                    self.after(0, lambda: self.lbl_stat_progress.configure(text="📊 Progress:\n1 / 1 image (100%)"))

                self.after(0, lambda: self.lbl_stat_fps.configure(text=f"⚡ Speed:\n{fps:.1f} FPS"))
                self.after(0, lambda: self.lbl_stat_time.configure(text=f"⏱ Time:\nElapsed: {elapsed:.2f}s | Done"))
                self.after(0, lambda: self.lbl_stat_detections.configure(text=f"🎯 Detections:\n{num_detections} objects"))
                self.after(0, lambda: self._log(f"Saved: {out_file.name} ({num_detections} detections)"))

        except Exception as e:
            self.after(0, lambda: self._log(f"Error processing image {source_path.name}: {e}"))

    def _process_single_video(self, model, source_path: Path, out_file: Path, conf: float, stride: int, device: str, batch_info=None):
        global cv2
        cap = cv2.VideoCapture(str(source_path.resolve()))
        if not cap.isOpened():
            self.after(0, lambda: self._log(f"Cannot open video source: {source_path.name}"))
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        effective_total = total_frames // max(stride, 1)

        out_fps = orig_fps / stride
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file.resolve()), fourcc, out_fps, (width, height))

        frame_idx = 0
        processed_count = 0
        total_detections_count = 0
        start_time = time.time()
        fps_tracker_time = time.time()
        fps_tracker_count = 0
        current_fps = 0.0

        try:
            while cap.isOpened():
                if self.stop_requested.is_set():
                    self.after(0, lambda: self._log(f"Stopped processing '{source_path.name}'."))
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % stride == 0:
                    res = model.predict(source=frame, conf=conf, device=device, verbose=False)
                    if res and len(res) > 0:
                        annotated = res[0].plot()
                        writer.write(annotated)
                        boxes = res[0].boxes
                        if boxes is not None:
                            total_detections_count += len(boxes)
                    else:
                        writer.write(frame)

                    processed_count += 1
                    fps_tracker_count += 1

                    now = time.time()
                    time_diff = now - fps_tracker_time
                    if time_diff >= 0.5:
                        current_fps = fps_tracker_count / time_diff
                        fps_tracker_time = now
                        fps_tracker_count = 0

                        file_progress = processed_count / max(effective_total, 1)
                        if batch_info:
                            b_idx, b_total = batch_info
                            overall_progress = ((b_idx - 1) + file_progress) / b_total
                        else:
                            overall_progress = file_progress

                        elapsed = now - start_time
                        eta_seconds = (elapsed / max(file_progress, 0.001)) - elapsed if file_progress < 1.0 else 0

                        elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
                        eta_str = time.strftime("%M:%S", time.gmtime(max(eta_seconds, 0)))

                        progress_text = (
                            f"📊 File {batch_info[0]}/{batch_info[1]}:\n{processed_count}/{effective_total} frames"
                            if batch_info else
                            f"📊 Progress:\n{processed_count} / {effective_total} frames ({int(file_progress * 100)}%)"
                        )

                        self._dispatch_metrics_update(
                            progress_ratio=overall_progress,
                            current_fps=current_fps,
                            progress_text=progress_text,
                            elapsed_str=elapsed_str,
                            eta_str=eta_str,
                            detections=total_detections_count
                        )

                frame_idx += 1

        finally:
            cap.release()
            writer.release()

        elapsed_total = time.time() - start_time
        self.after(0, lambda: self._log(f"Saved video to: {out_file.name} in {elapsed_total:.1f}s"))

    def _dispatch_metrics_update(self, progress_ratio: float, current_fps: float, progress_text: str,
                                elapsed_str: str, eta_str: str, detections: int):
        self.after(0, lambda: self.progress_bar.set(min(progress_ratio, 1.0)))
        self.after(0, lambda: self.lbl_stat_fps.configure(text=f"⚡ Speed:\n{current_fps:.1f} FPS"))
        self.after(0, lambda: self.lbl_stat_progress.configure(text=progress_text))
        self.after(0, lambda: self.lbl_stat_time.configure(text=f"⏱ Time:\nElapsed: {elapsed_str} | ETA: {eta_str}"))
        self.after(0, lambda: self.lbl_stat_detections.configure(text=f"🎯 Detections:\n{detections} objects"))

    def _on_inference_finished(self, success: bool, output_file: Optional[Path]):
        self.is_inferencing = False
        self.btn_run_inference.configure(state="normal")
        self.btn_stop_inference.configure(state="disabled")

        if success and output_file and output_file.exists():
            res = messagebox.askyesno(
                "Inference Complete!",
                f"Result saved successfully to:\n\n{output_file.resolve()}\n\nWould you like to open it in File Explorer now?"
            )
            if res:
                open_in_explorer(output_file)


if __name__ == "__main__":
    app = App()
    app.mainloop()
