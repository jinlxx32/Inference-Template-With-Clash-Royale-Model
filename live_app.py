"""
Clash Royale Live Studio - Direct Embedded Stream, YOLO Overlay & Card Tracker
Features:
- Headless Scrcpy Stream: No external floating window on your desktop
- Zero Occlusion: Other windows placed on top never block or disrupt the feed
- High-Performance Decoupled Pipeline (Always displays newest frame with 0ms delay buildup)
- Full Clash Royale 4-Card Cycle Tracker:
  * 🔴 Top Row: 4 Red Slots (In Cycle / Waiting to return)
  * 🟢 Bottom Row: 4 Green Slots (In Hand / Available for opponent to deploy now)
  * Real-time deployment debouncing (3.5s cooldown per card)
  * 204 official Clash Royale card portrait icons loaded in-memory
  * Live match deployment history feed with timestamps
- Dynamic YOLO Model Selector from models/ directory
- Live Confidence tuning slider (10% - 90%)
"""

import os
import re
import sys
import time
import threading
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Lazy imports
cv2 = None
np = None
torch = None
YOLO = None
av = None
win32pipe = None
win32file = None

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCRCPY_DIR = BASE_DIR / "tools" / "scrcpy"
SCRCPY_EXE = SCRCPY_DIR / "scrcpy.exe"
ADB_EXE = SCRCPY_DIR / "adb.exe"
CARDS_DIR = BASE_DIR / "assets" / "cards"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def import_dependencies():
    global cv2, np, torch, YOLO, av, win32pipe, win32file
    if cv2 is None:
        import cv2 as _cv2
        import numpy as _np
        import torch as _torch
        from ultralytics import YOLO as _YOLO
        import av as _av
        import win32pipe as _win32pipe
        import win32file as _win32file

        cv2 = _cv2
        np = _np
        torch = _torch
        YOLO = _YOLO
        av = _av
        win32pipe = _win32pipe
        win32file = _win32file


class NamedPipeStream:
    """Stream wrapper with non-seeking read support and thread-safe cancellation for PyAV."""
    def __init__(self, pipe_handle, stop_event: threading.Event):
        self.handle = pipe_handle
        self.stop_event = stop_event
        self.is_closed = False
        self.lock = threading.Lock()

    def read(self, n: int) -> bytes:
        if self.stop_event.is_set() or self.is_closed:
            return b""
        try:
            with self.lock:
                if self.is_closed or not self.handle:
                    return b""
                _, data = win32file.ReadFile(self.handle, n)
                return bytes(data)
        except Exception:
            return b""

    def close(self):
        with self.lock:
            self.is_closed = True
            if self.handle:
                try:
                    win32pipe.DisconnectNamedPipe(self.handle)
                except Exception:
                    pass
                try:
                    win32file.CloseHandle(self.handle)
                except Exception:
                    pass
                self.handle = None

    def seek(self, offset: int, whence: int = 0) -> int:
        return 0


def resolve_card_info(raw_name: str) -> Tuple[str, Optional[Path]]:
    """Convert raw model label (e.g. Knight_D, Pekka_D) to clean display name and image path."""
    clean = re.sub(r'_D$', '', raw_name)
    is_evo = bool(re.search(r'Evo(_D)?$', raw_name, re.IGNORECASE))

    clean_display = re.sub(r'_?Evo$', '', clean)
    clean_display = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', clean_display)
    if is_evo:
        clean_display += " (Evo)"

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', clean)
    slug = re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1).lower()
    slug = slug.replace('furnance', 'furnace')
    slug = slug.replace('battleram', 'battle-ram')
    slug = slug.replace('wallbreaker', 'wall-breakers')
    slug = slug.replace('spirit-empress-ground', 'spirit-empress')
    slug = slug.replace('spirit-empress-air', 'spirit-empress')
    slug = slug.replace('arrow', 'arrows')
    slug = slug.replace('rage-spell', 'rage')
    slug = slug.replace('monk-skill', 'monk')
    slug = slug.replace('_-evo', '-ev1')
    slug = slug.replace('_evo', '-ev1')

    candidates = [
        f"{slug}.png",
        f"{slug.replace('-evo', '-ev1')}.png",
        f"{slug.replace('-evo', '')}.png",
        f"{slug.rstrip('s')}.png",
        f"{slug}s.png",
    ]
    for c in candidates:
        p = CARDS_DIR / c
        if p.exists():
            return clean_display, p

    return clean_display, None


class CardTracker:
    """Clash Royale 4-card cycle manager and deployment debounce tracker."""
    def __init__(self, cooldown_sec: float = 3.5):
        self.cooldown_sec = cooldown_sec
        self.known_deck: List[str] = []         # Up to 8 unique card labels
        self.red_cycle: List[str] = []          # Up to 4 cards currently in cycle (out of hand)
        self.history: List[Tuple[str, str, str]] = [] # [(time_str, display_name, raw_label)]
        self.last_deployed: Dict[str, float] = {}

    def reset(self):
        self.known_deck.clear()
        self.red_cycle.clear()
        self.history.clear()
        self.last_deployed.clear()

    def record_deployment(self, raw_label: str) -> bool:
        """Process detected card deployment. Returns True if accepted, False if debounced."""
        now = time.time()
        last_t = self.last_deployed.get(raw_label, 0.0)
        if now - last_t < self.cooldown_sec:
            return False

        self.last_deployed[raw_label] = now

        # Add to known deck if not present (up to 8 cards)
        if raw_label not in self.known_deck and len(self.known_deck) < 8:
            self.known_deck.append(raw_label)

        # Update Red Cycle:
        # If card was already in red_cycle, remove it (it was played again after graduating)
        if raw_label in self.red_cycle:
            self.red_cycle.remove(raw_label)

        # Insert at front of red cycle (most recently deployed)
        self.red_cycle.insert(0, raw_label)

        # Keep red cycle at max 4 cards
        if len(self.red_cycle) > 4:
            self.red_cycle = self.red_cycle[:4]

        # Add to history
        time_str = time.strftime("%H:%M:%S", time.localtime(now))
        disp, _ = resolve_card_info(raw_label)
        self.history.insert(0, (time_str, disp, raw_label))
        if len(self.history) > 30:
            self.history = self.history[:30]

        return True

    def get_green_hand(self) -> List[Optional[str]]:
        """Return 4 cards currently in hand (cards in known_deck not in red_cycle)."""
        hand = [c for c in self.known_deck if c not in self.red_cycle]
        result = []
        for i in range(4):
            if i < len(hand):
                result.append(hand[i])
            else:
                result.append(None)
        return result

    def get_red_slots(self) -> List[Optional[str]]:
        """Return 4 red slots (0: just played ... 3: returns on next deployment)."""
        result = []
        for i in range(4):
            if i < len(self.red_cycle):
                result.append(self.red_cycle[i])
            else:
                result.append(None)
        return result


class LiveApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("📱 Clash Royale Live Studio - Real-Time Overlay & Card Tracker")
        self.geometry("1340x840")
        self.minsize(1080, 700)

        # Center on screen
        self._center_window(1340, 840)

        # State Variables
        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.stop_requested = threading.Event()
        self.scrcpy_proc: Optional[subprocess.Popen] = None
        self.pipe_handle = None
        self.stream_wrapper: Optional[NamedPipeStream] = None
        self.current_pipe_name: Optional[str] = None

        self.selected_model_name = ctk.StringVar()
        
        # Multi-Device Phone Selection
        self.device_map: Dict[str, str] = {}  # display_label -> serial
        self.selected_device_label = ctk.StringVar(value="Scanning phones...")
        self.selected_device_serial: Optional[str] = None

        # Hardware Compute Device Selection (NVIDIA GPU / CPU)
        self.active_compute_device = ctk.StringVar(value="auto")
        self.available_compute_devices: List[str] = ["auto"]


        self.confidence_var = ctk.DoubleVar(value=0.35)
        self.show_detections_var = ctk.BooleanVar(value=True)

        self.active_model = None
        self.active_model_path: Optional[Path] = None
        self.model_lock = threading.Lock()

        # Decoupled Real-Time Frame Sync (Zero Lag / Auto-Skipping)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.inference_thread: Optional[threading.Thread] = None

        # Telemetry
        self.current_fps = 0.0
        self.current_latency_ms = 0.0
        self.current_detection_count = 0

        # Card Tracker Engine & Image Cache
        self.card_tracker = CardTracker(cooldown_sec=3.5)
        self.card_ctk_cache: Dict[str, ctk.CTkImage] = {}
        self.card_thumb_cache: Dict[str, ctk.CTkImage] = {}
        self.placeholder_ctk_image: Optional[ctk.CTkImage] = None
        self._init_card_images()

        # Current photo image reference for canvas
        self.current_photo_image = None
        self.display_width = 460
        self.display_height = 720

        # Build UI
        self._build_ui()

        # Load dependencies and initialize in background
        threading.Thread(target=self._async_init, daemon=True).start()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _center_window(self, width: int, height: int):
        try:
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            x = max((screen_width - width) // 2, 0)
            y = max((screen_height - height) // 2, 0)
            self.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            pass

    def _init_card_images(self):
        """Preload card images and ensure assets exist with automatic fallback downloader."""
        try:
            ph = Image.new("RGBA", (54, 68), color=(35, 38, 45, 255))
            self.placeholder_ctk_image = ctk.CTkImage(light_image=ph, dark_image=ph, size=(54, 68))
        except Exception:
            pass

        self._load_cached_card_images()
        self._ensure_card_assets()

    def _load_cached_card_images(self):
        """Load card PNGs from assets/cards into in-memory CTkImages."""
        if not CARDS_DIR.exists():
            return

        for img_path in CARDS_DIR.glob("*.png"):
            if img_path.name not in self.card_ctk_cache:
                try:
                    pil_img = Image.open(img_path).convert("RGBA")
                    self.card_ctk_cache[img_path.name] = ctk.CTkImage(
                        light_image=pil_img, dark_image=pil_img, size=(54, 68)
                    )
                    self.card_thumb_cache[img_path.name] = ctk.CTkImage(
                        light_image=pil_img, dark_image=pil_img, size=(28, 35)
                    )
                except Exception:
                    pass

    def _ensure_card_assets(self):
        """If card images are missing, automatically download them from RoyaleAPI in the background."""
        CARDS_DIR.mkdir(parents=True, exist_ok=True)
        existing = list(CARDS_DIR.glob("*.png"))
        if len(existing) >= 50:
            return

        def download_worker():
            try:
                import urllib.request, json
                url = "https://api.github.com/repos/RoyaleAPI/cr-api-assets/contents/cards-75"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    items = json.loads(resp.read())

                for item in items:
                    name = item.get("name", "")
                    if name.endswith(".png"):
                        target = CARDS_DIR / name
                        if not target.exists():
                            urllib.request.urlretrieve(item["download_url"], target)

                self._load_cached_card_images()
                self.after(0, self._refresh_tracker_ui)
            except Exception:
                pass

        threading.Thread(target=download_worker, daemon=True).start()


    def _scan_compute_devices(self) -> List[str]:
        devices = []
        if torch is not None and torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                devices.append(f"cuda:{i} ({name})")
        devices.append("cpu")
        return devices

    def _get_active_torch_device(self) -> str:
        chosen = self.active_compute_device.get()
        if chosen == "cpu" or not torch.cuda.is_available():
            return "cpu"
        if chosen.startswith("cuda:"):
            return chosen.split()[0]
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _on_compute_device_selected(self, chosen: str):
        target_dev = "cpu" if chosen == "cpu" else chosen.split()[0]
        def worker():
            with self.model_lock:
                if self.active_model is not None:
                    try:
                        self.active_model.to(target_dev)
                        self.after(0, lambda: self.lbl_bottom_model.configure(
                            text=f"Model: {self.selected_model_name.get()} ({target_dev})", text_color="#81C784"
                        ))
                    except Exception as e:
                        self.after(0, lambda: self.lbl_bottom_model.configure(
                            text=f"Compute Error: {e}", text_color="#EF5350"
                        ))
        threading.Thread(target=worker, daemon=True).start()

    def _async_init(self):
        """Warm up deep learning libraries and refresh device/models."""
        try:
            import_dependencies()

            # Scan compute devices (GPU / CPU)
            devices = self._scan_compute_devices()
            default_dev = devices[0] if devices else "cpu"
            self.available_compute_devices = devices
            self.active_compute_device.set(default_dev)

            def update_compute_ui(devs=devices, def_d=default_dev):
                self.compute_dropdown.configure(values=devs)
                self.active_compute_device.set(def_d)

            self.after(0, update_compute_ui)

            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                gpu_name = torch.cuda.get_device_name(0)
                status_txt = f"⚡ CUDA: {gpu_name}"
            else:
                status_txt = "🖥 CPU Mode"

            self.after(0, lambda: self.lbl_engine_status.configure(
                text=status_txt, text_color="#81C784" if cuda_avail else "#FFB74D"
            ))

            self.refresh_models()
            self.check_usb_device()

            if self.selected_model_name.get():
                self._load_model_async(self.selected_model_name.get())

        except Exception as e:
            self.after(0, lambda: self.lbl_engine_status.configure(
                text=f"Engine Error: {e}", text_color="#EF5350"
            ))


    # ================= UI SETUP =================
    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # 1. Top Header & Control Toolbar
        self._build_top_controls()

        # 2. Main Middle Area (Phone screen in center + Card tracker on right)
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 10))
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)  # Middle phone display
        self.content_frame.grid_columnconfigure(1, weight=0)  # Right tracker panel

        self._build_phone_display()
        self._build_tracker_panel()

        # 3. Bottom Status Bar
        self._build_status_bar()

    def _build_top_controls(self):
        control_card = ctk.CTkFrame(self, corner_radius=10, fg_color=("#E0E0E0", "#1E1E1E"))
        control_card.grid(row=0, column=0, sticky="ew", padx=15, pady=12)

        # Row 0: App Title & Hardware Badge
        title_box = ctk.CTkFrame(control_card, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 8))
        
        lbl_title = ctk.CTkLabel(
            title_box, text="📱 Clash Royale Live Studio",
            font=ctk.CTkFont(size=18, weight="bold"), text_color="#90CAF9"
        )
        lbl_title.pack(side="left")

        self.lbl_engine_status = ctk.CTkLabel(
            title_box, text="🔄 Initializing AI Engine...",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#FFB74D"
        )
        self.lbl_engine_status.pack(side="right")

        # Row 1: Model Selection, Compute Device & Phone Selector
        row1_frame = ctk.CTkFrame(control_card, fg_color="transparent")
        row1_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 10))

        # --- Model Selector ---
        ctk.CTkLabel(row1_frame, text="🧠 Model:", font=ctk.CTkFont(size=13, weight="bold")).pack(
            side="left", padx=(0, 4)
        )
        self.model_dropdown = ctk.CTkOptionMenu(
            row1_frame, variable=self.selected_model_name,
            values=["Scanning models..."], width=150,
            command=self._on_model_selected
        )
        self.model_dropdown.pack(side="left", padx=4)

        btn_refresh_models = ctk.CTkButton(
            row1_frame, text="🔄", width=32, height=28,
            fg_color="#37474F", hover_color="#263238",
            command=self.refresh_models
        )
        btn_refresh_models.pack(side="left", padx=(2, 14))

        # --- Compute Device Selector (NVIDIA GPU / CPU) ---
        ctk.CTkLabel(row1_frame, text="⚡ Compute:", font=ctk.CTkFont(size=13, weight="bold")).pack(
            side="left", padx=(0, 4)
        )
        self.compute_dropdown = ctk.CTkOptionMenu(
            row1_frame, variable=self.active_compute_device,
            values=["auto"], width=175,
            command=self._on_compute_device_selected
        )
        self.compute_dropdown.pack(side="left", padx=4)

        # --- Phone Device Selector ---
        ctk.CTkLabel(row1_frame, text="📱 Phone:", font=ctk.CTkFont(size=13, weight="bold")).pack(
            side="left", padx=(14, 4)
        )
        self.device_dropdown = ctk.CTkOptionMenu(
            row1_frame, variable=self.selected_device_label,
            values=["Scanning phones..."], width=230,
            command=self._on_device_selected
        )
        self.device_dropdown.pack(side="left", padx=4)

        btn_scan_device = ctk.CTkButton(
            row1_frame, text="🔄 Scan", width=65, height=28,
            fg_color="#37474F", hover_color="#263238",
            command=self.check_usb_device
        )
        btn_scan_device.pack(side="left", padx=(2, 0))


        # Row 2: Live Controls (Connect/Disconnect, Conf Slider, Overlay Toggle, Reset)
        row2_frame = ctk.CTkFrame(control_card, fg_color="transparent")
        row2_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 10))

        # Connect & Stream Button
        self.btn_toggle_stream = ctk.CTkButton(
            row2_frame, text="▶ Connect & Stream Phone",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=34, width=200,
            fg_color="#2E7D32", hover_color="#1B5E20",
            command=self.toggle_stream
        )
        self.btn_toggle_stream.pack(side="left", padx=(0, 12))

        # Reset All Button
        self.btn_clear = ctk.CTkButton(
            row2_frame, text="🔄 Clear / Reset",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=34, width=120,
            fg_color="#455A64", hover_color="#37474F",
            command=self.reset_state
        )
        self.btn_clear.pack(side="left", padx=6)

        # AI Bounding Boxes Toggle
        self.switch_detections = ctk.CTkSwitch(
            row2_frame, text="Show AI Boxes",
            variable=self.show_detections_var,
            font=ctk.CTkFont(size=12)
        )
        self.switch_detections.pack(side="left", padx=15)

        # Confidence Slider
        conf_box = ctk.CTkFrame(row2_frame, fg_color="transparent")
        conf_box.pack(side="right", padx=(10, 0))

        self.lbl_conf_val = ctk.CTkLabel(
            conf_box, text=f"Conf: {int(self.confidence_var.get() * 100)}%",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#90CAF9", width=75
        )
        self.lbl_conf_val.pack(side="right")

        self.slider_conf = ctk.CTkSlider(
            conf_box, from_=0.10, to=0.90, number_of_steps=80,
            variable=self.confidence_var, width=130,
            command=self._on_conf_change
        )
        self.slider_conf.pack(side="right", padx=6)

        ctk.CTkLabel(conf_box, text="Confidence:", font=ctk.CTkFont(size=12)).pack(side="right")

    def _build_phone_display(self):
        """Center phone screen view with crisp canvas rendering."""
        display_outer = ctk.CTkFrame(self.content_frame, corner_radius=12, fg_color=("#D5D5D5", "#141414"))
        display_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        display_outer.grid_rowconfigure(1, weight=1)
        display_outer.grid_columnconfigure(0, weight=1)

        disp_head = ctk.CTkFrame(display_outer, height=36, fg_color=("#C0C0C0", "#1C1C1C"), corner_radius=8)
        disp_head.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        self.lbl_screen_title = ctk.CTkLabel(
            disp_head, text="📱 Live Phone Feed (Direct Non-Occluded Stream)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="gray75"
        )
        self.lbl_screen_title.pack(side="left", padx=12)

        self.lbl_live_stats = ctk.CTkLabel(
            disp_head, text="FPS: 0.0 | Latency: 0ms | Objects: 0",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color="#81C784"
        )
        self.lbl_live_stats.pack(side="right", padx=12)

        self.canvas_container = ctk.CTkFrame(display_outer, fg_color="black", corner_radius=8)
        self.canvas_container.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.canvas_container, bg="#0A0A0A", highlightthickness=0, bd=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self._show_idle_placeholder()

    def _show_idle_placeholder(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or 480
        ch = self.canvas.winfo_height() or 640
        self.canvas.create_text(
            cw // 2, ch // 2 - 25,
            text="📱 Phone Stream Idle",
            fill="#90CAF9", font=("Segoe UI", 16, "bold")
        )
        self.canvas.create_text(
            cw // 2, ch // 2 + 15,
            text="Click '▶ Connect & Stream Phone' to begin.\nYour phone screen will stream directly here with zero external windows.",
            fill="gray60", font=("Segoe UI", 11), justify="center"
        )

    def _on_canvas_resize(self, event):
        self.display_width = max(event.width, 200)
        self.display_height = max(event.height, 200)
        if not self.is_streaming:
            self._show_idle_placeholder()

    # ================= CARD TRACKER PANEL =================
    def _build_tracker_panel(self):
        """Right-hand sidebar: 4 Red (Top / Cycle) and 4 Green (Bottom / In Hand) Tracker."""
        self.tracker_panel = ctk.CTkFrame(
            self.content_frame, width=360, corner_radius=12,
            fg_color=("#E0E0E0", "#181818")
        )
        self.tracker_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 0), pady=0)
        self.tracker_panel.grid_rowconfigure(5, weight=1)
        self.tracker_panel.grid_propagate(False)

        # 1. Header with Title & Reset Match Button
        tracker_header = ctk.CTkFrame(self.tracker_panel, fg_color="transparent")
        tracker_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))

        lbl_track_title = ctk.CTkLabel(
            tracker_header, text="🃏 Opponent Deck Tracker",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#FFD54F"
        )
        lbl_track_title.pack(side="left")

        btn_reset_match = ctk.CTkButton(
            tracker_header, text="🔄 Reset Match", width=95, height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#D32F2F", hover_color="#9A0007",
            command=self.reset_tracker
        )
        btn_reset_match.pack(side="right")

        # Revealed Deck Status Badge
        self.lbl_deck_revealed = ctk.CTkLabel(
            self.tracker_panel,
            text="Deck Revealed: 0 / 8 Cards",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#90CAF9"
        )
        self.lbl_deck_revealed.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 6))

        # 2. 🔴 TOP SECTION: 4 Red Slots (In Cycle / Waiting to Return)
        red_box = ctk.CTkFrame(self.tracker_panel, corner_radius=10, fg_color=("#D6D6D6", "#201515"))
        red_box.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))

        red_head = ctk.CTkFrame(red_box, fg_color="transparent")
        red_head.pack(fill="x", padx=8, pady=(6, 4))
        ctk.CTkLabel(
            red_head, text="🔴 IN CYCLE (Out of Hand)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#EF5350"
        ).pack(side="left")
        ctk.CTkLabel(
            red_head, text="(Needs 4 cards to return)",
            font=ctk.CTkFont(size=10, slant="italic"), text_color="gray60"
        ).pack(side="right")

        self.red_slots_frame = ctk.CTkFrame(red_box, fg_color="transparent")
        self.red_slots_frame.pack(fill="x", padx=4, pady=(0, 6))
        for col in range(4):
            self.red_slots_frame.grid_columnconfigure(col, weight=1)

        self.red_slot_widgets = []
        for i in range(4):
            slot = self._create_card_slot_widget(self.red_slots_frame, col=i, is_red=True)
            self.red_slot_widgets.append(slot)

        # 3. ⬇ CYCLE ARROW DIVIDER ⬇
        cycle_div = ctk.CTkFrame(self.tracker_panel, fg_color="transparent")
        cycle_div.grid(row=3, column=0, sticky="ew", padx=12, pady=1)
        ctk.CTkLabel(
            cycle_div, text="⬇  Graduates into Hand after 4 plays  ⬇",
            font=ctk.CTkFont(size=10, weight="bold"), text_color="#81C784"
        ).pack()

        # 4. 🟢 BOTTOM SECTION: 4 Green Slots (In Hand / Available Now)
        green_box = ctk.CTkFrame(self.tracker_panel, corner_radius=10, fg_color=("#D6D6D6", "#152015"))
        green_box.grid(row=4, column=0, sticky="ew", padx=10, pady=(4, 6))

        green_head = ctk.CTkFrame(green_box, fg_color="transparent")
        green_head.pack(fill="x", padx=8, pady=(6, 4))
        ctk.CTkLabel(
            green_head, text="🟢 IN HAND (Ready to Deploy)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#66BB6A"
        ).pack(side="left")
        ctk.CTkLabel(
            green_head, text="(Opponent holds these now)",
            font=ctk.CTkFont(size=10, slant="italic"), text_color="gray60"
        ).pack(side="right")

        self.green_slots_frame = ctk.CTkFrame(green_box, fg_color="transparent")
        self.green_slots_frame.pack(fill="x", padx=4, pady=(0, 6))
        for col in range(4):
            self.green_slots_frame.grid_columnconfigure(col, weight=1)

        self.green_slot_widgets = []
        for i in range(4):
            slot = self._create_card_slot_widget(self.green_slots_frame, col=i, is_red=False)
            self.green_slot_widgets.append(slot)

        # 5. 📜 DEPLOYMENT LOG FEED
        history_box = ctk.CTkFrame(self.tracker_panel, corner_radius=10, fg_color=("#D6D6D6", "#1E1E1E"))
        history_box.grid(row=5, column=0, sticky="nsew", padx=10, pady=(4, 10))
        history_box.grid_rowconfigure(1, weight=1)
        history_box.grid_columnconfigure(0, weight=1)

        hist_head = ctk.CTkFrame(history_box, fg_color="transparent")
        hist_head.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        ctk.CTkLabel(
            hist_head, text="📜 Match Deployment Log",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="gray75"
        ).pack(side="left")

        self.history_scroll = ctk.CTkScrollableFrame(history_box, fg_color="transparent")
        self.history_scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        self.history_scroll.grid_columnconfigure(0, weight=1)

        self._refresh_tracker_ui()

    def _create_card_slot_widget(self, parent, col: int, is_red: bool) -> Dict:
        """Create a single card slot (Frame, Card Portrait Label, Name Label, Sub-badge)."""
        border_col = "#E53935" if is_red else "#43A047"
        slot_frame = ctk.CTkFrame(
            parent, width=74, height=115, corner_radius=8,
            border_width=2, border_color=border_col,
            fg_color=("#C8C8C8", "#121212")
        )
        slot_frame.grid(row=0, column=col, padx=3, pady=4, sticky="nsew")
        slot_frame.grid_propagate(False)

        # Sub-badge (e.g. "1/4", "READY")
        badge_text = f"{col + 1}/4" if is_red else "READY"
        badge_col = "#EF5350" if is_red else "#81C784"
        lbl_badge = ctk.CTkLabel(
            slot_frame, text=badge_text,
            font=ctk.CTkFont(size=9, weight="bold"), text_color=badge_col, height=14
        )
        lbl_badge.pack(pady=(2, 0))

        # Card Image Portrait
        lbl_img = ctk.CTkLabel(
            slot_frame, text="", image=self.placeholder_ctk_image,
            width=54, height=68
        )
        lbl_img.pack(pady=2)

        # Card Name Label
        lbl_name = ctk.CTkLabel(
            slot_frame, text="?", font=ctk.CTkFont(size=10, weight="bold"),
            text_color="gray70", height=16
        )
        lbl_name.pack(pady=(0, 2))

        return {
            "frame": slot_frame,
            "img": lbl_img,
            "name": lbl_name,
            "badge": lbl_badge,
            "is_red": is_red,
            "col": col
        }

    def _refresh_tracker_ui(self):
        """Sync the 4 Red and 4 Green slots with the CardTracker state."""
        red_cards = self.card_tracker.get_red_slots()
        green_cards = self.card_tracker.get_green_hand()

        # Update Deck Revealed Count
        num_known = len(self.card_tracker.known_deck)
        self.lbl_deck_revealed.configure(text=f"Deck Revealed: {num_known} / 8 Cards")

        # 1. Update 4 Red Slots (0: just played ... 3: returns on next deployment)
        for i, raw_card in enumerate(red_cards):
            slot = self.red_slot_widgets[i]
            if raw_card:
                disp, p = resolve_card_info(raw_card)
                slot["name"].configure(text=disp[:11], text_color="#FFCDD2")
                slot["badge"].configure(
                    text="Next!" if i == 3 else f"{i + 1}/4",
                    text_color="#FF5252" if i == 3 else "#EF5350"
                )
                slot["frame"].configure(border_color="#E53935")
                img_key = p.name if p else None
                if img_key and img_key in self.card_ctk_cache:
                    slot["img"].configure(image=self.card_ctk_cache[img_key])
                else:
                    slot["img"].configure(image=self.placeholder_ctk_image)
            else:
                slot["name"].configure(text="Empty", text_color="gray50")
                slot["badge"].configure(text=f"{i + 1}/4", text_color="gray50")
                slot["frame"].configure(border_color="#37474F")
                slot["img"].configure(image=self.placeholder_ctk_image)

        # 2. Update 4 Green Slots (Cards currently available to opponent)
        for i, raw_card in enumerate(green_cards):
            slot = self.green_slot_widgets[i]
            if raw_card:
                disp, p = resolve_card_info(raw_card)
                slot["name"].configure(text=disp[:11], text_color="#C8E6C9")
                slot["badge"].configure(text="READY", text_color="#81C784")
                slot["frame"].configure(border_color="#43A047")
                img_key = p.name if p else None
                if img_key and img_key in self.card_ctk_cache:
                    slot["img"].configure(image=self.card_ctk_cache[img_key])
                else:
                    slot["img"].configure(image=self.placeholder_ctk_image)
            else:
                slot["name"].configure(text="?", text_color="gray50")
                slot["badge"].configure(text="IN HAND", text_color="gray50")
                slot["frame"].configure(border_color="#37474F")
                slot["img"].configure(image=self.placeholder_ctk_image)

        # 3. Update History Log
        for child in self.history_scroll.winfo_children():
            child.destroy()

        if not self.card_tracker.history:
            ctk.CTkLabel(
                self.history_scroll, text="No deployments recorded yet.",
                font=ctk.CTkFont(size=11), text_color="gray50"
            ).pack(pady=15)
        else:
            for t_str, disp_name, raw in self.card_tracker.history[:15]:
                row = ctk.CTkFrame(self.history_scroll, fg_color=("#D0D0D0", "#181818"), height=32, corner_radius=6)
                row.pack(fill="x", pady=2)
                row.pack_propagate(False)

                _, img_p = resolve_card_info(raw)
                img_key = img_p.name if img_p else None
                if img_key and img_key in self.card_thumb_cache:
                    lbl_t = ctk.CTkLabel(row, text="", image=self.card_thumb_cache[img_key], width=30)
                    lbl_t.pack(side="left", padx=(4, 6))

                ctk.CTkLabel(
                    row, text=f"{t_str} - {disp_name}",
                    font=ctk.CTkFont(size=11, weight="bold"), text_color="#E0E0E0"
                ).pack(side="left")

    def reset_tracker(self):
        """Reset the match deck tracker state."""
        self.card_tracker.reset()
        self._refresh_tracker_ui()

    def reset_state(self):
        """Reset detections, telemetry and tracker."""
        self.current_fps = 0.0
        self.current_latency_ms = 0.0
        self.current_detection_count = 0
        self.lbl_live_stats.configure(text="FPS: 0.0 | Latency: 0ms | Objects: 0")
        self.reset_tracker()
        if not self.is_streaming:
            self._show_idle_placeholder()

    def _build_status_bar(self):
        status_bar = ctk.CTkFrame(self, height=28, fg_color=("#CCCCCC", "#161616"), corner_radius=0)
        status_bar.grid(row=2, column=0, sticky="ew")

        self.lbl_bottom_info = ctk.CTkLabel(
            status_bar,
            text="Ready. Click 'Connect & Stream Phone' to begin.",
            font=ctk.CTkFont(size=11), text_color="gray65"
        )
        self.lbl_bottom_info.pack(side="left", padx=15)

        self.lbl_bottom_model = ctk.CTkLabel(
            status_bar, text="Model: None",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#90CAF9"
        )
        self.lbl_bottom_model.pack(side="right", padx=15)

    # ================= SCANNING & SELECTION =================
    def refresh_models(self):
        models = [f.name for f in MODELS_DIR.glob("*.pt") if f.is_file()]
        if not models:
            self.model_dropdown.configure(values=["No .pt models found"])
            self.selected_model_name.set("No .pt models found")
            return

        self.model_dropdown.configure(values=models)
        default_m = "best_cr.pt" if "best_cr.pt" in models else models[0]
        if self.selected_model_name.get() not in models:
            self.selected_model_name.set(default_m)
            self._load_model_async(default_m)

    def _on_model_selected(self, model_name: str):
        if model_name and not model_name.startswith("No .pt"):
            self._load_model_async(model_name)

    def _load_model_async(self, model_name: str):
        def worker():
            global YOLO
            import_dependencies()
            path = MODELS_DIR / model_name
            if not path.exists():
                return
            try:
                self.after(0, lambda: self.lbl_bottom_model.configure(
                    text=f"Loading {model_name}...", text_color="#FFB74D"
                ))
                loaded = YOLO(str(path))
                device = self._get_active_torch_device()
                loaded.to(device)

                with self.model_lock:
                    self.active_model = loaded
                    self.active_model_path = path

                self.after(0, lambda: self.lbl_bottom_model.configure(
                    text=f"Model: {model_name} ({device})", text_color="#81C784"
                ))
            except Exception as e:
                self.after(0, lambda: self.lbl_bottom_model.configure(
                    text=f"Load Error: {e}", text_color="#EF5350"
                ))
        threading.Thread(target=worker, daemon=True).start()

    def check_usb_device(self):
        """Query ADB for all connected phones/emulators and populate dropdown."""
        def worker():
            if not ADB_EXE.exists():
                self.selected_device_label.set("ADB tool missing")
                return

            try:
                proc = subprocess.run(
                    [str(ADB_EXE), "devices", "-l"],
                    capture_output=True, text=True, timeout=5, check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
                new_map = {}
                labels = []
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "device":
                        serial = parts[0]
                        model_str = next((p for p in parts if p.startswith("model:")), "")
                        name = model_str.replace("model:", "").replace("_", " ") if model_str else serial
                        disp = f"{name} ({serial})"
                        new_map[disp] = serial
                        labels.append(disp)

                def update_ui(m=new_map, labs=labels):
                    self.device_map = m
                    if labs:
                        self.device_dropdown.configure(values=labs)
                        if self.selected_device_label.get() not in labs:
                            self.selected_device_label.set(labs[0])
                            self.selected_device_serial = m[labs[0]]
                        else:
                            self.selected_device_serial = m[self.selected_device_label.get()]
                        self.lbl_bottom_info.configure(
                            text=f"Selected phone: {self.selected_device_label.get()}",
                            text_color="#81C784"
                        )
                    else:
                        self.selected_device_serial = None
                        self.device_dropdown.configure(values=["No devices found (Check USB)"])
                        self.selected_device_label.set("No devices found (Check USB)")
                        self.lbl_bottom_info.configure(
                            text="No Android device detected over USB. Check USB debugging.",
                            text_color="#EF5350"
                        )

                self.after(0, update_ui)

            except Exception as e:
                self.after(0, lambda: self.selected_device_label.set(f"USB Scan Error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_device_selected(self, chosen: str):
        if chosen in self.device_map:
            self.selected_device_serial = self.device_map[chosen]
            self.lbl_bottom_info.configure(text=f"Target phone: {chosen}", text_color="#81C784")


    def _on_conf_change(self, val: float):
        self.lbl_conf_val.configure(text=f"Conf: {int(val * 100)}%")

    # ================= HEADLESS DIRECT STREAM =================
    def toggle_stream(self):
        if self.is_streaming:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        if self.is_streaming:
            return

        if not SCRCPY_EXE.exists():
            messagebox.showerror(
                "scrcpy Missing",
                f"scrcpy executable not found at:\n{SCRCPY_EXE}\n\nPlease check tools/scrcpy directory."
            )
            return

        self.is_streaming = True
        self.stop_requested.clear()
        self.new_frame_event.clear()
        with self.frame_lock:
            self.latest_frame = None

        self.btn_toggle_stream.configure(
            text="⏳ Connecting...",
            fg_color="#C62828", hover_color="#8E0000"
        )
        self.lbl_bottom_info.configure(text="Connecting to phone stream via ADB...", text_color="#FFB74D")

        # Start stream reader thread (Inference thread will start once connected)
        self.stream_thread = threading.Thread(target=self._stream_reader_worker, daemon=True)
        self.stream_thread.start()

    def stop_stream(self):
        if not self.is_streaming:
            return
        self.is_streaming = False
        self.stop_requested.set()
        self.new_frame_event.set()

        # Update UI instantly so there is zero delay or UI lockup
        self.btn_toggle_stream.configure(
            text="▶ Connect & Stream Phone",
            fg_color="#2E7D32", hover_color="#1B5E20"
        )
        self.lbl_bottom_info.configure(text="Stream disconnected.", text_color="gray70")
        self._show_idle_placeholder()

        # Run process termination and pipe teardown in background thread
        proc_to_kill = self.scrcpy_proc
        wrapper_to_close = self.stream_wrapper
        handle_to_close = self.pipe_handle
        self.scrcpy_proc = None
        self.stream_wrapper = None
        self.pipe_handle = None

        def cleanup():
            if proc_to_kill:
                try:
                    proc_to_kill.kill()
                except Exception:
                    pass

            if wrapper_to_close:
                try:
                    wrapper_to_close.close()
                except Exception:
                    pass

            if handle_to_close:
                try:
                    win32pipe.DisconnectNamedPipe(handle_to_close)
                except Exception:
                    pass
                try:
                    win32file.CloseHandle(handle_to_close)
                except Exception:
                    pass

        threading.Thread(target=cleanup, daemon=True).start()

    def _stream_reader_worker(self):
        """Ultra-fast pipe consumer: constantly reads PyAV frames and updates latest_frame."""
        global av, win32pipe, win32file
        import_dependencies()

        pipe_name = rf"\\.\pipe\cr_live_pipe_{int(time.time())}"
        self.current_pipe_name = pipe_name

        try:
            self.pipe_handle = win32pipe.CreateNamedPipe(
                pipe_name,
                win32pipe.PIPE_ACCESS_INBOUND,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                1, 2097152, 2097152, 0, None
            )
        except Exception as e:
            if not self.stop_requested.is_set():
                self.after(0, lambda: messagebox.showerror("Pipe Error", f"Could not create named pipe:\n{e}"))
                self.after(0, self.stop_stream)
            return

        cmd = [
            str(SCRCPY_EXE),
            "--no-window",
            "--no-audio",
            "--max-size", "1024",      # Downscales for 3x faster inference & decode
            "--max-fps", "30",         # Prevents frame accumulation
            "-r", pipe_name,
            "--record-format=mkv",
            "--stay-awake"
        ]
        if self.selected_device_serial:
            cmd.extend(["-s", self.selected_device_serial])


        try:
            self.scrcpy_proc = subprocess.Popen(
                cmd,
                cwd=str(SCRCPY_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
        except Exception as e:
            if not self.stop_requested.is_set():
                self.after(0, lambda: messagebox.showerror("Launch Error", f"Failed to start scrcpy:\n{e}"))
                self.after(0, self.stop_stream)
            return

        # Wait for scrcpy to connect to named pipe
        try:
            win32pipe.ConnectNamedPipe(self.pipe_handle, None)
        except Exception:
            if not self.stop_requested.is_set():
                self.after(0, self.stop_stream)
            return

        if self.stop_requested.is_set():
            return

        # Wrap pipe in thread-safe wrapper
        self.stream_wrapper = NamedPipeStream(self.pipe_handle, self.stop_requested)
        try:
            container = av.open(self.stream_wrapper, format="matroska", mode="r")
        except Exception as e:
            if not self.stop_requested.is_set():
                self.after(0, lambda: messagebox.showerror(
                    "Stream Decode Error", f"Could not open video stream:\n{e}"
                ))
                self.after(0, self.stop_stream)
            return

        if self.stop_requested.is_set():
            return

        def on_connected():
            if self.is_streaming and not self.stop_requested.is_set():
                self.btn_toggle_stream.configure(
                    text="⏹ Disconnect",
                    fg_color="#C62828", hover_color="#8E0000"
                )
                self.lbl_bottom_info.configure(
                    text="✅ Direct stream active (Real-time sync enabled).", text_color="#81C784"
                )

        self.after(0, on_connected)

        # Start parallel inference worker now that frames are streaming
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()

        try:
            for frame in container.decode(video=0):
                if self.stop_requested.is_set():
                    break

                # Convert decoded frame to BGR numpy array (~1ms)
                frame_bgr = frame.to_ndarray(format="bgr24")

                with self.frame_lock:
                    self.latest_frame = frame_bgr

                self.new_frame_event.set()

        except Exception:
            pass
        finally:
            if not self.stop_requested.is_set():
                self.after(0, self.stop_stream)

    def _inference_worker(self):
        """Asynchronous worker: consumes the newest available frame, triggers tracker."""
        fps_count = 0
        fps_timer = time.time()

        while not self.stop_requested.is_set():
            if not self.new_frame_event.wait(timeout=0.1):
                continue

            self.new_frame_event.clear()

            with self.frame_lock:
                frame = self.latest_frame
                if frame is None:
                    continue
                frame_bgr = frame.copy()

            det_count = 0
            inference_ms = 0.0

            with self.model_lock:
                model = self.active_model

            if model is not None and self.show_detections_var.get():
                t0 = time.time()
                try:
                    conf = self.confidence_var.get()
                    res = model.predict(frame_bgr, conf=conf, verbose=False)
                    if res and len(res) > 0:
                        frame_bgr = res[0].plot()
                        boxes = res[0].boxes
                        if boxes is not None and len(boxes) > 0:
                            det_count = len(boxes)
                            # Check detected boxes for deploying cards
                            tracker_updated = False
                            for b in boxes:
                                cls_id = int(b.cls[0].item())
                                if cls_id in model.names:
                                    raw_label = model.names[cls_id]
                                    if self.card_tracker.record_deployment(raw_label):
                                        tracker_updated = True

                            if tracker_updated:
                                self.after(0, self._refresh_tracker_ui)

                    inference_ms = (time.time() - t0) * 1000.0
                except Exception:
                    pass

            # Calculate FPS
            fps_count += 1
            now = time.time()
            time_diff = now - fps_timer
            if time_diff >= 0.5:
                self.current_fps = fps_count / time_diff
                fps_count = 0
                fps_timer = now
                self.current_latency_ms = inference_ms
                self.current_detection_count = det_count

            # Render immediately to canvas
            self._render_frame(frame_bgr, self.current_fps, self.current_latency_ms, det_count)

    def _render_frame(self, frame_bgr, fps: float, latency_ms: float, det_count: int):
        """Scale and render BGR frame onto the canvas cleanly preserving aspect ratio."""
        try:
            h, w = frame_bgr.shape[:2]
            target_w = self.display_width
            target_h = self.display_height

            scale = min(target_w / w, target_h / h)
            new_w = max(int(w * scale), 1)
            new_h = max(int(h * scale), 1)

            resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=img)

            pos_x = target_w // 2
            pos_y = target_h // 2

            def update_gui(p=photo, x=pos_x, y=pos_y, f=fps, l=latency_ms, c=det_count):
                self.current_photo_image = p
                self.canvas.delete("all")
                self.canvas.create_image(x, y, image=p, anchor="center")
                self.lbl_live_stats.configure(
                    text=f"FPS: {f:.1f} | Latency: {l:.1f}ms | Objects: {c}"
                )

            self.after(0, update_gui)

        except Exception:
            pass

    def _on_closing(self):
        self.stop_stream()
        self.destroy()


if __name__ == "__main__":
    app = LiveApp()
    app.mainloop()
