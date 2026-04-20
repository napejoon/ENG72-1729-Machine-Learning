"""
GUI Mode — SUT_RFDETR
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import time
import threading
import numpy as np
from PIL import Image, ImageTk
from collections import deque
import supervision as sv


def launch_gui(detector):

    root = tk.Tk()
    root.title("SUT RF-DETR  —  Object Detection")
    root.geometry("1300x850")
    root.configure(bg="#f0f2f5")
    root.minsize(1050, 720)

    C = {
        'bg': '#f0f2f5', 'card': '#ffffff', 'header': '#1e3a5f',
        'accent': '#2196F3', 'accent2': '#0d47a1',
        'success': '#4caf50', 'danger': '#f44336', 'warning': '#ff9800',
        'text': '#212121', 'text2': '#757575', 'border': '#e0e0e0',
        'input_bg': '#fafafa',
    }

    running = threading.Event()
    cap_ref = [None]
    latest_frame = [None]
    fps_buffer = deque(maxlen=30)
    polygon_points = []
    polygon_preview_img = [None]

    # ========================================
    # Header
    # ========================================
    header = tk.Frame(root, bg=C['header'], height=60)
    header.pack(fill='x')
    header.pack_propagate(False)

    tk.Label(header, text="🔍 SUT RF-DETR",
             font=('Segoe UI', 16, 'bold'), fg='#ffffff', bg=C['header']
             ).pack(side='left', padx=20)

    status_label = tk.Label(header, text="⏹ READY",
                            font=('Segoe UI', 10, 'bold'), fg='#81c784', bg=C['header'])
    status_label.pack(side='right', padx=15)

    def _hbtn(parent, text, bg, fg='#fff'):
        return tk.Button(parent, text=text, bg=bg, fg=fg,
                         font=('Segoe UI', 10, 'bold'), relief='flat',
                         cursor='hand2', bd=0, padx=14, pady=2)

    screenshot_btn = _hbtn(header, "📸 Save", '#546e7a')
    screenshot_btn.pack(side='right', padx=4, pady=12)
    stop_btn = _hbtn(header, "⏹ STOP", C['danger'])
    stop_btn.pack(side='right', padx=4, pady=12)
    start_btn = _hbtn(header, "▶ START", C['success'])
    start_btn.pack(side='right', padx=4, pady=12)
    tk.Label(header, text="🌐 localhost:5555", font=('Segoe UI', 8),
             fg='#90caf9', bg=C['header']).pack(side='right', padx=10)

    # ========================================
    # Info Bar
    # ========================================
    info_bar = tk.Frame(root, bg=C['card'], height=40,
                        highlightbackground=C['border'], highlightthickness=1)
    info_bar.pack(fill='x', padx=8, pady=(4, 0))
    info_bar.pack_propagate(False)

    fps_label = tk.Label(info_bar, text="FPS: -", fg=C['success'],
                         bg=C['card'], font=('Consolas', 11, 'bold'))
    fps_label.pack(side='left', padx=15)
    obj_label = tk.Label(info_bar, text="Objects: -", fg=C['accent'],
                         bg=C['card'], font=('Consolas', 11, 'bold'))
    obj_label.pack(side='left', padx=15)
    pass_label = tk.Label(info_bar, text="IN: 0 | OUT: 0 | Zone: 0", fg=C['warning'],
                          bg=C['card'], font=('Consolas', 11, 'bold'))
    pass_label.pack(side='left', padx=15)
    count_label = tk.Label(info_bar, text="", fg='#7b1fa2',
                           bg=C['card'], font=('Consolas', 11, 'bold'))
    count_label.pack(side='right', padx=15)

    # ========================================
    # Main
    # ========================================
    main_frame = tk.Frame(root, bg=C['bg'])
    main_frame.pack(fill='both', expand=True, padx=8, pady=6)

    left_panel = tk.Frame(main_frame, bg=C['card'], width=300,
                          highlightbackground=C['border'], highlightthickness=1)
    left_panel.pack(side='left', fill='y', padx=(0, 6))
    left_panel.pack_propagate(False)

    canvas_scroll = tk.Canvas(left_panel, bg=C['card'], highlightthickness=0)
    scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas_scroll.yview)
    scroll_frame = tk.Frame(canvas_scroll, bg=C['card'])
    scroll_frame.bind("<Configure>",
                      lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
    canvas_scroll.create_window((0, 0), window=scroll_frame, anchor="nw", width=278)
    canvas_scroll.configure(yscrollcommand=scrollbar.set)
    canvas_scroll.bind_all("<MouseWheel>",
                           lambda e: canvas_scroll.yview_scroll(int(-1 * (e.delta / 120)), "units"))
    canvas_scroll.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    video_container = tk.Frame(main_frame, bg='#e8eaf6',
                               highlightbackground=C['border'], highlightthickness=1)
    video_container.pack(side='right', fill='both', expand=True)
    video_label = tk.Label(video_container, bg='#e8eaf6',
                           text="📷 Video Preview\n\nLoad model → Click ▶ START",
                           fg=C['text2'], font=('Segoe UI', 14))
    video_label.pack(fill='both', expand=True)

    # ========================================
    # Widgets Helper
    # ========================================
    def section_header(p, t):
        f = tk.Frame(p, bg=C['card'])
        f.pack(fill='x', padx=12, pady=(12, 3))
        tk.Label(f, text=t, font=('Segoe UI', 11, 'bold'),
                 fg=C['accent2'], bg=C['card']).pack(anchor='w')
        tk.Frame(f, bg=C['accent'], height=2).pack(fill='x', pady=(3, 0))

    def sub_label(p, t):
        tk.Label(p, text=t, fg=C['text2'], bg=C['card'],
                 font=('Segoe UI', 9)).pack(anchor='w', padx=14, pady=(5, 1))

    def make_entry(p, tv):
        e = tk.Entry(p, textvariable=tv, width=26, bg=C['input_bg'], fg=C['text'],
                     insertbackground=C['text'], relief='solid', bd=1, font=('Segoe UI', 9))
        e.pack(anchor='w', padx=14, pady=2)
        return e

    def make_btn(p, t, cmd, bg_c=None):
        tk.Button(p, text=t, command=cmd, bg=bg_c or C['accent'], fg='#fff',
                  font=('Segoe UI', 9, 'bold'), relief='flat', cursor='hand2',
                  bd=0, padx=12, pady=3).pack(anchor='w', padx=14, pady=3)

    # ========================================
    # MODEL
    # ========================================
    section_header(scroll_frame, "📦 Model")
    sub_label(scroll_frame, "Model Size:")
    model_size_var = tk.StringVar(value="base (B/M)")
    ttk.Combobox(scroll_frame, textvariable=model_size_var,
                 values=["nano (N)", "small (S)", "base (B/M)", "large (L)", "xtralarge (X)"],
                 state='readonly', width=24, font=('Segoe UI', 9)).pack(anchor='w', padx=14, pady=2)
    size_map = {"nano (N)": "n", "small (S)": "s", "base (B/M)": "base",
                "large (L)": "large", "xtralarge (X)": "x"}

    sub_label(scroll_frame, "Model Type:")
    model_type_var = tk.StringVar(value="pretrained")
    ttk.Combobox(scroll_frame, textvariable=model_type_var,
                 values=["pretrained", "custom"], state='readonly', width=24,
                 font=('Segoe UI', 9)).pack(anchor='w', padx=14, pady=2)

    sub_label(scroll_frame, "Weights Path:")
    weights_var = tk.StringVar()
    make_entry(scroll_frame, weights_var)
    make_btn(scroll_frame, "📁 Browse",
             lambda: weights_var.set(
                 filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth *.pt")]) or weights_var.get()
             ), '#78909c')

    sub_label(scroll_frame, "Class Names (comma):")
    classes_var = tk.StringVar()
    make_entry(scroll_frame, classes_var)

    def load_model_gui():
        try:
            sz = size_map.get(model_size_var.get(), "base")
            mt = model_type_var.get()
            status_label.config(text="⏳ LOADING...", fg='#ffb74d')
            root.update()
            if mt == "custom":
                wp = weights_var.get().strip()
                cn = [c.strip() for c in classes_var.get().split(',') if c.strip()]
                if not wp or not cn:
                    messagebox.showerror("Error", "ใส่ Weights + Classes")
                    status_label.config(text="⏹ READY", fg='#81c784')
                    return
                detector.load_model(sz, "custom", wp, cn)
            else:
                detector.load_model(sz, "pretrained")
            status_label.config(text="✅ LOADED", fg='#81c784')
            messagebox.showinfo("OK", f"โหลด {sz} ({mt}) สำเร็จ!")
        except Exception as e:
            status_label.config(text="❌ ERROR", fg=C['danger'])
            messagebox.showerror("Error", str(e))

    make_btn(scroll_frame, "🚀 Load Model", load_model_gui)

    # ========================================
    # SETTINGS
    # ========================================
    section_header(scroll_frame, "⚙️ Settings")
    sub_label(scroll_frame, "Confidence:")
    threshold_var = tk.DoubleVar(value=0.5)
    tk.Scale(scroll_frame, from_=0.1, to=1.0, resolution=0.05, orient='horizontal',
             variable=threshold_var, bg=C['card'], fg=C['text'], highlightthickness=0,
             troughcolor=C['border'], length=180, font=('Segoe UI', 8)
             ).pack(anchor='w', padx=14, pady=2)

    sub_label(scroll_frame, "Filter Classes (comma):")
    filter_var = tk.StringVar()
    make_entry(scroll_frame, filter_var)

    # ========================================
    # SOURCE
    # ========================================
    section_header(scroll_frame, "📷 Source")
    source_var = tk.StringVar(value="webcam")
    for t, v in [("🎥 Webcam", "webcam"), ("🖼️ Image", "image"), ("🎬 Video", "video")]:
        tk.Radiobutton(scroll_frame, text=t, variable=source_var, value=v,
                       bg=C['card'], fg=C['text'], selectcolor=C['card'],
                       activebackground=C['card'], font=('Segoe UI', 10),
                       highlightthickness=0).pack(anchor='w', padx=14, pady=1)

    # ========================================
    # FUNCTION
    # ========================================
    section_header(scroll_frame, "🎯 Function")
    func_var = tk.StringVar(value="detect")
    for t, v in [("🔍 Detect", "detect"), ("🔢 Counting", "counting"),
                 ("🚦 Passing Check", "passing")]:
        tk.Radiobutton(scroll_frame, text=t, variable=func_var, value=v,
                       bg=C['card'], fg=C['text'], selectcolor=C['card'],
                       activebackground=C['card'], font=('Segoe UI', 10),
                       highlightthickness=0).pack(anchor='w', padx=14, pady=1)

    # ========================================
    # ZONE OPTIONS (ใช้ร่วม Counting + Passing)
    # ========================================
    section_header(scroll_frame, "📐 Zone Options")
    tk.Label(scroll_frame, text="(ใช้กับ Counting + Passing)",
             fg=C['text2'], bg=C['card'], font=('Segoe UI', 8)
             ).pack(anchor='w', padx=14)

    sub_label(scroll_frame, "Zone Type:")
    zone_type_var = tk.StringVar(value="none")
    for t, v in [("❌ None (Count All)", "none"), ("── Line", "line"),
                 ("▭ Rectangle", "rect"), ("⬠ Polygon (Draw)", "polygon")]:
        tk.Radiobutton(scroll_frame, text=t, variable=zone_type_var, value=v,
                       bg=C['card'], fg=C['text'], selectcolor=C['card'],
                       activebackground=C['card'], font=('Segoe UI', 10),
                       highlightthickness=0).pack(anchor='w', padx=14, pady=1)

    sub_label(scroll_frame, "Line Y (%):")
    line_y_var = tk.IntVar(value=50)
    tk.Scale(scroll_frame, from_=10, to=90, orient='horizontal', variable=line_y_var,
             bg=C['card'], fg=C['text'], highlightthickness=0,
             troughcolor=C['border'], length=180, font=('Segoe UI', 8)
             ).pack(anchor='w', padx=14, pady=2)

    sub_label(scroll_frame, "Rect (x1,y1,x2,y2):")
    rect_var = tk.StringVar(value="200,200,800,500")
    make_entry(scroll_frame, rect_var)

    sub_label(scroll_frame, "Polygon:")
    polygon_display = tk.Label(scroll_frame, text="(ยังไม่ได้วาด)", fg=C['text2'],
                               bg=C['card'], font=('Segoe UI', 8),
                               wraplength=250, justify='left')
    polygon_display.pack(anchor='w', padx=14, pady=2)

    # ========================================
    # Draw Polygon
    # ========================================
    def start_polygon_draw():
        polygon_points.clear()
        cap_temp = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap_temp.isOpened():
            cap_temp = cv2.VideoCapture(0)
        if not cap_temp.isOpened():
            messagebox.showerror("Error", "เปิดกล้องไม่ได้")
            return
        ret, frame = cap_temp.read()
        cap_temp.release()
        if not ret:
            messagebox.showerror("Error", "อ่านเฟรมไม่ได้")
            return
        polygon_preview_img[0] = frame.copy()

        dw = tk.Toplevel(root)
        dw.title("⬠ Draw Polygon")
        dw.configure(bg=C['bg'])
        dw.geometry("900x700")
        dw.grab_set()

        tk.Label(dw, text="🖱️ คลิกวางจุด → Done เมื่อเสร็จ",
                 font=('Segoe UI', 11), fg=C['text'], bg=C['bg']).pack(pady=8)

        bf = tk.Frame(dw, bg=C['bg'])
        bf.pack(fill='x', padx=10)
        cf = tk.Frame(dw, bg='#000')
        cf.pack(fill='both', expand=True, padx=10, pady=(4, 10))
        dc = tk.Canvas(cf, bg='#000', highlightthickness=0)
        dc.pack(fill='both', expand=True)

        ds = [1.0]
        off = [0, 0]
        pr = [None]

        def render():
            img = polygon_preview_img[0].copy()
            hi, wi = img.shape[:2]
            for i, pt in enumerate(polygon_points):
                cv2.circle(img, pt, 6, (0, 200, 255), -1)
                cv2.circle(img, pt, 6, (0, 0, 0), 2)
                if i > 0:
                    cv2.line(img, polygon_points[i - 1], pt, (0, 255, 0), 2)
                cv2.putText(img, str(i + 1), (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if len(polygon_points) >= 3:
                cv2.line(img, polygon_points[-1], polygon_points[0], (0, 255, 0), 2)
                ov = img.copy()
                cv2.fillPoly(ov, [np.array(polygon_points)], (0, 255, 0))
                img = cv2.addWeighted(ov, 0.2, img, 0.8, 0)
            cw = dc.winfo_width()
            ch = dc.winfo_height()
            if cw < 10: cw = 850
            if ch < 10: ch = 550
            sc = min(cw / wi, ch / hi)
            ds[0] = sc
            nw, nh = int(wi * sc), int(hi * sc)
            off[0] = (cw - nw) // 2
            off[1] = (ch - nh) // 2
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (nw, nh))
            photo = ImageTk.PhotoImage(Image.fromarray(resized))
            pr[0] = photo
            dc.delete("all")
            dc.create_image(off[0], off[1], anchor='nw', image=photo)

        def click(e):
            ix = int((e.x - off[0]) / max(ds[0], 0.01))
            iy = int((e.y - off[1]) / max(ds[0], 0.01))
            hi, wi = polygon_preview_img[0].shape[:2]
            if 0 <= ix < wi and 0 <= iy < hi:
                polygon_points.append((ix, iy))
                render()
                upd()

        def undo():
            if polygon_points:
                polygon_points.pop()
                render()
                upd()

        def clear():
            polygon_points.clear()
            render()
            upd()

        def done():
            if len(polygon_points) < 3:
                messagebox.showwarning("Warning", "ต้อง >= 3 จุด")
                return
            upd()
            dw.destroy()
            messagebox.showinfo("OK", f"{len(polygon_points)} จุด บันทึกแล้ว!")

        def upd():
            if polygon_points:
                t = ", ".join(f"({x},{y})" for x, y in polygon_points)
                polygon_display.config(text=f"[{len(polygon_points)} pts] {t}")
            else:
                polygon_display.config(text="(ยังไม่ได้วาด)")

        dc.bind("<Button-1>", click)
        tk.Button(bf, text="✅ Done", command=done, bg=C['success'], fg='#fff',
                  font=('Segoe UI', 10, 'bold'), relief='flat', padx=16).pack(side='left', padx=4)
        tk.Button(bf, text="↩ Undo", command=undo, bg=C['warning'], fg='#fff',
                  font=('Segoe UI', 10, 'bold'), relief='flat', padx=16).pack(side='left', padx=4)
        tk.Button(bf, text="🗑 Clear", command=clear, bg=C['danger'], fg='#fff',
                  font=('Segoe UI', 10, 'bold'), relief='flat', padx=16).pack(side='left', padx=4)
        dw.after(100, render)

    make_btn(scroll_frame, "🎨 Draw Polygon", start_polygon_draw, '#7b1fa2')

    # ========================================
    # Frame Queue
    # ========================================
    _pf = [None]
    _pi = [None]

    def sched():
        if _pf[0] is not None:
            _upd_preview(_pf[0])
            _pf[0] = None
        if _pi[0] is not None:
            fv, no, ct, zinfo = _pi[0]
            fps_label.config(text=f"FPS: {fv:.1f}")
            obj_label.config(text=f"Objects: {no}")
            pass_label.config(text=zinfo)
            count_label.config(text=ct)
            _pi[0] = None
        root.after(33, sched)

    def _upd_preview(frame):
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cw = video_container.winfo_width() - 4
            ch = video_container.winfo_height() - 4
            if cw > 10 and ch > 10:
                fh, fw = img.shape[:2]
                sc = min(cw / fw, ch / fh)
                nw = min(max(int(fw * sc), 1), cw)
                nh = min(max(int(fh * sc), 1), ch)
                img = cv2.resize(img, (nw, nh))
            photo = ImageTk.PhotoImage(Image.fromarray(img))
            video_label.configure(image=photo, text="")
            video_label.image = photo
        except Exception:
            pass

    def qf(f):
        _pf[0] = f
        latest_frame[0] = f

    def qi(fv, no, ct="", zinfo=""):
        _pi[0] = (fv, no, ct, zinfo)

    sched()

    # ========================================
    # Import PolygonZoneCounter
    # ========================================
    from sut_rfdetr import PolygonZoneCounter

    # ========================================
    # Start / Stop
    # ========================================
    def start_detection():
        if detector.model is None:
            messagebox.showwarning("Warning", "โหลดโมเดลก่อน!")
            return
        if running.is_set():
            return

        running.set()
        status_label.config(text="▶ RUNNING", fg='#81c784')
        detector.threshold = threshold_var.get()
        detector.show_display = False

        ft = filter_var.get().strip()
        fcls = [c.strip() for c in ft.split(',') if c.strip()] if ft else None
        src = source_var.get()
        func = func_var.get()
        zt = zone_type_var.get()

        def run():
            try:
                # ==================== IMAGE ====================
                if src == "image":
                    path = filedialog.askopenfilename(
                        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
                    if not path:
                        running.clear()
                        root.after(0, lambda: status_label.config(text="⏹ READY", fg='#81c784'))
                        return
                    frame = cv2.imread(path)
                    if frame is None:
                        raise FileNotFoundError(f"ไม่พบ: {path}")
                    raw, _, _ = detector._predict(frame, threshold_var.get())
                    dets = detector._clean_detections(raw)
                    if fcls:
                        dets = detector._filter_by_class(dets, fcls)
                    labels = detector._make_labels(dets)
                    cc = detector._count_classes(dets)
                    extra = f"Count: {sum(cc.values())}" if func == "counting" else ""
                    ann = detector._annotate(frame, dets, labels, 0, extra)
                    qf(ann)
                    ct = " | ".join(f"{k}:{v}" for k, v in cc.items())
                    qi(0, len(dets), ct, "")
                    running.clear()
                    root.after(0, lambda: status_label.config(text="✅ DONE", fg='#81c784'))
                    return

                # ==================== VIDEO / WEBCAM ====================
                if src == "video":
                    path = filedialog.askopenfilename(
                        filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
                    if not path:
                        running.clear()
                        root.after(0, lambda: status_label.config(text="⏹ READY", fg='#81c784'))
                        return
                    cap = cv2.VideoCapture(path)
                else:
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(0)

                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap_ref[0] = cap
                if not cap.isOpened():
                    raise RuntimeError("เปิดไม่ได้")

                fw_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fh_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # --- Zone Setup ---
                use_tracker = func in ("passing", "counting") and zt != "none"
                tracker = sv.ByteTrack() if use_tracker else None
                line_zone = None
                line_ann = None
                poly_counter = None
                poly_ann = None

                detector._pass_in_count = 0
                detector._pass_out_count = 0
                detector._zone_entered_count = 0

                if (func in ("passing", "counting")) and zt != "none":
                    if zt == "line":
                        ly = int(fh_ * line_y_var.get() / 100)
                        line_zone = sv.LineZone(
                            start=sv.Point(0, ly), end=sv.Point(fw_, ly))
                        line_ann = sv.LineZoneAnnotator(thickness=3)

                    elif zt == "rect":
                        try:
                            parts = [int(x.strip()) for x in rect_var.get().split(',')]
                            x1, y1, x2, y2 = parts
                            pnp = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                        except Exception:
                            pnp = np.array([[200, 200], [800, 200], [800, 500], [200, 500]])
                        poly_counter = PolygonZoneCounter(polygon=pnp)
                        poly_ann = sv.PolygonZoneAnnotator(
                            zone=poly_counter.zone,
                            color=sv.Color.from_hex("#2196F3"), thickness=3)

                    elif zt == "polygon":
                        if len(polygon_points) >= 3:
                            pnp = np.array(polygon_points)
                            poly_counter = PolygonZoneCounter(polygon=pnp)
                            poly_ann = sv.PolygonZoneAnnotator(
                                zone=poly_counter.zone,
                                color=sv.Color.from_hex("#7b1fa2"), thickness=3)
                        else:
                            ly = int(fh_ * line_y_var.get() / 100)
                            line_zone = sv.LineZone(
                                start=sv.Point(0, ly), end=sv.Point(fw_, ly))
                            line_ann = sv.LineZoneAnnotator(thickness=3)

                prev = time.time()

                # ==================== MAIN LOOP ====================
                while running.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        if src == "video":
                            break
                        continue

                    raw, _, _ = detector._predict(frame, threshold_var.get())
                    dets = detector._clean_detections(raw)

                    if fcls:
                        dets = detector._filter_by_class(dets, fcls)

                    # Track
                    if tracker:
                        if len(dets) > 0:
                            ti = sv.Detections(
                                xyxy=dets.xyxy.copy(),
                                confidence=dets.confidence.copy(),
                                class_id=dets.class_id.copy())
                            dets = tracker.update_with_detections(ti)
                        else:
                            tracker.update_with_detections(sv.Detections.empty())

                    cur = time.time()
                    fps = 1 / max(cur - prev, 0.001)
                    prev = cur
                    fps_buffer.append(fps)
                    avg = sum(fps_buffer) / len(fps_buffer)

                    extra_parts = []
                    zone_info = ""

                    # Line Zone
                    if line_zone:
                        if len(dets) > 0:
                            line_zone.trigger(detections=dets)
                        detector._pass_in_count = line_zone.in_count
                        detector._pass_out_count = line_zone.out_count
                        extra_parts.append(
                            f"IN: {line_zone.in_count} | OUT: {line_zone.out_count}")
                        zone_info = f"IN: {line_zone.in_count} | OUT: {line_zone.out_count}"

                    # Polygon Zone (สะสม)
                    if poly_counter:
                        poly_counter.update(dets)
                        detector._zone_entered_count = poly_counter.total_entered
                        extra_parts.append(
                            f"Now: {poly_counter.in_zone_count} | "
                            f"Entered: {poly_counter.total_entered} | "
                            f"Exited: {poly_counter.total_exited}")
                        zone_info = (
                            f"Now: {poly_counter.in_zone_count} | "
                            f"Entered: {poly_counter.total_entered} | "
                            f"Exited: {poly_counter.total_exited}")

                    # Counting (no zone)
                    cc = detector._count_classes(dets)
                    if not line_zone and not poly_counter:
                        extra_parts.append(f"Count: {sum(cc.values())}")
                        zone_info = f"Count: {sum(cc.values())}"

                    extra = " | ".join(extra_parts)
                    labels = detector._make_labels(dets)
                    ann = detector._annotate(frame, dets, labels, avg, extra)

                    if line_ann and line_zone:
                        ann = line_ann.annotate(ann, line_zone)
                    if poly_ann and poly_counter:
                        ann = poly_ann.annotate(ann)

                    # Class count overlay
                    y = 60
                    for cn, cnt in cc.items():
                        cv2.putText(ann, f"{cn}: {cnt}", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 180), 2)
                        y += 25

                    ct = " | ".join(f"{k}:{v}" for k, v in cc.items())
                    qf(ann)
                    qi(avg, len(dets), ct, zone_info)
                    detector._update_web(avg, dets, cc, source=src)

                cap.release()
                cap_ref[0] = None

            except Exception as e:
                import traceback
                err = f"{str(e)}\n\n{traceback.format_exc()}"
                root.after(0, lambda m=err: messagebox.showerror("Error", m))
            finally:
                running.clear()
                root.after(0, lambda: status_label.config(text="⏹ STOPPED", fg=C['danger']))

        threading.Thread(target=run, daemon=True).start()

    def stop_detection():
        running.clear()
        if cap_ref[0]:
            try:
                cap_ref[0].release()
            except Exception:
                pass
            cap_ref[0] = None
        status_label.config(text="⏹ STOPPED", fg=C['danger'])

    def save_screenshot():
        if latest_frame[0] is not None:
            p = filedialog.asksaveasfilename(defaultextension=".jpg",
                                             filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
            if p:
                cv2.imwrite(p, latest_frame[0])
                messagebox.showinfo("OK", f"บันทึกที่: {p}")
        else:
            messagebox.showinfo("Info", "ยังไม่มีภาพ")

    start_btn.config(command=start_detection)
    stop_btn.config(command=stop_detection)
    screenshot_btn.config(command=save_screenshot)

    def on_closing():
        running.clear()
        if cap_ref[0]:
            try:
                cap_ref[0].release()
            except Exception:
                pass
        try:
            detector.stop()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()