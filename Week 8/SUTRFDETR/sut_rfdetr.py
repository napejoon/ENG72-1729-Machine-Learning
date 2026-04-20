"""
SUT_RFDETR — Python Class API for RF-DETR Object Detection
Suranaree University of Technology
"""
import cv2
import time
import threading
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import supervision as sv


# ========================================
# Data Classes
# ========================================
@dataclass
class FrameResult:
    """ผลลัพธ์การตรวจจับในแต่ละเฟรม"""
    frame_number: int = 0
    class_ids: List[int] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    bboxes: np.ndarray = None
    confidences: np.ndarray = None
    class_count: Dict[str, int] = field(default_factory=dict)
    total_objects: int = 0
    fps: float = 0.0
    frame: np.ndarray = None
    raw_frame: np.ndarray = None


@dataclass
class DetectionResult:
    """ผลลัพธ์การตรวจจับรวม"""
    frame: np.ndarray = None
    raw_frame: np.ndarray = None
    detections: object = None
    labels: List[str] = field(default_factory=list)
    class_count: Dict[str, int] = field(default_factory=dict)
    total_objects: int = 0
    fps: float = 0.0
    class_ids: List[int] = field(default_factory=list)
    class_names_found: List[str] = field(default_factory=list)
    bboxes: np.ndarray = None
    confidences: np.ndarray = None
    frame_results: List[FrameResult] = field(default_factory=list)
    total_frames: int = 0


@dataclass
class CountingResult:
    total: int = 0
    by_class: Dict[str, int] = field(default_factory=dict)
    frame: Optional[np.ndarray] = None
    detections: object = None
    passed_in: int = 0
    passed_out: int = 0
    zone_total: int = 0


@dataclass
class PassingResult:
    passed_in: int = 0
    passed_out: int = 0
    total_passed: int = 0
    by_class: Dict[str, int] = field(default_factory=dict)
    frame: Optional[np.ndarray] = None


# ========================================
# Model Loader
# ========================================
def _load_rfdetr_model(size: str, num_classes: int = 80, weights: str = None):
    size = size.lower().strip()
    model_map = {}
    import_map = {
        "n": "RFDETRNano", "nano": "RFDETRNano",
        "s": "RFDETRSmall", "small": "RFDETRSmall",
        "m": "RFDETRMedium", "medium": "RFDETRMedium",
        "b": "RFDETRBase", "base": "RFDETRBase",
        "l": "RFDETRLarge", "large": "RFDETRLarge",
        "x": "RFDETRXtraLarge", "xl": "RFDETRXtraLarge",
        "xtralarge": "RFDETRXtraLarge",
    }
    import rfdetr
    for key, class_name in import_map.items():
        try:
            cls = getattr(rfdetr, class_name)
            model_map[key] = cls
        except AttributeError:
            pass
    if size not in model_map:
        available = sorted(set(k for k in model_map.keys() if len(k) <= 6))
        raise ValueError(
            f"❌ Model '{size}' not available. Available: {available}\n"
            f"   Installed: {[v.__name__ for v in set(model_map.values())]}"
        )
    model_cls = model_map[size]
    print(f"   → Using {model_cls.__name__}")
    if weights:
        model = model_cls(num_classes=num_classes, pretrain_weights=weights)
    else:
        model = model_cls()
    try:
        model.optimize_for_inference()
        print("   → Optimized ✅")
    except Exception:
        pass
    return model


def _get_coco_classes():
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
        return COCO_CLASSES
    except ImportError:
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES
            return COCO_CLASSES
        except ImportError:
            return [f"class_{i}" for i in range(80)]


# ========================================
# Polygon Zone Tracker
# ========================================
class PolygonZoneCounter:
    def __init__(self, polygon: np.ndarray):
        self.polygon_zone = sv.PolygonZone(polygon=polygon)
        self.counted_ids = set()
        self.in_zone_count = 0
        self.total_entered = 0
        self.total_exited = 0
        self._prev_in_zone_ids = set()

    def update(self, detections) -> int:
        if len(detections) == 0:
            newly_exited = self._prev_in_zone_ids
            self.total_exited += len(newly_exited)
            self._prev_in_zone_ids = set()
            self.in_zone_count = 0
            return 0
        is_in_zone = self.polygon_zone.trigger(detections=detections)
        if is_in_zone is None:
            return 0
        current_in_zone_ids = set()
        for i, in_zone in enumerate(is_in_zone):
            if not in_zone:
                continue
            tid = None
            if detections.tracker_id is not None and i < len(detections.tracker_id):
                tid = int(detections.tracker_id[i])
            if tid is not None:
                current_in_zone_ids.add(tid)
                if tid not in self.counted_ids:
                    self.counted_ids.add(tid)
                    self.total_entered += 1
        newly_exited = self._prev_in_zone_ids - current_in_zone_ids
        self.total_exited += len(newly_exited)
        self._prev_in_zone_ids = current_in_zone_ids
        self.in_zone_count = len(current_in_zone_ids)
        return self.in_zone_count

    @property
    def zone(self):
        return self.polygon_zone


# ========================================
# Main Class
# ========================================
class SUT_RFDETR:
    def __init__(self, mode: str = "code", web_monitor: bool = True, web_port: int = 5555):
        self.mode = mode.lower()
        self.model = None
        self.model_size = None
        self.model_type = None
        self.class_names: List[str] = []
        self.threshold = 0.5
        self.show_display = True
        self._box_annotator = sv.BoxAnnotator(thickness=2)
        self._label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self._pass_in_count = 0
        self._pass_out_count = 0
        self._zone_entered_count = 0
        self._web_monitor = None
        self._frame_count = 0

        # ===== Real-time result access =====
        self.latest_result: Optional[FrameResult] = None
        self._on_frame_callback: Optional[Callable] = None
        self._is_running = False

        if web_monitor:
            try:
                from web_monitor import WebMonitor
                self._web_monitor = WebMonitor(port=web_port)
                self._web_monitor.start()
            except Exception as e:
                print(f"⚠️ Web Monitor: {e}")

        self._stats = {
            'fps': 0, 'total_objects': 0, 'unique_classes': 0,
            'class_distribution': {}, 'pass_in': 0, 'pass_out': 0,
            'source': '', 'model': '', 'frame_count': 0
        }

        if self.mode == "gui":
            print("🖥️  GUI Mode — เรียก .launch() เพื่อเปิดหน้าต่าง")
        else:
            print("⌨️  Code Mode — พร้อมใช้งาน")

    # ========================================
    # Callback สำหรับรับผลแบบ Real-time
    # ========================================
    def on_frame(self, callback: Callable):
        """
        ตั้ง callback function ที่จะถูกเรียกทุกเฟรม
        callback จะได้รับ FrameResult

        Usage:
            def my_callback(result):
                print(f"Found {result.total_objects} objects")
                print(f"Classes: {result.class_names}")
                print(f"BBoxes: {result.bboxes}")

            det.on_frame(my_callback)
            det.detect_webcam(0)
        """
        self._on_frame_callback = callback
        return self

    # ========================================
    # Camera
    # ========================================
    def _open_camera(self, camera_id: int) -> cv2.VideoCapture:
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto"),
        ]
        for backend, name in backends:
            cap = cv2.VideoCapture(camera_id, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"📷 กล้อง {camera_id} (backend: {name})")
                return cap
            cap.release()
        raise RuntimeError(f"❌ เปิดกล้อง {camera_id} ไม่ได้")

    def _open_source(self, source) -> cv2.VideoCapture:
        if isinstance(source, int):
            return self._open_camera(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"เปิดไม่ได้: {source}")
        return cap

    # ========================================
    # Model
    # ========================================
    def load_model(self, size="base", model_type="pretrained",
                   weights_path=None, class_names=None, num_classes=None):
        print(f"📦 โหลด RF-DETR ({size})...")
        if model_type == "pretrained":
            self.model = _load_rfdetr_model(size)
            self.class_names = _get_coco_classes()
            self.model_type = "pretrained"
        elif model_type == "custom":
            if not weights_path:
                raise ValueError("ต้องระบุ weights_path")
            if not class_names:
                raise ValueError("ต้องระบุ class_names")
            n = num_classes or len(class_names)
            self.model = _load_rfdetr_model(size, n, weights_path)
            self.class_names = class_names
            self.model_type = "custom"
        else:
            raise ValueError("model_type: 'pretrained' หรือ 'custom'")
        self.model_size = size
        self._stats['model'] = f"RF-DETR-{size} ({model_type})"
        print(f"✅ สำเร็จ! ({len(self.class_names)} classes)")
        self._log("Model Loaded", f"{size}/{model_type}")
        return self

    def set_classes(self, class_names):
        self.class_names = class_names
        return self

    def set_threshold(self, threshold):
        self.threshold = threshold
        return self

    def set_display(self, show):
        self.show_display = show
        return self

    # ========================================
    # Core
    # ========================================
    def _predict(self, frame, threshold=None):
        if self.model is None:
            raise RuntimeError("โหลดโมเดลก่อน!")
        th = threshold or self.threshold
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = self.model.predict(rgb, threshold=th)
        return dets, self._make_labels(dets), self._count_classes(dets)

    def _make_labels(self, dets):
        if dets is None or len(dets) == 0:
            return []
        labels = []
        for cid, conf in zip(dets.class_id, dets.confidence):
            name = self.class_names[cid] if cid < len(self.class_names) else f"id:{cid}"
            labels.append(f"{name} {conf:.2f}")
        return labels

    def _count_classes(self, dets):
        cc = defaultdict(int)
        if dets is None or len(dets) == 0:
            return dict(cc)
        for cid in dets.class_id:
            name = self.class_names[cid] if cid < len(self.class_names) else f"id:{cid}"
            cc[name] += 1
        return dict(cc)

    def _filter_by_class(self, dets, classes):
        if not classes or dets is None or len(dets) == 0:
            return dets
        keep = []
        for i, cid in enumerate(dets.class_id):
            name = self.class_names[cid] if cid < len(self.class_names) else f"id:{cid}"
            if name in classes:
                keep.append(i)
        if not keep:
            return sv.Detections.empty()
        keep = np.array(keep)
        return sv.Detections(
            xyxy=dets.xyxy[keep].copy(),
            confidence=dets.confidence[keep].copy(),
            class_id=dets.class_id[keep].copy()
        )

    def _clean_detections(self, dets):
        if dets is None or len(dets) == 0:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=dets.xyxy.copy(),
            confidence=dets.confidence.copy(),
            class_id=dets.class_id.copy()
        )

    def _extract_detection_data(self, dets):
        if dets is None or len(dets) == 0:
            return [], [], np.empty((0, 4)), np.empty(0)
        class_ids = dets.class_id.tolist()
        class_names_found = [
            self.class_names[cid] if cid < len(self.class_names) else f"id:{cid}"
            for cid in class_ids
        ]
        bboxes = dets.xyxy.copy()
        confidences = dets.confidence.copy()
        return class_ids, class_names_found, bboxes, confidences

    def _annotate(self, frame, dets, labels, fps=0, extra=""):
        annotated = frame.copy()
        if dets is not None and len(dets) > 0:
            annotated = self._box_annotator.annotate(annotated, dets)
            annotated = self._label_annotator.annotate(annotated, dets, labels)
        info = f"FPS: {fps:.1f} | Objects: {len(dets) if dets else 0}"
        if extra:
            info += f" | {extra}"
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated

    def _update_web(self, fps, dets, cc, source=""):
        self._frame_count += 1
        self._stats.update({
            'fps': round(fps, 1),
            'total_objects': len(dets) if dets else 0,
            'unique_classes': len(cc),
            'class_distribution': cc,
            'pass_in': self._pass_in_count,
            'pass_out': self._pass_out_count,
            'source': source, 'frame_count': self._frame_count
        })
        if self._web_monitor:
            self._web_monitor.send_update(self._stats)

    def _log(self, event, detail=""):
        if self._web_monitor:
            self._web_monitor.send_log(event, detail)

    # ========================================
    # Zone Setup Helper
    # ========================================
    def _setup_zones(self, w, h, line_start=None, line_end=None,
                     roi_polygon=None, roi_rect=None):
        line_zone = None
        line_ann = None
        poly_counter = None
        poly_ann = None
        if line_start is None and line_end is None and roi_polygon is None and roi_rect is None:
            line_start = (0, h // 2)
            line_end = (w, h // 2)
        if line_start and line_end:
            line_zone = sv.LineZone(
                start=sv.Point(line_start[0], line_start[1]),
                end=sv.Point(line_end[0], line_end[1])
            )
            line_ann = sv.LineZoneAnnotator(thickness=3)
        if roi_rect:
            x1, y1, x2, y2 = roi_rect
            roi_polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if roi_polygon:
            poly_np = np.array(roi_polygon, dtype=np.int32)
            poly_counter = PolygonZoneCounter(polygon=poly_np)
            poly_ann = sv.PolygonZoneAnnotator(
                zone=poly_counter.zone,
                color=sv.Color.from_hex("#00d4ff"),
                thickness=3
            )
        return line_zone, line_ann, poly_counter, poly_ann

    # ========================================
    # Detection Loop — Real-time callback
    # ========================================
    def _run_detection_loop(
        self, cap, threshold=None, classes=None,
        line_zone=None, line_ann=None,
        poly_counter=None, poly_ann=None,
        use_tracker=False, mode_name="Detect",
        save_path=None, source_label=""
    ) -> DetectionResult:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_path:
            writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h)
            )
        tracker = sv.ByteTrack() if use_tracker else None
        self._pass_in_count = 0
        self._pass_out_count = 0
        self._zone_entered_count = 0
        self._is_running = True
        prev = time.time()

        all_frame_results: List[FrameResult] = []
        total_class_count = defaultdict(int)
        frame_number = 0
        fps_list = []
        last_frame = None
        last_raw_frame = None
        last_dets = None
        last_labels = []

        print(f"🚀 {mode_name} — {source_label} — กด 'q' เพื่อหยุด")

        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                if not isinstance(source_label, str) or "Cam" not in source_label:
                    break
                continue

            frame_number += 1

            # Step 1: Detect + Clean
            raw, _, _ = self._predict(frame, threshold)
            dets = self._clean_detections(raw)

            # Step 2: Filter class
            if classes:
                dets = self._filter_by_class(dets, classes)

            # Step 3: Track
            if tracker:
                if len(dets) > 0:
                    track_in = sv.Detections(
                        xyxy=dets.xyxy.copy(),
                        confidence=dets.confidence.copy(),
                        class_id=dets.class_id.copy()
                    )
                    dets = tracker.update_with_detections(track_in)
                else:
                    tracker.update_with_detections(sv.Detections.empty())

            # FPS
            cur = time.time()
            fps = 1 / max(cur - prev, 0.001)
            prev = cur
            fps_list.append(fps)

            # ===== สร้าง FrameResult =====
            cids, cnames, bboxes, confs = self._extract_detection_data(dets)
            cc = self._count_classes(dets)

            extra_parts = []

            # Step 4: Line Zone
            if line_zone:
                if len(dets) > 0:
                    line_zone.trigger(detections=dets)
                self._pass_in_count = line_zone.in_count
                self._pass_out_count = line_zone.out_count
                extra_parts.append(
                    f"IN: {line_zone.in_count} | OUT: {line_zone.out_count}"
                )

            # Step 5: Polygon Zone
            if poly_counter:
                poly_counter.update(dets)
                self._zone_entered_count = poly_counter.total_entered
                extra_parts.append(
                    f"Zone Now: {poly_counter.in_zone_count} | "
                    f"Entered: {poly_counter.total_entered} | "
                    f"Exited: {poly_counter.total_exited}"
                )

            # Step 6: Counting
            if not line_zone and not poly_counter:
                extra_parts.append(f"Count: {sum(cc.values())}")
            extra = " | ".join(extra_parts)

            # Step 7: Annotate
            labels = self._make_labels(dets)
            annotated = self._annotate(frame, dets, labels, fps, extra)

            if line_ann and line_zone:
                annotated = line_ann.annotate(annotated, line_zone)
            if poly_ann and poly_counter:
                annotated = poly_ann.annotate(annotated)

            y_pos = 60
            for cn, cnt in cc.items():
                cv2.putText(annotated, f"{cn}: {cnt}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 180), 2)
                y_pos += 25

            # ===== สร้าง FrameResult พร้อมข้อมูลครบ =====
            frame_result = FrameResult(
                frame_number=frame_number,
                class_ids=cids,
                class_names=cnames,
                bboxes=bboxes,
                confidences=confs,
                class_count=cc,
                total_objects=len(dets) if dets else 0,
                fps=round(fps, 1),
                frame=annotated,
                raw_frame=frame
            )

            # ===== อัปเดต latest_result ทันที =====
            self.latest_result = frame_result

            # ===== เรียก callback ทุกเฟรม =====
            if self._on_frame_callback:
                try:
                    self._on_frame_callback(frame_result)
                except Exception as e:
                    print(f"⚠️ Callback error: {e}")

            # ===== Print ผลทุกเฟรม (real-time) =====
            if frame_number % 30 == 1:
                print(
                    f"  📊 Frame {frame_number} | "
                    f"FPS: {fps:.1f} | "
                    f"Objects: {frame_result.total_objects} | "
                    f"Classes: {cnames}"
                )

            all_frame_results.append(frame_result)
            for cn, cnt in cc.items():
                total_class_count[cn] += cnt

            last_frame = annotated
            last_raw_frame = frame
            last_dets = dets
            last_labels = labels

            self._update_web(fps, dets, cc, source=source_label)

            if writer:
                writer.write(annotated)

            if self.show_display:
                cv2.imshow(f"SUT RF-DETR - {mode_name}", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self._is_running = False
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # ===== Summary =====
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        last_cids, last_cnames, last_bboxes, last_confs = \
            self._extract_detection_data(last_dets)

        print(f"\n{'='*50}")
        print(f"📊 สรุปผล {mode_name}")
        print(f"{'='*50}")
        print(f"  Total Frames : {frame_number}")
        print(f"  Average FPS  : {avg_fps:.1f}")
        print(f"  Last Objects : {len(last_dets) if last_dets else 0}")
        print(f"  Total Count  : {dict(total_class_count)}")
        if self._pass_in_count or self._pass_out_count:
            print(f"  Pass IN      : {self._pass_in_count}")
            print(f"  Pass OUT     : {self._pass_out_count}")
        if self._zone_entered_count:
            print(f"  Zone Entered : {self._zone_entered_count}")
        print(f"{'='*50}\n")

        return DetectionResult(
            frame=last_frame,
            raw_frame=last_raw_frame,
            detections=last_dets,
            labels=last_labels,
            class_count=dict(total_class_count),
            total_objects=len(last_dets) if last_dets else 0,
            fps=round(avg_fps, 1),
            class_ids=last_cids,
            class_names_found=last_cnames,
            bboxes=last_bboxes,
            confidences=last_confs,
            frame_results=all_frame_results,
            total_frames=frame_number
        )

    # ========================================
    # Detect Image
    # ========================================
    def detect_image(self, image_path, threshold=None, save_path=None) -> DetectionResult:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"ไม่พบ: {image_path}")

        t0 = time.time()
        dets, labels, cc = self._predict(frame, threshold)
        fps = 1 / max(time.time() - t0, 0.001)
        annotated = self._annotate(frame, dets, labels, fps)
        self._update_web(fps, dets, cc, source=f"Image: {Path(image_path).name}")

        if save_path:
            cv2.imwrite(save_path, annotated)
        if self.show_display:
            cv2.imshow("SUT RF-DETR - Image", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        class_ids, class_names_found, bboxes, confidences = \
            self._extract_detection_data(dets)

        result = DetectionResult(
            frame=annotated, raw_frame=frame, detections=dets,
            labels=labels, class_count=cc,
            total_objects=len(dets), fps=fps,
            class_ids=class_ids,
            class_names_found=class_names_found,
            bboxes=bboxes, confidences=confidences,
            frame_results=[], total_frames=1
        )

        # ===== เรียก callback =====
        if self._on_frame_callback:
            frame_result = FrameResult(
                frame_number=1,
                class_ids=class_ids,
                class_names=class_names_found,
                bboxes=bboxes, confidences=confidences,
                class_count=cc,
                total_objects=len(dets),
                fps=fps, frame=annotated, raw_frame=frame
            )
            self.latest_result = frame_result
            try:
                self._on_frame_callback(frame_result)
            except Exception as e:
                print(f"⚠️ Callback error: {e}")

        # ===== Print ผลทันที =====
        print(f"\n📊 ผลการตรวจจับ: {Path(image_path).name}")
        print(f"  Objects : {len(dets)}")
        print(f"  Classes : {class_names_found}")
        print(f"  Count   : {cc}")
        print(f"  FPS     : {fps:.1f}")

        return result

    # ========================================
    # Detect Video
    # ========================================
    def detect_video(self, video_path, threshold=None, save_path=None) -> DetectionResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"เปิดไม่ได้: {video_path}")
        result = self._run_detection_loop(
            cap, threshold=threshold, mode_name="Video",
            save_path=save_path, source_label=f"Video: {Path(video_path).name}"
        )
        return result

    # ========================================
    # Detect Webcam
    # ========================================
    def detect_webcam(self, camera_id=0, threshold=None, save_path=None) -> DetectionResult:
        cap = self._open_camera(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        result = self._run_detection_loop(
            cap, threshold=threshold, mode_name="Webcam",
            save_path=save_path, source_label=f"Cam {camera_id}"
        )
        return result

    # ========================================
    # Counting
    # ========================================
    def counting(self, source, classes=None, threshold=None, save_path=None,
                 line_start=None, line_end=None,
                 roi_polygon=None, roi_rect=None):
        has_zone = any([line_start, line_end, roi_polygon, roi_rect])

        if not has_zone:
            if isinstance(source, np.ndarray):
                frame = source
            elif isinstance(source, str):
                frame = cv2.imread(source)
                if frame is None:
                    raise FileNotFoundError(f"ไม่พบ: {source}")
            elif isinstance(source, int):
                cap = self._open_camera(source)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError("อ่านเฟรมไม่ได้")
            else:
                raise ValueError("source ไม่ถูกต้อง")

            dets, labels, cc = self._predict(frame, threshold)
            if classes:
                dets = self._filter_by_class(dets, classes)
                labels = self._make_labels(dets)
                cc = self._count_classes(dets)
            total = sum(cc.values())
            annotated = self._annotate(frame, dets, labels, extra=f"Count: {total}")
            y = 60
            for cn, cnt in cc.items():
                cv2.putText(annotated, f"{cn}: {cnt}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 180), 2)
                y += 25
            if save_path:
                cv2.imwrite(save_path, annotated)
            if self.show_display:
                cv2.imshow("SUT RF-DETR - Counting", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return CountingResult(total=total, by_class=cc, frame=annotated, detections=dets)

        cap = self._open_source(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        lz, la, pc, pa = self._setup_zones(w, h, line_start, line_end, roi_polygon, roi_rect)
        self._run_detection_loop(
            cap, threshold=threshold, classes=classes,
            line_zone=lz, line_ann=la,
            poly_counter=pc, poly_ann=pa,
            use_tracker=True, mode_name="Counting + Zone",
            save_path=save_path, source_label=f"Counting: {source}"
        )
        return CountingResult(
            total=self._pass_in_count + self._zone_entered_count,
            passed_in=self._pass_in_count,
            passed_out=self._pass_out_count,
            zone_total=self._zone_entered_count
        )

    def counting_realtime(self, camera_id=0, classes=None, threshold=None,
                          save_path=None, line_start=None, line_end=None,
                          roi_polygon=None, roi_rect=None):
        return self.counting(
            source=camera_id, classes=classes, threshold=threshold,
            save_path=save_path, line_start=line_start, line_end=line_end,
            roi_polygon=roi_polygon, roi_rect=roi_rect
        )

    # ========================================
    # Passing Check
    # ========================================
    def passing_check(self, source=0, line_start=None, line_end=None,
                      roi_polygon=None, roi_rect=None,
                      classes=None, threshold=None, save_path=None):
        if self.model is None:
            raise RuntimeError("โหลดโมเดลก่อน!")
        cap = self._open_source(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        lz, la, pc, pa = self._setup_zones(w, h, line_start, line_end, roi_polygon, roi_rect)
        src_name = f"Cam {source}" if isinstance(source, int) else Path(source).name
        self._run_detection_loop(
            cap, threshold=threshold, classes=classes,
            line_zone=lz, line_ann=la,
            poly_counter=pc, poly_ann=pa,
            use_tracker=True, mode_name="Passing Check",
            save_path=save_path, source_label=f"Passing: {src_name}"
        )
        return PassingResult(
            passed_in=self._pass_in_count + (pc.total_entered if pc else 0),
            passed_out=self._pass_out_count + (pc.total_exited if pc else 0),
            total_passed=self._pass_in_count + self._pass_out_count + (
                pc.total_entered if pc else 0
            ),
        )

    # ========================================
    # GUI
    # ========================================
    def launch(self):
        if self.mode != "gui":
            print("⚠️ ต้องสร้างด้วย mode='gui'")
            return
        from gui_app import launch_gui
        launch_gui(self)

    def stop(self):
        self._is_running = False
        if self._web_monitor:
            self._web_monitor.stop()
        cv2.destroyAllWindows()
        print("👋 หยุดแล้ว")