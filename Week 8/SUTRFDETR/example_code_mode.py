"""
SUT_RFDETR — Code Mode Examples
"""
from sut_rfdetr import SUT_RFDETR
import cv2


# ============================================================
# สร้าง + โหลดโมเดล
# ============================================================

det = SUT_RFDETR(mode="code", web_monitor=True)
det.load_model("base", model_type="pretrained")
det.set_threshold(0.5)
det.set_display(True)


# ============================================================
# Detect — รูปภาพ / วิดีโอ / กล้อง
# ============================================================

result = det.detect_image("test.jpg")
print(result.total_objects, result.class_count)

result = det.detect_image("test.jpg", threshold=0.3, save_path="output.jpg")

det.detect_video("test.mp4", save_path="output.mp4")

det.detect_webcam(camera_id=0)


# ============================================================
# Counting — นับวัตถุ (ไม่ใช้ Zone)
# ============================================================

result = det.counting("test.jpg")
print(result.total, result.by_class)

result = det.counting("test.jpg", classes=["person", "car"])
print(result.total, result.by_class)

result = det.counting(0)
print(result.by_class)

frame = cv2.imread("test.jpg")
result = det.counting(frame, classes=["person"], save_path="count.jpg")


# ============================================================
# Counting + Zone — นับด้วยเส้น / พื้นที่
# ============================================================

det.counting(0, line_start=(0, 400), line_end=(1280, 400))

det.counting(0, roi_rect=(200, 200, 800, 500))

det.counting(0, roi_polygon=[
    (100, 100), (600, 100), (600, 500), (100, 500)
])

det.counting(0, roi_polygon=[
    (100, 100), (600, 100), (600, 500), (100, 500)
], classes=["person"])


# ============================================================
# Passing Check — Line Crossing
# ============================================================

result = det.passing_check(
    source=0,
    line_start=(0, 400),
    line_end=(1280, 400)
)
print(result.passed_in, result.passed_out, result.total_passed)

result = det.passing_check(
    source=0,
    line_start=(0, 350),
    line_end=(1280, 350),
    classes=["person"],
    threshold=0.4
)

result = det.passing_check(
    source="conveyor.mp4",
    line_start=(0, 300),
    line_end=(1280, 300),
    save_path="passing_output.mp4"
)


# ============================================================
# Passing Check — Rectangle Zone
# ============================================================

result = det.passing_check(
    source=0,
    roi_rect=(200, 200, 800, 500)
)


# ============================================================
# Passing Check — Polygon Zone (สะสมเมื่อเข้า ไม่ reset เมื่อออก)
# ============================================================

result = det.passing_check(
    source=0,
    roi_polygon=[
        (100, 100), (600, 80), (650, 500), (80, 480)
    ]
)
print(result.passed_in, result.passed_out)

result = det.passing_check(
    source="conveyor.mp4",
    roi_polygon=[
        (200, 150), (700, 150), (750, 550), (150, 550)
    ],
    classes=["product_a", "product_b"],
    save_path="polygon_check.mp4"
)


# ============================================================
# Headless Mode (ไม่เปิดหน้าต่าง)
# ============================================================

det.set_display(False)

result = det.counting("test.jpg", classes=["person"])
print(result.total, result.by_class)
cv2.imwrite("headless_output.jpg", result.frame)


# ============================================================
# Custom Model — งาน Conveyor
# ============================================================

det2 = SUT_RFDETR(mode="code", web_monitor=False)
det2.load_model(
    size="base",
    model_type="custom",
    weights_path="models/conveyor_best.pth",
    class_names=["good_product", "defect", "empty_box"]
)
det2.set_threshold(0.4)

det2.counting(0, classes=["good_product", "defect"],
              line_start=(0, 400), line_end=(1280, 400))

result = det2.passing_check(
    source=0,
    roi_polygon=[(300, 200), (900, 200), (950, 600), (250, 600)],
    classes=["good_product"]
)
print(result.passed_in, result.passed_out)


# ============================================================
# Pipeline อัตโนมัติ
# ============================================================

def auto_inspect(image_path):
    d = SUT_RFDETR(mode="code", web_monitor=False)
    d.load_model("base", "pretrained")
    d.set_display(False)
    result = d.counting(image_path)
    return {
        "total": result.total,
        "details": result.by_class,
        "status": "PASS" if result.total > 0 else "FAIL"
    }


# ============================================================
# หยุด
# ============================================================

det.stop()