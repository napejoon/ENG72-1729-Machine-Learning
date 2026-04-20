from sut_rfdetr import SUT_RFDETR

det = SUT_RFDETR(mode="code", web_monitor=False)
det.load_model("n", model_type="pretrained")
det.set_threshold(0.5)

def on_detect(result):
    print(f"Frame {result.frame_number}: "
          f"{result.total_objects} objects | "
          f"Classes: {result.class_names} | "
          f"FPS: {result.fps}")

    for i in range(result.total_objects):
        print(f"  [{result.class_names[i]}] "
              f"bbox={result.bboxes[i]} "
              f"conf={result.confidences[i]:.2f}")

det.on_frame(on_detect)
res = det.detect_webcam(camera_id=0)