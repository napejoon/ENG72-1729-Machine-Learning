from sut_rfdetr import SUT_RFDETR
import cv2

det = SUT_RFDETR(mode="code")
det.load_model("n", model_type="pretrained")
det.set_threshold(0.5)
det.set_display(True)
res = det.detect_webcam(camera_id=0)


