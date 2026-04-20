"""
ตัวอย่างเปิด GUI Mode
"""
from sut_rfdetr import SUT_RFDETR

# สร้างและเปิด GUI
det = SUT_RFDETR(mode="gui", web_monitor=True)
det.launch()