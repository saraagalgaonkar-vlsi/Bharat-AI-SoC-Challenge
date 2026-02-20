import sys

sys.path.append(r"D:\arm_accelerator\yolo_nngen\PyTorch-YOLOv3")

from pytorchyolo.models import Darknet

model = Darknet(r"D:\arm_accelerator\yolo_nngen\cfg\yolov3-tiny.cfg")
model.load_darknet_weights(r"D:\arm_accelerator\yolo_nngen\weights\yolov3-tiny.weights")

model.eval()

print("YOLO Tiny Model Loaded Successfully")
