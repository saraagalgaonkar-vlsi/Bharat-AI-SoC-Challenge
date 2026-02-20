import sys
import torch
import torch.nn as nn

sys.path.append(r"D:\arm_accelerator\yolo_nngen\PyTorch-YOLOv3")

from pytorchyolo.models import Darknet


class YoloBackboneOnly(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = Darknet(
            r"D:\arm_accelerator\yolo_nngen\cfg\yolov3-tiny.cfg"
        )

        self.model.load_darknet_weights(
            r"D:\arm_accelerator\yolo_nngen\weights\yolov3-tiny.weights"
        )

        self.model.eval()

        # Keep only backbone layers (cut before YOLO detection head)
        # Usually last few layers are YOLO + routing
        self.backbone = self.model.module_list[:13]   # <-- SAFE CUT (we can tune)

    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        return x


if __name__ == "__main__":
    net = YoloBackboneOnly()
    dummy = torch.randn(1,3,416,416)
    out = net(dummy)
    print("Backbone Output Shape:", out.shape)
