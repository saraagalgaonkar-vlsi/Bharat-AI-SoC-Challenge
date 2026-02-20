import torch
from yolo_backbone_split import YoloBackboneOnly

model = YoloBackboneOnly()
model.eval()

dummy = torch.randn(1,3,416,416)

torch.onnx.export(
    model,
    dummy,
    r"D:\arm_accelerator\yolo_nngen\onnx_models\yolo_split_backbone.onnx",
    opset_version=11
)

print("Split Backbone ONNX Export Done")
