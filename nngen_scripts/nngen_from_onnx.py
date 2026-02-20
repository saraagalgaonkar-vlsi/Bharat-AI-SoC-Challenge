import nngen as ng

onnx_path = r"D:\arm_accelerator\yolo_nngen\onnx_models\yolo_split_backbone.onnx"

outputs = ng.from_onnx(onnx_path)

print("NNgen Graph Created Successfully")
