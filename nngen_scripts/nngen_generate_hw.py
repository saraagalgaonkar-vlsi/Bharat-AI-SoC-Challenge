import nngen as ng

onnx_path = r"D:\arm_accelerator\yolo_nngen\onnx_models\yolo_split_backbone.onnx"

(
    outputs,
    placeholders,
    variables,
    constants,
    operators
) = ng.from_onnx(onnx_path)

input_scale_factors = {
    list(placeholders.keys())[0]: 1.0/128.0
}

ng.quantize(list(outputs.values()), input_scale_factors)

ng.to_verilog(
    list(outputs.values()),
    name="yolo_backbone_accel",
    filename="yolo_backbone_accelerator.v"
)

print("Hardware Verilog Generated")
