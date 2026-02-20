import onnxruntime as ort
import numpy as np

model_path = r"D:\arm_accelerator\yolo_nngen\onnx_models\yolov3_tiny.onnx"

session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name

dummy = np.random.randn(1, 3, 416, 416).astype(np.float32)

outputs = session.run(None, {input_name: dummy})

print("ONNX Runtime Inference Successful")
print("Number of outputs:", len(outputs))
