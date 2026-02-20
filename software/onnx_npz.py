import onnx
import numpy as np
from onnx import numpy_helper

onnx_path = "yolo_split_backbone.onnx"     
npz_path  = "tiny_yolov3_weights.npz"

model = onnx.load(onnx_path)

params = {}

for init in model.graph.initializer:
    name = init.name
    array = numpy_helper.to_array(init)
    params[name] = array
    print(f"{name:40s} {array.shape}")

np.savez(npz_path, **params)

print("\nSaved weights to:", npz_path)
print("Total tensors:", len(params))
