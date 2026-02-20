import numpy as np

p = np.load("tiny_yolov3_weights.npz")
print(len(p.files))
print(p.files[:20]) 