import cv2
import numpy as np
import time

# -------------------------------
# Load YOLO-Tiny model
# -------------------------------
net = cv2.dnn.readNetFromDarknet("yolov3-tiny.cfg", "yolov3-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get output layer names
layer_names = net.getLayerNames()
out_layer_ids = net.getUnconnectedOutLayers()
out_layer_names = [layer_names[i - 1] for i in out_layer_ids.flatten()]

# Load class names
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# -------------------------------
# Open webcam (0 = default camera)
# -------------------------------
cap = cv2.VideoCapture(0)

CONF_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w, _ = frame.shape

    # -------------------------------
    # Create blob and forward pass
    # -------------------------------
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    outputs = net.forward(out_layer_names)
    end = time.time()

    # -------------------------------
    # Post-processing
    # -------------------------------
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            obj_score = detection[4]

            confidence = float(obj_score)

            if confidence > CONF_THRESHOLD:
                cx, cy, bw, bh = detection[0:4]

                x = int((cx - bw / 2) * w)
                y = int((cy - bh / 2) * h)
                bw = int(bw * w)
                bh = int(bh * h)

                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    # -------------------------------
    # Draw results
    # -------------------------------
    if len(indexes) > 0:
        indexes = indexes.flatten()
        for i in indexes:
            x, y, bw, bh = boxes[i]

            # Clamp safely
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)

            bw = max(1, x2 - x)
            bh = max(1, y2 - y)

            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # -------------------------------
    # Show FPS
    # -------------------------------
    fps = 1 / (end - start + 1e-6)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv3-Tiny Real-Time", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
