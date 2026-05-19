from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture("input_video/shrt_video.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output/output.mp4", fourcc, fps, (w, h))

# ROI threshold (crowd is usually top area)
FIELD_Y_THRESHOLD = int(h * 0.45)

# confidence threshold
CONF_THRES = 0.3

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=0)  # person class only
    result = results[0]

    if result.boxes is None:
        continue

    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        # 🔥 confidence filter
        if conf < CONF_THRES:
            continue

        # center point
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 🔥 ROI filter (remove crowd)
        if cy < FIELD_Y_THRESHOLD:
            continue

        # label
        label = f"player {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()