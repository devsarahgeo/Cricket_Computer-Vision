import cv2
import numpy as np
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
video_path = os.path.join(base_dir, "input_video", "shrt_video.mp4")
save_path = os.path.join(base_dir, "models", "homography.npy")

points = []

# ---------------------------
# CLICK FUNCTION
# ---------------------------
def click_event(event, x, y, flags, param):
    global points, img

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point added: {x}, {y}")

        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

# ---------------------------
# LOAD FIRST FRAME
# ---------------------------
cap = cv2.VideoCapture(video_path)
ret, img = cap.read()
cap.release()

cv2.namedWindow("Select 4 Pitch Points")
cv2.setMouseCallback("Select 4 Pitch Points", click_event)

print("\nCLICK THESE 4 POINTS IN ORDER:")
print("1. Bottom-left pitch corner")
print("2. Bottom-right pitch corner")
print("3. Top-right pitch corner")
print("4. Top-left pitch corner")

while True:
    cv2.imshow("Select 4 Pitch Points", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(points) == 4:
        break

cv2.destroyAllWindows()

# ---------------------------
# REAL WORLD PITCH COORDS
# ---------------------------
world_pts = np.array([
    [0, 0],
    [3.05, 0],
    [3.05, 20.12],
    [0, 20.12]
], dtype=np.float32)

img_pts = np.array(points, dtype=np.float32)

# ---------------------------
# HOMOGRAPHY MATRIX
# ---------------------------
H, _ = cv2.findHomography(img_pts, world_pts)

# SAVE
np.save(save_path, H)

print("\nSaved homography matrix →", save_path)