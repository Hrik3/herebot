import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import time
import os
import urllib.request
import sys
import argparse
# ------------------------------------------------------------
# 1. Download and load MiDaS v2.1 small model (ONNX)
# ------------------------------------------------------------
model_path = "model-small.onnx"
if not os.path.exists(model_path):
    print("Downloading model (35MB)...")
    url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete")

net = cv2.dnn.readNetFromONNX(model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

input_height, input_width = 256, 256

# ------------------------------------------------------------
# 2. Camera intrinsics (adjust to your camera if needed)
# ------------------------------------------------------------
focal_length_x = 640.0
focal_length_y = 640.0
principal_x = 320.0
principal_y = 240.0

# ------------------------------------------------------------
# 3. Voxel filter and volume functions
# ------------------------------------------------------------
# ------------------------------------------------------------
# 4. Guided depth upsampling (requires opencv-contrib-python)
# ------------------------------------------------------------
def guided_depth_upsample(depth_low, color_high, low_size=(256,256)):
    depth_up = cv2.resize(depth_low,
                          (color_high.shape[1], color_high.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    try:
        guide = color_high.astype(np.float32) / 255.0
        depth_up = cv2.ximgproc.jointBilateralFilter(
            guide,
            depth_up.astype(np.float32),
            d=7,
            sigmaColor=50,
            sigmaSpace=5
        )
    except AttributeError:
        depth_up = cv2.bilateralFilter(depth_up.astype(np.float32),
                                       d=7,
                                       sigmaColor=50,
                                       sigmaSpace=5)
    return depth_up

# ------------------------------------------------------------
# Depth objectness helper functions
# ------------------------------------------------------------
def compute_depth_contrast(depth_map, cnt):
    mask = np.zeros(depth_map.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    inside = cv2.mean(depth_map, mask=mask)[0]

    kernel = np.ones((15,15), np.uint8)
    border = cv2.dilate(mask, kernel, iterations=1)
    border = cv2.subtract(border, mask)

    border_val = cv2.mean(depth_map, mask=border)[0]

    return inside - border_val


def compute_depth_variation(depth_map, cnt):
    mask = np.zeros(depth_map.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    pixels = depth_map[mask > 0]
    if len(pixels) == 0:
        return 0
    return np.std(pixels)

# ------------------------------------------------------------
# 5. Argument parsing and IPC setup
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true',
                    help='Enable IPC via named pipe (/tmp/vision_pipe)')
args = parser.parse_args()

pipe_path = '/tmp/vision_pipe'
pipe_out = None
if args.run:
    # Create the named pipe if it doesn't exist
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    # Open the pipe for writing (blocks until a reader connects)
    print(f"Waiting for reader on {pipe_path}...", file=sys.stderr)
    pipe_out = open(pipe_path, 'w')
    print("Reader connected.", file=sys.stderr)

# ------------------------------------------------------------
# 6. Main loop (camera)
# ------------------------------------------------------------
cap = cv2.VideoCapture("/dev/video10", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

fps = 0
frame_count = 0
start_time = time.time()

depth_smooth = None
prev_gray = None

# -------- persistent ID tracking ----------
object_memory = {}
next_object_id = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # Depth estimation
    # --------------------------------------------------------
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_width, input_height),
                                 (0.485, 0.456, 0.406), swapRB=True, crop=False)
    net.setInput(blob)
    depth_map = net.forward()
    depth_map = depth_map.squeeze()

    depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    depth_map = guided_depth_upsample(depth_map, frame, (input_width, input_height))

    # ---- Motion-aware temporal smoothing ----
    gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if depth_smooth is None:
        depth_smooth = depth_map.copy()
    if prev_gray is None:
        prev_gray = gray_now.copy()
    frame_diff = cv2.absdiff(gray_now, prev_gray)
    motion_level = np.mean(frame_diff)
    alpha = 0.55 if motion_level > 8 else 0.90
    depth_smooth = alpha * depth_smooth + (1 - alpha) * depth_map
    depth_map = depth_smooth
    prev_gray = gray_now.copy()

    # ---- Compute depth gradient magnitude for edge/object discrimination ----
    depth_float = depth_map.astype(np.float32)
    grad_x = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    # Normalize gradient to [0,1] for thresholding
    max_grad = np.max(grad_mag) + 1e-6
    grad_norm = grad_mag / max_grad

    depth_disp = cv2.normalize(depth_map, None, 0, 255,
                               cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # --------- Detect nearest object directly from depth ----------
    # Detect brightest (nearest) region from normalized depth
    near_thresh = 240   # brightest pixels
    _, near_mask = cv2.threshold(depth_disp, near_thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((9,9), np.uint8)
    near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_CLOSE, kernel)
    near_mask = cv2.dilate(near_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(near_mask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    detected_boxes = []
    depth_gradient_threshold = 0.05   # fraction of max gradient; adjust as needed

    if contours:
        # sort contours by mean depth brightness (nearest first)
        contour_depth = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 300:
                continue
            # ---- Depth gradient consistency check ----
            # Sample gradient along contour
            contour_points = cnt.squeeze().reshape(-1,2)
            if len(contour_points.shape) < 2:  # single point
                continue
            # Take up to 30 sample points
            if len(contour_points) > 30:
                indices = np.linspace(0, len(contour_points)-1, 30, dtype=int)
                sample_pts = contour_points[indices]
            else:
                sample_pts = contour_points
            grad_vals = []
            for pt in sample_pts:
                px, py = int(pt[0]), int(pt[1])
                if 0 <= px < w and 0 <= py < h:
                    grad_vals.append(grad_norm[py, px])
            if len(grad_vals) == 0:
                continue
            avg_grad = np.mean(grad_vals)
            if avg_grad < depth_gradient_threshold:
                continue   # likely a shadow or flat region

            # If passed gradient check, compute mean depth for sorting
            mask = np.zeros_like(depth_disp)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(depth_disp, mask=mask)[0]
            contour_depth.append((mean_val, cnt))

        # ---------------------------------------------------
        # Professional objectness scoring (depth-object score)
        # ---------------------------------------------------
        scored_contours = []

        for mean_val, cnt in contour_depth:

            # proximity (normalized)
            depth_score = mean_val / 255.0

            # depth contrast vs surroundings
            contrast = compute_depth_contrast(depth_disp, cnt)
            contrast_score = np.clip(contrast / 25.0, 0, 1)

            # internal depth variation
            variation = compute_depth_variation(depth_disp, cnt)
            variation_score = np.clip(variation / 20.0, 0, 1)

            # boundary strength from depth gradient
            mask = np.zeros_like(depth_disp)
            cv2.drawContours(mask, [cnt], -1, 255, 1)
            boundary_strength = cv2.mean(grad_norm, mask=mask)[0]
            boundary_score = np.clip(boundary_strength * 2.0, 0, 1)

            # final objectness score
            object_score = (
                0.35 * depth_score +
                0.25 * contrast_score +
                0.20 * variation_score +
                0.20 * boundary_score
            )

            scored_contours.append((object_score, cnt))

        scored_contours.sort(key=lambda x: x[0], reverse=True)

        for i, (_, cnt) in enumerate(scored_contours[:3]):
            x,y,wc,hc = cv2.boundingRect(cnt)
            detected_boxes.append((i, x, y, wc, hc))


    else:
        print("DEBUG: no contours in near_mask", file=sys.stderr)

    # ---------- draw depth detected object immediately ----------
    overlay = frame.copy()

    for assigned_id, x, y, wc, hc in detected_boxes:

        centroid = (x+wc//2, y+hc//2)
        object_memory[assigned_id] = centroid

        cv2.rectangle(overlay,(x,y),(x+wc,y+hc),(0,0,255),2)

        # Output the four corners of the bounding box
        if args.run and pipe_out:
            pipe_out.write(f"{x} {y} {x+wc} {y} {x+wc} {y+hc} {x} {y+hc}\n")
            pipe_out.flush()
        else:
            print(f"{x} {y} {x+wc} {y} {x+wc} {y+hc} {x} {y+hc}")
            sys.stdout.flush()
        cv2.putText(overlay,f"ID {assigned_id}",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2)

    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # --------------------------------------------------------
    # Display FPS
    # --------------------------------------------------------
    frame_count += 1
    if time.time() - start_time >= 1.0:
        fps = frame_count
        frame_count = 0
        start_time = time.time()

    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imshow("Depth Map", depth_disp)
    cv2.imshow("Volumetric Analysis", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

if pipe_out:
    pipe_out.close()

cap.release()
cv2.destroyAllWindows()