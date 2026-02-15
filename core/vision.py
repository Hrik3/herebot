import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import time
import os
import urllib.request

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
def voxel_filter(points, colors, voxel_size=0.03):   # finer voxel (3 cm)
    if len(points) == 0:
        return points, colors
    indices = np.floor(points / voxel_size).astype(int)
    unique_keys, inverse = np.unique(indices, axis=0, return_inverse=True)
    down_points = np.zeros((len(unique_keys), 3))
    down_colors = np.zeros((len(unique_keys), 3))
    for i in range(len(unique_keys)):
        mask = inverse == i
        down_points[i] = points[mask].mean(axis=0)
        down_colors[i] = colors[mask].mean(axis=0)
    return down_points, down_colors

def oriented_bbox_volume(points):
    if len(points) < 4:
        return 0.0
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    projected = centered @ evecs
    min_vals = projected.min(axis=0)
    max_vals = projected.max(axis=0)
    extents = max_vals - min_vals
    return extents[0] * extents[1] * extents[2]

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
# 5. Main loop
# ------------------------------------------------------------
cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
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

    nearest_box = None
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 300:
            x,y,wc,hc = cv2.boundingRect(cnt)
            nearest_box = (x,y,wc,hc)

    # ---------- draw depth detected object immediately ----------
    overlay = frame.copy()

    if nearest_box is not None:
        x,y,wc,hc = nearest_box
        centroid = (x+wc//2, y+hc//2)

        assigned_id = 0
        object_memory = {0: centroid}


        cv2.rectangle(overlay,(x,y),(x+wc,y+hc),(0,0,255),3)
        cv2.putText(overlay,f"ID {assigned_id}",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2)

    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # --------------------------------------------------------
    # Generate 3D point cloud with color (subsampled)
    # --------------------------------------------------------
    step = 2
    u = np.arange(0, w, step, dtype=np.float32)
    v = np.arange(0, h, step, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)
    Z = depth_map[::step, ::step]
    scale = 5.0
    Z = scale / (Z + 1e-6)
    Z = np.clip(Z, 0, 10.0)

    X = (u_grid - principal_x) * Z / focal_length_x
    Y = (v_grid - principal_y) * Z / focal_length_y
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)[::step, ::step]
    L = frame_lab[:,:,0] / 100.0
    a = frame_lab[:,:,1] / 127.0
    b = frame_lab[:,:,2] / 127.0
    colors = np.stack((L, a, b), axis=-1).reshape(-1, 3)

    valid_mask = (Z.flatten() > 0.1) & (Z.flatten() < 10.0)
    points = points[valid_mask]
    colors = colors[valid_mask]

    # --------------------------------------------------------
    # Downsample and cluster
    # --------------------------------------------------------
    if len(points) > 0:
        points_down, colors_down = voxel_filter(points, colors, voxel_size=0.03)

        if len(points_down) > 50:
            color_weight = 0.8
            features = np.hstack((points_down, colors_down * color_weight))
            clustering = DBSCAN(eps=0.25, min_samples=10).fit(features)
            labels = clustering.labels_

            # ----------------------------------------------------
            # Find the nearest cluster (smallest mean Z)
            # ----------------------------------------------------
            nearest_label = None
            min_dist = float('inf')
            for label in set(labels):
                if label == -1:
                    continue
                cluster_pts = points_down[labels == label]
                if len(cluster_pts) < 20:
                    continue
                mean_z = np.mean(cluster_pts[:, 2])
                if mean_z < min_dist:
                    min_dist = mean_z
                    nearest_label = label

            overlay = frame.copy()
            nearest_box = (x,y,wc,hc)

            # ---------- draw depth detected nearest object (ALWAYS) ----------
            if nearest_box is not None:
                x,y,wc,hc = nearest_box
                centroid = (x+wc//2, y+hc//2)

                assigned_id = 0
                object_memory = {0: centroid}


                cv2.rectangle(overlay,(x,y),(x+wc,y+hc),(0,0,255),3)
                cv2.putText(overlay,f"ID {assigned_id}",
                            (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,0,255),
                            2)


            for label in set(labels):
                if label == -1:
                    continue
                cluster_pts = points_down[labels == label]
                if len(cluster_pts) < 20:
                    continue

                vol = oriented_bbox_volume(cluster_pts)

                # Project all cluster points back to 2D
                xs, ys = [], []
                for pt in cluster_pts:
                    Xc, Yc, Zc = pt
                    if Zc > 0.1:
                        u_ = int(Xc * focal_length_x / Zc + principal_x)
                        v_ = int(Yc * focal_length_y / Zc + principal_y)
                        if 0 <= u_ < w and 0 <= v_ < h:
                            xs.append(u_)
                            ys.append(v_)

                if len(xs) >= 3:
                    pts_2d = np.column_stack((xs, ys)).astype(np.int32)
                    hull = cv2.convexHull(pts_2d)

                    if label == nearest_label:

                        # -------- compute bounding box ----------
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)

                        centroid = (int(np.mean(xs)), int(np.mean(ys)))

                        # -------- ID assignment ----------
                        assigned_id = None
                        for oid, (px, py) in object_memory.items():
                            if np.hypot(px-centroid[0], py-centroid[1]) < 80:
                                assigned_id = oid
                                break

                        if assigned_id is None:
                            assigned_id = next_object_id
                            next_object_id += 1

                        object_memory = {assigned_id: centroid}

                        # -------- draw bounding box ----------
                        cv2.rectangle(overlay,
                                    (x_min, y_min),
                                    (x_max, y_max),
                                    (0,0,255),
                                    3)

                        cv2.putText(overlay,
                                    f"ID {assigned_id}  {min_dist:.2f}m",
                                    (x_min, y_min-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0,0,255),
                                    2)

                    else:
                        # Normal objects
                        np.random.seed(label)
                        color = tuple(np.random.randint(0, 255, 3).tolist())
                        cv2.drawContours(overlay, [hull], -1, color, 2)
                        cx = int(np.mean(xs))
                        cy = int(np.mean(ys))
                        cv2.putText(overlay, f"{vol:.2f} m3", (cx+10, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Blend overlay
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

cap.release()
cv2.destroyAllWindows()