import cv2
import numpy as np
import time
from scipy.spatial import distance as dist

# ---------- Centroid Tracker ----------
class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        if len(centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        if len(self.objects) == 0:
            for c in centroids:
                self.register(c)
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())
            D = dist.cdist(np.array(obj_centroids), np.array(centroids))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                obj_id = obj_ids[row]
                self.objects[obj_id] = centroids[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(len(obj_centroids))) - used_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            for col in set(range(len(centroids))) - used_cols:
                self.register(centroids[col])

        return self.objects


# ---------- Colour‑Based Segmenter with Denoising ----------
class ColorSegmenter:
    def __init__(self,
                 resize_width=320,
                 spatial_radius=10,
                 color_radius=20,
                 quant_levels=8,
                 min_area=500,
                 denoise=True,
                 denoise_method='nlmeans',
                 use_gpu=False):
        self.resize_width = resize_width
        self.spatial_radius = spatial_radius
        self.color_radius = color_radius
        self.quant_levels = quant_levels
        self.min_area = min_area
        self.denoise = denoise
        self.denoise_method = denoise_method
        self.use_gpu = use_gpu

    def segment(self, frame):
        h, w = frame.shape[:2]
        scale = self.resize_width / w
        new_h = int(h * scale)
        small = cv2.resize(frame, (self.resize_width, new_h))

        # ----- Denoising -----
        if self.denoise:
            if self.denoise_method == 'nlmeans':
                small = cv2.fastNlMeansDenoisingColored(small, None, 10, 10, 7, 21)
            elif self.denoise_method == 'median':
                small = cv2.medianBlur(small, 5)

        # Edge‑preserving smoothing
        filtered = cv2.pyrMeanShiftFiltering(small, self.spatial_radius, self.color_radius)
        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2Lab)

        L, a, b = cv2.split(lab)
        bin_width = 256 // self.quant_levels
        L_quant = (L // bin_width).astype(np.int32)
        a_quant = (a // bin_width).astype(np.int32)
        b_quant = (b // bin_width).astype(np.int32)

        Q = self.quant_levels
        label_img = L_quant * (Q * Q) + a_quant * Q + b_quant

        unique_labels = np.unique(label_img)
        objects = []
        factor = 1.0 / scale

        for colour_val in unique_labels:
            mask = (label_img == colour_val).astype(np.uint8) * 255
            num_labels, labels_cc = cv2.connectedComponents(mask, connectivity=8)

            for comp_id in range(1, num_labels):
                comp_mask = (labels_cc == comp_id).astype(np.uint8) * 255
                contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue

                mean_bgr = cv2.mean(small, mask=comp_mask)[:3]
                mean_bgr = tuple(map(int, mean_bgr))

                x, y, bw, bh = cv2.boundingRect(contour)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + bw//2, y + bh//2

                contour_orig = (contour * factor).astype(np.int32)
                centroid_orig = (int(cx * factor), int(cy * factor))
                bbox_orig = (int(x * factor), int(y * factor),
                             int(bw * factor), int(bh * factor))

                objects.append({
                    'contour': contour_orig,
                    'centroid': centroid_orig,
                    'bbox': bbox_orig,
                    'area': area * (factor ** 2),
                    'mean_bgr': mean_bgr
                })

        return objects


# ---------- Grouping Function ----------
def group_objects(objects, spatial_thresh=50, color_thresh=30):
    """
    Group objects that are spatially close and have similar color.
    spatial_thresh: max Euclidean distance between centroids (pixels)
    color_thresh: max Euclidean distance in BGR color space
    Returns list of groups, each with 'bbox', 'centroid', 'mean_bgr', 'objects'
    """
    if not objects:
        return []
    obj_list = sorted(objects, key=lambda x: x['area'], reverse=True)
    n = len(obj_list)
    used = [False] * n
    groups = []
    for i in range(n):
        if used[i]:
            continue
        group = {
            'objects': [obj_list[i]],
            'bbox': list(obj_list[i]['bbox']),
            'centroids': [obj_list[i]['centroid']],
            'mean_bgrs': [obj_list[i]['mean_bgr']],
            'areas': [obj_list[i]['area']]
        }
        used[i] = True
        # Try to add other objects
        for j in range(i+1, n):
            if used[j]:
                continue
            # Spatial distance
            dx = obj_list[i]['centroid'][0] - obj_list[j]['centroid'][0]
            dy = obj_list[i]['centroid'][1] - obj_list[j]['centroid'][1]
            dist_spatial = np.sqrt(dx*dx + dy*dy)
            if dist_spatial > spatial_thresh:
                continue
            # Color distance
            color_dist = np.linalg.norm(np.array(obj_list[i]['mean_bgr']) -
                                         np.array(obj_list[j]['mean_bgr']))
            if color_dist > color_thresh:
                continue
            # Merge
            group['objects'].append(obj_list[j])
            group['centroids'].append(obj_list[j]['centroid'])
            group['mean_bgrs'].append(obj_list[j]['mean_bgr'])
            group['areas'].append(obj_list[j]['area'])
            # Update bounding box to union
            x1 = min(group['bbox'][0], obj_list[j]['bbox'][0])
            y1 = min(group['bbox'][1], obj_list[j]['bbox'][1])
            x2 = max(group['bbox'][0] + group['bbox'][2],
                     obj_list[j]['bbox'][0] + obj_list[j]['bbox'][2])
            y2 = max(group['bbox'][1] + group['bbox'][3],
                     obj_list[j]['bbox'][1] + obj_list[j]['bbox'][3])
            group['bbox'] = [x1, y1, x2 - x1, y2 - y1]
            used[j] = True
        # Compute group centroid (weighted by area)
        total_area = sum(group['areas'])
        if total_area > 0:
            cx = sum(c[0] * a for c, a in zip(group['centroids'], group['areas'])) / total_area
            cy = sum(c[1] * a for c, a in zip(group['centroids'], group['areas'])) / total_area
        else:
            cx = group['bbox'][0] + group['bbox'][2] // 2
            cy = group['bbox'][1] + group['bbox'][3] // 2
        # Compute group mean color (weighted by area)
        mean_bgr = [0, 0, 0]
        if total_area > 0:
            for c, a in zip(group['mean_bgrs'], group['areas']):
                mean_bgr[0] += c[0] * a
                mean_bgr[1] += c[1] * a
                mean_bgr[2] += c[2] * a
            mean_bgr = [int(m / total_area) for m in mean_bgr]
        else:
            mean_bgr = group['mean_bgrs'][0]
        groups.append({
            'bbox': group['bbox'],
            'centroid': (int(cx), int(cy)),
            'mean_bgr': tuple(mean_bgr),
            'objects': group['objects']
        })
    return groups


# ---------- Main Program ----------
def main():
    # Grouping thresholds
    SPATIAL_THRESH = 50
    COLOR_THRESH   = 30

    # Focus mode parameters
    FOCUS_CALIB_SEC = 10
    MOTION_THRESH = 1.0
    FOCUS_MARGIN = 50
    CELL_SIZE = 20

    # Create segmenter with denoising enabled
    segmenter = ColorSegmenter(resize_width=320,
                               spatial_radius=5,
                               color_radius=10,
                               quant_levels=20,
                               min_area=200,
                               denoise=True,
                               denoise_method='nlmeans')   # or 'median' for speed

    tracker = CentroidTracker(max_disappeared=40, max_distance=50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, _ = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width} x {height}")

    cv2.namedWindow("Object Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Tracking", 800, 600)

    print("Press 'q' to quit. Press 'f' to toggle focus mode.")
    print("Make sure the video window is active (click on it) before pressing keys.")

    fps = 0
    frame_count = 0
    start_time = time.time()

    object_colors = {}
    motion_history = {}
    last_velocities = {}
    last_positions = {}
    last_sizes = {}

    focus_mode = False
    focus_calibrating = False
    focus_calib_start = 0
    focus_region = None
    motion_points = []
    previous_alive_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        if np.mean(frame) < 5:
            print("Warning: Frame is almost black.")

        try:
            objects = segmenter.segment(frame)
        except Exception as e:
            print(f"Segmentation error: {e}")
            objects = []

        groups = group_objects(objects, spatial_thresh=SPATIAL_THRESH, color_thresh=COLOR_THRESH)
        centroids = [g['centroid'] for g in groups]
        tracked = tracker.update(centroids)

        centroid_to_group = {g['centroid']: g for g in groups}

        # ----- Motion Tracking & Prediction -----
        alive_ids = set(tracker.objects.keys()) | set(tracker.disappeared.keys())
        current_positions = {}

        for group_id, centroid in tracked.items():
            if centroid in centroid_to_group:
                group = centroid_to_group[centroid]
                if group_id not in motion_history:
                    motion_history[group_id] = []
                motion_history[group_id].append(centroid)
                if len(motion_history[group_id]) > 5:
                    motion_history[group_id].pop(0)

                if len(motion_history[group_id]) >= 2:
                    prev = motion_history[group_id][-2]
                    vx = centroid[0] - prev[0]
                    vy = centroid[1] - prev[1]
                    last_velocities[group_id] = (vx, vy)

                last_positions[group_id] = centroid
                w, h = group['bbox'][2], group['bbox'][3]
                last_sizes[group_id] = (w, h)
                current_positions[group_id] = centroid

        for group_id, missing_count in tracker.disappeared.items():
            if group_id in tracked:
                continue
            if group_id in last_positions and group_id in last_velocities:
                last = last_positions[group_id]
                vel = last_velocities[group_id]
                pred_x = last[0] + vel[0] * missing_count
                pred_y = last[1] + vel[1] * missing_count
                pred = (int(pred_x), int(pred_y))
                current_positions[group_id] = pred

        for gid in list(motion_history.keys()):
            if gid not in alive_ids:
                del motion_history[gid]
        for gid in list(last_velocities.keys()):
            if gid not in alive_ids:
                del last_velocities[gid]
        for gid in list(last_positions.keys()):
            if gid not in alive_ids:
                del last_positions[gid]
        for gid in list(last_sizes.keys()):
            if gid not in alive_ids:
                del last_sizes[gid]

        # ----- Focus Mode Calibration and Locking -----
        if focus_mode:
            if not focus_calibrating and focus_region is None:
                focus_calibrating = True
                focus_calib_start = time.time()
                motion_points = []
                previous_alive_ids = set()

            if focus_calibrating:
                # Movement points
                for gid in alive_ids:
                    if gid in last_velocities:
                        vx, vy = last_velocities[gid]
                        mag = np.sqrt(vx*vx + vy*vy)
                        if mag > MOTION_THRESH and gid in current_positions:
                            motion_points.append(current_positions[gid])

                # Appearance points
                new_ids = alive_ids - previous_alive_ids
                for gid in new_ids:
                    if gid in current_positions:
                        motion_points.append(current_positions[gid])

                # Disappearance points
                disappeared_ids = previous_alive_ids - alive_ids
                for gid in disappeared_ids:
                    if gid in last_positions:
                        motion_points.append(last_positions[gid])

                previous_alive_ids = alive_ids.copy()
                print(f"Motion points: {len(motion_points)}")

                if time.time() - focus_calib_start > FOCUS_CALIB_SEC:
                    focus_calibrating = False
                    if len(motion_points) > 10:
                        cols = (width + CELL_SIZE - 1) // CELL_SIZE
                        rows = (height + CELL_SIZE - 1) // CELL_SIZE
                        grid = np.zeros((rows, cols), dtype=np.int32)
                        for (x, y) in motion_points:
                            if 0 <= x < width and 0 <= y < height:
                                grid[y // CELL_SIZE, x // CELL_SIZE] += 1

                        max_count = np.max(grid)
                        if max_count > 0:
                            threshold = max_count * 0.7
                            high_density_mask = (grid >= threshold).astype(np.uint8)
                            num_labels, labels = cv2.connectedComponents(high_density_mask, connectivity=8)

                            largest_label = 0
                            largest_size = 0
                            for label in range(1, num_labels):
                                size = np.sum(labels == label)
                                if size > largest_size:
                                    largest_size = size
                                    largest_label = label

                            if largest_label > 0:
                                cells = np.argwhere(labels == largest_label)
                                min_row = np.min(cells[:, 0])
                                max_row = np.max(cells[:, 0])
                                min_col = np.min(cells[:, 1])
                                max_col = np.max(cells[:, 1])

                                x1 = max(0, min_col * CELL_SIZE - FOCUS_MARGIN)
                                y1 = max(0, min_row * CELL_SIZE - FOCUS_MARGIN)
                                x2 = min(width, (max_col + 1) * CELL_SIZE + FOCUS_MARGIN)
                                y2 = min(height, (max_row + 1) * CELL_SIZE + FOCUS_MARGIN)
                                focus_region = (x1, y1, x2, y2)
                            else:
                                focus_region = (0, 0, width, height)
                        else:
                            focus_region = (0, 0, width, height)
                    else:
                        focus_region = (0, 0, width, height)
                    print(f"Focus locked on region: {focus_region}")

        # ----- Apply edge highlighting effect -----
        if focus_mode:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (31, 31), 0)
            normalized_bg = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_mag = np.uint8(np.clip(grad_mag, 0, 255))

            _, edge_mask = cv2.threshold(grad_mag, 10, 255, cv2.THRESH_BINARY)
            edge_mask = cv2.dilate(edge_mask, None, iterations=2)

            edge_mask_3c = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR) / 255.0
            frame = (frame * edge_mask_3c + normalized_bg * (1 - edge_mask_3c)).astype(np.uint8)

            if focus_calibrating:
                cv2.putText(frame, "FOCUS CALIBRATING...", (frame.shape[1]-300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                remaining = max(0, FOCUS_CALIB_SEC - (time.time() - focus_calib_start))
                cv2.putText(frame, f"{remaining:.1f}s", (frame.shape[1]-300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "FOCUS LOCKED", (frame.shape[1]-200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if focus_region:
                    overlay = frame.copy()
                    x1,y1,x2,y2 = focus_region
                    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,255), -1)
                    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

        # ----- Draw results -----
        if focus_mode and not focus_calibrating and focus_region is not None:
            x1r, y1r, x2r, y2r = focus_region
            for group_id, centroid in current_positions.items():
                if x1r <= centroid[0] <= x2r and y1r <= centroid[1] <= y2r:
                    if group_id in last_sizes:
                        w, h = last_sizes[group_id]
                        x = centroid[0] - w//2
                        y = centroid[1] - h//2
                        if group_id not in object_colors:
                            np.random.seed(group_id)
                            color = tuple(np.random.randint(0, 255, 3).tolist())
                            object_colors[group_id] = color
                        else:
                            color = object_colors[group_id]

                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, str(group_id), (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        if group_id not in tracked:
                            cv2.putText(frame, "predicted", (x, y+h+15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            for group_id, centroid in tracked.items():
                if centroid in centroid_to_group:
                    group = centroid_to_group[centroid]
                    if group_id not in object_colors:
                        np.random.seed(group_id)
                        color = tuple(np.random.randint(0, 255, 3).tolist())
                        object_colors[group_id] = color
                    else:
                        color = object_colors[group_id]

                    x, y, w, h = group['bbox']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, str(group_id), (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ----- On‑Screen Instructions -----
        cv2.putText(frame, "Press 'f' for focus, 'q' to quit", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ----- FPS Display -----
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = end_time

        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Object Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            focus_mode = not focus_mode
            if not focus_mode:
                focus_calibrating = False
                focus_region = None
                motion_points = []
                previous_alive_ids = set()
            else:
                focus_calibrating = False
                focus_region = None
                motion_points = []
                previous_alive_ids = set()
            print(f"Focus mode {'enabled' if focus_mode else 'disabled'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()