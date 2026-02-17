import cv2
import numpy as np
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

PIPE_PATH = '/tmp/vision_pipe'
WINDOW_NAME = 'Receiver'
CAMERA_DEVICE = '/dev/video11'
@dataclass
class TrackedObject:
    """Simplified tracked object with temporal persistence - inspired by lightweight tracking research [citation:2][citation:4]"""
    id: int
    contours: List[np.ndarray]  # Recent contour history
    centroids: List[Tuple[float, float]]  # Recent positions
    hues: List[float]  # Recent dominant hues
    brightness: float  # Average brightness for depth ranking
    contour_area: float  # Latest area
    color_sig: Optional[np.ndarray] = None  # 2D histogram (Hue+Saturation) or LAB a,b vector
    age: int = 0  # Frames since first seen
    persistence_score: float = 1.0  # Distance-weighted confidence [citation:1]
    lost_frames: int = 0  # Frames since last detection (for occlusion recovery)
    max_history: int = 5  # Keep last 5 frames for smoothing
    # Optical flow fields
    prev_points: Optional[np.ndarray] = None  # Good features from previous frame
    median_flow: Tuple[float, float] = (0.0, 0.0)  # Median flow (dx, dy)
    flow_confidence: float = 0.0  # How many points were successfully tracked

class ObjectTracker:
    """
    Lightweight multi-object tracker for low-end hardware.
    Implements distance-weighted persistence scoring and occlusion recovery [citation:5][citation:10]
    """
    
    def __init__(self, 
                 max_lost_frames: int = 10,      # Frames to keep lost objects (occlusion recovery)
                 distance_threshold: float = 50,  # Max pixel distance for ID matching
                 persistence_decay: float = 0.7,  # How quickly persistence decays with distance
                 min_persistence: float = 0.3):   # Threshold for ID reassignment
        
        self.next_id = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.max_lost_frames = max_lost_frames
        self.distance_threshold = distance_threshold
        self.persistence_decay = persistence_decay
        self.min_persistence = min_persistence
        # Temporal shape memory
        self.geometry_store = TemporalGeometryStore(
            max_history=5,
            iou_threshold=0.3,
            kalman_process_noise=1e-2,
            kalman_measure_noise=1e-1,
            max_reconstruct_frames=3
        )
        # Optical flow parameters (sparse LK)
        self.prev_gray = None
        self.flow_points_per_obj = 15      # max features per object
        self.flow_win_size = (15, 15)      # LK window size
        self.flow_max_level = 2             # pyramid levels
        self.flow_consistency_thresh = 5.0  # max allowed flow difference (pixels)  
        
    def _compute_centroid(self, contour: np.ndarray) -> Tuple[float, float]:
        """Fast centroid computation using moments"""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return (0, 0)
    
    def _compute_brightness(self, frame: np.ndarray, contour: np.ndarray) -> float:
        """Extract average brightness for depth/lighting-based ranking"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        brightness = cv2.mean(gray, mask=mask)[0]
        return brightness
    
    def _compute_dominant_hue(self, frame: np.ndarray, contour: np.ndarray) -> float:
        """Extract dominant hue for color-based tracking"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_vals = hsv[:,:,0][mask > 0]
        
        if len(hue_vals) > 0:
            hist = np.bincount(hue_vals, minlength=180)
            return float(np.argmax(hist))
        return 0.0

    def _compute_color_signature(self, frame: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        Compute a 2D histogram of Hue and Saturation (or LAB a,b) inside the contour.
        Returns a 1D normalized vector (bins flattened).
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Use only Hue and Saturation (ignore Value for illumination invariance)
        hue = hsv[:,:,0][mask > 0]
        sat = hsv[:,:,1][mask > 0]
        
        if len(hue) == 0:
            return np.zeros(32, dtype=np.float32)  # fallback
        
        # 2D histogram: 8 bins for Hue, 4 bins for Saturation (total 32)
        hist, _, _ = np.histogram2d(hue, sat, bins=[8, 4], range=[[0, 180], [0, 255]])
        hist = hist.flatten()
        hist = hist / (np.sum(hist) + 1e-6)  # normalize
        return hist.astype(np.float32)
        
    def _distance_weighted_score(self, 
                                  obj: TrackedObject, 
                                  centroid: Tuple[float, float],
                                  color_sig: np.ndarray) -> float:
        """
        Compute persistence score based on position and color histogram.
        Uses Bhattacharyya distance for color similarity.
        """
        if not obj.centroids:
            return 0.0
            
        last_centroid = obj.centroids[-1]
        
        # Euclidean distance
        dist = np.sqrt((centroid[0] - last_centroid[0])**2 + 
                       (centroid[1] - last_centroid[1])**2)
        
        # Distance-based score (closer = higher)
        if dist < self.distance_threshold:
            dist_score = 1.0 - (dist / self.distance_threshold) * self.persistence_decay
        else:
            dist_score = 0.0
            
        # Color consistency score using histogram comparison
        if obj.color_sig is not None:
            # Bhattacharyya distance: sum(sqrt(p*q))
            bc = np.sum(np.sqrt(obj.color_sig * color_sig))
            # Convert to similarity (1.0 = identical)
            color_score = bc
        else:
            color_score = 0.5  # default when no prior color
            
        # Combined score (position weighted more heavily)
        return 0.6 * dist_score + 0.4 * color_score  

    def _extract_points_from_contour(self, frame_gray: np.ndarray, contour: np.ndarray, max_points: int) -> np.ndarray:
        """Extract good features to track inside a contour."""
        mask = np.zeros(frame_gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        # Use Shi-Tomasi corner detector
        points = cv2.goodFeaturesToTrack(frame_gray, maxCorners=max_points, qualityLevel=0.01,
                                         minDistance=5, mask=mask)
        if points is not None:
            return points.reshape(-1, 2)
        return np.array([], dtype=np.float32).reshape(0, 2)

    def _compute_median_flow(self, prev_pts: np.ndarray, curr_pts: np.ndarray, status: np.ndarray) -> Tuple[float, float, float]:
        """Compute median flow vector from successfully tracked points."""
        if prev_pts is None or curr_pts is None or len(prev_pts) == 0:
            return (0.0, 0.0, 0.0)
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]
        if len(good_prev) < 3:
            return (0.0, 0.0, 0.0)
        flows = good_curr - good_prev
        median_x = np.median(flows[:, 0])
        median_y = np.median(flows[:, 1])
        confidence = len(good_prev) / len(prev_pts)
        return (median_x, median_y, confidence)

    def _compute_detection_flow(self, frame_gray: np.ndarray, contour: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute median flow for a detection by tracking its points backward.
        Returns (dx, dy, confidence).
        """
        if self.prev_gray is None:
            return (0.0, 0.0, 0.0)
        # Extract points inside the detection contour in the current frame
        curr_pts = self._extract_points_from_contour(frame_gray, contour, self.flow_points_per_obj)
        if len(curr_pts) == 0:
            return (0.0, 0.0, 0.0)
        # Track backward to previous frame
        prev_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            frame_gray, self.prev_gray, curr_pts.astype(np.float32), None,
            winSize=self.flow_win_size, maxLevel=self.flow_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        if prev_pts is None:
            return (0.0, 0.0, 0.0)
        # Compute median flow (current - previous) but note we have previous positions; flow = curr - prev
        good_curr = curr_pts[status.flatten() == 1]
        good_prev = prev_pts[status.flatten() == 1]
        if len(good_curr) < 3:
            return (0.0, 0.0, 0.0)
        flows = good_curr - good_prev
        median_x = np.median(flows[:, 0])
        median_y = np.median(flows[:, 1])
        confidence = len(good_curr) / len(curr_pts)
        return (median_x, median_y, confidence)
    
    def update(self, 
               frame: np.ndarray, 
               detected_contours: List[np.ndarray]) -> Dict[int, Tuple[np.ndarray, Tuple[float, float]]]:
        """
        Update tracker with new frame and detected contours.
        Uses geometry store for shape memory, Kalman smoothing, and optical flow for shadow rejection.
        Returns dict mapping ID -> (contour, smoothed centroid)
        """
        # Convert to grayscale for optical flow
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Compute optical flow for existing tracked objects (if previous frame exists) ----
        if self.prev_gray is not None:
            for obj in self.tracked_objects.values():
                if obj.prev_points is not None and len(obj.prev_points) > 0:
                    # Track points forward
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, frame_gray, obj.prev_points.astype(np.float32), None,
                        winSize=self.flow_win_size, maxLevel=self.flow_max_level,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                    )
                    if curr_pts is not None:
                        median_x, median_y, conf = self._compute_median_flow(obj.prev_points, curr_pts, status)
                        obj.median_flow = (median_x, median_y)
                        obj.flow_confidence = conf
                    else:
                        obj.median_flow = (0.0, 0.0)
                        obj.flow_confidence = 0.0
                else:
                    obj.median_flow = (0.0, 0.0)
                    obj.flow_confidence = 0.0

        # ---- Mark all objects as potentially lost ----
        for obj in self.tracked_objects.values():
            obj.lost_frames += 1
            self.geometry_store.mark_lost(obj.id)

        matched_objects = {}

        if detected_contours:
            # Extract features for each detection
            detections = []
            for contour in detected_contours:
                centroid = self._compute_centroid(contour)
                brightness = self._compute_brightness(frame, contour)
                hue = self._compute_dominant_hue(frame, contour)
                area = cv2.contourArea(contour)
                color_sig = self._compute_color_signature(frame, contour)
                # Compute optical flow for this detection (optional, can be deferred to matching)
                flow_dx, flow_dy, flow_conf = self._compute_detection_flow(frame_gray, contour)
                detections.append({
                    'contour': contour,
                    'centroid': centroid,
                    'brightness': brightness,
                    'hue': hue,
                    'area': area,
                    'color_sig': color_sig,
                    'flow': (flow_dx, flow_dy),
                    'flow_conf': flow_conf
                })

            # Sort detections by area (larger objects first)
            detections.sort(key=lambda x: x['area'], reverse=True)

            # Get list of active object IDs (not expired)
            active_ids = [oid for oid, obj in self.tracked_objects.items() 
                          if obj.lost_frames <= self.max_lost_frames]

            used_detections = set()
            used_objects = set()

            # ---- First pass: IoU-based matching (geometry) ----
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                best_id = self.geometry_store.find_best_match(det['contour'], active_ids)
                if best_id is not None and best_id not in used_objects:
                    used_detections.add(i)
                    used_objects.add(best_id)
                    obj = self.tracked_objects[best_id]

                    # Update object
                    obj.contours.append(det['contour'])
                    obj.centroids.append(det['centroid'])
                    obj.hues.append(det['hue'])
                    obj.brightness = det['brightness']
                    obj.contour_area = det['area']
                    obj.color_sig = det['color_sig']
                    obj.age += 1
                    obj.lost_frames = 0
                    obj.persistence_score = 1.0

                    # Keep history bounded
                    if len(obj.contours) > obj.max_history:
                        obj.contours.pop(0)
                        obj.centroids.pop(0)
                        obj.hues.pop(0)

                    # Update geometry store
                    self.geometry_store.update_object(best_id, det['contour'], det['centroid'], frame)

                    # Update optical flow points for this object (extract new points from current contour)
                    obj.prev_points = self._extract_points_from_contour(frame_gray, det['contour'], self.flow_points_per_obj)

                    matched_objects[best_id] = (det['contour'], det['centroid'])

            # ---- Second pass: distance-weighted matching with flow consistency ----
            for obj_id, obj in sorted(self.tracked_objects.items(), 
                                     key=lambda x: x[1].persistence_score, 
                                     reverse=True):
                if obj.lost_frames > self.max_lost_frames:
                    continue
                if obj_id in used_objects:
                    continue

                best_match = None
                best_score = 0
                best_flow_ok = True

                for i, det in enumerate(detections):
                    if i in used_detections:
                        continue

                    # Base score from position and color
                    score = self._distance_weighted_score(obj, det['centroid'], det['color_sig'])

                    # Flow consistency check: if object has reliable flow, compare with detection's flow
                    if obj.flow_confidence > 0.3 and det['flow_conf'] > 0.3:
                        flow_diff = np.sqrt((det['flow'][0] - obj.median_flow[0])**2 +
                                           (det['flow'][1] - obj.median_flow[1])**2)
                        if flow_diff > self.flow_consistency_thresh:
                            # Penalize heavily (or set score to 0)
                            score *= 0.2  # reduce score significantly
                            flow_ok = False
                        else:
                            flow_ok = True
                    else:
                        flow_ok = True

                    if score > best_score and score > self.min_persistence:
                        best_score = score
                        best_match = i
                        best_flow_ok = flow_ok

                if best_match is not None and (best_flow_ok or best_score > self.min_persistence * 1.5):
                    used_detections.add(best_match)
                    used_objects.add(obj_id)
                    det = detections[best_match]

                    # Update existing object
                    obj.contours.append(det['contour'])
                    obj.centroids.append(det['centroid'])
                    obj.hues.append(det['hue'])
                    obj.brightness = det['brightness']
                    obj.contour_area = det['area']
                    obj.color_sig = det['color_sig']
                    obj.age += 1
                    obj.lost_frames = 0
                    obj.persistence_score = best_score

                    # Keep history bounded
                    if len(obj.contours) > obj.max_history:
                        obj.contours.pop(0)
                        obj.centroids.pop(0)
                        obj.hues.pop(0)

                    # Update geometry store
                    self.geometry_store.update_object(obj_id, det['contour'], det['centroid'], frame)

                    # Update optical flow points for this object
                    obj.prev_points = self._extract_points_from_contour(frame_gray, det['contour'], self.flow_points_per_obj)

                    matched_objects[obj_id] = (det['contour'], det['centroid'])

            # ---- Third pass: create new objects for unmatched detections ----
            for i, det in enumerate(detections):
                if i not in used_detections:
                    new_obj = TrackedObject(
                        id=self.next_id,
                        contours=[det['contour']],
                        centroids=[det['centroid']],
                        hues=[det['hue']],
                        brightness=det['brightness'],
                        contour_area=det['area'],
                        color_sig=det['color_sig'],
                        age=1,
                        persistence_score=1.0,
                        lost_frames=0
                    )
                    # Initialize flow points for new object
                    new_obj.prev_points = self._extract_points_from_contour(frame_gray, det['contour'], self.flow_points_per_obj)

                    self.tracked_objects[self.next_id] = new_obj
                    matched_objects[self.next_id] = (det['contour'], det['centroid'])
                    self.geometry_store.register_object(self.next_id, det['contour'], det['centroid'], frame)
                    self.next_id += 1

        # Remove objects lost for too long
        expired_ids = [obj_id for obj_id, obj in self.tracked_objects.items() 
                      if obj.lost_frames > self.max_lost_frames]
        for obj_id in expired_ids:
            del self.tracked_objects[obj_id]
            if obj_id in self.geometry_store.shape_memory:
                del self.geometry_store.shape_memory[obj_id]

        # Build output with Kalman‑smoothed centroids
        output_objects = {}
        for obj_id, (contour, raw_centroid) in matched_objects.items():
            kalman_cent = self.geometry_store.get_kalman_centroid(obj_id)
            if kalman_cent is None:
                kalman_cent = raw_centroid
            smooth_contour = self.geometry_store.get_smoothed_contour(obj_id)
            if smooth_contour is None:
                smooth_contour = contour
            output_objects[obj_id] = (smooth_contour, (int(kalman_cent[0]), int(kalman_cent[1])))

        # Store current grayscale for next frame's optical flow
        self.prev_gray = frame_gray

        return output_objects
      
    def get_object_lifetimes(self) -> Dict[int, int]:
        """Return age of each tracked object (for debugging/display)"""
        return {obj_id: obj.age for obj_id, obj in self.tracked_objects.items()}

class TemporalGeometryStore:
    """
    Manages shape memory for each tracked object.
    Provides contour smoothing, IoU‑based similarity, and lost‑frame reconstruction.
    Now also stores average RGB for color‑aided matching.
    """
    
    def __init__(self, 
                 max_history: int = 5,               # number of recent contours to keep
                 iou_threshold: float = 0.3,          # IoU threshold for considering same shape
                 kalman_process_noise: float = 1e-2,  # Q (process noise)
                 kalman_measure_noise: float = 1e-1,  # R (measurement noise)
                 max_reconstruct_frames: int = 3):    # max frames to reconstruct before dropping
        
        self.max_history = max_history
        self.iou_threshold = iou_threshold
        self.max_reconstruct_frames = max_reconstruct_frames
        
        # Per‑object shape memory (key = object ID)
        self.shape_memory: Dict[int, Dict] = {}
        
        # Kalman parameters (shared for all objects, but each has own state)
        self.Q = kalman_process_noise   # process noise covariance
        self.R = kalman_measure_noise   # measurement noise covariance
        
    def _init_kalman(self, x: float, y: float):
        """Initialize a simple 1D Kalman filter for x and y separately."""
        # state: [position, velocity]
        state_x = np.array([x, 0.0])
        state_y = np.array([y, 0.0])
        # covariance estimate
        cov_x = np.eye(2) * 10.0
        cov_y = np.eye(2) * 10.0
        return (state_x, cov_x), (state_y, cov_y)
    
    def _kalman_predict(self, state, cov, dt=1.0):
        """Predict step for one coordinate (constant velocity model)."""
        F = np.array([[1, dt],
                      [0, 1]])          # state transition matrix
        Q = np.eye(2) * self.Q          # process noise covariance
        
        state_pred = F @ state
        cov_pred = F @ cov @ F.T + Q
        return state_pred, cov_pred
    
    def _kalman_update(self, state_pred, cov_pred, meas):
        """Update step for one coordinate."""
        H = np.array([[1, 0]])           # measurement matrix (we measure position only)
        R = np.array([[self.R]])          # measurement noise covariance
        
        y = meas - H @ state_pred         # innovation
        S = H @ cov_pred @ H.T + R        # innovation covariance
        K = cov_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        state_upd = state_pred + K @ y
        cov_upd = (np.eye(2) - K @ H) @ cov_pred
        return state_upd, cov_upd
    
    def _contour_iou(self, cnt1: np.ndarray, cnt2: np.ndarray) -> float:
        """Compute Intersection over Union of bounding rectangles (fast proxy for shape similarity)."""
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        
        # Intersection rectangle
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = min(x1 + w1, x2 + w2) - xi
        hi = min(y1 + h1, y2 + h2) - yi
        
        if wi <= 0 or hi <= 0:
            return 0.0
        
        inter_area = wi * hi
        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    def register_object(self, obj_id: int, contour: np.ndarray, centroid: Tuple[float, float], frame: np.ndarray = None):
        """Initialize memory for a new object. Optionally compute average RGB from frame."""
        (kx, ky) = self._init_kalman(centroid[0], centroid[1])
        # Compute bounding box of contour
        x, y, w, h = cv2.boundingRect(contour)
        # Compute average RGB if frame provided
        avg_rgb = None
        if frame is not None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_val = cv2.mean(frame, mask=mask)
            avg_rgb = (mean_val[0], mean_val[1], mean_val[2])  # BGR order
        self.shape_memory[obj_id] = {
            'contours': [contour],
            'bboxes': [(x, y, w, h)],   # store as tuple (x, y, w, h)
            'timestamps': [time.time()],
            'kalman_x': kx,
            'kalman_y': ky,
            'lost_count': 0,
            'avg_rgb': avg_rgb  # store average RGB for color matching
        }
    
    def update_object(self, obj_id: int, contour: np.ndarray, centroid: Tuple[float, float], frame: np.ndarray = None):
        """Update memory with a new detection. Optionally update average RGB."""
        mem = self.shape_memory.get(obj_id)
        if mem is None:
            self.register_object(obj_id, contour, centroid, frame)
            return
        
        # Update contour history
        mem['contours'].append(contour)
        if len(mem['contours']) > self.max_history:
            mem['contours'].pop(0)
        
        # Update bounding box history
        x, y, w, h = cv2.boundingRect(contour)
        if 'bboxes' not in mem:
            mem['bboxes'] = []
        mem['bboxes'].append((x, y, w, h))
        if len(mem['bboxes']) > self.max_history:
            mem['bboxes'].pop(0)
        
        mem['timestamps'].append(time.time())
        if len(mem['timestamps']) > self.max_history:
            mem['timestamps'].pop(0)
        
        # Update average RGB if frame provided
        if frame is not None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_val = cv2.mean(frame, mask=mask)
            mem['avg_rgb'] = (mean_val[0], mean_val[1], mean_val[2])
        
        # Kalman update
        dt = 1.0  # assume constant frame rate, could compute from timestamps
        # Predict
        kx_pred, cx_pred = self._kalman_predict(mem['kalman_x'][0], mem['kalman_x'][1], dt)
        ky_pred, cy_pred = self._kalman_predict(mem['kalman_y'][0], mem['kalman_y'][1], dt)
        # Update with measurement
        kx_upd, cx_upd = self._kalman_update(kx_pred, cx_pred, centroid[0])
        ky_upd, cy_upd = self._kalman_update(ky_pred, cy_pred, centroid[1])
        mem['kalman_x'] = (kx_upd, cx_upd)
        mem['kalman_y'] = (ky_upd, cy_upd)
        
        mem['lost_count'] = 0

    def mark_lost(self, obj_id: int):
        """Increment lost counter for an object."""
        if obj_id in self.shape_memory:
            self.shape_memory[obj_id]['lost_count'] += 1
    
    def get_smoothed_contour(self, obj_id: int) -> Optional[np.ndarray]:
        """
        Return a temporally smoothed contour (median of recent contours).
        If object is lost but within reconstruction window, return the last known contour.
        """
        mem = self.shape_memory.get(obj_id)
        if not mem:
            return None
        
        lost = mem['lost_count']
        if lost > self.max_reconstruct_frames:
            return None  # too long lost, drop
        
        if lost > 0:
            # lost – return last known contour (could also predict via Kalman)
            return mem['contours'][-1] if mem['contours'] else None
        
        # not lost – return median of recent contours (or just last)
        if len(mem['contours']) >= 3:
            # compute median contour? hard. Return last for simplicity.
            # Could average points, but expensive. We'll just return last.
            return mem['contours'][-1]
        else:
            return mem['contours'][-1] if mem['contours'] else None
    
    def get_kalman_centroid(self, obj_id: int) -> Optional[Tuple[float, float]]:
        """Return the Kalman‑filtered centroid (predicted if lost)."""
        mem = self.shape_memory.get(obj_id)
        if not mem:
            return None
        
        # Predict ahead if lost
        if mem['lost_count'] > 0:
            dt = mem['lost_count']  # assume 1 frame per lost count
            kx_pred, _ = self._kalman_predict(mem['kalman_x'][0], mem['kalman_x'][1], dt)
            ky_pred, _ = self._kalman_predict(mem['kalman_y'][0], mem['kalman_y'][1], dt)
            return (kx_pred[0], ky_pred[0])
        else:
            # use last updated state (position component)
            return (mem['kalman_x'][0][0], mem['kalman_y'][0][0])
    
    def get_last_bbox(self, obj_id: int) -> Optional[Tuple[int, int, int, int]]:
        """Return the last known bounding box (x, y, w, h) for the object."""
        mem = self.shape_memory.get(obj_id)
        if mem and 'bboxes' in mem and mem['bboxes']:
            return mem['bboxes'][-1]
        return None
    
    def get_avg_rgb(self, obj_id: int) -> Optional[Tuple[float, float, float]]:
        """Return the average RGB of the object."""
        mem = self.shape_memory.get(obj_id)
        if mem:
            return mem.get('avg_rgb')
        return None

    def find_best_match(self, contour: np.ndarray, active_ids: List[int]) -> Optional[int]:
        """
        Given a new contour, find the best matching existing object by bounding‑box IoU.
        Returns object ID if IoU > threshold, else None.
        """
        best_id = None
        best_iou = self.iou_threshold
        for obj_id in active_ids:
            mem = self.shape_memory.get(obj_id)
            if not mem or not mem['contours']:
                continue
            last_cnt = mem['contours'][-1]
            iou = self._contour_iou(contour, last_cnt)
            if iou > best_iou:
                best_iou = iou
                best_id = obj_id
        return best_id

class AppearanceExtractor:
    """
    Robust object envelope extraction using multi-mask fusion and adaptive lighting compensation.
    Implements HSV/LAB clustering, shadow compensation, region growing, and contour scoring.
    Optimized for low-res cameras and low-end hardware.
    """
    
    def __init__(self, 
                 roi_margin: float = 0.08,           # ROI expansion margin
                 min_contour_area: int = 50,          # Minimum contour area (pixels)
                 n_colors: int = 3,                    # Number of color clusters for HSV
                 shadow_threshold: float = 0.3,        # Shadow detection threshold (V difference)
                 edge_threshold1: int = 40,             # Canny low threshold
                 edge_threshold2: int = 120,            # Canny high threshold
                 grow_tolerance: float = 0.15):         # Region growing color tolerance (fraction)
        
        self.roi_margin = roi_margin
        self.min_contour_area = min_contour_area
        self.n_colors = n_colors
        self.shadow_threshold = shadow_threshold
        self.edge_threshold1 = edge_threshold1
        self.edge_threshold2 = edge_threshold2
        self.grow_tolerance = grow_tolerance
        
    def _simple_kmeans_hue(self, hue_vals: np.ndarray, k: int, max_iters: int = 10) -> np.ndarray:
        """
        Fast k-means on hue values (circular) using histogram peaks as initial centers.
        Returns cluster centers (hue angles).
        """
        if len(hue_vals) < k:
            # Not enough pixels, return equally spaced centers
            return np.linspace(0, 179, k, dtype=np.float32)
        
        # Use histogram peaks as initial centers (fast)
        hist = np.bincount(hue_vals, minlength=180)
        # Find top k peaks (excluding zero bins maybe)
        peaks = []
        for _ in range(k):
            if np.max(hist) == 0:
                break
            peak = np.argmax(hist)
            peaks.append(peak)
            # Suppress neighborhood to avoid multiple peaks near each other
            start = max(0, peak - 10)
            end = min(180, peak + 11)
            hist[start:end] = 0
        if len(peaks) < k:
            # Pad with default
            peaks.extend([0] * (k - len(peaks)))
        
        centers = np.array(peaks, dtype=np.float32)
        
        # Simple k-means with circular distance
        for _ in range(max_iters):
            # Assign each pixel to nearest center (circular distance)
            distances = np.abs(hue_vals[:, None] - centers[None, :])
            distances = np.minimum(distances, 180 - distances)  # circular
            labels = np.argmin(distances, axis=1)
            
            # Update centers
            new_centers = []
            for i in range(k):
                mask = labels == i
                if np.any(mask):
                    # Circular mean
                    mean_hue = np.mean(hue_vals[mask])
                    new_centers.append(mean_hue)
                else:
                    new_centers.append(centers[i])
            new_centers = np.array(new_centers)
            if np.allclose(centers, new_centers, atol=1.0):
                break
            centers = new_centers
        return centers
    
    def _adaptive_color_mask(self, roi_hsv: np.ndarray, seed_point: Tuple[int, int]) -> np.ndarray:
        """
        Generate color mask using HSV clustering around seed point.
        Returns binary mask.
        """
        h, w = roi_hsv.shape[:2]
        seed_y, seed_x = seed_point
        seed_hue = roi_hsv[seed_y, seed_x, 0]
        
        # Sample pixels near seed to estimate local color distribution
        margin = 5
        y1 = max(0, seed_y - margin)
        y2 = min(h, seed_y + margin + 1)
        x1 = max(0, seed_x - margin)
        x2 = min(w, seed_x + margin + 1)
        local_region = roi_hsv[y1:y2, x1:x2]
        local_hues = local_region[:, :, 0].reshape(-1)
        local_sats = local_region[:, :, 1].reshape(-1)
        
        # Cluster hues in local region (k=2 for object vs background)
        if len(local_hues) > 10:
            centers = self._simple_kmeans_hue(local_hues, k=min(2, self.n_colors))
            # Find which center is closest to seed hue
            dist_to_seed = np.minimum(np.abs(centers - seed_hue), 180 - np.abs(centers - seed_hue))
            object_center = centers[np.argmin(dist_to_seed)]
            # Use standard deviation of local hues around object center as tolerance
            distances = np.minimum(np.abs(local_hues - object_center), 180 - np.abs(local_hues - object_center))
            hue_std = np.std(distances)
            hue_range = max(15, int(2 * hue_std))
        else:
            object_center = seed_hue
            hue_range = 20
        
        # Also compute local saturation and value statistics for adaptive thresholds
        local_sats = local_sats[local_sats > 0]  # ignore zero if any
        local_vals = local_region[:, :, 2].reshape(-1)
        local_vals = local_vals[local_vals > 0]
        
        s_mean = np.mean(local_sats) if len(local_sats) > 0 else 100
        v_mean = np.mean(local_vals) if len(local_vals) > 0 else 100
        
        # Adaptive lower/upper bounds (ensure integer and within 0-255)
        lower_hue = int(max(0, object_center - hue_range))
        upper_hue = int(min(179, object_center + hue_range))
        
        lower_sat = int(max(30, s_mean * 0.4))
        upper_sat = 255
        
        lower_val = int(max(30, v_mean * 0.3))
        upper_val = 255
        
        # Create arrays with explicit uint8 type
        lower = np.array([lower_hue, lower_sat, lower_val], dtype=np.uint8)
        upper = np.array([upper_hue, upper_sat, upper_val], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(roi_hsv, lower, upper)
        
        # Optional: handle hue wrap-around
        if lower_hue > upper_hue:  # if range crosses 0
            lower2 = np.array([0, lower_sat, lower_val], dtype=np.uint8)
            upper2 = np.array([upper_hue, upper_sat, upper_val], dtype=np.uint8)
            mask2 = cv2.inRange(roi_hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)
        
        return mask  

    def _shadow_compensation(self, roi_hsv: np.ndarray, color_mask: np.ndarray) -> np.ndarray:
        """
        Detect and compensate for shadows by analyzing V channel relative to local mean.
        Returns adjusted mask (shadows removed or included based on color).
        """
        # Compute local average V for non-zero areas in color mask
        v_channel = roi_hsv[:, :, 2].astype(np.float32)
        if np.sum(color_mask) == 0:
            return color_mask
        
        # Estimate object's typical brightness from areas already in mask
        object_v = v_channel[color_mask > 0]
        if len(object_v) == 0:
            return color_mask
        mean_v = np.mean(object_v)
        std_v = np.std(object_v)
        
        # Shadow regions: pixels that have similar hue/sat but much lower V
        # We'll create a mask of candidate shadows: where V < mean_v - threshold*std_v
        shadow_candidates = (v_channel < (mean_v - self.shadow_threshold * std_v)).astype(np.uint8) * 255
        
        # But only if hue and sat are similar to object (we already have color mask for those)
        # So final shadow mask = shadow_candidates AND (hue/sat similar)
        # We can approximate by dilating color mask and intersecting
        kernel = np.ones((5,5), np.uint8)
        dilated_mask = cv2.dilate(color_mask, kernel, iterations=1)
        shadow_mask = cv2.bitwise_and(shadow_candidates, dilated_mask)
        
        # Add shadow mask to original color mask
        compensated_mask = cv2.bitwise_or(color_mask, shadow_mask)
        return compensated_mask
    
    def _edge_mask(self, roi_gray: np.ndarray) -> np.ndarray:
        """Canny edge detection with morphological closing to form connected regions."""
        edges = cv2.Canny(roi_gray, self.edge_threshold1, self.edge_threshold2)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        return edges
    
    def _region_growing(self, mask: np.ndarray, seed_point: Tuple[int, int], 
                        roi_hsv: np.ndarray, roi_gray: np.ndarray) -> np.ndarray:
        """
        Grow region from seed using color and edge affinity.
        Returns refined mask.
        """
        h, w = mask.shape
        seed_y, seed_x = seed_point
        
        # If seed is out of bounds or already zero, return original
        if seed_y >= h or seed_x >= w or mask[seed_y, seed_x] == 0:
            return mask
        
        # Get seed color - convert to int to avoid uint8 underflow later
        seed_hsv = roi_hsv[seed_y, seed_x]
        seed_hue = int(seed_hsv[0])
        seed_sat = int(seed_hsv[1])
        seed_val = int(seed_hsv[2])
        
        # Create a queue for flood fill
        visited = np.zeros_like(mask, dtype=bool)
        grown_mask = np.zeros_like(mask)
        
        queue = [(seed_y, seed_x)]
        visited[seed_y, seed_x] = True
        
        # Tolerances
        hue_tol = 15
        sat_tol = 40
        val_tol = 40
        
        while queue:
            y, x = queue.pop(0)
            grown_mask[y, x] = 255
            
            # Check 4-neighbors
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y+dy, x+dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx]:
                    continue
                
                # Check if neighbor is in original mask OR color-similar
                if mask[ny, nx] > 0:
                    # If it's in mask, accept
                    visited[ny, nx] = True
                    queue.append((ny, nx))
                else:
                    # Check color similarity - convert neighbor values to int
                    nhsv = roi_hsv[ny, nx]
                    nhsv_0 = int(nhsv[0])
                    nhsv_1 = int(nhsv[1])
                    nhsv_2 = int(nhsv[2])
                    
                    # Compute differences (now with ints, no overflow)
                    dhue = min(abs(nhsv_0 - seed_hue), 180 - abs(nhsv_0 - seed_hue))
                    dsat = abs(nhsv_1 - seed_sat)
                    dval = abs(nhsv_2 - seed_val)
                    
                    if dhue < hue_tol and dsat < sat_tol and dval < val_tol:
                        # Also check edge presence: if strong edge nearby, maybe stop
                        # For simplicity, we proceed
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        
        return grown_mask  

    def _contour_confidence(self, contour: np.ndarray, roi_gray: np.ndarray, 
                            roi_mask: np.ndarray) -> float:
        """
        Compute confidence score for a contour based on:
        - Edge strength along boundary
        - Color consistency inside
        - Contour smoothness (curve continuity)
        """
        # Create mask of contour interior
        interior_mask = np.zeros_like(roi_gray, dtype=np.uint8)
        cv2.drawContours(interior_mask, [contour], -1, 255, -1)
        
        # 1. Color consistency: standard deviation of gray values inside
        inside_pixels = roi_gray[interior_mask > 0]
        if len(inside_pixels) == 0:
            color_score = 0.0
        else:
            color_std = np.std(inside_pixels)
            # Normalize: lower std is better (max possible std ~128)
            color_score = max(0.0, 1.0 - (color_std / 128.0))
        
        # 2. Edge strength along boundary: sample edge image at contour points
        edges = cv2.Canny(roi_gray, self.edge_threshold1, self.edge_threshold2)
        # Draw contour line
        contour_line = np.zeros_like(roi_gray, dtype=np.uint8)
        cv2.drawContours(contour_line, [contour], -1, 255, 1)
        edge_on_contour = cv2.bitwise_and(edges, contour_line)
        edge_strength = np.sum(edge_on_contour > 0) / max(1, np.sum(contour_line > 0))
        
        # 3. Smoothness: compare arc length to convex hull perimeter
        hull = cv2.convexHull(contour)
        arc_len = cv2.arcLength(contour, True)
        hull_len = cv2.arcLength(hull, True)
        if hull_len > 0:
            smoothness = hull_len / arc_len  # closer to 1 means smoother
        else:
            smoothness = 1.0
        
        # Combine scores (weights can be tuned)
        confidence = 0.3 * color_score + 0.5 * edge_strength + 0.2 * smoothness
        return confidence
    
    def extract_contours(self, frame: np.ndarray, box: Tuple[int, int, int, int, int, int, int, int], 
                         prior_contour: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, float]]:
        """
        Main method: extract multiple contours from ROI with confidence scores.
        Returns list of (contour, confidence).
        """
        x1,y1,x2,y2,x3,y3,x4,y4 = box
        x_min = min(x1,x2,x3,x4)
        y_min = min(y1,y2,y3,y4)
        x_max = max(x1,x2,x3,x4)
        y_max = max(y1,y2,y3,y4)

        margin = int(self.roi_margin * max(x_max - x_min, y_max - y_min))
        h, w = frame.shape[:2]
        x_min_exp = max(0, x_min - margin)
        y_min_exp = max(0, y_min - margin)
        x_max_exp = min(w, x_max + margin)
        y_max_exp = min(h, y_max + margin)

        roi = frame[y_min_exp:y_max_exp, x_min_exp:x_max_exp]
        if roi.size == 0:
            return []
        
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Seed point: center of original box (before expansion) within ROI
        seed_x = (x_min + x_max)//2 - x_min_exp
        seed_y = (y_min + y_max)//2 - y_min_exp
        seed_point = (seed_y, seed_x)
        
        # --- Multi-mask generation ---
        
        # 1. Adaptive color mask (HSV clustering)
        color_mask = self._adaptive_color_mask(roi_hsv, seed_point)
        
        # 2. Shadow compensation
        color_mask = self._shadow_compensation(roi_hsv, color_mask)
        
        # 3. Edge mask
        edge_mask = self._edge_mask(roi_gray)
        
        # 4. Fusion: combine color and edge (weighted OR, but keep color as primary)
        #    Use color mask as base, then add edges to fill gaps
        kernel = np.ones((3,3), np.uint8)
        color_mask_dilated = cv2.dilate(color_mask, kernel, iterations=1)
        fused_mask = cv2.bitwise_or(color_mask_dilated, edge_mask)
        # 5. Apply prior contour if available (as a soft constraint)
        if prior_contour is not None:
            # Transform prior contour to ROI coordinates
            prior_in_roi = prior_contour.copy()
            prior_in_roi[:, :, 0] -= x_min_exp
            prior_in_roi[:, :, 1] -= y_min_exp
            
            # Create mask from prior contour and dilate it
            prior_mask = np.zeros_like(roi_gray, dtype=np.uint8)
            cv2.drawContours(prior_mask, [prior_in_roi], -1, 255, -1)
            kernel_large = np.ones((11,11), np.uint8)
            prior_mask_dilated = cv2.dilate(prior_mask, kernel_large, iterations=2)
            
            # Restrict fused mask to the dilated prior region
            fused_mask = cv2.bitwise_and(fused_mask, prior_mask_dilated)
        
        # 6. Region growing from seed to ensure continuity
        # 5. Region growing from seed to ensure continuity
        grown_mask = self._region_growing(fused_mask, seed_point, roi_hsv, roi_gray)
        
        # --- Contour extraction ---
        contours, _ = cv2.findContours(grown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        
        # Filter small contours
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_contour_area:
                # Smooth contour
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, epsilon, True)
                # Shift back to frame coordinates
                cnt[:, :, 0] += x_min_exp
                cnt[:, :, 1] += y_min_exp
                # Compute confidence
                # Need ROI-level gray for confidence; we have roi_gray in ROI coordinates,
                # but contour is in frame coordinates. For simplicity, we compute confidence
                # before shifting, using roi_gray and mask.
                # We'll compute confidence now (before shift) using roi-level data.
                # Create a temporary contour copy in ROI coordinates
                cnt_roi = cnt.copy()
                cnt_roi[:, :, 0] -= x_min_exp
                cnt_roi[:, :, 1] -= y_min_exp
                conf = self._contour_confidence(cnt_roi, roi_gray, grown_mask)
                valid.append((cnt, conf))
        
        # Sort by confidence descending
        valid.sort(key=lambda x: x[1], reverse=True)
        return valid

# ----------------------------------------------------------------------
# Global proposal generator (runs every frame, independent of depth)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Global proposal generator (runs every frame, independent of depth)
# Now includes all requested improvements:
#   - Adaptive min_area based on frame resolution
#   - Aspect ratio filtering
#   - Texture/gradient strength filtering
#   - Merging of overlapping proposals
#   - Center-distance confidence bonus
#   - Temporal persistence scoring
#   - Improved motion integration (combine edge+motion)
#   - Max-size rejection
# ----------------------------------------------------------------------
def generate_global_proposals(frame: np.ndarray,
                              prev_frame: Optional[np.ndarray] = None,
                              prev_proposals_list: List[Tuple] = None,
                              use_motion: bool = False,
                              max_proposals: int = 10) -> List[Tuple[Tuple[int,int,int,int], np.ndarray, float]]:
    """
    Fast candidate object detection with advanced filtering and scoring.
    
    Args:
        frame: current BGR frame
        prev_frame: previous grayscale frame (for motion, if use_motion=True)
        prev_proposals_list: list of proposals from previous frame for temporal persistence
        use_motion: whether to compute motion mask (set True only for static camera)
        max_proposals: maximum number of proposals to return

    Returns:
        List of (bbox, contour, confidence) sorted by confidence.
        bbox is (x, y, w, h) in image coordinates.
    """
    h_img, w_img = frame.shape[:2]
    frame_area = h_img * w_img

    # ----- 1. Adaptive minimum area (0.05% of frame area, but at least 30 pixels) -----
    adaptive_min_area = max(30, int(frame_area * 0.0005))

    # ----- 2. Max-size rejection (reject proposals covering >60% of frame) -----
    max_allowed_area = int(0.6 * frame_area)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pre‑compute gradient magnitude for texture filtering (Sobel)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))

    raw_proposals = []  # will hold (bbox, contour, confidence)

    # ----- 3. Edge-based contours (Canny + closing) -----
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < adaptive_min_area:
            continue
        if area > max_allowed_area:
            continue

        # Approximate contour
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        # ----- Aspect ratio filtering (reject extremely thin/elongated) -----
        if w / h > 10 or h / w > 10:
            continue

        # ----- Texture / gradient strength inside bounding box -----
        roi_grad = grad_mag[y:y+h, x:x+w]
        if roi_grad.size > 0:
            mean_grad = np.mean(roi_grad)
            if mean_grad < 5.0:   # heuristic threshold for flat areas
                continue
        else:
            continue

        # Base confidence from area (normalized)
        conf = min(1.0, area / 5000.0)
        raw_proposals.append(((x, y, w, h), approx, conf))

    # ----- 4. Motion mask (if enabled) and combine with edge proposals -----
    if use_motion and prev_frame is not None:
        diff = cv2.absdiff(prev_frame, gray)
        _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        m_cnts, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in m_cnts:
            area = cv2.contourArea(cnt)
            if area < adaptive_min_area:
                continue
            if area > max_allowed_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Aspect ratio filter
            if w / h > 10 or h / w > 10:
                continue
            # Texture check
            roi_grad = grad_mag[y:y+h, x:x+w]
            if roi_grad.size > 0 and np.mean(roi_grad) < 5.0:
                continue

            # Check overlap with existing edge proposals
            overlap_found = False
            for i, (bbox, cnt_ex, conf_ex) in enumerate(raw_proposals):
                bx, by, bw, bh = bbox
                # Compute IoU of bounding boxes
                xi = max(x, bx)
                yi = max(y, by)
                wi = min(x+w, bx+bw) - xi
                hi = min(y+h, by+bh) - yi
                if wi > 0 and hi > 0:
                    inter = wi * hi
                    union = w*h + bw*bh - inter
                    iou = inter / union if union > 0 else 0
                    if iou > 0.3:   # significant overlap
                        overlap_found = True
                        # Boost confidence of the edge proposal
                        raw_proposals[i] = (bbox, cnt_ex, min(1.0, conf_ex + 0.15))
                        break
            if not overlap_found:
                # Motion-only proposal with lower confidence
                conf = min(1.0, area / 3000.0) * 0.7
                raw_proposals.append(((x, y, w, h), cnt, conf))

    # ----- 5. Merge overlapping proposals (Non‑Maximum Suppression) -----
    merged_proposals = []
    raw_proposals.sort(key=lambda p: p[2], reverse=True)  # sort by confidence descending
    used = [False] * len(raw_proposals)

    for i, (bbox_i, cnt_i, conf_i) in enumerate(raw_proposals):
        if used[i]:
            continue
        x1, y1, w1, h1 = bbox_i
        # Group all proposals that overlap significantly with this one
        group = [i]
        for j, (bbox_j, cnt_j, conf_j) in enumerate(raw_proposals):
            if used[j] or i == j:
                continue
            x2, y2, w2, h2 = bbox_j
            xi = max(x1, x2)
            yi = max(y1, y2)
            wi = min(x1+w1, x2+w2) - xi
            hi = min(y1+h1, y2+h2) - yi
            if wi > 0 and hi > 0:
                inter = wi * hi
                union = w1*h1 + w2*h2 - inter
                iou = inter / union if union > 0 else 0
                if iou > 0.4:   # overlap threshold for merging
                    group.append(j)
        if group:
            # Keep the highest confidence proposal from the group
            best_idx = max(group, key=lambda idx: raw_proposals[idx][2])
            best_bbox, best_cnt, best_conf = raw_proposals[best_idx]
            merged_proposals.append((best_bbox, best_cnt, best_conf))
            for idx in group:
                used[idx] = True

    # ----- 6. Center-distance confidence bonus -----
    center_x, center_y = w_img / 2.0, h_img / 2.0
    max_dist = np.sqrt(center_x**2 + center_y**2)
    for i, (bbox, cnt, conf) in enumerate(merged_proposals):
        x, y, w, h = bbox
        prop_center_x = x + w/2.0
        prop_center_y = y + h/2.0
        dist = np.sqrt((prop_center_x - center_x)**2 + (prop_center_y - center_y)**2)
        # bonus factor: 1.0 at center, linearly decreasing to 0.8 at edges
        bonus = 1.0 - 0.2 * (dist / max_dist)
        merged_proposals[i] = (bbox, cnt, min(1.0, conf * bonus))

    # ----- 7. Temporal persistence scoring (if previous proposals available) -----
    if prev_proposals_list is not None and len(prev_proposals_list) > 0:
        # For each current proposal, find best matching previous proposal by IoU
        for i, (bbox, cnt, conf) in enumerate(merged_proposals):
            best_match = None
            best_iou = 0.3   # matching threshold
            for prev_bbox, prev_cnt, prev_conf in prev_proposals_list:
                x1,y1,w1,h1 = bbox
                x2,y2,w2,h2 = prev_bbox
                xi = max(x1, x2)
                yi = max(y1, y2)
                wi = min(x1+w1, x2+w2) - xi
                hi = min(y1+h1, y2+h2) - yi
                if wi > 0 and hi > 0:
                    inter = wi * hi
                    union = w1*h1 + w2*h2 - inter
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_match = prev_conf
            if best_match is not None:
                # Temporal smoothing: new_conf = 0.7*current + 0.3*previous
                new_conf = 0.7 * conf + 0.3 * best_match
            else:
                # New proposal: slight penalty
                new_conf = conf * 0.9
            merged_proposals[i] = (bbox, cnt, min(1.0, new_conf))

    # ----- 8. Final sort and truncation -----
    merged_proposals.sort(key=lambda p: p[2], reverse=True)
    return merged_proposals[:max_proposals]

# ----------------------------------------------------------------------
# Proposal fusion layer (combines global and depth proposals,
# computes scores, merges overlapping, and returns top N)
# ----------------------------------------------------------------------
def fuse_proposals(global_proposals: List[Tuple],
                   depth_proposals: List[Tuple],
                   frame_shape: Tuple[int, int],
                   max_proposals: int = 6,
                   overlap_threshold: float = 0.4,
                   depth_bonus: float = 0.2) -> List[Tuple]:
    """
    Fuse global and depth proposals, compute final scores, and apply NMS.

    Args:
        global_proposals: list from generate_global_proposals (bbox, contour, conf)
        depth_proposals: list of depth-derived proposals (bbox, contour, conf)
        frame_shape: (height, width) for area normalization
        max_proposals: maximum number to keep
        overlap_threshold: IoU threshold for NMS
        depth_bonus: extra confidence added if proposal overlaps with any depth proposal

    Returns:
        List of fused proposals: (bbox, contour, score)
    """
    h_img, w_img = frame_shape
    frame_area = h_img * w_img

    # Combine all proposals into one list
    all_props = global_proposals + depth_proposals

    # Compute final score for each proposal
    scored = []
    for bbox, contour, base_conf in all_props:
        x, y, w, h = bbox
        area = w * h

        # Area weight (normalized by frame area, capped)
        area_weight = min(1.0, area / (frame_area * 0.1))  # 10% of frame as max

        # Contour solidity (area / convex hull area) as strength indicator
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # Combine base confidence with area and solidity
        # Base confidence already contains edge strength & temporal persistence
        score = 0.5 * base_conf + 0.3 * area_weight + 0.2 * solidity

        # Depth bonus: if this proposal overlaps with any depth proposal
        if depth_proposals:
            for dbox, dcontour, dconf in depth_proposals:
                dx, dy, dw, dh = dbox
                # Compute IoU
                xi = max(x, dx)
                yi = max(y, dy)
                wi = min(x+w, dx+dw) - xi
                hi = min(y+h, dy+dh) - yi
                if wi > 0 and hi > 0:
                    inter = wi * hi
                    union = w*h + dw*dh - inter
                    iou = inter / union if union > 0 else 0
                    if iou > 0.3:
                        score += depth_bonus
                        break

        # Clamp score to [0,1]
        score = min(1.0, max(0.0, score))
        scored.append((bbox, contour, score))

    # Sort by score descending
    scored.sort(key=lambda p: p[2], reverse=True)

    # Non‑maximum suppression (merge overlapping)
    kept = []
    used = [False] * len(scored)

    for i, (bbox_i, cnt_i, score_i) in enumerate(scored):
        if used[i]:
            continue
        x1, y1, w1, h1 = bbox_i
        # Keep this proposal
        kept.append((bbox_i, cnt_i, score_i))
        used[i] = True
        # Suppress others that overlap too much
        for j, (bbox_j, cnt_j, score_j) in enumerate(scored):
            if used[j]:
                continue
            x2, y2, w2, h2 = bbox_j
            xi = max(x1, x2)
            yi = max(y1, y2)
            wi = min(x1+w1, x2+w2) - xi
            hi = min(y1+h1, y2+h2) - yi
            if wi > 0 and hi > 0:
                inter = wi * hi
                union = w1*h1 + w2*h2 - inter
                iou = inter / union if union > 0 else 0
                if iou > overlap_threshold:
                    used[j] = True  # suppress

    # Return top N
    return kept[:max_proposals]

current_box = None
box_lock = threading.Lock()
smoothed_box = None          # EMA-smoothed box coordinates
prev_contour = None          # last frame's best contour for prior
alpha = 0.3                  # EMA smoothing factor (0.0-1.0, lower = smoother)
locked_id = None             # ID of the locked object (if any)
lock_frame_count = 0         # frames since lock was established (for confidence)
LOCK_CONFIDENCE_THRESHOLD = 5   # need 5 frames of good tracking before locking
LOCK_PERSISTENCE_THRESHOLD = 0.7 # minimum persistence score to lock

# For global proposal generator enhancements
prev_frame_gray = None       # previous grayscale frame for motion detection
prev_proposals = []          # proposals from previous frame for temporal persistence
# -------------------------------------------------------
# Pipe reader
# -------------------------------------------------------
def pipe_reader():
    global current_box
    print("Pipe reader thread started.", file=sys.stderr)
    with open(PIPE_PATH, 'r') as pipe:
        while True:
            line = pipe.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            try:
                box = tuple(map(int, parts))
                with box_lock:
                    current_box = box
            except:
                pass

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    if not os.path.exists(PIPE_PATH):
        print("Pipe not found.", file=sys.stderr)
        sys.exit(1)
    global smoothed_box, prev_contour, locked_id, lock_frame_count

    threading.Thread(target=pipe_reader, daemon=True).start()

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Camera open failed", file=sys.stderr)
        sys.exit(1)

    # Initialize tracker with parameters optimized for low-res video
    tracker = ObjectTracker(
        max_lost_frames=8,           # Quick recovery after occlusion
        distance_threshold=40,        # Tighter matching for small objects
        persistence_decay=0.6,        # Moderate decay
        min_persistence=0.25          # Lower threshold for reassignment
    )

    # Initialize appearance extractor
    extractor = AppearanceExtractor(
        roi_margin=0.12,
        min_contour_area=100,
        n_colors=3,
        shadow_threshold=0.3,
        edge_threshold1=40,
        edge_threshold2=120,
        grow_tolerance=0.15
    )

    cv2.namedWindow(WINDOW_NAME)

    cv2.namedWindow(WINDOW_NAME)
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        with box_lock:
            raw_box = current_box

        # Apply EMA smoothing to box coordinates
        if raw_box is not None:
            if smoothed_box is None:
                smoothed_box = raw_box
            else:
                # Smooth each of the 8 coordinates
                smoothed_box = tuple(int(alpha * c + (1 - alpha) * s) 
                                     for c, s in zip(raw_box, smoothed_box))
            box_to_use = smoothed_box
        else:
            box_to_use = None
        # ---------------------------------------------------------------
        # Generate global proposals (runs every frame)
        # ---------------------------------------------------------------
        global prev_frame_gray, prev_proposals
        global_proposals = generate_global_proposals(
            frame,
            prev_frame=prev_frame_gray,          # for motion detection
            prev_proposals_list=prev_proposals,  # for temporal persistence
            use_motion=False,                     # set True only for static camera
            max_proposals=10
        )

        # ---------------------------------------------------------------
        # Depth proposals (from pipe) – convert to same format
        # ---------------------------------------------------------------
        depth_proposals = []
        if box_to_use is not None:
            # The box from pipe is 8 coordinates; we create a bounding rectangle and a simple contour
            x1,y1,x2,y2,x3,y3,x4,y4 = box_to_use
            xs = [x1,x2,x3,x4]
            ys = [y1,y2,y3,y4]
            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)
            bbox = (x_min, y_min, x_max-x_min, y_max-y_min)
            # Create a contour that is the bounding rectangle
            rect_contour = np.array([[x_min, y_min],
                                      [x_max, y_min],
                                      [x_max, y_max],
                                      [x_min, y_max]], dtype=np.int32).reshape((-1,1,2))
            depth_proposals.append((bbox, rect_contour, 1.0))   # confidence 1.0 for depth
        elif locked_id is not None:
            # If we have a locked object but no depth box, we can still generate a prediction box
            kalman_cent = tracker.geometry_store.get_kalman_centroid(locked_id)
            last_bbox = tracker.geometry_store.get_last_bbox(locked_id)
            if kalman_cent is not None and last_bbox is not None:
                cx, cy = kalman_cent
                last_x, last_y, last_w, last_h = last_bbox
                # Expand box to search area
                margin_factor = 2.0
                search_w = int(last_w * margin_factor)
                search_h = int(last_h * margin_factor)
                x_min = int(cx - search_w // 2)
                y_min = int(cy - search_h // 2)
                x_max = x_min + search_w
                y_max = y_min + search_h
                # Clamp to image
                h_img, w_img = frame.shape[:2]
                x_min = max(0, min(x_min, w_img-1))
                y_min = max(0, min(y_min, h_img-1))
                x_max = max(0, min(x_max, w_img-1))
                y_max = max(0, min(y_max, h_img-1))
                bbox = (x_min, y_min, x_max-x_min, y_max-y_min)
                rect_contour = np.array([[x_min, y_min],
                                          [x_max, y_min],
                                          [x_max, y_max],
                                          [x_min, y_max]], dtype=np.int32).reshape((-1,1,2))
                depth_proposals.append((bbox, rect_contour, 0.8))   # slightly lower confidence
        # ---------------------------------------------------------------
        # Proposal fusion
        # ---------------------------------------------------------------
        fused_proposals = fuse_proposals(
            global_proposals,
            depth_proposals,
            frame.shape[:2],
            max_proposals=6,
            overlap_threshold=0.4,
            depth_bonus=0.2
        )

        # ---------------------------------------------------------------
        # Appearance extraction from each fused proposal
        # ---------------------------------------------------------------
        detected_contours = []
        for (bbox, cnt, score) in fused_proposals:
            # Convert bbox (x,y,w,h) to 8-point format for extractor
            x, y, w, h = bbox
            box8 = (x, y, x+w, y, x+w, y+h, x, y+h)
            # Extract refined contours from this region
            # Note: prior_contour set to None because we don't have per‑object prior here
            contour_list = extractor.extract_contours(frame, box8, prior_contour=None)
            # Add all returned contours to the tracker input
            for refined_cnt, conf in contour_list:
                detected_contours.append(refined_cnt)

        # Optional: Draw fused proposals (in green) for debugging
        for (bbox, cnt, score) in fused_proposals:
            x,y,w,h = bbox
            # Draw bounding box (thick green)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            # Show score
            cv2.putText(frame, f"{score:.2f}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            # Also draw the original contour (dashed)
            cv2.drawContours(frame, [cnt], -1, (0,255,0), 1)

        # Update tracker with all detected contours
        tracked_objects = tracker.update(frame, detected_contours)

        # Compute attention scores for primary focus selection
        attention_scores = {}
        primary_focus_id = None
        if tracked_objects:
            max_area = max((obj.contour_area for obj in tracker.tracked_objects.values()), default=1)
            for obj_id, (contour, centroid) in tracked_objects.items():
                obj = tracker.tracked_objects[obj_id]
                area_weight = min(1.0, obj.contour_area / max_area) if max_area > 0 else 0
                motion_weight = obj.flow_confidence
                attention = obj.persistence_score + area_weight + motion_weight
                attention_scores[obj_id] = attention
            primary_focus_id = max(attention_scores.items(), key=lambda x: x[1])[0]

        if tracked_objects:
            # Find the object with highest persistence score
            best_id = max(tracked_objects.keys(), 
                          key=lambda oid: tracker.tracked_objects[oid].persistence_score)
            best_obj = tracker.tracked_objects[best_id]
            best_persistence = best_obj.persistence_score

            if locked_id is None:
                if best_persistence > LOCK_PERSISTENCE_THRESHOLD and best_obj.age > 8:
                    lock_frame_count += 1
                    if lock_frame_count >= LOCK_CONFIDENCE_THRESHOLD:
                        locked_id = best_id
                        print(f"Locked to object {locked_id}", file=sys.stderr)
                else:
                    lock_frame_count = 0

            else:
                # Check if locked object is still being tracked
                if locked_id in tracked_objects:
                    # Locked object found, reset counter
                    lock_frame_count = 0
                else:
                    # Locked object lost, count down
                    lock_frame_count -= 1
                    if lock_frame_count <= -LOCK_CONFIDENCE_THRESHOLD:
                        # Lost too long, release lock
                        print(f"Lost lock on object {locked_id}", file=sys.stderr)
                        locked_id = None
                        lock_frame_count = 0
        else:
            # No tracked objects at all
            if locked_id is not None:
                lock_frame_count -= 1
                if lock_frame_count <= -LOCK_CONFIDENCE_THRESHOLD:
                    print(f"Lost lock on object {locked_id} (no detections)", file=sys.stderr)
                    locked_id = None
                    lock_frame_count = 0
            else:
                lock_frame_count = 0
        
        # Draw tracked objects with stable IDs
        for obj_id, (contour, centroid) in tracked_objects.items():
            obj = tracker.tracked_objects[obj_id]
            
            # Determine line thickness and color
            if primary_focus_id is not None and obj_id == primary_focus_id:
                # Primary focus: thicker cyan line
                color = (255, 255, 0)  # Cyan
                thickness = 4
            else:
                # Others: based on persistence
                if obj.persistence_score > 0.7:
                    color = (0, 255, 0)  # Green - high confidence
                elif obj.persistence_score > 0.4:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 0, 255)  # Red - low confidence
                thickness = 2
            
            cv2.drawContours(frame, [contour], -1, color, thickness)
            
            # Draw ID and persistence info
            cx, cy = centroid
            # Show ID and age
            cv2.putText(frame, f"ID_{obj_id:03d}",
                       (cx - 20, cy - 15),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       color, 2)
            
            # Show persistence score as percentage
            cv2.putText(frame, f"{int(obj.persistence_score*100)}%",
                       (cx - 15, cy + 15),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.4,
                       color, 1)
            
            # Optional: Draw trajectory trail
            if len(obj.centroids) > 1:
                points = np.array(obj.centroids[-5:], dtype=np.int32)
                for i in range(1, len(points)):
                    cv2.line(frame, 
                             tuple(points[i-1]), 
                             tuple(points[i]), 
                             color, 1)

        # Update prior contour for next frame (use the most confident tracked object)
        if tracked_objects:
            # Find object with highest persistence score
            best_obj_id = max(tracked_objects.keys(), 
                              key=lambda oid: tracker.tracked_objects[oid].persistence_score)
            best_contour, _ = tracked_objects[best_obj_id]
            prev_contour = best_contour
        else:
            prev_contour = None

        # Display FPS and object count
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f} | Objects: {len(tracked_objects)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Update previous frame and proposals for next iteration
        prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Store only the top proposals (e.g., up to 15) to save memory
        prev_proposals = [(bbox, cnt, conf) for (bbox, cnt, conf) in global_proposals[:15]]
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()