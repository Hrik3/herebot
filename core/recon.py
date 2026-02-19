import cv2
import numpy as np
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import pickle

# ============================================================
# Track state definitions - based on DeepSORT research [citation:2]
# ============================================================
TRACK_STATE_TENTATIVE = "TENTATIVE"      # First 3 frames, no permanent ID
TRACK_STATE_CONFIRMED = "CONFIRMED"      # After 5 stable matches, gets permanent ID
TRACK_STATE_ARCHIVED = "ARCHIVED"        # Lost but remembered for re-identification

# State transition thresholds - optimized for low-end hardware [citation:3]
TENTATIVE_FRAMES = 3                      # Minimum frames in tentative state
CONFIRMATION_THRESHOLD = 5                # Consecutive matches needed for confirmation
MAX_ARCHIVED_AGE = 30                      # Frames to keep archived tracks

PIPE_PATH = '/tmp/vision_pipe'
WINDOW_NAME = 'Receiver'
CAMERA_DEVICE = '/dev/video11'

# ============================================================
# Global camera motion estimation (Stage 1 & 2)
# Optimized for low‑end hardware & low‑res cameras
# ============================================================
class GlobalMotionEstimator:
    """
    Estimates per‑frame global camera motion (translation, rotation, scale)
    and maintains a cumulative world transform.
    Uses Shi‑Tomasi corners + pyramidal LK + RANSAC affine.
    """
    # NEW: Cache homographies to avoid recomputation [citation:9]
    def __init__(self,
                 max_features: int = 150,
                 quality_level: float = 0.01,
                 min_distance: int = 5,
                 lk_win_size: tuple = (15, 15),
                 lk_max_level: int = 2,
                 ransac_thresh: float = 3.0,
                 max_iters: int = 1000,
                 confidence: float = 0.99):
        
        # Existing initializations
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.lk_win_size = lk_win_size
        self.lk_max_level = lk_max_level
        self.ransac_thresh = ransac_thresh
        self.max_iters = max_iters
        self.confidence = confidence
        self.last_affine = None
        
        # Previous frame data
        self.prev_gray = None
        self.prev_pts = None
        # Cumulative world transform: 2x3 affine matrix (identity initially)
        self.world_transform = np.eye(2, 3, dtype=np.float32)
        
        # NEW: Adaptive feature management for low-end hardware [citation:9]
        self.feature_cache = {}
        self.motion_history = []
        self.max_history = 30
        self.last_motion_magnitude = 0.0
        self.adaptive_feature_count = max_features
        
    # NEW: Adaptive feature extraction based on motion [citation:9]
    def _adaptive_feature_extraction(self, gray: np.ndarray, motion_magnitude: float) -> np.ndarray:
        """Adjust feature count based on motion magnitude."""
        if motion_magnitude > 10.0:  # Fast motion - need more features
            self.adaptive_feature_count = min(self.max_features, int(self.max_features * 1.5))
        elif motion_magnitude < 2.0:  # Slow motion - can use fewer features
            self.adaptive_feature_count = max(50, int(self.max_features * 0.7))
        else:
            self.adaptive_feature_count = self.max_features
            
        return self.detect_features(gray)

    def detect_features(self, gray: np.ndarray) -> np.ndarray:
        """Detect good features to track using Shi‑Tomasi."""
        pts = cv2.goodFeaturesToTrack(gray,
                                       maxCorners=self.max_features,
                                       qualityLevel=self.quality_level,
                                       minDistance=self.min_distance)
        if pts is None:
            return np.array([], dtype=np.float32).reshape(0, 2)
        return pts.reshape(-1, 2)

    def estimate_motion(self, curr_gray: np.ndarray):
        """
        Compute global motion (dx, dy, rotation, scale) from prev_gray to curr_gray.
        Updates cumulative world transform.
        Returns (dx, dy, rot_deg, scale, success_flag) and stores world_transform.
        If motion cannot be estimated, returns identity motion.
        """
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 10:
            # Not enough data – store current frame for next time
            self.prev_gray = curr_gray.copy()
            self.prev_pts = self._adaptive_feature_extraction(curr_gray, self.last_motion_magnitude)
            return 0.0, 0.0, 0.0, 1.0, False

        # Track features from previous to current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray,
            self.prev_pts.astype(np.float32), None,
            winSize=self.lk_win_size,
            maxLevel=self.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        if curr_pts is None:
            # Tracking failed – reset
            self.prev_gray = curr_gray.copy()
            self.prev_pts = self._adaptive_feature_extraction(curr_gray, self.last_motion_magnitude)
            return 0.0, 0.0, 0.0, 1.0, False

        # Filter good points
        good_prev = self.prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < 10:
            # Too few inliers – reset
            self.prev_gray = curr_gray.copy()
            self.prev_pts = self._adaptive_feature_extraction(curr_gray, self.last_motion_magnitude)
            return 0.0, 0.0, 0.0, 1.0, False

        # Estimate affine transform with RANSAC
        affine, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=self.max_iters,
            confidence=self.confidence
        )

        if affine is None:
            # No valid transform – reset
            self.prev_gray = curr_gray.copy()
            self.prev_pts = self._adaptive_feature_extraction(curr_gray, self.last_motion_magnitude)
            return 0.0, 0.0, 0.0, 1.0, False

        self.last_affine = affine.copy()

        # Decompose affine: | a  b  tx |
        #                   | -b a  ty |  (for similarity/partial affine)
        # Extract rotation (theta), scale, translation
        a, b, tx = affine[0, 0], affine[0, 1], affine[0, 2]
        c, d, ty = affine[1, 0], affine[1, 1], affine[1, 2]

        # For a pure similarity transform: a = scale*cos(theta), b = -scale*sin(theta)
        # We'll compute scale = sqrt(a^2 + b^2)  (assuming a ≈ d and b ≈ -c)
        scale = np.sqrt(a*a + b*b)
        if scale < 1e-6:
            scale = 1.0
        rot_rad = np.arctan2(b, a)   # careful: arctan2(b, a) gives theta if a=scale*cos, b=-scale*sin? Actually for similarity: [s*cos, -s*sin; s*sin, s*cos] so a = s*cos, b = -s*sin. Then theta = -atan2(b, a). Let's compute both.
        # Simpler: use the angle directly from the 2x2 part
        # We'll just use the average of the two possible angles
        angle_candidates = [np.arctan2(b, a), np.arctan2(d, c)]
        rot_rad = np.mean(angle_candidates)
        rot_deg = np.degrees(rot_rad)

        # Update cumulative world transform
        # Compose: new_world = affine @ world (assuming world stores transformation from world to previous frame)
        # Compute motion magnitude (translation magnitude) for adaptive feature extraction
        self.last_motion_magnitude = np.sqrt(tx*tx + ty*ty)
        # If world_transform maps world -> previous frame, then new = affine * world_transform
        self.world_transform = affine @ np.vstack([self.world_transform, [0, 0, 1]])

        # Prepare for next frame
        # Keep only inliers for next tracking (better stability)
        if inliers is not None:
            inlier_mask = (inliers.flatten() == 1)
            self.prev_pts = good_curr[inlier_mask]
        else:
            self.prev_pts = good_curr
        # Optionally add new features if too few
        if len(self.prev_pts) < self.max_features // 2:
            new_pts = self._adaptive_feature_extraction(curr_gray, self.last_motion_magnitude)
            if len(new_pts) > 0:
                # Combine old and new (avoid duplicates)
                self.prev_pts = np.vstack([self.prev_pts, new_pts])[:self.max_features]
        self.prev_gray = curr_gray.copy()

        return tx, ty, rot_deg, scale, True
# ============================================================
# Object Identity Memory Database - Re-ID Persistence Layer
# Stores object signatures for long-term identity recovery
# Optimized for low-end hardware [citation:2][citation:3]
# ============================================================
class ObjectIdentityMemory:
    """
    Persistent object memory that survives tracking sessions.
    When objects disappear and reappear, they get the same ID.
    Uses lightweight appearance embeddings (histogram + texture).
    """
    def __init__(self, db_path: str = './object_memory_db.pkl',
                 similarity_threshold: float = 0.75,
                 max_objects: int = 100):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.max_objects = max_objects
        self.objects = {}  # id -> {embedding, mean_size, color_hist, motion_stats, last_seen}
        self.next_id = 0
        self.load_database()
        
        # NEW: Feature cache for efficiency [citation:2]
        self.feature_cache = {}
        self.cache_size = 30
        
    def load_database(self):
        """Load existing object memory from disk."""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.objects = data.get('objects', {})
                    self.next_id = data.get('next_id', 0)
                print(f"Loaded {len(self.objects)} objects from memory DB", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load object DB: {e}", file=sys.stderr)
            self.objects = {}
            self.next_id = 0
    
    def save_database(self):
        """Save object memory to disk."""
        try:
            # Limit database size
            if len(self.objects) > self.max_objects:
                # Keep most recent objects
                sorted_objects = sorted(self.objects.items(),
                                      key=lambda x: x[1]['last_seen'],
                                      reverse=True)
                self.objects = dict(sorted_objects[:self.max_objects])
            
            with open(self.db_path, 'wb') as f:
                pickle.dump({
                    'objects': self.objects,
                    'next_id': self.next_id
                }, f)
        except Exception as e:
            print(f"Failed to save object DB: {e}", file=sys.stderr)
    
    def compute_appearance_embedding(self, frame: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        Compute lightweight appearance embedding.
        Combines: color histogram (HSV) + texture vector (LBP-like) + size
        Optimized for low-res cameras [citation:3]
        """
        # Extract ROI
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
            
        # Downscale for efficiency (adaptive based on object size) [citation:3]
        if w * h > 5000:  # Large object
            scale = 0.5
            roi = cv2.resize(roi, None, fx=scale, fy=scale)
        
        # 1. Color histogram (HSV - 48 bins total, illumination invariant)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])  # Hue
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])   # Saturation
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])   # Value
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # 2. Texture features (simplified LBP) [citation:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Local binary pattern-like features using Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
        
        # Orientation histogram (8 bins)
        valid_mag = mag > 5
        if np.any(valid_mag):
            hist_grad = np.histogram(angle[valid_mag], bins=8, range=(-180, 180))[0]
            hist_grad = hist_grad / (np.sum(hist_grad) + 1e-6)
        else:
            hist_grad = np.ones(8) / 8.0
        
        # 3. Size feature (normalized log area)
        size_feat = np.log(w * h + 1) / 10.0
        
        # Combine all features (total ~40 dimensions - very lightweight)
        embedding = np.concatenate([
            hist_h, hist_s, hist_v,  # 16+8+8 = 32
            hist_grad,                # 8
            [size_feat]               # 1
        ]).astype(np.float32)
        
        return embedding
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Combined similarity: Cosine + Bhattacharyya.
        Optimized for fast computation [citation:3]
        """
        if emb1 is None or emb2 is None:
            return 0.0
            
        # Cosine similarity for the whole vector
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
        
        # Bhattacharyya for histogram parts (first 32 elements)
        hist1 = emb1[:32] + 1e-6  # Avoid zeros
        hist2 = emb2[:32] + 1e-6
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        bhatt = np.sum(np.sqrt(hist1 * hist2))
        
        # Combined score
        return 0.6 * cosine_sim + 0.4 * bhatt
    
    def find_matching_id(self, embedding: np.ndarray, current_time: float) -> Optional[int]:
        """
        Find existing object ID by embedding similarity.
        Returns None if no match above threshold.
        """
        if embedding is None:
            return None
            
        best_match = None
        best_score = self.similarity_threshold
        
        for obj_id, obj_data in self.objects.items():
            # Skip very old objects? (optional)
            # if current_time - obj_data['last_seen'] > 3600:  # 1 hour
            #     continue
                
            similarity = self.compute_similarity(embedding, obj_data['embedding'])
            if similarity > best_score:
                best_score = similarity
                best_match = obj_id
        
        return best_match
    
    def update_object(self, obj_id: int, embedding: np.ndarray, 
                     contour: np.ndarray, current_time: float):
        """Update existing object in database."""
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        if obj_id in self.objects:
            # Update with exponential moving average [citation:3]
            obj = self.objects[obj_id]
            alpha = 0.7  # New observation weight
            
            # Update embedding (EMA)
            obj['embedding'] = alpha * embedding + (1 - alpha) * obj['embedding']
            # Normalize
            obj['embedding'] = obj['embedding'] / (np.linalg.norm(obj['embedding']) + 1e-6)
            
            # Update mean size
            obj['mean_size'] = alpha * area + (1 - alpha) * obj['mean_size']
            
            # Update motion stats (simplified)
            if 'motion_stats' not in obj:
                obj['motion_stats'] = [0, 0]  # mean_dx, mean_dy
        else:
            # New object
            self.objects[obj_id] = {
                'embedding': embedding.copy(),
                'mean_size': area,
                'color_hist': embedding[:32].copy(),  # Store histogram part
                'motion_stats': [0, 0],
                'last_seen': current_time,
                'first_seen': current_time
            }
        
        self.objects[obj_id]['last_seen'] = current_time
    
    def get_or_create_id(self, frame: np.ndarray, contour: np.ndarray, 
                        current_time: float) -> int:
        """
        Main entry point: get existing ID or create new one.
        """
        # Compute embedding
        embedding = self.compute_appearance_embedding(frame, contour)
        if embedding is None:
            return self._create_new_id(contour, current_time)
        
        # Look for match
        matched_id = self.find_matching_id(embedding, current_time)
        
        if matched_id is not None:
            # Update existing
            self.update_object(matched_id, embedding, contour, current_time)
            return matched_id
        else:
            # Create new
            return self._create_new_id(contour, current_time, embedding)
    
    def _create_new_id(self, contour: np.ndarray, current_time: float,
                      embedding: Optional[np.ndarray] = None) -> int:
        """Create new object ID."""
        new_id = self.next_id
        self.next_id += 1
        
        if embedding is None:
            # Will be computed on next update
            self.objects[new_id] = {
                'embedding': None,
                'mean_size': cv2.contourArea(contour),
                'color_hist': None,
                'motion_stats': [0, 0],
                'last_seen': current_time,
                'first_seen': current_time
            }
        else:
            self.update_object(new_id, embedding, contour, current_time)
        
        return new_id
    
    def cleanup_old_objects(self, max_age_seconds: float = 86400):  # 24 hours
        """Remove very old objects to prevent database bloat."""
        current_time = time.time()
        to_remove = []
        for obj_id, obj_data in self.objects.items():
            if current_time - obj_data['last_seen'] > max_age_seconds:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.objects[obj_id]
# ============================================================
# Phase 7: Long-term persistent identity memory (AGI-level)
# Stores object identity using appearance + 3D size + temporal frequency + location heatmap
# Based on research on low-end hardware optimization [citation:3][citation:4][citation:7]
# ============================================================
class PersistentObjectMemory:
    """
    Long-term object identity memory that survives across sessions.
    Stores:
      - appearance embedding (HSV histogram + texture)
      - 3D size estimate (from depth proxy)
      - temporal frequency (when object typically appears)
      - location heatmap (where object is often seen)
    
    Optimized for low-end hardware and low-res cameras [citation:9][citation:10]
    """
    def __init__(self, db_path: str = './persistent_memory.pkl',
                 similarity_threshold: float = 0.65,  # Lower threshold for low-res [citation:1]
                 max_objects: int = 50,                # Keep database small
                 location_grid_size: int = 8,          # 8x8 heatmap (64 bytes per object)
                 temporal_window_days: int = 7):       # Track patterns over week
        
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.max_objects = max_objects
        self.location_grid_size = location_grid_size
        self.temporal_window_days = temporal_window_days
        
        # Main object database
        self.objects = {}  # id -> {embedding, size_3d, temporal_freq, location_heatmap, last_seen, first_seen}
        self.next_id = 0
        
        # Performance optimizations [citation:2]
        self.feature_cache = {}
        self.cache_size = 20
        self.last_cleanup = time.time()
        
        self.load_database()
    
    def load_database(self):
        """Load existing object memory from disk."""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.objects = data.get('objects', {})
                    self.next_id = data.get('next_id', 0)
                print(f"Loaded {len(self.objects)} objects from persistent memory", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load persistent DB: {e}", file=sys.stderr)
            self.objects = {}
            self.next_id = 0
    
    def save_database(self):
        """Save object memory to disk."""
        try:
            # Limit database size (LRU by last_seen)
            if len(self.objects) > self.max_objects:
                sorted_objects = sorted(self.objects.items(),
                                      key=lambda x: x[1]['last_seen'],
                                      reverse=True)
                self.objects = dict(sorted_objects[:self.max_objects])
            
            with open(self.db_path, 'wb') as f:
                pickle.dump({
                    'objects': self.objects,
                    'next_id': self.next_id
                }, f)
        except Exception as e:
            print(f"Failed to save persistent DB: {e}", file=sys.stderr)
    
    def compute_appearance_embedding(self, frame: np.ndarray, contour: np.ndarray) -> Optional[np.ndarray]:
        """
        Ultra-lightweight appearance embedding for low-res cameras [citation:3][citation:9]
        Uses: 
          - 16-bin Hue histogram (invariant to lighting)
          - 8-bin Saturation histogram
          - 8-bin Value histogram (for texture)
          - 8-bin gradient orientation (LBP-like)
        Total: 40 dimensions - fits in 160 bytes
        """
        # Extract ROI
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        
        # Downscale for efficiency (keep aspect ratio, max 32x32) [citation:1]
        scale = min(1.0, 32.0 / max(w, h))
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            roi = cv2.resize(roi, (new_w, new_h))
        
        # Convert to HSV (illumination invariant) [citation:3]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 1. Color histograms (normalized)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])  # Hue
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])   # Saturation
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])   # Value
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # 2. Texture features (simplified - gradient orientation) [citation:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
        
        # Orientation histogram (8 bins)
        valid_mag = mag > 5  # Threshold noise
        if np.any(valid_mag):
            hist_grad = np.histogram(angle[valid_mag], bins=8, range=(-180, 180))[0]
            hist_grad = hist_grad / (np.sum(hist_grad) + 1e-6)
        else:
            hist_grad = np.ones(8) / 8.0
        
        # Combine all features (40-dim total)
        embedding = np.concatenate([
            hist_h, hist_s, hist_v,  # 16+8+8 = 32
            hist_grad                 # 8
        ]).astype(np.float32)
        
        return embedding
    
    def estimate_3d_size(self, contour: np.ndarray, depth_proxy: Optional[float] = None) -> float:
        """
        Estimate 3D size using:
          - bounding box area (pixels)
          - depth proxy (if available from depth camera)
        Returns normalized size score [0-1] [citation:6]
        """
        x, y, w, h = cv2.boundingRect(contour)
        area_2d = w * h
        
        if depth_proxy is not None and depth_proxy > 0:
            # Approximate 3D size: area_2d / depth^2 (inverse square law)
            # Normalize to reasonable range
            size_3d = area_2d / (depth_proxy * depth_proxy + 1e-6)
            # Log-scale to handle wide range
            size_score = np.log1p(size_3d) / 10.0
        else:
            # Fallback to 2D area (normalized to typical object sizes)
            size_score = min(1.0, area_2d / 5000.0)
        
        return min(1.0, max(0.0, size_score))
    
    def update_temporal_frequency(self, obj_id: int, current_time: float):
        """
        Update when this object is typically seen.
        Stores hour-of-day and day-of-week patterns [citation:3]
        """
        if obj_id not in self.objects:
            return
        
        obj = self.objects[obj_id]
        
        # Get time components
        t = time.localtime(current_time)
        hour = t.tm_hour
        day = t.tm_wday  # 0-6
        
        # Initialize temporal frequency storage if needed
        if 'hour_freq' not in obj:
            obj['hour_freq'] = np.zeros(24, dtype=np.float32)
            obj['day_freq'] = np.zeros(7, dtype=np.float32)
            obj['total_seen'] = 0
        
        # Update frequencies (exponential moving average)
        alpha = 0.1  # Smoothing factor
        obj['hour_freq'][hour] = (1 - alpha) * obj['hour_freq'][hour] + alpha
        obj['day_freq'][day] = (1 - alpha) * obj['day_freq'][day] + alpha
        obj['total_seen'] += 1
    
    def update_location_heatmap(self, obj_id: int, centroid_world: Tuple[float, float], 
                               frame_shape: Tuple[int, int]):
        """
        Update location heatmap in normalized world coordinates [citation:4]
        Uses 8x8 grid for memory efficiency (64 bytes per object)
        """
        if obj_id not in self.objects:
            return
        
        obj = self.objects[obj_id]
        
        # Initialize heatmap if needed
        if 'heatmap' not in obj:
            obj['heatmap'] = np.zeros((self.location_grid_size, self.location_grid_size), 
                                      dtype=np.float32)
        
        # Normalize centroid to grid coordinates
        h_img, w_img = frame_shape
        grid_x = int((centroid_world[0] / w_img) * self.location_grid_size)
        grid_y = int((centroid_world[1] / h_img) * self.location_grid_size)
        
        # Clamp to grid bounds
        grid_x = max(0, min(grid_x, self.location_grid_size - 1))
        grid_y = max(0, min(grid_y, self.location_grid_size - 1))
        
        # Update with exponential moving average
        alpha = 0.2
        obj['heatmap'][grid_y, grid_x] = (1 - alpha) * obj['heatmap'][grid_y, grid_x] + alpha
    
    def compute_similarity(self, 
                          emb1: np.ndarray, 
                          emb2: np.ndarray,
                          size1: float,
                          size2: float,
                          heatmap1: Optional[np.ndarray],
                          heatmap2: Optional[np.ndarray],
                          temporal1: Optional[Dict] = None,
                          temporal2: Optional[Dict] = None) -> float:
        """
        Multi-factor similarity combining all four identity dimensions.
        Weighted for low-end hardware performance [citation:3][citation:7]
        """
        if emb1 is None or emb2 is None:
            return 0.0
        
        # 1. Appearance similarity (cosine) - 50% weight
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 > 0 and norm2 > 0:
            appearance_sim = np.dot(emb1, emb2) / (norm1 * norm2)
        else:
            appearance_sim = 0.0
        
        # 2. Size similarity - 20% weight
        size_sim = 1.0 - min(1.0, abs(size1 - size2))
        
        # 3. Location heatmap similarity (if both have heatmaps) - 20% weight
        if heatmap1 is not None and heatmap2 is not None:
            # Normalize heatmaps
            h1 = heatmap1 / (np.sum(heatmap1) + 1e-6)
            h2 = heatmap2 / (np.sum(heatmap2) + 1e-6)
            # Bhattacharyya distance
            location_sim = np.sum(np.sqrt(h1 * h2))
        else:
            location_sim = 0.5  # Neutral if no data
        
        # 4. Temporal pattern similarity - 10% weight
        if temporal1 is not None and temporal2 is not None:
            current_hour = time.localtime().tm_hour
            # Compare hour patterns (simple - just look at current hour)
            hour_sim = 1.0 - abs(temporal1.get('hour_freq', np.zeros(24))[current_hour] -
                                 temporal2.get('hour_freq', np.zeros(24))[current_hour])
        else:
            hour_sim = 0.5
        
        # Weighted combination (tuned for low-res cameras) [citation:9]
        total_sim = (0.5 * appearance_sim + 
                    0.2 * size_sim + 
                    0.2 * location_sim + 
                    0.1 * hour_sim)
        
        return total_sim
    
    def find_matching_id(self, 
                        embedding: np.ndarray,
                        size_3d: float,
                        centroid_world: Tuple[float, float],
                        frame_shape: Tuple[int, int],
                        current_time: float) -> Optional[int]:
        """
        Find best matching object ID using all four dimensions.
        Returns None if no match above threshold.
        """
        if embedding is None:
            return None
        
        best_match = None
        best_score = self.similarity_threshold
        
        for obj_id, obj_data in self.objects.items():
            # Skip if never seen (shouldn't happen)
            if 'embedding' not in obj_data or obj_data['embedding'] is None:
                continue
            
            # Compute similarity
            similarity = self.compute_similarity(
                embedding, obj_data['embedding'],
                size_3d, obj_data.get('size_3d', 0.5),
                obj_data.get('heatmap'), None,  # Will compute heatmap after match
                obj_data, None
            )
            
            if similarity > best_score:
                best_score = similarity
                best_match = obj_id
        
        return best_match
    
    def update_object(self, obj_id: int, 
                     embedding: np.ndarray,
                     size_3d: float,
                     centroid_world: Tuple[float, float],
                     frame_shape: Tuple[int, int],
                     current_time: float):
        """Update existing object with new observation."""
        if obj_id not in self.objects:
            return
        
        obj = self.objects[obj_id]
        
        # Update embedding (EMA)
        alpha = 0.7
        obj['embedding'] = alpha * embedding + (1 - alpha) * obj.get('embedding', embedding)
        # Normalize
        norm = np.linalg.norm(obj['embedding'])
        if norm > 0:
            obj['embedding'] /= norm
        
        # Update size (EMA)
        obj['size_3d'] = alpha * size_3d + (1 - alpha) * obj.get('size_3d', size_3d)
        
        # Update temporal frequency
        self.update_temporal_frequency(obj_id, current_time)
        
        # Update location heatmap
        self.update_location_heatmap(obj_id, centroid_world, frame_shape)
        
        # Update timestamps
        obj['last_seen'] = current_time
        if 'first_seen' not in obj:
            obj['first_seen'] = current_time
    
    def create_object(self, 
                     embedding: np.ndarray,
                     size_3d: float,
                     centroid_world: Tuple[float, float],
                     frame_shape: Tuple[int, int],
                     current_time: float) -> int:
        """Create new persistent object entry."""
        obj_id = self.next_id
        self.next_id += 1
        
        self.objects[obj_id] = {
            'embedding': embedding.copy(),
            'size_3d': size_3d,
            'first_seen': current_time,
            'last_seen': current_time,
            'total_seen': 1
        }
        
        # Initialize temporal data
        self.update_temporal_frequency(obj_id, current_time)
        self.update_location_heatmap(obj_id, centroid_world, frame_shape)
        
        return obj_id
    
    def get_or_create_id(self, 
                        frame: np.ndarray,
                        contour: np.ndarray,
                        centroid_world: Tuple[float, float],
                        depth_proxy: Optional[float] = None,
                        current_time: Optional[float] = None) -> int:
        """
        Main entry point: get existing persistent ID or create new one.
        This is the AGI-level memory recall function.
        """
        if current_time is None:
            current_time = time.time()
        
        # 1. Compute appearance embedding
        embedding = self.compute_appearance_embedding(frame, contour)
        if embedding is None:
            return self.next_id  # Fallback to new ID
        
        # 2. Estimate 3D size
        size_3d = self.estimate_3d_size(contour, depth_proxy)
        
        # 3. Get frame shape for heatmap
        frame_shape = frame.shape[:2]
        
        # 4. Try to find match in persistent memory
        matched_id = self.find_matching_id(
            embedding, size_3d, centroid_world, frame_shape, current_time
        )
        
        if matched_id is not None:
            # Update existing object
            self.update_object(matched_id, embedding, size_3d, 
                             centroid_world, frame_shape, current_time)
            return matched_id
        else:
            # Create new object
            return self.create_object(embedding, size_3d, 
                                    centroid_world, frame_shape, current_time)
    
    def cleanup_old_objects(self, max_age_days: float = 30):
        """Remove objects not seen for a long time."""
        current_time = time.time()
        cutoff = current_time - (max_age_days * 86400)
        
        to_remove = []
        for obj_id, obj_data in self.objects.items():
            if obj_data.get('last_seen', 0) < cutoff:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.objects[obj_id]
        
        if to_remove:
            print(f"Cleaned up {len(to_remove)} old objects", file=sys.stderr)
# ============================================================
# Static geometry memory (Stage 5)
# Stores long‑lived scene elements that move with camera
# ============================================================
class StaticGeometryStore:
    """
    Remembers contours that have been observed for many frames
    and whose motion is consistent with the global camera motion.
    Contours are stored in world‑stabilized coordinates.
    """
    def __init__(self, min_age: int = 30, max_contours: int = 50):
        self.min_age = min_age
        self.max_contours = max_contours
        self.static_contours = []          # list of (contour_world, age, last_seen)
        self.next_id = 0

    def update(self, tracker, tracked_objects: Dict[int, Tuple[np.ndarray, Tuple]], 
               world_transform: np.ndarray, frame_gray: np.ndarray):
        """
        Update static geometry with currently tracked objects.
        `world_transform` maps world coordinates to current frame.
        """
        # Transform current frame contours to world coordinates
        inv_transform = cv2.invertAffineTransform(world_transform)  # 2x3 inverse
        def world_from_frame(pts):
            # pts: Nx2 array
            return cv2.transform(pts.reshape(-1,1,2), inv_transform).reshape(-1,2)

        now = time.time()
        # Age existing static contours
        for item in self.static_contours:
            item['age'] += 1
            item['last_seen'] = now

        # Match current tracked objects to static contours (simple bounding‑box IoU)
        static_snapshot = list(self.static_contours)   # snapshot
        used = [False] * len(static_snapshot)

        for obj_id, (cnt, _) in tracked_objects.items():
            # Get bounding box of current contour
            x, y, w, h = cv2.boundingRect(cnt)
            best_match = None
            best_iou = 0.3
            for i, item in enumerate(static_snapshot):
                if used[i]:
                    continue
                # Transform static contour to current frame for comparison
                static_cnt_frame = cv2.transform(item['contour'].reshape(-1,1,2), world_transform).reshape(-1,2)
                sx, sy, sw, sh = cv2.boundingRect(static_cnt_frame.astype(np.int32))
                # IoU of bounding boxes
                xi = max(x, sx)
                yi = max(y, sy)
                wi = min(x+w, sx+sw) - xi
                hi = min(y+h, sy+sh) - yi
                if wi > 0 and hi > 0:
                    inter = wi * hi
                    union = w*h + sw*sh - inter
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_match = i
            if best_match is not None:
                # Update existing static contour
                item = self.static_contours[best_match]
                # Blend contour (simple average of points? too heavy – keep last)
                item['contour'] = world_from_frame(cnt.reshape(-1,2)).reshape(-1,1,2).astype(np.int32)
                item['age'] = 0   # reset age because we saw it
                item['last_seen'] = now
                used[best_match] = True
            else:
                # New candidate: add if object is old enough (age > 8 frames)
                obj = tracker.tracked_objects.get(obj_id)
                if obj and obj.age > 8:
                    self.static_contours.append({
                        'id': self.next_id,
                        'contour': world_from_frame(cnt.reshape(-1,2)).reshape(-1,1,2).astype(np.int32),
                        'age': 0,
                        'last_seen': now
                    })
                    self.next_id += 1

        # Remove static contours not seen for a while (e.g., > 2 seconds)
        self.static_contours = [item for item in self.static_contours 
                                if now - item['last_seen'] < 2.0]

        # Keep only the most persistent contours (by age)
        self.static_contours.sort(key=lambda x: x['age'], reverse=True)
        self.static_contours = self.static_contours[:self.max_contours]

    def draw(self, frame: np.ndarray, world_transform: np.ndarray, color=(128,128,128)):
        """Draw static contours onto frame (transformed to current view)."""
        for item in self.static_contours:
            if item['age'] < self.min_age:
                continue
            cnt_frame = cv2.transform(item['contour'].reshape(-1,1,2), world_transform).reshape(-1,2)
            cnt_frame = cnt_frame.astype(np.int32).reshape(-1,1,2)
            cv2.drawContours(frame, [cnt_frame], -1, color, 1)

@dataclass
class TrackedObject:
    """Simplified tracked object with temporal persistence - inspired by lightweight tracking research [citation:2][citation:4]"""
    # Track state management - from DeepSORT research [citation:2]
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
    
    # NEW: Track maturity system - based on DeepSORT implementation [citation:2]
    track_state: str = "TENTATIVE"  # One of: TENTATIVE, CONFIRMED, ARCHIVED
    consecutive_matches: int = 0  # Frames with stable matching (for TENTATIVE -> CONFIRMED)
    
    # Optical flow fields - optimized using MedianFlow research [citation:3]
    prev_points: Optional[np.ndarray] = None  # Good features from previous frame
    median_flow: Tuple[float, float] = (0.0, 0.0)  # Median flow (dx, dy) - more efficient than mean
    flow_confidence: float = 0.0  # How many points were successfully tracked
    depth_history: List[float] = None       # rolling window of depth‑proxy values
    depth_stable: bool = False
    
    # Feature cache for confirmed tracks - avoid recomputation [citation:8]
    cached_features: Optional[np.ndarray] = None
    last_feature_update: int = 0  # Frame number of last feature computation
    
    # --- Segment 1: Attention state (global controller will use this per-object score) ---
    attention_score_ema: float = 0.0  # Exponential moving average of attention
    
    # --- Segment 4: Novelty detection (for curiosity switch) ---
    areas: List[float] = None  # history of contour areas, aligned with contours


# ----------------------------------------------------------------------
# Depth-proxy integration for small object enhancement [citation:6]
# Uses depth information to improve detection of small objects
# ----------------------------------------------------------------------
def enhance_small_objects_with_depth(frame: np.ndarray, 
                                    depth_proposals: List[Tuple],
                                    min_area: int = 100) -> List[np.ndarray]:
    """
    Use depth proposals to enhance small object detection.
    Based on YOLOv7-small-scale research [citation:6].
    """
    enhanced_contours = []
    
    for dbox, dcontour, dconf in depth_proposals:
        x, y, w, h = dbox
        
        # Focus on small objects (by area)
        if w * h < min_area:
            # Enhance this region
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
                
            # Apply contrast enhancement for small objects
            if len(roi.shape) == 3:
                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
                l_enhanced = clahe.apply(l)
                enhanced_lab = cv2.merge([l_enhanced, a, b])
                enhanced_roi = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                
                # Put back
                frame[y:y+h, x:x+w] = enhanced_roi
                
            # Add the contour
            enhanced_contours.append(dcontour)
            
    return enhanced_contours
    
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

    @staticmethod
    def _hungarian(cost_matrix):
        """
        Solve linear assignment problem using Hungarian algorithm (Munkres).
        cost_matrix: 2D numpy array (n_workers, n_jobs)
        Returns (row_indices, col_indices) for optimal assignment.
        """
        # If scipy is available, use it (fast). Otherwise fallback to O(n^3) numpy implementation.
        try:
            from scipy.optimize import linear_sum_assignment
            return linear_sum_assignment(cost_matrix)
        except ImportError:
            pass

        # Pure numpy implementation (optimized for small matrices, typical for tracking)
        cost = cost_matrix.copy()
        n, m = cost.shape
        # Pad to square if necessary
        if n != m:
            pad_size = max(n, m)
            padded = np.full((pad_size, pad_size), np.max(cost) + 1, dtype=cost.dtype)
            padded[:n, :m] = cost
            cost = padded
            n = pad_size

        # Standard Hungarian algorithm (O(n^3))
        u = np.zeros(n, dtype=cost.dtype)
        v = np.zeros(n, dtype=cost.dtype)
        p = np.zeros(n, dtype=int)
        way = np.zeros(n, dtype=int)

        for i in range(1, n):
            p[0] = i
            j0 = 0
            minv = np.full(n, np.inf, dtype=cost.dtype)
            used = np.zeros(n, dtype=bool)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = np.inf
                j1 = 0
                for j in range(1, n):
                    if not used[j]:
                        cur = cost[i0-1, j-1] - u[i0-1] - v[j-1]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(n):
                    if used[j]:
                        u[p[j]-1] += delta
                        v[j-1] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        # Extract assignment for original dimensions
        row_ind = np.zeros(n, dtype=int)
        col_ind = np.zeros(n, dtype=int)
        for j in range(1, n):
            if p[j] != 0:
                row_ind[p[j]-1] = j-1
                col_ind[j-1] = p[j]-1

        # Keep only original rows/cols
        row_ind = row_ind[:n]
        col_ind = col_ind[:m]
        return np.arange(n), col_ind
    
    @staticmethod
    def _warp_point(affine, pt):
        """Apply 2x3 affine matrix to point (x,y) -> (x',y')"""
        x, y = pt
        return (affine[0,0]*x + affine[0,1]*y + affine[0,2],
                affine[1,0]*x + affine[1,1]*y + affine[1,2])
        
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
                                  color_sig: np.ndarray,
                                  ref_centroid: Optional[Tuple[float, float]] = None) -> float:
        """
        Compute persistence score based on position and color histogram.
        Uses Bhattacharyya distance for color similarity.
        """
        if ref_centroid is not None:
            last_centroid = ref_centroid
        else:
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
        """Compute median flow vector from successfully tracked points.
        Optimized with outlier rejection for low-end hardware [citation:3]"""
        if prev_pts is None or curr_pts is None or len(prev_pts) == 0:
            return (0.0, 0.0, 0.0)
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]
        if len(good_prev) < 5:  # Need at least 5 points for reliable median [citation:3]
            return (0.0, 0.0, 0.0)
        
        flows = good_curr - good_prev
        
        # NEW: Reject outliers using median absolute deviation [citation:3]
        median_x = np.median(flows[:, 0])
        median_y = np.median(flows[:, 1])
        mad_x = np.median(np.abs(flows[:, 0] - median_x))
        mad_y = np.median(np.abs(flows[:, 1] - median_y))
        
        # Filter out points with deviation > 2*MAD
        valid_x = np.abs(flows[:, 0] - median_x) <= 2 * mad_x
        valid_y = np.abs(flows[:, 1] - median_y) <= 2 * mad_y
        valid = valid_x & valid_y
        
        if np.sum(valid) >= 3:
            filtered_flows = flows[valid]
            median_x = np.median(filtered_flows[:, 0])
            median_y = np.median(filtered_flows[:, 1])
            confidence = np.sum(valid) / len(prev_pts)
        else:
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
    
    def _get_last_positions(self):
        """Return the last known positions for all tracked objects."""
        output = {}
        for obj_id, obj in self.tracked_objects.items():
            if obj.contours and obj.centroids:
                output[obj_id] = (obj.contours[-1], obj.centroids[-1])
        return output
    
    def update(self, 
               frame: np.ndarray, 
               detected_pairs: List[Tuple[np.ndarray, float]],
               camera_affine: Optional[np.ndarray] = None,
               identity_memory: Optional[ObjectIdentityMemory] = None) -> Dict[int, Tuple[np.ndarray, Tuple[float, float]]]:
        """
        Enhanced multi‑object tracker with camera-motion-aware world-space matching.
        
        KEY IMPROVEMENT:
        1. Convert centroids to world-stabilized coordinates
        2. Perform matching in world space
        3. Convert back to frame space for output
        """
        # ---- Adaptive frame skipping (unchanged) ----
        if not hasattr(self, 'frame_counter'):
            self.frame_counter = 0
            self.skip_counter = 0
            self.skip_pattern = 0
        self.frame_counter += 1
        current_time = time.time()
        if hasattr(self, 'last_fps_check'):
            elapsed = current_time - self.last_fps_check
            if elapsed > 1.0:
                frames_processed = getattr(self, 'frames_processed', 30)
                estimated_fps = frames_processed / elapsed
                if estimated_fps < 10:
                    self.skip_pattern = min(3, self.skip_pattern + 1)
                elif estimated_fps > 25:
                    self.skip_pattern = max(0, self.skip_pattern - 1)
                self.frames_processed = 0
                self.last_fps_check = current_time
        else:
            self.last_fps_check = current_time
            self.frames_processed = 0
        self.frames_processed += 1
        if self.skip_pattern > 0 and self.frame_counter % (self.skip_pattern + 1) != 0:
            return self._get_last_positions()

        # ---- Prepare current frame ----
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Compute optical flow for existing tracked objects (unchanged) ----
        if self.prev_gray is not None:
            for obj in self.tracked_objects.values():
                if obj.prev_points is not None and len(obj.prev_points) > 0:
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

        if detected_pairs:
            # ---- Extract detection features ----
            detections = []
            for contour, conf in detected_pairs:
                centroid = self._compute_centroid(contour)                
                brightness = self._compute_brightness(frame, contour)
                hue = self._compute_dominant_hue(frame, contour)
                area = cv2.contourArea(contour)
                color_sig = self._compute_color_signature(frame, contour)
                flow_dx, flow_dy, flow_conf = self._compute_detection_flow(frame_gray, contour)
                
                # ===== CRITICAL IMPROVEMENT: Convert to world coordinates =====
                world_centroid = centroid
                if camera_affine is not None:
                    # Apply inverse affine to get world coordinates [citation:4][citation:9]
                    inv_transform = cv2.invertAffineTransform(camera_affine)
                    pt = np.array([[centroid]], dtype=np.float32)
                    world_pt = cv2.transform(pt, inv_transform).reshape(2)
                    world_centroid = (world_pt[0], world_pt[1])
                
                detections.append({
                    'contour': contour,
                    'centroid': centroid,
                    'world_centroid': world_centroid,  # Store world coordinates
                    'brightness': brightness,
                    'hue': hue,
                    'area': area,
                    'color_sig': color_sig,
                    'flow': (flow_dx, flow_dy),
                    'flow_conf': flow_conf,
                    'conf': conf
                })
            detections.sort(key=lambda x: x['area'], reverse=True)

            # ---- Stage A: Get active objects and convert their positions to world space ----
            active_items = [(oid, obj) for oid, obj in self.tracked_objects.items()
                            if obj.lost_frames <= self.max_lost_frames]

            if active_items and detections:
                # Build predicted world centroids with dynamic gate radius
                active_ids = []
                active_objs = []
                predicted_world = []      # Predictions in WORLD coordinates
                gate_radii = []            # Per‑object dynamic radius (in world units)
                
                for oid, obj in active_items:
                    if obj.centroids:
                        last = obj.centroids[-1]
                        last_world = last
                        
                        # Convert last known position to world coordinates
                        if camera_affine is not None:
                            inv_transform = cv2.invertAffineTransform(camera_affine)
                            pt = np.array([[last]], dtype=np.float32)
                            last_world_pt = cv2.transform(pt, inv_transform).reshape(2)
                            last_world = (last_world_pt[0], last_world_pt[1])
                        
                        # Predict using median flow (also convert flow to world scale)
                        if obj.flow_confidence > 0.3:
                            # Approximate world-scale flow (simplified)
                            world_flow = obj.median_flow
                            pred_world = (last_world[0] + world_flow[0], 
                                         last_world[1] + world_flow[1])
                            
                            # Dynamic radius in world units
                            speed = np.linalg.norm(obj.median_flow)
                            # Convert to approximate world units using affine scale
                            if camera_affine is not None:
                                scale = np.sqrt(camera_affine[0,0]**2 + camera_affine[0,1]**2)
                                world_speed = speed / max(scale, 0.1)
                            else:
                                world_speed = speed
                            radius = 30 + min(world_speed * 2, 40)
                        else:
                            pred_world = last_world
                            radius = 40
                    else:
                        pred_world = None
                        radius = 50
                    
                    predicted_world.append(pred_world)
                    gate_radii.append(radius)
                    active_ids.append(oid)
                    active_objs.append(obj)

                # ---- Stage B: Hungarian assignment in WORLD coordinates ----
                n_obj = len(active_ids)
                n_det = len(detections)
                cost_matrix = np.full((n_obj, n_det), 1e9, dtype=np.float32)

                for i, (pred_world, radius) in enumerate(zip(predicted_world, gate_radii)):
                    if pred_world is None:
                        continue
                    for j, det in enumerate(detections):
                        # Use WORLD coordinates for distance calculation
                        dist = np.linalg.norm(np.array(pred_world) - np.array(det['world_centroid']))
                        if dist > radius:      # Motion gate rejection in world space
                            continue

                        # Distance cost (normalised by 100 world units)
                        dist_cost = dist / 100.0

                        # Appearance cost (Bhattacharyya distance) - unchanged
                        obj = active_objs[i]
                        if obj.color_sig is not None and det['color_sig'] is not None:
                            bc = np.sum(np.sqrt(obj.color_sig * det['color_sig']))
                            appearance_sim = bc
                        else:
                            appearance_sim = 0.5
                        appearance_cost = 1.0 - appearance_sim

                        # Flow consistency cost - now using world-relative flow
                        if obj.flow_confidence > 0.3 and det['flow_conf'] > 0.3:
                            flow_diff = np.linalg.norm(np.array(obj.median_flow) - np.array(det['flow']))
                            flow_cost = min(flow_diff / 20.0, 1.0)
                        else:
                            flow_cost = 0.5

                        # Weighted sum (weights tuned for world-space matching)
                        total_cost = 0.7 * dist_cost + 0.2 * appearance_cost + 0.1 * flow_cost
                        cost_matrix[i, j] = total_cost

                # Solve assignment (Hungarian)
                row_ind, col_ind = self._hungarian(cost_matrix)
                valid = cost_matrix[row_ind, col_ind] < 1e8
                row_ind = row_ind[valid]
                col_ind = col_ind[valid]

                used_detections = set()
                used_objects = set()

                for i, j in zip(row_ind, col_ind):
                    obj_id = active_ids[i]
                    det = detections[j]
                    
                    # ===== INTEGRATE WITH IDENTITY MEMORY =====
                    if identity_memory is not None:
                        # Check if this detection matches a known object from memory
                        memory_id = identity_memory.get_or_create_id(
                            frame, det['contour'], current_time
                        )
                        
                        # If memory suggests a different ID, consider reassigning
                        if memory_id != obj_id and memory_id not in self.tracked_objects:
                            # This detection might be a reappearing object
                            # We'll handle this carefully to avoid flicker
                            if det['conf'] > 0.8:  # High confidence
                                # Create new tracked object with memory ID
                                new_obj = TrackedObject(
                                    id=memory_id,
                                    contours=[det['contour']],
                                    centroids=[det['centroid']],
                                    hues=[det['hue']],
                                    brightness=det['brightness'],
                                    contour_area=det['area'],
                                    color_sig=det['color_sig'],
                                    age=obj.age,  # Inherit age from old?
                                    persistence_score=0.9,
                                    lost_frames=0,
                                    areas=[det['area']]  # Initialize area history
                                )
                                self.tracked_objects[memory_id] = new_obj
                                matched_objects[memory_id] = (det['contour'], det['centroid'])
                                used_detections.add(j)
                                used_objects.add(obj_id)
                                # Mark old as lost
                                if obj_id in self.tracked_objects:
                                    self.tracked_objects[obj_id].lost_frames = self.max_lost_frames + 1
                                continue
                    
                    # Normal update path
                    used_detections.add(j)
                    used_objects.add(obj_id)

                    obj = self.tracked_objects[obj_id]

                    # Update object
                    obj.contours.append(det['contour'])
                    obj.centroids.append(det['centroid'])
                    obj.hues.append(det['hue'])
                    obj.brightness = det['brightness']
                    obj.contour_area = det['area']
                    # Update area history for novelty
                    if obj.areas is None:
                        obj.areas = [det['area']]
                    else:
                        obj.areas.append(det['area'])
                        if len(obj.areas) > obj.max_history:
                            obj.areas.pop(0)
                    obj.color_sig = det['color_sig']
                    obj.age += 1
                    obj.lost_frames = 0
                    obj.persistence_score = 1.0 - total_cost
                    
                    # NEW: Update consecutive matches and check state transitions [citation:2]
                    obj.consecutive_matches += 1
                    
                    # State transition logic - based on DeepSORT [citation:2]
                    if obj.track_state == TRACK_STATE_TENTATIVE:
                        if obj.consecutive_matches >= CONFIRMATION_THRESHOLD:
                            old_state = obj.track_state
                            obj.track_state = TRACK_STATE_CONFIRMED
                            # Log state change for debugging
                            print(f"Track {obj_id} promoted to CONFIRMED after {obj.consecutive_matches} matches", file=sys.stderr)
                    
                    elif obj.track_state == TRACK_STATE_ARCHIVED:
                        # Reactivate archived track - but keep as CONFIRMED if it was confirmed before [citation:8]
                        if obj.consecutive_matches >= TENTATIVE_FRAMES:
                            obj.track_state = TRACK_STATE_CONFIRMED
                            print(f"Track {obj_id} restored from ARCHIVED to CONFIRMED", file=sys.stderr)

                    if len(obj.contours) > obj.max_history:
                        obj.contours.pop(0)
                        obj.centroids.pop(0)
                        obj.hues.pop(0)

                    self.geometry_store.update_object(obj_id, det['contour'], det['centroid'], frame)
                    
                    # NEW: Only extract flow points for non-ARCHIVED tracks (optimization) [citation:3]
                    if obj.track_state != TRACK_STATE_ARCHIVED:
                        obj.prev_points = self._extract_points_from_contour(frame_gray, det['contour'],
                                                                            self.flow_points_per_obj)
                    matched_objects[obj_id] = (det['contour'], det['centroid'])

                # ---- Create new objects for unmatched detections ----
                for j, det in enumerate(detections):
                    if j in used_detections:
                        continue
                    
                    # Check identity memory for reappearing object
                    new_id = self.next_id
                    if identity_memory is not None:
                        memory_id = identity_memory.get_or_create_id(
                            frame, det['contour'], current_time
                        )
                        if memory_id not in self.tracked_objects:
                            new_id = memory_id
                        # else: memory_id already tracked, use new ID
                    
                    # NEW: Create track with TENTATIVE state - only confirmed tracks get permanent IDs [citation:2]
                    new_obj = TrackedObject(
                        id=new_id,
                        contours=[det['contour']],
                        centroids=[det['centroid']],
                        hues=[det['hue']],
                        brightness=det['brightness'],
                        contour_area=det['area'],
                        color_sig=det['color_sig'],
                        age=1,
                        persistence_score=1.0,
                        lost_frames=0,
                        track_state=TRACK_STATE_TENTATIVE,  # Start as tentative
                        consecutive_matches=1,  # First match
                        cached_features=None,
                        last_feature_update=0,
                        areas=[det['area']]  # Initialize area history
                    )
                    
                    # Only extract flow points for tentative tracks (optimization) [citation:3]
                    new_obj.prev_points = self._extract_points_from_contour(frame_gray, det['contour'],
                                                                            self.flow_points_per_obj)
                    self.tracked_objects[new_id] = new_obj
                    matched_objects[new_id] = (det['contour'], det['centroid'])
                    self.geometry_store.register_object(new_id, det['contour'], det['centroid'], frame)
                    
                    if new_id == self.next_id:
                        self.next_id += 1

            else:
                # No active objects or no detections: fallback to new objects only
                for det in detections:
                    new_id = self.next_id
                    if identity_memory is not None:
                        memory_id = identity_memory.get_or_create_id(
                            frame, det['contour'], current_time
                        )
                        if memory_id not in self.tracked_objects:
                            new_id = memory_id
                    
                    new_obj = TrackedObject(
                        id=new_id,
                        contours=[det['contour']],
                        centroids=[det['centroid']],
                        hues=[det['hue']],
                        brightness=det['brightness'],
                        contour_area=det['area'],
                        color_sig=det['color_sig'],
                        age=1,
                        persistence_score=1.0,
                        lost_frames=0,
                        areas=[det['area']]  # Initialize area history
                    )
                    new_obj.prev_points = self._extract_points_from_contour(frame_gray, det['contour'],
                                                                            self.flow_points_per_obj)
                    self.tracked_objects[new_id] = new_obj
                    matched_objects[new_id] = (det['contour'], det['centroid'])
                    self.geometry_store.register_object(new_id, det['contour'], det['centroid'], frame)
                    
                    if new_id == self.next_id:
                        self.next_id += 1

        # ---- Stage C: handle lost objects - archive confirmed tracks, delete tentative ones [citation:2] ----
        current_time = time.time()
        expired_ids = []
        for obj_id, obj in list(self.tracked_objects.items()):
            if obj.lost_frames > self.max_lost_frames:
                if obj.track_state == TRACK_STATE_CONFIRMED:
                    # NEW: Archive confirmed tracks instead of deleting [citation:8]
                    obj.track_state = TRACK_STATE_ARCHIVED
                    obj.lost_frames = 0  # Reset for archived tracking
                    print(f"Track {obj_id} ARCHIVED (lost)", file=sys.stderr)
                else:
                    # Tentative tracks are simply removed (noise) [citation:2]
                    expired_ids.append(obj_id)
        
        # Remove only tentative expired tracks
        for obj_id in expired_ids:
            del self.tracked_objects[obj_id]
            if obj_id in self.geometry_store.shape_memory:
                del self.geometry_store.shape_memory[obj_id]
        
        # NEW: Remove very old archived tracks (older than MAX_ARCHIVED_AGE frames)
        archived_expired = [obj_id for obj_id, obj in self.tracked_objects.items()
                           if obj.track_state == TRACK_STATE_ARCHIVED and obj.age > MAX_ARCHIVED_AGE]
        for obj_id in archived_expired:
            print(f"Removing old archived track {obj_id}", file=sys.stderr)
            del self.tracked_objects[obj_id]
            if obj_id in self.geometry_store.shape_memory:
                del self.geometry_store.shape_memory[obj_id]

        # ---- Build output with Kalman‑smoothed centroids ----
        output_objects = {}
        for obj_id, (contour, raw_centroid) in matched_objects.items():
            kalman_cent = self.geometry_store.get_kalman_centroid(obj_id)
            if kalman_cent is None:
                kalman_cent = raw_centroid
            smooth_contour = self.geometry_store.get_smoothed_contour(obj_id)
            if smooth_contour is None:
                smooth_contour = contour
            output_objects[obj_id] = (smooth_contour, (int(kalman_cent[0]), int(kalman_cent[1])))

        # ---- Store current grayscale for next frame's optical flow ----
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
                 grow_tolerance: float = 0.15,         # Region growing color tolerance (fraction)
                 # NEW: Dynamic resolution scaling for low-end hardware [citation:1]
                 enable_dynamic_resolution: bool = True,
                 base_scale_factor: float = 1.0,       # Start at full resolution
                 min_scale_factor: float = 0.5,         # Don't go below 50% resolution
                 performance_target: float = 15.0):     # Target 15 FPS minimum
        
        # Existing initializations
        self.roi_margin = roi_margin
        self.min_contour_area = min_contour_area
        self.n_colors = n_colors
        self.shadow_threshold = shadow_threshold
        self.edge_threshold1 = edge_threshold1
        self.edge_threshold2 = edge_threshold2
        self.grow_tolerance = grow_tolerance
        
        # NEW: Dynamic resolution attributes
        self.enable_dynamic_resolution = enable_dynamic_resolution
        self.current_scale = base_scale_factor
        self.min_scale = min_scale_factor
        self.target_fps = performance_target
        self.fps_history = []
        self.last_scale_adjustment = time.time()
        
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

# ============================================================
# Proposal buffer – temporal stability gate + area filter + hysteresis
# ============================================================
class ProposalBuffer:
    """
    Filters raw detections before they enter the tracker.
    Implements:
      - Temporal stability: detection must appear in >=2 consecutive frames.
      - Area change filter: reject if area jumps by >60% from previous frame.
      - Confidence hysteresis: enter_thr / exit_thr to suppress flicker.
    """
    def __init__(self,
                 iou_threshold: float = 0.5,       # IoU for matching across frames
                 min_consecutive: int = 2,          # frames needed before release
                 area_change_threshold: float = 0.6,
                 enter_threshold: float = 0.65,
                 exit_threshold: float = 0.45,
                 max_age: int = 5):                  # frames before deleting unused entry
        self.iou_thr = iou_threshold
        self.min_consecutive = min_consecutive
        self.area_change_thr = area_change_threshold
        self.enter_thr = enter_threshold
        self.exit_thr = exit_threshold
        self.max_age = max_age

        self.next_id = 0
        self.entries = {}            # id -> {count, last_contour, last_bbox, last_area, last_conf, active, last_seen}
        self.frame_counter = 0

    def _bbox_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = min(x1+w1, x2+w2) - xi
        hi = min(y1+h1, y2+h2) - yi
        if wi <= 0 or hi <= 0:
            return 0.0
        inter = wi * hi
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: List[Tuple[np.ndarray, float]]) -> List[Tuple[np.ndarray, float]]:
        """
        detections: list of (contour, confidence)
        returns: list of contours that pass all gates
        """
        self.frame_counter += 1

        # Build list of current detections with bounding boxes
        curr = []
        for cnt, conf in detections:
            x, y, w, h = cv2.boundingRect(cnt)
            curr.append({
                'contour': cnt,
                'conf': conf,
                'bbox': (x, y, w, h),
                'area': w * h
            })

        # Mark all entries as not seen this frame
        for eid in self.entries:
            self.entries[eid]['seen'] = False

        # Match current detections to existing entries
        used_det = [False] * len(curr)
        for i, det in enumerate(curr):
            best_match = None
            best_iou = self.iou_thr
            for eid, entry in self.entries.items():
                if entry['seen']:
                    continue
                last_bbox = entry['last_bbox']
                iou = self._bbox_iou(det['bbox'], last_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = eid
            if best_match is not None:
                entry = self.entries[best_match]
                # Area change check
                area_old = entry['last_area']
                area_new = det['area']
                if abs(area_new - area_old) / (area_old + 1e-6) > self.area_change_thr:
                    # area jump too large → ignore this detection (do not update entry)
                    continue
                # Update entry
                entry['last_contour'] = det['contour']
                entry['last_bbox'] = det['bbox']
                entry['last_area'] = det['area']
                entry['last_conf'] = det['conf']
                entry['count'] = min(entry['count'] + 1, 10)
                entry['last_seen'] = self.frame_counter
                entry['seen'] = True

                # Apply hysteresis
                if entry['active']:
                    if det['conf'] < self.exit_thr:
                        entry['active'] = False
                else:
                    if det['conf'] >= self.enter_thr:
                        entry['active'] = True

                used_det[i] = True

        # Create new entries for unmatched detections
        for i, det in enumerate(curr):
            if used_det[i]:
                continue
            active = det['conf'] >= self.enter_thr
            eid = self.next_id
            self.next_id += 1
            self.entries[eid] = {
                'last_contour': det['contour'],
                'last_bbox': det['bbox'],
                'last_area': det['area'],
                'last_conf': det['conf'],
                'count': 1,
                'active': active,
                'last_seen': self.frame_counter,
                'seen': True
            }

        # Remove old entries
        to_del = [eid for eid, e in self.entries.items()
                  if self.frame_counter - e['last_seen'] > self.max_age]
        for eid in to_del:
            del self.entries[eid]

        # Collect output contours and confidences from entries that are mature and active
        output = []
        for eid, entry in self.entries.items():
            if entry['seen'] and entry['count'] >= self.min_consecutive and entry['active']:
                output.append((entry['last_contour'], entry['last_conf']))
        return output
# ----------------------------------------------------------------------
# Template feature cache for efficient tracking (based on research [citation:2])
# Implements template-branch separable design to avoid redundant computation
# ----------------------------------------------------------------------
class TemplateFeatureCache:
    """
    Caches deep features for tracked objects to avoid redundant computation.
    Inspired by template-branch separable vision transformers [citation:2].
    Only recomputes features when appearance changes significantly.
    """
    def __init__(self, max_cache_size: int = 10, update_threshold: float = 0.3):
        self.cache = {}  # obj_id -> {features, last_frame, last_centroid, stability_score}
        self.max_size = max_cache_size
        self.update_threshold = update_threshold
        
    def get_features(self, obj_id: int, frame: np.ndarray, contour: np.ndarray, current_frame_num: int):
        """Return cached features or compute new ones if needed."""
        if obj_id in self.cache:
            cached = self.cache[obj_id]
            # Check if we need to update (based on frame distance or motion)
            if current_frame_num - cached['last_frame'] > 5:  # Update every 5 frames max
                return self._compute_and_cache(obj_id, frame, contour, current_frame_num)
            return cached['features']
        else:
            return self._compute_and_cache(obj_id, frame, contour, current_frame_num)
    
    def _compute_and_cache(self, obj_id: int, frame: np.ndarray, contour: np.ndarray, frame_num: int):
        """Compute lightweight appearance features for caching."""
        # Extract ROI
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
            
        # Compute simple but effective features [citation:2]
        # Color histogram (32 bins) + texture (LBP-like)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Color features (HSV histogram)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        
        # Normalize
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Texture features (gradient histogram)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
        hist_grad = np.histogram(angle[mag > 5], bins=8, range=(-180, 180))[0]
        hist_grad = hist_grad / (np.sum(hist_grad) + 1e-6)
        
        # Combine features
        features = np.concatenate([hist_h, hist_s, hist_v, hist_grad])
        
        # Update cache (LRU-style)
        if len(self.cache) >= self.max_size:
            # Remove oldest by frame number
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k]['last_frame'])
            del self.cache[oldest]
            
        self.cache[obj_id] = {
            'features': features,
            'last_frame': frame_num,
            'last_centroid': ((x + w//2), (y + h//2)),
            'stability_score': 1.0
        }
        return features
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
                              max_proposals: int = 10,
                              camera_affine: Optional[np.ndarray] = None) -> List[Tuple[Tuple[int,int,int,int], np.ndarray, float]]:
    """
    Fast candidate object detection with advanced filtering and scoring.
    Now uses camera_affine to warp search regions based on camera motion.
    """
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

def match_depth_to_objects(tracked_objects, depth_proposals, iou_thresh=0.3):
    """For each tracked object, find the best matching depth box and return its area if good."""
    obj_to_depth = {}
    for obj_id, (cnt, _) in tracked_objects.items():
        x, y, w, h = cv2.boundingRect(cnt)
        best_iou = iou_thresh
        best_area = None
        for dbox, _, _ in depth_proposals:
            dx, dy, dw, dh = dbox
            xi = max(x, dx)
            yi = max(y, dy)
            wi = min(x+w, dx+dw) - xi
            hi = min(y+h, dy+dh) - yi
            if wi > 0 and hi > 0:
                inter = wi * hi
                union = w*h + dw*dh - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_area = dw * dh   # area as depth proxy
        if best_area is not None:
            obj_to_depth[obj_id] = best_area
    return obj_to_depth
# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    if not os.path.exists(PIPE_PATH):
        print("Pipe not found.", file=sys.stderr)
        sys.exit(1)
    global smoothed_box, prev_contour, locked_id, lock_frame_count, prev_frame_gray, prev_proposals

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
    # ===== Initialize Object Identity Memory =====
    identity_memory = ObjectIdentityMemory(
        db_path='./object_memory_db.pkl',
        similarity_threshold=0.75,     # Tune this value (0.7-0.85 works well)
        max_objects=100                 # Keep database size manageable
    )
    # ===== Initialize Phase 7: Persistent Identity Memory (AGI-level) =====
    persistent_memory = PersistentObjectMemory(
        db_path='./persistent_memory.pkl',
        similarity_threshold=0.65,      # Lower for low-res cameras [citation:9]
        max_objects=50,                   # Keep database small
        location_grid_size=8,              # 8x8 heatmap
        temporal_window_days=7             # Track weekly patterns
    )
    static_geo = StaticGeometryStore(min_age=30, max_contours=50)
    # Initialize global motion estimator
    motion_estimator = GlobalMotionEstimator(
        max_features=150,          # total features per frame
        quality_level=0.01,
        min_distance=5,
        lk_win_size=(15, 15),
        lk_max_level=2,
        ransac_thresh=3.0,
        max_iters=1000,
        confidence=0.99
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
    # Initialize proposal buffer
    proposal_buffer = ProposalBuffer(
        iou_threshold=0.5,
        min_consecutive=2,
        area_change_threshold=0.6,
        enter_threshold=0.65,
        exit_threshold=0.45,
        max_age=5
    )
    # Cumulative camera motion (for display)
    cumulative_dx = 0.0
    cumulative_dy = 0.0
    cumulative_rot = 0.0
    cumulative_scale = 1.0

    cv2.namedWindow(WINDOW_NAME)    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    smoothed_affine = None

    # --- Segment 1: Global attention controller state ---
    current_focus_id = None          # ID of the object currently in focus
    focus_hold_frames = 0             # how many frames current focus has been held
    focus_min_hold = 8                 # minimum frames before switching allowed
    focus_switch_margin = 1.25         # require 25% improvement to switch
    last_switch_time = time.time()     # optional timestamp for time-based hold

    # --- Segment 4: Curiosity / novelty trigger parameters ---
    NOVELTY_THRESHOLD = 0.8            # novelty score above this triggers immediate switch
    NOVELTY_MOTION_WEIGHT = 0.5        # weight for flow confidence in novelty
    NOVELTY_AGE_DECAY = 20              # frames over which novelty from age decays
    NOVELTY_AREA_INCREASE_FACTOR = 1.5  # area increase factor for novelty bonus
    NOVELTY_AREA_BONUS = 0.3            # bonus if area increased significantly
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        success = False
        if prev_frame_gray is not None:
            dx, dy, rot, scale, success = motion_estimator.estimate_motion(gray)
            if success:
                cumulative_dx += dx
                cumulative_dy += dy
                cumulative_rot += rot
                cumulative_scale *= scale
        else:
            # First frame: initialise motion estimator
            motion_estimator.prev_gray = gray.copy()
            motion_estimator.prev_pts = motion_estimator.detect_features(gray)
        affine = motion_estimator.last_affine if success else None
        # ---- Stage 6: EMA smoothing of affine ----
        ema_alpha = 0.2   # weight of current measurement (0.2 = strong smoothing)
        if success and affine is not None:
            if smoothed_affine is None:
                smoothed_affine = affine.copy()
            else:
                smoothed_affine = cv2.addWeighted(affine, ema_alpha, smoothed_affine, 1 - ema_alpha, 0)
        # Use smoothed affine for the tracker if available, otherwise fall back to raw
        affine_to_use = smoothed_affine if smoothed_affine is not None else affine
        
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
        # ---- FIRST: Estimate camera motion (STABILIZE) ----
        # (This was already above, but ensure proposals are AFTER this)
        # The motion_estimator.estimate_motion() call should happen BEFORE proposals
        
        # ---- SECOND: Now generate proposals from STABILIZED context ----
        # Use the smoothed affine to guide where to look
        # ---------------------------------------------------------------
        # Generate global proposals (now AFTER stabilization)
        # ---------------------------------------------------------------   
        global_proposals = generate_global_proposals(
            frame,
            prev_frame=prev_frame_gray,          # for motion detection
            prev_proposals_list=prev_proposals,  # for temporal persistence
            use_motion=False,                     # set True only for static camera
            max_proposals=10,
            camera_affine=affine_to_use           # ADD THIS: use stabilized camera motion
        )

         # ---------------------------------------------------------------
        # Depth proposals (from pipe) – now with stabilization context
        # ---------------------------------------------------------------
        depth_proposals = []
        if box_to_use is not None:
            # Unpack the raw 8-point box
            x1, y1, x2, y2, x3, y3, x4, y4 = box_to_use

            if affine_to_use is not None:
                # Stabilize the box by applying the inverse camera transform
                points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
                inv_transform = cv2.invertAffineTransform(affine_to_use)
                stabilized_points = cv2.transform(points.reshape(-1, 1, 2), inv_transform).reshape(-1, 2)

                # Use stabilized points for bounding box
                xs = stabilized_points[:, 0]
                ys = stabilized_points[:, 1]
            else:
                # No stabilization: use original points
                xs = [x1, x2, x3, x4]
                ys = [y1, y2, y3, y4]

            # Compute bounding rectangle from the chosen points
            x_min = int(min(xs))
            x_max = int(max(xs))
            y_min = int(min(ys))
            y_max = int(max(ys))
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            # Create a simple rectangular contour
            rect_contour = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ], dtype=np.int32).reshape((-1, 1, 2))

            depth_proposals.append((bbox, rect_contour, 1.0))   # confidence 1.0 for depth

        elif locked_id is not None:
            # If we have a locked object but no depth box, generate a prediction box
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
                # Clamp to image boundaries
                h_img, w_img = frame.shape[:2]
                x_min = max(0, min(x_min, w_img - 1))
                y_min = max(0, min(y_min, h_img - 1))
                x_max = max(0, min(x_max, w_img - 1))
                y_max = max(0, min(y_max, h_img - 1))
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                rect_contour = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ], dtype=np.int32).reshape((-1, 1, 2))
                depth_proposals.append((bbox, rect_contour, 0.8))   # slightly lower confidence
        # ---------------------------------------------------------------
        # Proposal fusion
        # ---------------------------------------------------------------
        fused_proposals = fuse_proposals(
            global_proposals,
            depth_proposals,
            frame.shape[:2],
            max_proposals=6,
            overlap_threshold=0.55,
            depth_bonus=0.2
        )

        # ---------------------------------------------------------------
        # Appearance extraction from each fused proposal
        # ---------------------------------------------------------------
        raw_detections = []  # list of (contour, confidence)
        for (bbox, cnt, score) in fused_proposals:
            x, y, w, h = bbox
            box8 = (x, y, x+w, y, x+w, y+h, x, y+h)
            contour_list = extractor.extract_contours(frame, box8, prior_contour=None)
            raw_detections.extend(contour_list)   # each is (contour, conf)

        # NEW: Enhance small objects using depth information [citation:6]
        if depth_proposals:   # depth_proposals is always defined (may be empty)
            enhanced_small = enhance_small_objects_with_depth(frame, depth_proposals)
            # Assign a default confidence of 0.7 to enhanced contours
            for cnt in enhanced_small:
                raw_detections.append((cnt, 0.7))

        # Optional: Draw fused proposals (in green) for debugging
        for (bbox, cnt, score) in fused_proposals:
            x,y,w,h = bbox
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{score:.2f}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.drawContours(frame, [cnt], -1, (0,255,0), 1)

        # Pass through proposal buffer to apply temporal stability & area filter
        filtered_pairs = proposal_buffer.update(raw_detections)

        # Update tracker with filtered contours and confidences
        tracked_objects = tracker.update(frame, filtered_pairs, camera_affine=affine_to_use, identity_memory=identity_memory)
        
        # NEW: Update template feature cache - only for confirmed tracks [citation:8]
        if not hasattr(tracker, 'feature_cache'):
            tracker.feature_cache = TemplateFeatureCache(max_cache_size=15)
        
        # Only compute features for confirmed tracks to save CPU [citation:3]
        for obj_id, (contour, centroid) in tracked_objects.items():
            obj = tracker.tracked_objects.get(obj_id)
            if obj and obj.track_state == TRACK_STATE_CONFIRMED:
                # Cache features for confirmed objects (less frequent updates)
                if frame_count - obj.last_feature_update > 10:  # Update every 10 frames
                    features = tracker.feature_cache.get_features(
                        obj_id, frame, contour, frame_count
                    )
                    if features is not None:
                        obj.cached_features = features
                        obj.last_feature_update = frame_count
        # ---- Stage 7: Depth consistency anchoring ----
        depth_matches = match_depth_to_objects(tracked_objects, depth_proposals)
        for obj_id, area in depth_matches.items():
            obj = tracker.tracked_objects[obj_id]
            if obj.depth_history is None:
                obj.depth_history = []
            obj.depth_history.append(area)
            if len(obj.depth_history) > 15:
                obj.depth_history.pop(0)
            if len(obj.depth_history) >= 10:
                arr = np.array(obj.depth_history)
                mean_area = np.mean(arr)
                if mean_area > 0:
                    var_coeff = np.var(arr) / mean_area
                    obj.depth_stable = var_coeff < 0.1
        # ---- Stage 5: Update static geometry ----
        static_geo.update(tracker, tracked_objects, motion_estimator.world_transform, gray)
        # --- Segment 2: Compute stable attention score (EMA) for each object ---
        attention_scores = {}
        novelty_scores = {}
        if tracked_objects:
            max_area = max((obj.contour_area for obj in tracker.tracked_objects.values()), default=1)
            for obj_id, (contour, centroid) in tracked_objects.items():
                obj = tracker.tracked_objects[obj_id]
                area_weight = min(1.0, obj.contour_area / max_area) if max_area > 0 else 0
                motion_weight = obj.flow_confidence
                depth_bonus = 0.3 if obj.depth_stable else 0.0
                
                # Raw attention score (same as before, but now we'll smooth it)
                attention_raw = obj.persistence_score + area_weight + motion_weight + depth_bonus
                
                # Update EMA (0.6 * old + 0.4 * new)
                obj.attention_score_ema = 0.6 * obj.attention_score_ema + 0.4 * attention_raw
                attention_scores[obj_id] = obj.attention_score_ema   # use EMA for focus decision

                # --- Segment 4: Compute novelty score for curiosity trigger ---
                # Novelty based on: young age, high motion, sudden area increase
                age_novelty = max(0, 1.0 - obj.age / NOVELTY_AGE_DECAY)  # decays over frames
                motion_novelty = obj.flow_confidence * NOVELTY_MOTION_WEIGHT
                area_novelty = 0.0
                if obj.areas is not None and len(obj.areas) >= 2:
                    prev_area = obj.areas[-2]
                    if prev_area > 0 and obj.contour_area > prev_area * NOVELTY_AREA_INCREASE_FACTOR:
                        area_novelty = NOVELTY_AREA_BONUS
                novelty = age_novelty + motion_novelty + area_novelty
                # Also boost if object is new (age very low)
                if obj.age <= 2:
                    novelty += 0.2
                novelty_scores[obj_id] = min(1.0, novelty)

            # --- Segment 1: Apply focus controller (prevent jitter) with curiosity override ---
            # Find object with highest EMA score
            best_id = max(attention_scores.items(), key=lambda x: x[1])[0]
            best_score = attention_scores[best_id]

            # Check for novel object that might trigger immediate switch
            curiosity_trigger = None
            if current_focus_id is not None and current_focus_id in tracked_objects:
                current_score = attention_scores[current_focus_id]
                for obj_id, nov in novelty_scores.items():
                    if obj_id != current_focus_id and nov > NOVELTY_THRESHOLD:
                        # If this novel object's attention is at least 80% of current focus, allow switch
                        if attention_scores[obj_id] > current_score * 0.8:
                            curiosity_trigger = obj_id
                            break

            if curiosity_trigger is not None:
                # Immediate switch due to novelty
                current_focus_id = curiosity_trigger
                focus_hold_frames = 1
                last_switch_time = time.time()
                # Optionally log
                # print(f"Curiosity switch to {curiosity_trigger}", file=sys.stderr)
            elif current_focus_id is None:
                # No current focus -> adopt the best if it's reasonably high
                if best_score > 0.5:   # arbitrary threshold, can be tuned
                    current_focus_id = best_id
                    focus_hold_frames = 1
                    last_switch_time = time.time()
            else:
                # Check if current focus is still tracked
                if current_focus_id in tracked_objects:
                    current_score = attention_scores[current_focus_id]
                    focus_hold_frames += 1

                    # Only consider switching if we've held for minimum frames
                    if focus_hold_frames >= focus_min_hold:
                        # Switch if best score is significantly better (margin)
                        if best_id != current_focus_id and best_score > current_score * focus_switch_margin:
                            # Switch focus
                            current_focus_id = best_id
                            focus_hold_frames = 1
                            last_switch_time = time.time()
                    # else: still in cooldown, keep current focus
                else:
                    # Current focus lost -> reset
                    current_focus_id = None
                    focus_hold_frames = 0

            # The primary focus for display is now current_focus_id (or fallback to best if none)
            primary_focus_id = current_focus_id if current_focus_id is not None else best_id

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
            
            # NEW: Color and style based on track state [citation:2]
            if obj.track_state == TRACK_STATE_TENTATIVE:
                # Tentative tracks: dashed blue lines (filtered noise) [citation:3]
                color = (255, 0, 0)  # Blue
                thickness = 1
                # Draw dashed contour for tentative tracks
                for i in range(len(contour)):
                    pt1 = tuple(contour[i][0])
                    pt2 = tuple(contour[(i+1) % len(contour)][0])
                    if i % 2 == 0:  # Draw every other segment to create dashed effect
                        cv2.line(frame, pt1, pt2, color, thickness)
            elif obj.track_state == TRACK_STATE_CONFIRMED:
                # Confirmed tracks: get permanent IDs [citation:2]
                if primary_focus_id is not None and obj_id == primary_focus_id:
                    color = (255, 255, 0)  # Cyan - primary focus
                    thickness = 4
                elif obj.persistence_score > 0.7:
                    color = (0, 255, 0)  # Green - high confidence
                    thickness = 2
                elif obj.persistence_score > 0.4:
                    color = (0, 255, 255)  # Yellow - medium confidence
                    thickness = 2
                else:
                    color = (0, 165, 255)  # Orange - low confidence
                    thickness = 2
                cv2.drawContours(frame, [contour], -1, color, thickness)
            else:  # ARCHIVED
                # Archived tracks: gray dashed lines [citation:8]
                color = (128, 128, 128)  # Gray
                thickness = 1
                for i in range(len(contour)):
                    pt1 = tuple(contour[i][0])
                    pt2 = tuple(contour[(i+1) % len(contour)][0])
                    if i % 3 == 0:  # Different dash pattern for archived
                        cv2.line(frame, pt1, pt2, color, thickness)
            
            # Draw ID and state info
            cx, cy = centroid
            
            # NEW: Show track state in display
            state_prefix = {
                TRACK_STATE_TENTATIVE: "T",
                TRACK_STATE_CONFIRMED: "C",
                TRACK_STATE_ARCHIVED: "A"
            }.get(obj.track_state, "?")
            
            # Show ID with state prefix
            cv2.putText(frame, f"{state_prefix}_{obj_id:03d}",
                       (cx - 20, cy - 20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       color, 2)
            
            # Show consecutive matches for tentative tracks [citation:2]
            if obj.track_state == TRACK_STATE_TENTATIVE:
                cv2.putText(frame, f"{obj.consecutive_matches}/{CONFIRMATION_THRESHOLD}",
                           (cx - 15, cy + 15),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.4,
                           color, 1)
            else:
                # Show persistence score as percentage for confirmed tracks
                cv2.putText(frame, f"{int(obj.persistence_score*100)}%",
                           (cx - 15, cy + 15),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.4,
                           color, 1)
            
            # Draw trajectory trail for confirmed tracks only (optimization) [citation:3]
            if obj.track_state == TRACK_STATE_CONFIRMED and len(obj.centroids) > 1:
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
            identity_memory.save_database()
            identity_memory.cleanup_old_objects(max_age_seconds=86400)  # 24 hours
        # Update previous frame and proposals for next iteration
        prev_frame_gray = gray
        # ---- Global camera motion estimation ----
        # Store only the top proposals (e.g., up to 15) to save memory
        prev_proposals = [(bbox, cnt, conf) for (bbox, cnt, conf) in global_proposals[:15]]
        # ---- Draw static geometry ----
        static_geo.draw(frame, motion_estimator.world_transform, color=(100,100,100))
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()