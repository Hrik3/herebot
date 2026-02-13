Good — this is the **correct moment to formalize the vision worker contract**, because once the interface is stable your cognitive engine can evolve without touching perception code.

Below is a **clean AGI-ready specification** for `vision.py` live stream output.

Think of this as the **perceptual state packet** emitted every frame.

---

# vision.py — Goal

Continuously read video frames and output a structured **PerceptualFrame** object containing:

* raw perceptual data
* compressed interpretable representations
* region annotations
* motion / temporal signals
* embedding summaries
* interpretable symbolic hints usable by cognition engine

This allows your main AGI script to reason **without directly touching pixels**.

---

# Core streaming object

Every frame should output something like:

```
PerceptualFrame = {
    "timestamp": float,
    "frame_id": int,

    "resolution": (width, height),

    "pixel_stats": {...},
    "color_fields": {...},
    "regions": [...],
    "motion": {...},
    "objects": [...],
    "scene_embedding": [...],
    "saliency_map": [...],
    "attention_regions": [...],
    "temporal_context": {...}
}
```

---

# 1. Pixel-level statistics

(very important for early grounding learning)

```
pixel_stats:
    mean_rgb
    std_rgb
    brightness
    contrast
    entropy
    dominant_color
    color_histogram (coarse bins)
```

---

# 2. Color / field segmentation

This is what will let your AGI learn “sky”, “grass”, etc.

```
color_fields:
    field_id
    avg_color
    area_percentage
    centroid
    boundary_polygon
```

---

# 3. Regions (interpretable patches)

Split frame into patches (grid or segmentation):

```
regions:
    region_id
    bbox
    avg_color
    texture_score
    edge_density
    embedding_vector
    motion_score
```

This allows the cognition engine to say:

> region_17 = sky

and reinforce it.

---

# 4. Motion features (critical for video cognition)

```
motion:
    global_motion_vector
    optical_flow_map (compressed)
    moving_regions_ids
    motion_intensity
```

Motion is essential for **object permanence learning** later.

---

# 5. Object candidates (even without detector)

Early stage can be unsupervised clusters:

```
objects:
    object_id
    bbox
    region_ids
    object_embedding
    persistence_score
```

Persistence over frames helps the AGI understand **identity**.

---

# 6. Scene embedding

Global vector representing the entire frame:

```
scene_embedding:
    128–512 dim vector
```

Used later for:

* episodic indexing
* world-state transitions
* planning

---

# 7. Saliency / attention hints

```
saliency_map
attention_regions:
    region_id
    saliency_score
```

Lets curiosity engine focus perception.

---

# 8. Temporal context

Important for continuity learning:

```
temporal_context:
    frame_delta_time
    scene_change_score
    object_births
    object_deaths
```

---

# 9. Optional advanced signals (future ready)

You don't need them immediately, but reserve keys:

```
depth_estimate
surface_normals
lighting_estimate
camera_motion
semantic_hypotheses
```

---

# 10. Streaming method (recommended)

vision.py should continuously:

```
capture frame
process frame
emit JSON / shared memory / socket message
```

Main AGI script subscribes and converts regions → symbolic learning:

Example:

```
user: "this region is sky"
AGI: binds region embedding → concept "sky"
```

This is exactly **developmental perception learning**, not pretrained recognition.

---

# 11. Important architectural insight

Your design (separate vision worker → cognition engine) is **exactly how large cognitive robotics systems are structured**:

* perception workers
* world-state stream
* cognition loop consuming state

This is the scalable design.

---

# 12. Very important next step (high impact)

Before writing CV models, first implement:

```
vision.py → dummy generator
```

that streams:

* random regions
* fake motion
* random embeddings

Then integrate it into AGI pipeline.

This prevents months of integration pain later.

---

If you want, next I can design the **exact minimal v1 vision.py implementation (200-line starter)** that already streams usable perceptual packets and plugs directly into your episodic memory system.
