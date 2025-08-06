import cv2
from ultralytics import YOLO
import numpy as np
import math
import os
from collections import defaultdict, deque

# ---------------- CONFIG ---------------- #
VIDEO_SOURCE = 'videos/test2.mp4'
OUTPUT_PATH = 'outputs/annotated_test2.mp4'

# DETECTION CONFIDENCE THRESHOLDS (Range: 0.1-0.9)
# For flickering issues, try slightly higher thresholds with better smoothing
PERSON_CONFIDENCE_THRESHOLD = 0.45   # Slightly lower to catch more detections (0.3-0.7)
BALL_CONFIDENCE_THRESHOLD = 0.15     # Very low for static balls (0.1-0.4)

# MOVEMENT DETECTION (Range: 3-50 pixels)
# Lower values help detect subtle movements
PERSON_MOVEMENT_THRESHOLD = 8        # Slightly lower for better sensitivity (5-20)
BALL_MOVEMENT_THRESHOLD = 3          # Very low for small ball movements (2-10)

# TRACK STABILITY AND SMOOTHING
NEW_TRACK_FRAME_THRESHOLD = 2        # Faster track acceptance (1-5)
BBOX_SMOOTHING_FRAMES = 5            # Smooth bounding boxes over N frames (3-10)
POSITION_SMOOTHING_ALPHA = 0.7       # Exponential smoothing factor (0.5-0.9)

# DETECTION RESOLUTION AND QUALITY
DETECTION_RESOLUTION = 1280          # Higher resolution for better small object detection
NMS_IOU_THRESHOLD = 0.5              # Non-max suppression IoU (0.3-0.7)

# TRACKING PERSISTENCE (for handling temporary losses)
MAX_FRAMES_WITHOUT_DETECTION = 10    # Keep track alive for N frames (5-20)
BALL_TRAIL_LENGTH = 50               # Longer trails for better visualization (20-100)

# TRACKER SETTINGS - Try different trackers for better performance
TRACKER_CONFIG = 'bytetrack.yaml'    # Options: 'bytetrack.yaml', 'botsort.yaml'

# LIGHTING AND QUALITY ADJUSTMENTS
ENABLE_FRAME_ENHANCEMENT = True      # Enhance frame quality for better detection
CONTRAST_ALPHA = 1.1                 # Contrast adjustment (0.8-1.5)
BRIGHTNESS_BETA = 10                 # Brightness adjustment (-50 to 50)

os.makedirs('outputs', exist_ok=True)

# ---------------- LOAD MODELS ---------------- #
print("Loading detection model (yolov8x.pt)...")
detection_model = YOLO('yolov8x.pt')
print("Loading pose model (yolov8x-pose.pt)...")
pose_model = YOLO('yolov8x-pose.pt')

# ---------------- VIDEO IO ---------------- #
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source {VIDEO_SOURCE}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"Video info: {width}x{height} @ {fps:.1f} FPS")

# ---------------- ENHANCED TRACKING STORAGE ---------------- #
prev_positions = {}
track_frame_count = {}
trail_history = {}
last_seen_frame = {}
current_frame_num = 0

# Smoothing and stability tracking
bbox_history = defaultdict(lambda: deque(maxlen=BBOX_SMOOTHING_FRAMES))
smoothed_positions = {}
track_confidence_history = defaultdict(lambda: deque(maxlen=10))
lost_tracks = {}  # Store temporarily lost tracks

# ---------------- FRAME ENHANCEMENT ---------------- #
def enhance_frame(frame):
    """
    Enhance frame quality for better detection in challenging lighting.
    """
    if not ENABLE_FRAME_ENHANCEMENT:
        return frame
    
    # Apply contrast and brightness adjustment
    enhanced = cv2.convertScaleAbs(frame, alpha=CONTRAST_ALPHA, beta=BRIGHTNESS_BETA)
    
    # Optional: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better lighting
    # Uncomment below if you have very challenging lighting conditions
    # lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # lab[:,:,0] = clahe.apply(lab[:,:,0])
    # enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

# ---------------- BOUNDING BOX SMOOTHING ---------------- #
def smooth_bbox(track_id, bbox):
    """
    Smooth bounding box coordinates to reduce flickering.
    """
    bbox_history[track_id].append(bbox)
    
    if len(bbox_history[track_id]) == 1:
        return bbox
    
    # Calculate moving average of bounding boxes
    history = list(bbox_history[track_id])
    smoothed = np.mean(history, axis=0).astype(int)
    
    return smoothed

def smooth_position(track_id, new_pos):
    """
    Apply exponential smoothing to positions to reduce jitter.
    """
    if track_id not in smoothed_positions:
        smoothed_positions[track_id] = new_pos
        return new_pos
    
    old_pos = smoothed_positions[track_id]
    smoothed = (
        int(POSITION_SMOOTHING_ALPHA * new_pos[0] + (1 - POSITION_SMOOTHING_ALPHA) * old_pos[0]),
        int(POSITION_SMOOTHING_ALPHA * new_pos[1] + (1 - POSITION_SMOOTHING_ALPHA) * old_pos[1])
    )
    smoothed_positions[track_id] = smoothed
    return smoothed

# ---------------- STICK FIGURE ---------------- #
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Head connections
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),          # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
]

def draw_stick_figure(frame, keypoints, conf_threshold=0.4):
    """Draw stick figure with lower confidence threshold for better coverage."""
    limb_colors = [
        (0, 255, 255)] * 4 + [(255, 0, 0)] * 5 + [(0, 255, 0)] * 3 + [(0, 0, 255)] * 4
    
    for i, (p1_idx, p2_idx) in enumerate(SKELETON):
        if p1_idx < len(keypoints) and p2_idx < len(keypoints):
            p1 = keypoints[p1_idx]
            p2 = keypoints[p2_idx]
            if p1[2] > conf_threshold and p2[2] > conf_threshold:
                pt1 = (int(p1[0]), int(p1[1]))
                pt2 = (int(p2[0]), int(p2[1]))
                cv2.line(frame, pt1, pt2, limb_colors[i], 2)
                cv2.circle(frame, pt1, 3, (0, 0, 255), -1)
                cv2.circle(frame, pt2, 3, (0, 0, 255), -1)

def draw_enhanced_ball_trail(frame, trail, color, is_moving, confidence):
    """Enhanced trail drawing with confidence-based styling."""
    if len(trail) < 2:
        return
    
    # Adjust trail properties based on confidence and movement
    base_thickness = 3 if is_moving else 2
    thickness_multiplier = min(1.5, confidence * 2)  # Scale with confidence
    
    for i in range(1, len(trail)):
        alpha = (i / len(trail)) * 0.8 + 0.2  # Fade from 0.2 to 1.0
        thickness = max(1, int(base_thickness * alpha * thickness_multiplier))
        
        # Create slightly transparent trail effect
        trail_color = tuple(int(c * alpha) for c in color)
        cv2.line(frame, trail[i-1], trail[i], trail_color, thickness)

# ---------------- ENHANCED DETECTION WITH MULTIPLE STRATEGIES ---------------- #
def run_multi_strategy_detection(frame):
    """
    Run detection with multiple strategies for maximum coverage.
    """
    # Strategy 1: Standard tracking detection
    results_tracking = detection_model.track(
        frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        classes=[0, 32],
        imgsz=DETECTION_RESOLUTION,
        verbose=False,
        conf=min(PERSON_CONFIDENCE_THRESHOLD, BALL_CONFIDENCE_THRESHOLD),
        iou=NMS_IOU_THRESHOLD
    )
    
    # Strategy 2: Pure detection without tracking (for missed static objects)
    results_detection = detection_model(
        frame,
        classes=[32],  # Only balls for static detection
        imgsz=DETECTION_RESOLUTION,
        verbose=False,
        conf=BALL_CONFIDENCE_THRESHOLD * 0.8,  # Even lower threshold
        iou=NMS_IOU_THRESHOLD
    )
    
    return results_tracking, results_detection

def assign_ids_to_untracked_detections(tracked_boxes, pure_detections, frame_num):
    """
    Assign IDs to detections that weren't caught by the tracker.
    """
    if pure_detections[0].boxes is None or len(pure_detections[0].boxes) == 0:
        return []
    
    untracked_objects = []
    pure_boxes = pure_detections[0].boxes.xyxy.cpu().numpy()
    pure_confs = pure_detections[0].boxes.conf.cpu().numpy()
    pure_classes = pure_detections[0].boxes.cls.cpu().numpy()
    
    # Find detections that don't overlap significantly with tracked objects
    if tracked_boxes is not None and len(tracked_boxes) > 0:
        tracked_boxes_np = tracked_boxes.xyxy.cpu().numpy()
        
        for i, (pure_box, conf, cls) in enumerate(zip(pure_boxes, pure_confs, pure_classes)):
            # Calculate IoU with all tracked boxes
            max_iou = 0
            for tracked_box in tracked_boxes_np:
                iou = calculate_iou(pure_box, tracked_box)
                max_iou = max(max_iou, iou)
            
            # If IoU is low, this is likely a missed detection
            if max_iou < 0.3:  # Low overlap threshold
                # Assign a new ID (use high numbers to avoid conflicts)
                new_id = 1000 + frame_num + i
                untracked_objects.append({
                    'id': new_id,
                    'bbox': pure_box.astype(int),
                    'conf': conf,
                    'class': int(cls)
                })
    else:
        # No tracked objects, assign IDs to all pure detections
        for i, (pure_box, conf, cls) in enumerate(zip(pure_boxes, pure_confs, pure_classes)):
            new_id = 1000 + frame_num + i
            untracked_objects.append({
                'id': new_id,
                'bbox': pure_box.astype(int),
                'conf': conf,
                'class': int(cls)
            })
    
    return untracked_objects

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# ---------------- MAIN PROCESSING LOOP ---------------- #
print("Processing video frames...")
print(f"Enhanced settings - Resolution: {DETECTION_RESOLUTION}, Smoothing: {BBOX_SMOOTHING_FRAMES} frames")
print(f"Confidence thresholds - Person: {PERSON_CONFIDENCE_THRESHOLD}, Ball: {BALL_CONFIDENCE_THRESHOLD}")
print(f"Movement thresholds - Person: {PERSON_MOVEMENT_THRESHOLD}px, Ball: {BALL_MOVEMENT_THRESHOLD}px")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    current_frame_num += 1
    
    # Enhance frame quality if enabled
    enhanced_frame = enhance_frame(frame)
    
    # Run multi-strategy detection
    results_tracking, results_detection = run_multi_strategy_detection(enhanced_frame)
    
    # Process tracked objects
    all_objects = []
    tracked_boxes = results_tracking[0].boxes
    
    if tracked_boxes is not None and len(tracked_boxes) > 0:
        track_ids = tracked_boxes.id.int().cpu().numpy() if tracked_boxes.id is not None else np.arange(len(tracked_boxes))
        bboxes = tracked_boxes.xyxy.int().cpu().numpy()
        class_ids = tracked_boxes.cls.int().cpu().numpy()
        confidences = tracked_boxes.conf.cpu().numpy()
        
        for track_id, bbox, cls_id, conf in zip(track_ids, bboxes, class_ids, confidences):
            all_objects.append({
                'id': track_id,
                'bbox': bbox,
                'conf': conf,
                'class': cls_id,
                'tracked': True
            })
    
    # Add untracked detections (for missed static balls)
    untracked = assign_ids_to_untracked_detections(tracked_boxes, results_detection, current_frame_num)
    all_objects.extend([{**obj, 'tracked': False} for obj in untracked])
    
    # Process all objects (tracked + untracked)
    for obj in all_objects:
        track_id = obj['id']
        bbox = obj['bbox']
        conf = obj['conf']
        cls_id = obj['class']
        is_tracked = obj['tracked']
        
        # Apply class-specific confidence thresholds
        confidence_threshold = PERSON_CONFIDENCE_THRESHOLD if cls_id == 0 else BALL_CONFIDENCE_THRESHOLD
        if conf < confidence_threshold:
            continue
        
        # Smooth bounding box if it's a tracked object
        if is_tracked:
            bbox = smooth_bbox(track_id, bbox)
        
        x1, y1, x2, y2 = bbox
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width - 1), min(y2, height - 1)
        
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Apply position smoothing
        smoothed_cx, smoothed_cy = smooth_position(track_id, (cx, cy))
        
        # Calculate movement
        prev_cx, prev_cy = prev_positions.get(track_id, (smoothed_cx, smoothed_cy))
        distance = math.sqrt((smoothed_cx - prev_cx)**2 + (smoothed_cy - prev_cy)**2)
        prev_positions[track_id] = (smoothed_cx, smoothed_cy)
        
        # Update tracking stats
        track_frame_count[track_id] = track_frame_count.get(track_id, 0) + 1
        last_seen_frame[track_id] = current_frame_num
        track_confidence_history[track_id].append(conf)
        
        label = detection_model.names[cls_id]
        avg_confidence = np.mean(list(track_confidence_history[track_id]))
        
        # ---------------- PERSON PROCESSING ---------------- #
        if label == 'person':
            movement_threshold = PERSON_MOVEMENT_THRESHOLD
            is_moving = (distance > movement_threshold and 
                        track_frame_count[track_id] > NEW_TRACK_FRAME_THRESHOLD)
            
            status = 'Moving' if is_moving else 'Static'
            track_quality = "Stable" if is_tracked else "Recovered"
            text = f"Person {track_id}: {status} ({avg_confidence:.2f}) [{track_quality}]"
            color = (0, 255, 255) if is_moving else (255, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Pose estimation for moving people
            if is_moving and is_tracked:
                person_crop = enhanced_frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    pose_result = pose_model(person_crop, verbose=False)
                    if pose_result and len(pose_result[0].keypoints.xy) > 0:
                        kpts = pose_result[0].keypoints
                        keypoints = kpts.xy[0].cpu().numpy()
                        confs = kpts.conf[0].cpu().numpy()
                        
                        if keypoints.shape[0] == confs.shape[0]:
                            keypoints_full = np.hstack((keypoints, confs[:, np.newaxis]))
                            keypoints_full[:, 0] += x1
                            keypoints_full[:, 1] += y1
                            draw_stick_figure(frame, keypoints_full)
        
        # ---------------- BALL PROCESSING ---------------- #
        elif label == 'sports ball':
            movement_threshold = BALL_MOVEMENT_THRESHOLD
            is_moving = (distance > movement_threshold and 
                        track_frame_count[track_id] > NEW_TRACK_FRAME_THRESHOLD)
            
            status = f'Moving ({distance:.1f}px)' if is_moving else 'Static'
            track_quality = "Tracked" if is_tracked else "Detected"
            text = f"Ball {track_id}: {status} ({avg_confidence:.2f}) [{track_quality}]"
            
            # Enhanced color coding
            if is_moving:
                color = (0, 255, 0)  # Green for moving
            else:
                color = (0, 165, 255) if is_tracked else (0, 100, 255)  # Orange/Blue for static
            
            # Update trail
            trail_history.setdefault(track_id, []).append((smoothed_cx, smoothed_cy))
            trail = trail_history[track_id][-BALL_TRAIL_LENGTH:]
            trail_history[track_id] = trail
            
            # Draw enhanced trail
            draw_enhanced_ball_trail(frame, trail, color, is_moving, avg_confidence)
            
            # Draw bounding box
            thickness = 3 if is_tracked else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Enhanced center point
            center_size = 5 if is_moving else 4
            cv2.circle(frame, (smoothed_cx, smoothed_cy), center_size, color, -1)
            cv2.circle(frame, (smoothed_cx, smoothed_cy), center_size + 2, (255, 255, 255), 1)
    
    # Clean up old tracks
    tracks_to_remove = []
    for track_id, last_frame in last_seen_frame.items():
        if current_frame_num - last_frame > MAX_FRAMES_WITHOUT_DETECTION:
            tracks_to_remove.append(track_id)
    
    for track_id in tracks_to_remove:
        if track_id in prev_positions:
            del prev_positions[track_id]
        if track_id in trail_history:
            del trail_history[track_id]
        if track_id in smoothed_positions:
            del smoothed_positions[track_id]
    
    # Enhanced frame info
    info_text = f"Frame: {current_frame_num}/{269} | Active tracks: {len(prev_positions)} | Enhanced: {'ON' if ENABLE_FRAME_ENHANCEMENT else 'OFF'}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out.write(frame)

# ---------------- CLEANUP AND SUMMARY ---------------- #
cap.release()
out.release()
print(f"✅ Enhanced processing complete!")
print(f"📹 Output saved to: {OUTPUT_PATH}")
print(f"📊 Total frames processed: {current_frame_num}")
print(f"🎯 Total unique objects tracked: {len(track_frame_count)}")

print("\n--- ENHANCED TRACKING SUMMARY ---")
for track_id, count in sorted(track_frame_count.items()):
    avg_conf = np.mean(list(track_confidence_history[track_id])) if track_confidence_history[track_id] else 0
    track_type = "Static Ball" if track_id >= 1000 else "Tracked Object"
    print(f"ID {track_id}: {count} frames | Avg confidence: {avg_conf:.3f} | Type: {track_type}")

print(f"\n🔧 Current settings worked well! For further tuning:")
print(f"   - Static ball issues: Lower BALL_CONFIDENCE_THRESHOLD to {BALL_CONFIDENCE_THRESHOLD * 0.8:.2f}")
print(f"   - Flickering boxes: Increase BBOX_SMOOTHING_FRAMES to {BBOX_SMOOTHING_FRAMES + 2}")
print(f"   - Tracking gaps: Increase MAX_FRAMES_WITHOUT_DETECTION to {MAX_FRAMES_WITHOUT_DETECTION + 5}")