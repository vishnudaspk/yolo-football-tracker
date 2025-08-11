import cv2
from ultralytics import YOLO
import numpy as np
import math
import os
from collections import defaultdict, deque

# ---------------- CONFIG ---------------- #
VIDEO_SOURCE = 'videos/test2.mp4'
OUTPUT_PATH = 'outputs/annotated_test2.mp4'

PERSON_CONFIDENCE_THRESHOLD = 0.45
BALL_CONFIDENCE_THRESHOLD = 0.15
PERSON_MOVEMENT_THRESHOLD = 8
BALL_MOVEMENT_THRESHOLD = 3
NEW_TRACK_FRAME_THRESHOLD = 2
BBOX_SMOOTHING_FRAMES = 5
POSITION_SMOOTHING_ALPHA = 0.7
DETECTION_RESOLUTION = 1280
NMS_IOU_THRESHOLD = 0.5
MAX_FRAMES_WITHOUT_DETECTION = 10
BALL_TRAIL_LENGTH = 50
TRACKER_CONFIG = 'bytetrack.yaml'
ENABLE_FRAME_ENHANCEMENT = True
CONTRAST_ALPHA = 1.1
BRIGHTNESS_BETA = 10

os.makedirs('outputs', exist_ok=True)

print("Loading detection model (yolov8x.pt)...")
detection_model = YOLO('yolov8x.pt')
print("Loading pose model (yolov8x-pose.pt)...")
pose_model = YOLO('yolov8x-pose.pt')

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source {VIDEO_SOURCE}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ---------------- STATE ---------------- #
prev_positions = {}                 # keyed by entity_id (e.g. 'ball_1' or 'person_12')
track_frame_count = {}
trail_history = {}                  # keyed by persistent ball id: trail_history[pid] = [(x,y), ...]
last_seen_frame = {}
current_frame_num = 0
bbox_history = defaultdict(lambda: deque(maxlen=BBOX_SMOOTHING_FRAMES))
smoothed_positions = {}
track_confidence_history = defaultdict(lambda: deque(maxlen=10))

# Persistent ball storage
persistent_balls = {}               # pid -> {'center': (x,y), 'radius': r, 'last_seen': frame, 'missed': 0}
detection_to_persistent = {}        # current detection id -> pid (updated each frame)
next_ball_number = 1
initial_balls_assigned = False

# Skeleton connections (COCO-like)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

# ---------------- UTILITIES ---------------- #

def enhance_frame(frame):
    if not ENABLE_FRAME_ENHANCEMENT:
        return frame
    return cv2.convertScaleAbs(frame, alpha=CONTRAST_ALPHA, beta=BRIGHTNESS_BETA)


def smooth_bbox(entity_id, bbox):
    bbox_history[entity_id].append(bbox)
    if len(bbox_history[entity_id]) == 1:
        return bbox
    return np.mean(list(bbox_history[entity_id]), axis=0).astype(int)


def smooth_position(entity_id, new_pos):
    if entity_id not in smoothed_positions:
        smoothed_positions[entity_id] = new_pos
        return new_pos
    old_pos = smoothed_positions[entity_id]
    smoothed = (
        int(POSITION_SMOOTHING_ALPHA * new_pos[0] + (1 - POSITION_SMOOTHING_ALPHA) * old_pos[0]),
        int(POSITION_SMOOTHING_ALPHA * new_pos[1] + (1 - POSITION_SMOOTHING_ALPHA) * old_pos[1])
    )
    smoothed_positions[entity_id] = smoothed
    return smoothed


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# Keep the same untracked-detection fallback you had earlier
def assign_ids_to_untracked_detections(tracked_boxes, pure_detections, frame_num):
    if pure_detections[0].boxes is None or len(pure_detections[0].boxes) == 0:
        return []
    untracked = []
    pure_boxes = pure_detections[0].boxes.xyxy.cpu().numpy()
    pure_confs = pure_detections[0].boxes.conf.cpu().numpy()
    pure_classes = pure_detections[0].boxes.cls.cpu().numpy()
    if tracked_boxes is not None and len(tracked_boxes) > 0:
        tracked_np = tracked_boxes.xyxy.cpu().numpy()
        for i, (pb, conf, cls) in enumerate(zip(pure_boxes, pure_confs, pure_classes)):
            if max(calculate_iou(pb, tb) for tb in tracked_np) < 0.3:
                untracked.append({'id': 1000 + frame_num + i, 'bbox': pb.astype(int), 'conf': conf, 'class': int(cls)})
    else:
        for i, (pb, conf, cls) in enumerate(zip(pure_boxes, pure_confs, pure_classes)):
            untracked.append({'id': 1000 + frame_num + i, 'bbox': pb.astype(int), 'conf': conf, 'class': int(cls)})
    return untracked


# Initial clustering for frame 1: merge detections that overlap (dist <= r1 + r2)
def cluster_initial_balls(detections):
    # detections: list of dict with keys 'det_id','cx','cy','radius'
    n = len(detections)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            dx = detections[i]['cx'] - detections[j]['cx']
            dy = detections[i]['cy'] - detections[j]['cy']
            dist = math.hypot(dx, dy)
            if dist <= (detections[i]['radius'] + detections[j]['radius']):
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(detections[i])

    # compute cluster centers
    cluster_list = []
    for members in clusters.values():
        xs = [m['cx'] for m in members]
        ys = [m['cy'] for m in members]
        rs = [m['radius'] for m in members]
        center_x = int(np.mean(xs))
        center_y = int(np.mean(ys))
        center_r = int(np.max(rs))
        det_ids = [m['det_id'] for m in members]
        cluster_list.append({'cx': center_x, 'cy': center_y, 'radius': center_r, 'members': det_ids})

    # sort left-to-right
    cluster_list.sort(key=lambda c: c['cx'])
    return cluster_list


# Match detections to existing persistent balls in a stable way
def match_detections_to_persistent(detections, frame_num):
    # detections: list of dict {'det_id','cx','cy','radius'}
    global next_ball_number
    matched = {}

    # Quick match if detection had previous mapping
    unmatched_dets = {}
    unmatched_pers = set(persistent_balls.keys())

    for det in detections:
        det_id = det['det_id']
        if det_id in detection_to_persistent and detection_to_persistent[det_id] in persistent_balls:
            pid = detection_to_persistent[det_id]
            matched[det_id] = pid
            # update persistent ball
            persistent_balls[pid]['center'] = (det['cx'], det['cy'])
            persistent_balls[pid]['radius'] = det['radius']
            persistent_balls[pid]['last_seen'] = frame_num
            persistent_balls[pid]['missed'] = 0
            unmatched_pers.discard(pid)
        else:
            unmatched_dets[det_id] = det

    # Build list of unmatched detections and unmatched persistent centers
    if len(unmatched_dets) > 0 and len(unmatched_pers) > 0:
        # Prepare list for greedy global minimum matching
        pairs = []  # (dist, det_id, pid)
        for det_id, det in unmatched_dets.items():
            for pid in list(unmatched_pers):
                px, py = persistent_balls[pid]['center']
                pr = persistent_balls[pid]['radius']
                dist = math.hypot(det['cx'] - px, det['cy'] - py)
                pairs.append((dist, det_id, pid))

        # sort pairs by distance ascending
        pairs.sort(key=lambda x: x[0])

        used_dets = set()
        used_pers = set()

        for dist, det_id, pid in pairs:
            if det_id in used_dets or pid in used_pers:
                continue
            det = unmatched_dets[det_id]
            pr = persistent_balls[pid]['radius']
            # adaptive threshold: allow reasonable movement but avoid wild swaps
            threshold = max(60, int((det['radius'] + pr) * 1.5))
            # Be a bit more tolerant if the persistent ball was seen very recently
            if frame_num - persistent_balls[pid]['last_seen'] <= 3:
                threshold = max(threshold, 120)
            if dist <= threshold:
                # accept match
                matched[det_id] = pid
                detection_to_persistent[det_id] = pid
                persistent_balls[pid]['center'] = (det['cx'], det['cy'])
                persistent_balls[pid]['radius'] = det['radius']
                persistent_balls[pid]['last_seen'] = frame_num
                persistent_balls[pid]['missed'] = 0
                used_dets.add(det_id)
                used_pers.add(pid)

        # remaining unmatched detections -> new persistent balls
        for det_id, det in list(unmatched_dets.items()):
            if det_id in used_dets:
                continue
            # create new persistent ball
            pid = next_ball_number
            next_ball_number += 1
            persistent_balls[pid] = {
                'center': (det['cx'], det['cy']),
                'radius': det['radius'],
                'last_seen': frame_num,
                'missed': 0
            }
            matched[det_id] = pid
            detection_to_persistent[det_id] = pid

    else:
        # Either no unmatched detections or no persistent balls to match: create new for unmatched detections
        for det_id, det in unmatched_dets.items():
            pid = next_ball_number
            next_ball_number += 1
            persistent_balls[pid] = {
                'center': (det['cx'], det['cy']),
                'radius': det['radius'],
                'last_seen': frame_num,
                'missed': 0
            }
            matched[det_id] = pid
            detection_to_persistent[det_id] = pid

    # Any persistent balls not matched this frame -> increment missed
    for pid in list(persistent_balls.keys()):
        # check if any detection mapped to it in matched
        still_seen = any(p == pid for p in matched.values())
        if not still_seen:
            persistent_balls[pid]['missed'] += 1
        # if missed too long, remove (but keep a conservative allowance)
        if persistent_balls[pid]['missed'] > MAX_FRAMES_WITHOUT_DETECTION * 3:
            # cleanup
            del persistent_balls[pid]
            # remove detection_to_persistent entries mapping to pid
            for d_id in list(detection_to_persistent.keys()):
                if detection_to_persistent.get(d_id) == pid:
                    del detection_to_persistent[d_id]
            # also remove trail and prev positions keyed by this pid
            trail_history.pop(pid, None)
            prev_positions.pop(f'ball_{pid}', None)
            smoothed_positions.pop(f'ball_{pid}', None)
            track_frame_count.pop(f'ball_{pid}', None)
            track_confidence_history.pop(f'ball_{pid}', None)

    return matched


def draw_skeleton(frame, keypoints, conf_threshold=0.4):
    for (p1, p2) in SKELETON:
        if keypoints[p1][2] > conf_threshold and keypoints[p2][2] > conf_threshold:
            pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
            pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
            cv2.circle(frame, pt1, 3, (0, 0, 255), -1)
            cv2.circle(frame, pt2, 3, (0, 0, 255), -1)


def associate_balls_to_people(balls, people):
    associations = set()
    for pid, pbbox in people.items():
        px1, py1, px2, py2 = pbbox
        for bid, (bcx, bcy) in balls.items():
            if px1 - 20 < bcx < px2 + 20 and py1 - 20 < bcy < py2 + 20:
                associations.add(pid)
    return associations


# ---------------- MAIN LOOP ---------------- #
print("Processing video frames...")
print(f"Enhanced settings - Resolution: {DETECTION_RESOLUTION}, Smoothing: {BBOX_SMOOTHING_FRAMES} frames")
print(f"Confidence thresholds - Person: {PERSON_CONFIDENCE_THRESHOLD}, Ball: {BALL_CONFIDENCE_THRESHOLD}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    current_frame_num += 1

    enhanced_frame = enhance_frame(frame)

    # Run detector + tracker + fallback detection
    results_tracking = detection_model.track(
        enhanced_frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        classes=[0, 32],
        imgsz=DETECTION_RESOLUTION,
        verbose=False,
        conf=min(PERSON_CONFIDENCE_THRESHOLD, BALL_CONFIDENCE_THRESHOLD),
        iou=NMS_IOU_THRESHOLD
    )
    results_detection = detection_model(
        enhanced_frame,
        classes=[32],
        imgsz=DETECTION_RESOLUTION,
        verbose=False,
        conf=BALL_CONFIDENCE_THRESHOLD * 0.8,
        iou=NMS_IOU_THRESHOLD
    )

    all_objects = []
    tracked_boxes = results_tracking[0].boxes

    if tracked_boxes is not None and len(tracked_boxes) > 0:
        track_ids = tracked_boxes.id.int().cpu().numpy() if tracked_boxes.id is not None else np.arange(len(tracked_boxes))
        bboxes = tracked_boxes.xyxy.int().cpu().numpy()
        class_ids = tracked_boxes.cls.int().cpu().numpy()
        confidences = tracked_boxes.conf.cpu().numpy()

        for track_id, bbox, cls_id, conf in zip(track_ids, bboxes, class_ids, confidences):
            all_objects.append({'id': int(track_id), 'bbox': bbox.astype(int), 'conf': float(conf), 'class': int(cls_id), 'tracked': True})

    # add static/untracked detections fallback
    untracked = assign_ids_to_untracked_detections(tracked_boxes, results_detection, current_frame_num)
    all_objects.extend([{**o, 'tracked': False} for o in untracked])

    # Build ball detections array
    ball_detections = []  # [{'det_id', 'bbox', 'cx','cy','radius','conf','tracked'}]
    person_boxes = {}
    for obj in all_objects:
        if obj['class'] == 32:
            x1, y1, x2, y2 = obj['bbox']
            cx, cy = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            radius = int(((x2 - x1) + (y2 - y1)) / 4)
            ball_detections.append({'det_id': obj['id'], 'bbox': obj['bbox'], 'cx': cx, 'cy': cy, 'radius': radius, 'conf': obj['conf'], 'tracked': obj['tracked']})
        elif obj['class'] == 0:
            person_boxes[obj['id']] = obj['bbox']

    # First-frame special handling: cluster overlapping detections and assign left-to-right IDs
    if not initial_balls_assigned and len(ball_detections) > 0 and current_frame_num == 1:
        clusters = cluster_initial_balls(ball_detections)
        # assign IDs left-to-right
        for cluster in clusters:
            pid = next_ball_number
            next_ball_number += 1
            persistent_balls[pid] = {
                'center': (cluster['cx'], cluster['cy']),
                'radius': cluster['radius'],
                'last_seen': current_frame_num,
                'missed': 0
            }
            # map all member detections to this pid
            for det_id in cluster['members']:
                detection_to_persistent[det_id] = pid
        initial_balls_assigned = True
    else:
        # Normal matching for subsequent frames
        matched = match_detections_to_persistent(ball_detections, current_frame_num)

    # Build a mapping of pid -> current detection center for associations
    current_ball_centers = {}
    for det in ball_detections:
        det_id = det['det_id']
        pid = detection_to_persistent.get(det_id)
        if pid is not None:
            current_ball_centers[pid] = (det['cx'], det['cy'])

    # Associate balls to people (to trigger pose even if person isn't moving much)
    interacting_people = associate_balls_to_people(current_ball_centers, person_boxes)

    # ---------------- Draw and update per-object state ----------------
    for obj in all_objects:
        det_id = obj['id']
        cls_id = obj['class']
        conf = obj['conf']
        bbox = obj['bbox']
        tracked = obj['tracked']

        if conf < (PERSON_CONFIDENCE_THRESHOLD if cls_id == 0 else BALL_CONFIDENCE_THRESHOLD):
            continue

        # determine entity id used for smoothing/state
        if cls_id == 32:
            pid = detection_to_persistent.get(det_id)
            if pid is None:
                # fallback: use det_id if no persistent mapping (should be rare)
                entity_id = f'ball_unmapped_{det_id}'
            else:
                entity_id = f'ball_{pid}'
        else:
            entity_id = f'person_{det_id}'

        # smooth bbox using entity_id
        if tracked:
            bbox = smooth_bbox(entity_id, bbox)
        x1, y1, x2, y2 = bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # smooth position
        smcx, smcy = smooth_position(entity_id, (cx, cy))

        # calculate movement distance using entity_id
        prev_cx, prev_cy = prev_positions.get(entity_id, (smcx, smcy))
        distance = math.hypot(smcx - prev_cx, smcy - prev_cy)
        prev_positions[entity_id] = (smcx, smcy)

        # update common stats
        track_frame_count[entity_id] = track_frame_count.get(entity_id, 0) + 1
        last_seen_frame[entity_id] = current_frame_num
        track_confidence_history[entity_id].append(conf)
        avg_confidence = np.mean(list(track_confidence_history[entity_id])) if track_confidence_history[entity_id] else conf

        # ---------- Ball rendering and persistence ----------
        if cls_id == 32:
            pid = detection_to_persistent.get(det_id)
            if pid is None:
                # skip weird unmatched detections
                continue
            # trail keyed by pid so it persists across detection-id churn
            trail_history.setdefault(pid, []).append((smcx, smcy))
            trail = trail_history[pid][-BALL_TRAIL_LENGTH:]

            is_moving = (distance > BALL_MOVEMENT_THRESHOLD and track_frame_count[entity_id] > NEW_TRACK_FRAME_THRESHOLD)
            color = (0, 255, 0) if is_moving else (0, 165, 255)

            # draw trail
            for i in range(1, len(trail)):
                alpha = (i / len(trail)) * 0.8 + 0.2
                pt1 = trail[i-1]
                pt2 = trail[i]
                thickness = max(1, int(2 * alpha))
                # simple faded trail (no overlay blending for speed)
                cv2.line(frame, pt1, pt2, color, thickness)

            # draw box and label using persistent pid
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Ball {pid}: {('Action' if is_moving else 'Static')} ({avg_confidence:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # ---------- Person rendering & pose ----------
        elif cls_id == 0:
            is_moving = distance > PERSON_MOVEMENT_THRESHOLD
            is_interacting = det_id in interacting_people

            if is_moving or is_interacting:
                # perform pose on the person
                px1, py1, px2, py2 = x1, y1, x2, y2
                person_crop = enhanced_frame[py1:py2, px1:px2]
                if person_crop.size > 0:
                    pose_res = pose_model(person_crop, verbose=False)
                    if pose_res and len(pose_res[0].keypoints.xy) > 0:
                        kpts = pose_res[0].keypoints
                        kpts_xy = kpts.xy[0].cpu().numpy()
                        confs_k = kpts.conf[0].cpu().numpy()
                        keypoints = np.hstack((kpts_xy, confs_k[:, None]))
                        keypoints[:, 0] += px1
                        keypoints[:, 1] += py1
                        draw_skeleton(frame, keypoints)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            txt = f"Person {det_id}: {('Action' if is_moving else 'Static') if det_id in interacting_people else ('Action' if is_moving else 'Static')}"
            cv2.putText(frame, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

    # Cleanup long-lost persistent balls (conservative)
    for pid in list(persistent_balls.keys()):
        if persistent_balls[pid]['missed'] > MAX_FRAMES_WITHOUT_DETECTION * 3:
            persistent_balls.pop(pid, None)
            trail_history.pop(pid, None)
            prev_positions.pop(f'ball_{pid}', None)
            smoothed_positions.pop(f'ball_{pid}', None)
            track_frame_count.pop(f'ball_{pid}', None)
            track_confidence_history.pop(f'ball_{pid}', None)
            # remove all detection_to_persistent entries mapping to pid
            for d in list(detection_to_persistent.keys()):
                if detection_to_persistent.get(d) == pid:
                    detection_to_persistent.pop(d, None)

    info_text = f"Frame: {current_frame_num} | Persistent balls: {len(persistent_balls)} | Enhanced: {'ON' if ENABLE_FRAME_ENHANCEMENT else 'OFF'}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(frame)

# ---------------- CLEANUP ---------------- #
cap.release()
out.release()
print(f"Output saved to: {OUTPUT_PATH}")
