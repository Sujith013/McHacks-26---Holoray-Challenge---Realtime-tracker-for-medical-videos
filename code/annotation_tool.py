"""Advanced medical video annotation and tracking tool for the HoloXR challenge."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackerContext:
    """State container for per-annotation tracking."""

    template: Optional[np.ndarray] = None
    template_size: Tuple[int, int] = (0, 0)
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    track_id: Optional[int] = None
    status: str = "initializing"
    lost_frames: int = 0
    total_offset: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    last_score: float = 0.0


@dataclass
class TrackState:
    """Internal state for a ByteTrack-managed track."""

    track_id: int
    annotation_id: int
    bbox: Tuple[float, float, float, float]
    score: float
    hits: int = 1
    time_since_update: int = 0
    state: str = "tracked"


def _iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two (x, y, w, h) boxes."""

    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    ax1, ay1 = x1, y1
    ax2, ay2 = x1 + w1, y1 + h1
    bx1, by1 = x2, y2
    bx2, by2 = x2 + w2, y2 + h2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = w1 * h1
    area_b = w2 * h2
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class ByteTracker:
    """Lightweight ByteTrack-inspired multi-object tracker.

    Tracks detections across frames using IoU matching and a simple buffer to
    retain lost tracks for a short time window.
    """

    def __init__(self, match_thresh: float = 0.4, track_buffer: int = 30) -> None:
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.tracks: List[TrackState] = []
        self.next_id: int = 1

    def update(
        self,
        detections: List[Tuple[int, Tuple[float, float, float, float], float]]
    ) -> Dict[int, TrackState]:
        """Update tracker with current detections.

        Args:
            detections: list of (annotation_id, bbox(x,y,w,h), score)

        Returns:
            Mapping from annotation_id to the latest TrackState.
        """

        # Age existing tracks.
        for track in self.tracks:
            track.time_since_update += 1
            if track.time_since_update > 0 and track.state == "tracked":
                track.state = "lost"

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))
        matches: List[Tuple[int, int]] = []

        # Greedy match detections to tracks by IoU.
        for det_idx in sorted(unmatched_dets, key=lambda idx: detections[idx][2], reverse=True):
            ann_id, det_bbox, _ = detections[det_idx]
            best_idx = -1
            best_iou = 0.0
            for track_idx in list(unmatched_tracks):
                track = self.tracks[track_idx]
                if track.annotation_id != ann_id:
                    continue
                iou_val = _iou(track.bbox, det_bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = track_idx
            if best_idx >= 0 and best_iou >= self.match_thresh:
                matches.append((best_idx, det_idx))
                unmatched_tracks.discard(best_idx)
                unmatched_dets.discard(det_idx)

        # Update matched tracks.
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            ann_id, det_bbox, det_score = detections[det_idx]
            track.bbox = det_bbox
            track.score = det_score
            track.hits += 1
            track.time_since_update = 0
            track.state = "tracked"

        # Handle unmatched detections (spawn new tracks).
        for det_idx in list(unmatched_dets):
            ann_id, det_bbox, det_score = detections[det_idx]
            new_track = TrackState(
                track_id=self.next_id,
                annotation_id=ann_id,
                bbox=det_bbox,
                score=det_score,
                hits=1,
                time_since_update=0,
                state="tracked"
            )
            self.tracks.append(new_track)
            self.next_id += 1

        # Handle unmatched tracks.
        for track_idx in list(unmatched_tracks):
            track = self.tracks[track_idx]
            if track.time_since_update > self.track_buffer:
                track.state = "removed"

        # Filter out removed tracks.
        self.tracks = [track for track in self.tracks if track.state != "removed"]

        return {track.annotation_id: track for track in self.tracks}


class MedicalAnnotationTool:
    """Interactive annotation + motion tracking tool built from scratch for HoloXR."""

    def __init__(self, dataset_root: Optional[str] = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self.project_root = base_dir
        self.dataset_root = Path(dataset_root).expanduser() if dataset_root else base_dir / "Dataset"
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.window_name = "HoloXR Annotation Tracker"

        self.current_frame: Optional[np.ndarray] = None
        self.frame_number: int = 0
        self.total_frames: int = 0
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0

        self.paused: bool = False
        self.tracking_enabled: bool = True

        self.annotations: List[Dict[str, object]] = []
        self.current_draft: List[List[int]] = []
        self.next_annotation_id: int = 1
        self.drawing: bool = False

        self.max_lost_frames: int = 20
        self.template_match_threshold: float = 0.6
        self.search_scale: float = 2.0
        self.byte_tracker = ByteTracker(match_thresh=0.3, track_buffer=self.max_lost_frames)

    # ------------------------------------------------------------------
    # Video lifecycle
    # ------------------------------------------------------------------
    def select_video(self) -> bool:
        """List available videos and let the user pick one."""

        print("\n" + "=" * 72)
        print("HoloXR :: Video Selection")
        print("=" * 72)

        if not self.dataset_root.exists():
            print(f"Dataset directory not found: {self.dataset_root}")
            return False

        dataset_root_str = str(self.dataset_root)
        video_files: List[tuple[str, str]] = []
        for root, _, files in os.walk(dataset_root_str):
            for file in files:
                if file.lower().endswith((".mp4", ".avi", ".mov")):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, dataset_root_str)
                    video_files.append((rel_path, full_path))

        if not video_files:
            print("No medical videos found. Check dataset path.")
            return False

        for idx, (rel_path, _) in enumerate(sorted(video_files), start=1):
            print(f"{idx:2d}. {rel_path}")

        while True:
            try:
                choice = input(f"\nSelect video (1-{len(video_files)}): ").strip()
                vid_idx = int(choice) - 1
                if 0 <= vid_idx < len(video_files):
                    self.video_path = sorted(video_files)[vid_idx][1]
                    print(f"\nSelected: {sorted(video_files)[vid_idx][0]}")
                    return True
                print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a numeric option.")
            except KeyboardInterrupt:
                print("\nCancelled by user.")
                return False

    def load_video(self) -> bool:
        """Open the selected video and cache basic properties."""

        if not self.video_path:
            print("No video selected yet.")
            return False

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap or not self.cap.isOpened():
            print(f"Failed to open video: {self.video_path}")
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("\n" + "=" * 72)
        print("HoloXR :: Video Properties")
        print("=" * 72)
        print(f"Path       : {self.video_path}")
        print(f"Resolution : {self.width} x {self.height}")
        print(f"Frame rate : {self.fps:.2f} fps")
        print(f"Frame count: {self.total_frames}")
        print(f"Duration   : {self.total_frames / max(self.fps, 1.0):.2f} s")

        return True
    
    # ------------------------------------------------------------------
    # Annotation creation
    # ------------------------------------------------------------------
    def mouse_callback(self, event: int, x: int, y: int, flags: int, _params: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_draft = [[x, y]]
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and (flags & cv2.EVENT_FLAG_LBUTTON):
            if not self.current_draft or (abs(self.current_draft[-1][0] - x) + abs(self.current_draft[-1][1] - y)) > 1:
                self.current_draft.append([x, y])
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if len(self.current_draft) >= 2:
                self.complete_annotation(rectangle_mode=False)
            else:
                self.current_draft.clear()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.current_draft.clear()
            self.drawing = False
    
    def complete_annotation(self, rectangle_mode: bool) -> None:
        if len(self.current_draft) < 2:
            print("Need at least two points to define a region.")
            return

        polygon = self._build_polygon(self.current_draft, rectangle_mode)
        meta = self._compute_annotation_metadata(polygon)

        annotation = {
            "id": self.next_annotation_id,
            "type": "rectangle" if rectangle_mode else "polygon",
            "polygon": polygon,
            "source_points": [pt[:] for pt in self.current_draft],
            "start_frame": self.frame_number,
            "timestamp": float(self.frame_number) / max(self.fps, 1.0),
            "metadata": meta,
            "tracker": TrackerContext()
        }

        self.annotations.append(annotation)
        self.next_annotation_id += 1
        print(f"\n✓ Annotation #{annotation['id']} created with {polygon.shape[0]} vertices")

        if self.current_frame is not None:
            self._initialize_tracker(annotation, self.current_frame)
        else:
            annotation["tracker"].status = "ready"

        self.current_draft.clear()
    
    # ------------------------------------------------------------------
    # Tracking logic
    # ------------------------------------------------------------------
    def _initialize_tracker(self, annotation: Dict[str, object], frame: np.ndarray) -> None:
        ctx: TrackerContext = annotation["tracker"]

        if frame is None or frame.size == 0:
            ctx.status = "unavailable"
            return

        bbox = self._bbox_from_polygon(annotation["polygon"])
        template = self._extract_template(frame, bbox)

        if template is None:
            ctx.template = None
            ctx.template_size = (0, 0)
            ctx.last_bbox = None
            ctx.status = "insufficient"
            return

        ctx.template = template
        ctx.template_size = (template.shape[1], template.shape[0])
        ctx.last_bbox = bbox
        ctx.status = "ready"
        ctx.lost_frames = 0
        ctx.last_score = 1.0
        ctx.total_offset = np.zeros(2, dtype=np.float32)

    def track_annotations(self, frame: np.ndarray) -> None:
        if not self.tracking_enabled:
            for ann in self.annotations:
                ann["tracker"].status = "paused"
            return

        detections = self._generate_detections(frame)
        track_map = self.byte_tracker.update(detections)

        for annotation in self.annotations:
            ctx: TrackerContext = annotation["tracker"]
            track_state = track_map.get(annotation["id"])

            if track_state and track_state.state == "tracked":
                ctx.track_id = track_state.track_id
                ctx.last_bbox = track_state.bbox
                ctx.last_score = track_state.score
                ctx.status = "tracking"
                ctx.lost_frames = 0
                self._apply_bbox_update(annotation, track_state.bbox)
            else:
                ctx.lost_frames += 1
                if ctx.lost_frames > self.max_lost_frames:
                    ctx.status = "lost"
                else:
                    ctx.status = "reinit"

    def _generate_detections(self, frame: np.ndarray) -> List[Tuple[int, Tuple[float, float, float, float], float]]:
        detections: List[Tuple[int, Tuple[float, float, float, float], float]] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for annotation in self.annotations:
            ctx: TrackerContext = annotation["tracker"]
            if ctx.template is None or ctx.template_size == (0, 0):
                continue

            bbox = ctx.last_bbox if ctx.last_bbox is not None else self._bbox_from_polygon(annotation["polygon"])
            match_bbox, score = self._match_template(gray, ctx, bbox)

            if match_bbox is None or score < self.template_match_threshold:
                ctx.status = "insufficient"
                continue

            ctx.last_bbox = match_bbox
            ctx.last_score = score
            detections.append((annotation["id"], match_bbox, score))

        return detections

    def _match_template(
        self,
        gray_frame: np.ndarray,
        ctx: TrackerContext,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
        x, y, w, h = bbox
        template = ctx.template
        if template is None:
            return None, 0.0

        search_w = int(w * self.search_scale)
        search_h = int(h * self.search_scale)
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        x1 = max(0, center_x - search_w // 2)
        y1 = max(0, center_y - search_h // 2)
        x2 = min(self.width, x1 + search_w)
        y2 = min(self.height, y1 + search_h)

        search_region = gray_frame[y1:y2, x1:x2]
        if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
            search_region = gray_frame
            x1, y1 = 0, 0

        
        blurred_region = cv2.GaussianBlur(search_region, (3, 3), 0)
        blurred_template = cv2.GaussianBlur(template, (3, 3), 0)
        result = cv2.matchTemplate(blurred_region, blurred_template, cv2.TM_CCOEFF_NORMED)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = (x1 + max_loc[0], y1 + max_loc[1])

        new_bbox = (
            float(top_left[0]),
            float(top_left[1]),
            float(ctx.template_size[0]),
            float(ctx.template_size[1])
        )

        return new_bbox, float(max_val)

    def _extract_template(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        x1 = int(max(0, x))
        y1 = int(max(0, y))
        x2 = int(min(self.width, x + w))
        y2 = int(min(self.height, y + h))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    def _bbox_from_polygon(self, polygon: np.ndarray) -> Tuple[float, float, float, float]:
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(polygon.astype(np.int32))
        return float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h)

    def _apply_bbox_update(self, annotation: Dict[str, object], bbox: Tuple[float, float, float, float]) -> None:
        x, y, w, h = bbox
        polygon = annotation["polygon"].astype(np.float32)
        if polygon.size == 0:
            return

        new_center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
        current_center = np.mean(polygon, axis=0)
        translation = new_center - current_center

        transformed = polygon + translation
        self._update_annotation_geometry(annotation, transformed)
    
    # ------------------------------------------------------------------
    # Geometry + metadata updates
    # ------------------------------------------------------------------
    def _build_polygon(self, points: List[List[int]], rectangle_mode: bool) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        if rectangle_mode and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            polygon = np.array(
                [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ],
                dtype=np.float32
            )
            return polygon
        if len(pts) >= 3:
            curve = pts.reshape(-1, 1, 2)
            epsilon = max(2.0, 0.01 * cv2.arcLength(curve, False))
            polygon = cv2.approxPolyDP(curve, epsilon, True).reshape(-1, 2)
            if polygon.shape[0] >= 3:
                return polygon.astype(np.float32)
            hull = cv2.convexHull(pts)
            if hull.shape[0] >= 3:
                return hull.reshape(-1, 2).astype(np.float32)
        if len(pts) >= 2:
            x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
            if w == 0:
                w = 2
            if h == 0:
                h = 2
            return np.array(
                [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ],
                dtype=np.float32
            )
        return pts

    def _compute_annotation_metadata(self, polygon: np.ndarray) -> Dict[str, object]:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

        region_pixels = self.current_frame[mask > 0] if self.current_frame is not None else np.empty((0, 3))
        if region_pixels.size == 0:
            region_pixels = np.zeros((1, 3), dtype=np.uint8)

        color_stats = {
            "mean_rgb": [float(np.mean(region_pixels[:, c])) for c in range(3)],
            "std_rgb": [float(np.std(region_pixels[:, c])) for c in range(3)],
            "median_rgb": [float(np.median(region_pixels[:, c])) for c in range(3)]
        }

        gray_pixels = cv2.cvtColor(region_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1)
        intensity_stats = {
            "mean": float(np.mean(gray_pixels)),
            "std": float(np.std(gray_pixels)),
            "min": int(np.min(gray_pixels)),
            "max": int(np.max(gray_pixels)),
            "median": float(np.median(gray_pixels))
        }

        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(polygon.astype(np.int32))
        center = np.mean(polygon, axis=0)
        area_pixels = float(cv2.contourArea(polygon.astype(np.float32)))

        return {
            "bounding_box": {
                "x": int(bbox_x),
                "y": int(bbox_y),
                "width": int(bbox_w),
                "height": int(bbox_h)
            },
            "initial_center": {"x": float(center[0]), "y": float(center[1])},
            "current_center": {"x": float(center[0]), "y": float(center[1])},
            "offset": {"x": 0.0, "y": 0.0},
            "area_pixels": area_pixels,
            "num_vertices": int(polygon.shape[0]),
            "color_statistics": color_stats,
            "intensity_statistics": intensity_stats,
            "region_pixel_count": int(mask.sum() // 255),
            "initial_polygon": polygon.tolist()
        }

    def _update_annotation_geometry(self, annotation: Dict[str, object], new_polygon: np.ndarray) -> None:
        new_polygon[:, 0] = np.clip(new_polygon[:, 0], 0, self.width - 1)
        new_polygon[:, 1] = np.clip(new_polygon[:, 1], 0, self.height - 1)

        annotation["polygon"] = new_polygon.astype(np.float32)

        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(annotation["polygon"].astype(np.int32))
        center = np.mean(annotation["polygon"], axis=0)

        meta = annotation["metadata"]
        meta["bounding_box"].update({
            "x": int(bbox_x),
            "y": int(bbox_y),
            "width": int(bbox_w),
            "height": int(bbox_h)
        })
        meta["current_center"] = {"x": float(center[0]), "y": float(center[1])}

        initial_center = meta["initial_center"]
        offset_x = center[0] - initial_center["x"]
        offset_y = center[1] - initial_center["y"]
        meta["offset"] = {"x": float(offset_x), "y": float(offset_y)}
        meta["area_pixels"] = float(cv2.contourArea(annotation["polygon"].astype(np.float32)))

        ctx: TrackerContext = annotation["tracker"]
        ctx.total_offset = np.array([offset_x, offset_y], dtype=np.float32)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()

        for annotation in self.annotations:
            polygon = annotation["polygon"].astype(np.int32)
            ctx: TrackerContext = annotation["tracker"]
            status = ctx.status

            if status == "tracking":
                color = (0, 200, 0)
            elif status == "reinit":
                color = (0, 200, 255)
            elif status == "insufficient":
                color = (0, 165, 255)
            elif status == "lost":
                color = (0, 0, 255)
            elif status == "paused":
                color = (180, 180, 180)
            else:
                color = (120, 120, 120)

            cv2.polylines(display, [polygon], True, color, 2)

            centroid = annotation["metadata"]["current_center"]
            label = f"#{annotation['id']} [{status.upper()}]"
            cv2.putText(
                display,
                label,
                (polygon[0][0], max(15, polygon[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

            offset = annotation["metadata"]["offset"]
            offset_text = f"Δ({offset['x']:.1f}, {offset['y']:.1f})"
            cv2.putText(
                display,
                offset_text,
                (polygon[0][0], polygon[0][1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA
            )

        if self.current_draft:
            draft_pts = np.array(self.current_draft, dtype=np.int32)
            for pt in draft_pts:
                cv2.circle(display, tuple(pt), 4, (255, 0, 0), -1)
            if len(draft_pts) > 1:
                cv2.polylines(display, [draft_pts], False, (255, 0, 0), 1)

        return display

    def add_overlay(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

        y = 28
        cv2.putText(display, "HoloXR Motion-Tracked Annotation Tool", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28
        cv2.putText(display, "SPACE: Pause/Play   T: Toggle Tracking   C: Clear Draft   S: Save   Q/ESC: Quit", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 24
        status_text = "PAUSED" if self.paused else "PLAYING"
        tracking_text = "ON" if self.tracking_enabled else "OFF"
        cv2.putText(display, f"Status: {status_text}   Tracking: {tracking_text}   Frame: {self.frame_number}/{self.total_frames}   FPS: {self.fps:.2f}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
        y += 22
        cv2.putText(display, f"Active annotations: {len(self.annotations)}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return display

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_annotations(self) -> None:
        if not self.annotations:
            print("\nNo annotations to save.")
            return

        output_dir = self.project_root / "annotations"
        output_dir.mkdir(exist_ok=True)

        video_name = Path(self.video_path).stem if self.video_path else "video"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{video_name}_tracked_annotations_{timestamp}.json"

        serializable: List[Dict[str, object]] = []
        for ann in self.annotations:
            ctx: TrackerContext = ann["tracker"]
            serializable.append({
                "id": ann["id"],
                "type": ann["type"],
                "polygon": ann["polygon"].astype(float).tolist(),
                "source_points": ann["source_points"],
                "start_frame": ann["start_frame"],
                "timestamp": ann["timestamp"],
                "metadata": ann["metadata"],
                "tracker_status": ctx.status,
                "tracker_offset": ctx.total_offset.astype(float).tolist()
            })

        payload = {
            "video_path": self.video_path,
            "video_properties": {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "total_frames": self.total_frames
            },
            "annotation_count": len(serializable),
            "annotations": serializable,
            "exported_at": datetime.now().isoformat()
        }

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        print(f"\n✓ Saved annotations to {output_path}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        if not self.cap:
            raise RuntimeError("Video not loaded. Call load_video() first.")

        print("\n" + "=" * 72)
        print("HoloXR :: Instructions")
        print("=" * 72)
        print("• Pause the video (SPACE) before drawing.")
        print("• Left click twice for rectangles, right click to close polygons.")
        print("• Tracking updates automatically when video resumes.")

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.paused = True

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("\nEnd of video reached.")
                    break
                self.current_frame = frame
                self.frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.track_annotations(self.current_frame)
            else:
                if self.current_frame is None:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    self.current_frame = frame
                    self.frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            if self.current_frame is None:
                continue

            display = self.draw_annotations(self.current_frame)
            display = self.add_overlay(display)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1 if self.paused else 10) & 0xFF

            if key == ord(' '):
                self.paused = not self.paused
                state = "PAUSED" if self.paused else "PLAYING"
                print(f"\n{state}")
            elif key in (ord('t'), ord('T')):
                self.tracking_enabled = not self.tracking_enabled
                state = "ENABLED" if self.tracking_enabled else "DISABLED"
                print(f"\nTracking {state}")
            elif key in (ord('c'), ord('C')):
                if self.current_draft:
                    self.current_draft.clear()
                    print("\nDraft cleared")
            elif key in (ord('s'), ord('S')):
                self.save_annotations()
            elif key in (ord('q'), 27):
                print("\nExiting...")
                break

        cv2.destroyAllWindows()
        if self.cap:
            self.cap.release()

        if self.annotations:
            save = input("\nSave annotations before exit? (y/n): ").strip().lower()
            if save == 'y':
                self.save_annotations()


def main() -> None:
    tool = MedicalAnnotationTool()
    if not tool.select_video():
        return
    if not tool.load_video():
        return
    tool.run()


if __name__ == "__main__":
    main()
    