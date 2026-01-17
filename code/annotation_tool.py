"""Advanced medical video annotation and tracking tool for the HoloXR challenge."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


# Lucas-Kanade parameters tuned for medical video motion.
LK_PARAMS: Dict[str, object] = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)


@dataclass
class TrackerContext:
    """State container for per-annotation tracking."""

    prev_points: Optional[np.ndarray] = None
    status: str = "initializing"
    lost_frames: int = 0
    total_offset: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    reinit_frame: int = -1


class MedicalAnnotationTool:
    """Interactive annotation + motion tracking tool built from scratch for HoloXR."""

    def __init__(self, dataset_root: str = "/Users/santhosh/Desktop/hackathon/holoray/Dataset") -> None:
        self.dataset_root = dataset_root
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.window_name = "HoloXR Annotation Tracker"

        self.current_frame: Optional[np.ndarray] = None
        self.current_gray: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
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

        self.max_features: int = 120
        self.min_feature_threshold: int = 6
        self.forward_backward_tol: float = 1.5
        self.max_lost_frames: int = 20

    # ------------------------------------------------------------------
    # Video lifecycle
    # ------------------------------------------------------------------
    def select_video(self) -> bool:
        """List available videos and let the user pick one."""

        print("\n" + "=" * 72)
        print("HoloXR :: Video Selection")
        print("=" * 72)

        video_files: List[tuple[str, str]] = []
        for root, _, files in os.walk(self.dataset_root):
            for file in files:
                if file.lower().endswith((".mp4", ".avi", ".mov")):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.dataset_root)
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
    def mouse_callback(self, event: int, x: int, y: int, _flags: int, _params: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_draft.append([x, y])
            if len(self.current_draft) == 2:
                self.complete_annotation(rectangle_mode=True)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_draft) >= 3:
                self.complete_annotation(rectangle_mode=False)
            else:
                print("Need at least three points for polygon.")
    
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
            "tracker": TrackerContext(reinit_frame=self.frame_number)
        }

        self.annotations.append(annotation)
        self.next_annotation_id += 1
        print(f"\n✓ Annotation #{annotation['id']} created with {polygon.shape[0]} vertices")

        if self.current_gray is not None:
            self._initialize_tracker(annotation, self.current_gray)

        self.current_draft.clear()
    
    # ------------------------------------------------------------------
    # Tracking logic
    # ------------------------------------------------------------------
    def _initialize_tracker(self, annotation: Dict[str, object], gray: np.ndarray) -> None:
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [annotation["polygon"].astype(np.int32)], 255)

        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask
        )

        ctx: TrackerContext = annotation["tracker"]
        if features is not None and len(features) >= self.min_feature_threshold:
            ctx.prev_points = features.astype(np.float32)
            ctx.status = "tracking"
            ctx.lost_frames = 0
            ctx.total_offset = np.zeros(2, dtype=np.float32)
        else:
            ctx.prev_points = None
            ctx.status = "insufficient"
            ctx.lost_frames += 1

    def track_annotations(self, prev_gray: np.ndarray, gray: np.ndarray) -> None:
        if not self.tracking_enabled:
            for ann in self.annotations:
                ann["tracker"].status = "paused"
            return

        for annotation in self.annotations:
            ctx: TrackerContext = annotation["tracker"]

            if ctx.prev_points is None:
                if (self.frame_number - ctx.reinit_frame) >= 3:
                    ctx.reinit_frame = self.frame_number
                    self._initialize_tracker(annotation, prev_gray)
                continue

            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                ctx.prev_points,
                None,
                **LK_PARAMS
            )

            if next_points is None:
                ctx.prev_points = None
                ctx.status = "reinit"
                ctx.lost_frames += 1
                continue

            back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                prev_gray,
                next_points,
                None,
                **LK_PARAMS
            )

            fb_error = np.linalg.norm(ctx.prev_points - back_points, axis=2)
            good_mask = (
                (status.flatten() == 1)
                & (back_status.flatten() == 1)
                & (fb_error.flatten() < self.forward_backward_tol)
            )

            prev_good = ctx.prev_points[good_mask]
            next_good = next_points[good_mask]

            if prev_good is None or len(prev_good) < self.min_feature_threshold:
                ctx.prev_points = None
                ctx.status = "reinit"
                ctx.lost_frames += 1
                continue

            matrix, inliers = cv2.estimateAffinePartial2D(
                prev_good,
                next_good,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0
            )

            if matrix is None or inliers is None or int(inliers.sum()) < self.min_feature_threshold:
                ctx.prev_points = None
                ctx.status = "reinit"
                ctx.lost_frames += 1
                continue

            new_polygon = cv2.transform(
                annotation["polygon"].reshape(-1, 1, 2),
                matrix
            ).reshape(-1, 2)

            self._update_annotation_geometry(annotation, new_polygon)
            ctx.prev_points = next_good.reshape(-1, 1, 2).astype(np.float32)
            ctx.status = "tracking"
            ctx.lost_frames = 0

            if len(ctx.prev_points) < self.min_feature_threshold + 4:
                ctx.reinit_frame = self.frame_number
                self._initialize_tracker(annotation, gray)

            if ctx.lost_frames > self.max_lost_frames:
                ctx.status = "lost"
                ctx.prev_points = None
    
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
        cv2.putText(display, f"Status: {status_text}   Tracking: {tracking_text}   Frame: {self.frame_number}/{self.total_frames}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
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

        output_dir = Path("/Users/santhosh/Desktop/code/annotations")
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
        self.prev_gray = None

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("\nEnd of video reached.")
                    break
                self.current_frame = frame
                self.frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.current_gray = gray

                if self.prev_gray is not None:
                    self.track_annotations(self.prev_gray, gray)
                self.prev_gray = gray
            else:
                if self.current_frame is None:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    self.current_frame = frame
                    self.frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if self.current_gray is None and self.current_frame is not None:
                    self.current_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                if self.prev_gray is None and self.current_gray is not None:
                    self.prev_gray = self.current_gray.copy()

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
    
    def draw_annotations_on_frame(self, frame):
        """Draw all annotations on the current frame with CSRT tracking status."""
        display = frame.copy()
        
        # Draw all completed annotations
        for i, ann in enumerate(self.annotations):
            points = np.array(ann['points'], dtype=np.int32)
            ann_id = ann['id']
            
            # Color based on CSRT tracking status
            status = self.tracking_status.get(ann_id, 'unknown')
            if status == 'tracking':
                color = (0, 255, 0)  # Green - tracking active
                status_text = "CSRT"
            elif status == 'lost':
                color = (0, 0, 255)  # Red - tracking lost
                status_text = "LOST"
            elif status == 'initialized':
                color = (0, 255, 255)  # Yellow - initialized, waiting
                status_text = "INIT"
            elif status == 'ready':
                color = (200, 200, 200)  # Gray - ready to track
                status_text = "READY"
            else:
                color = (100, 100, 100)  # Dark gray - unknown
                status_text = "?"
            
            if not self.tracking_enabled:
                color = (128, 128, 128)  # Gray - tracking disabled
                status_text = "OFF"
            
            # Draw rectangle
            if ann['type'] == 'rectangle' and len(points) == 2:
                cv2.rectangle(display, tuple(points[0]), tuple(points[1]), color, 2)
                cv2.putText(display, f"#{i+1}", tuple(points[0]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.polylines(display, [points], True, color, 2)
                cv2.putText(display, f"#{i+1}", tuple(points[0]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show offset and tracking status
            offset_text = f"({ann['offset']['x']:.0f},{ann['offset']['y']:.0f}) [{status_text}]"
            cv2.putText(display, offset_text, (points[0][0], points[0][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw current annotation being drawn
        if len(self.current_annotation) > 0:
            points = np.array(self.current_annotation, dtype=np.int32)
            for i, pt in enumerate(points):
                cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
                # Show point numbers
                cv2.putText(display, str(i+1), (pt[0]+10, pt[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            if len(points) > 1:
                # Draw lines connecting points
                cv2.polylines(display, [points], False, (0, 0, 255), 2)
                
                # Show preview of closing line for polygons (3+ points)
                if len(points) >= 3:
                    cv2.line(display, tuple(points[-1]), tuple(points[0]), (255, 0, 255), 1)
        
        return display
    
    def add_ui_overlay(self, frame):
        """Add UI instructions and CSRT tracking info overlay."""
        display = frame.copy()
        
        # Semi-transparent panel at top
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        # Instructions
        y = 25
        cv2.putText(display, "CONTROLS:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        cv2.putText(display, "SPACE: Pause/Play  |  LEFT CLICK: Add Point  |  RIGHT CLICK: Complete", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20
        cv2.putText(display, "T: Toggle Tracking  |  C: Clear Current  |  S: Save  |  Q/ESC: Quit", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Status and Algorithm Info
        y += 25
        status = "PAUSED" if self.paused else "PLAYING"
        status_color = (0, 255, 255) if self.paused else (0, 255, 0)
        tracking_status = "ON" if self.tracking_enabled else "OFF"
        cv2.putText(display, f"Status: {status}  |  CSRT-Like Tracking: {tracking_status}  |  Frame: {self.frame_number}/{self.total_frames}  |  Annotations: {len(self.annotations)}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # CSRT Algorithm Info
        y += 18
        csrt_info = "CSRT-Like: Dense Optical Flow + Kalman Filtering"
        cv2.putText(display, csrt_info, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        
        return display
        return display
    
    def run(self):
        """Run the annotation tool."""
        # Select and load video (skip selection if path already set)
        if not self.video_path:
            if not self.select_video():
                return
        
        if not self.load_video():
            return
        
        print("\n" + "=" * 60)
        print("ANNOTATION TOOL STARTED")
        print("=" * 60)
        print("\nCONTROLS:")
        print("  SPACE        - Pause/Play video")
        print("  LEFT CLICK   - Add annotation point")
        print("  RIGHT CLICK  - Complete annotation (2+ points)")
        print("  C            - Clear current annotation")
        print("  S            - Save annotations to JSON")
        print("  Q or ESC     - Quit")
        print("\nTIP: Pause the video (SPACE) before annotating")
        print("=" * 60 + "\n")
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Start paused
        self.paused = True
        
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("\nEnd of video reached.")
                    break
                self.current_frame = frame
                self.frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            if self.current_frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.current_frame = frame
                self.frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Track annotations using CSRT
            if not self.paused and self.tracking_enabled:
                self.track_annotations(self.current_frame)
            
            # Draw annotations and UI
            display = self.draw_annotations_on_frame(self.current_frame)
            display = self.add_ui_overlay(display)
            
            cv2.imshow(self.window_name, display)
            
            # Handle keyboard input
            key = cv2.waitKey(30 if not self.paused else 1) & 0xFF
            
            if key == ord(' '):  # Space - toggle pause
                self.paused = not self.paused
                print(f"\n{'PAUSED' if self.paused else 'PLAYING'}")
            
            elif key == ord('t') or key == ord('T'):  # Toggle tracking
                self.tracking_enabled = not self.tracking_enabled
                print(f"\nCSRT Tracking {'ENABLED' if self.tracking_enabled else 'DISABLED'}")
            
            elif key == ord('c') or key == ord('C'):  # Clear current annotation
                if self.current_annotation:
                    self.current_annotation = []
                    self.drawing = False
                    print("\nCurrent annotation cleared")
            
            elif key == ord('s') or key == ord('S'):  # Save annotations
                self.save_annotations()
            
            elif key == ord('q') or key == 27:  # Q or ESC - quit
                print("\nQuitting...")
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Ask to save before exit
        if self.annotations:
            print(f"\nYou have {len(self.annotations)} annotation(s).")
            save = input("Save annotations before exit? (y/n): ")
            if save.lower() == 'y':
                self.save_annotations()
    
    def save_annotations(self):
        """Save annotations to JSON file."""
        if not self.annotations:
            print("\nNo annotations to save!")
            return
        
        # Create output directory
        output_dir = Path("/Users/santhosh/Desktop/code/annotations")
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        video_name = Path(self.video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{video_name}_annotations_{timestamp}.json"
        
        # Prepare data
        data = {
            'video_path': self.video_path,
            'video_name': Path(self.video_path).name,
            'video_properties': {
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'total_frames': self.total_frames
            },
            'annotation_count': len(self.annotations),
            'annotations': self.annotations,
            'created_at': datetime.now().isoformat()
        }
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Annotations saved to: {output_file}")
        print(f"  Total annotations: {len(self.annotations)}")
        
        # Print summary
        print("\nAnnotation Summary:")
        for i, ann in enumerate(self.annotations, 1):
            print(f"  #{i}: Frame {ann['frame']}, Type: {ann['type']}, Points: {len(ann['points'])}")


def main():
    """Main entry point."""
    import sys
    
    tool = AnnotationTool()
    
    # If video path provided as argument, use it directly
    if len(sys.argv) > 1:
        tool.video_path = sys.argv[1]
        print(f"\nUsing video: {tool.video_path}")
    
    tool.run()


if __name__ == "__main__":
    main()
