import cv2
import os
import json
import argparse
import sys
import torch


OSTRACK_ROOT = '/Users/sujith/Projects/McHacks 26 - Holoray Challenge/OSTrack'

# Add OSTrack to Python path
if not os.path.exists(OSTRACK_ROOT):
    raise FileNotFoundError(f"OSTrack directory not found at:  {OSTRACK_ROOT}\nPlease update OSTRACK_ROOT variable.")

sys.path.insert(0, OSTRACK_ROOT)
print(f"Added OSTrack path:  {OSTRACK_ROOT}")

# Now import OSTrack modules
try:
    from lib.test.tracker.ostrack import OSTrack
    from lib.test.utils import TrackerParams
    from lib.config.ostrack.config import cfg, update_config_from_file
    from lib.train.data.processing_utils import sample_target
    from lib.test.tracker.data_utils import Preprocessor
    print("OSTrack modules imported successfully!")
except ImportError as e:
    print(f"Error importing OSTrack modules: {e}")
    print(f"Make sure OSTrack is properly installed at: {OSTRACK_ROOT}")
    sys.exit(1)

# Global variables
paused = False
drawing = False
start_point = None
current_mouse_pos = None
current_bbox = None

frame_idx = 0
current_frame = None

tracker = None
tracking_active = False
no_more_pause = False

annotations = {}
tracked_bboxes = {}


def mouse_handler(event, x, y, flags, param):
    global drawing, start_point, current_mouse_pos, current_bbox

    if not paused:
        return

    if event == cv2.EVENT_LBUTTONDOWN: 
        drawing = True
        start_point = (x, y)
        current_mouse_pos = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_mouse_pos = (x, y)

    elif event == cv2.EVENT_LBUTTONUP: 
        drawing = False
        x0, y0 = start_point
        x1, y1 = current_mouse_pos
        current_bbox = [x0, y0, x1 - x0, y1 - y0]
        current_mouse_pos = None


def initialize_ostrack(yaml_name='vitb_256_mae_ce_32x4_ep300'):
    """
    Initialize OSTrack tracker
    
    Args:
        yaml_name: Configuration file name (without .yaml extension)
    """
    print(f"Initializing OSTrack with config: {yaml_name}")
    
    # Set up parameters
    params = TrackerParams()
    
    # Update config from yaml file
    yaml_file = os.path.join(OSTRACK_ROOT, 'experiments/ostrack', f'{yaml_name}.yaml')
    
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Config file not found: {yaml_file}")
    
    print(f"Loading config from: {yaml_file}")
    update_config_from_file(yaml_file)
    params.cfg = cfg
    
    # Template and search region parameters
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    
    # Network checkpoint path
    checkpoint_path = os.path.join(
        OSTRACK_ROOT, 
        "output/checkpoints/train/ostrack",
        yaml_name,
        f"OSTrack_ep{cfg.TEST.EPOCH:04d}.pth.tar"
    )
    
    params.checkpoint = checkpoint_path
    
    # Check if checkpoint exists
    if not os.path.exists(params.checkpoint):
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"Checkpoint not found:  {params.checkpoint}\n"
            f"{'='*60}\n"
            f"Please download the model weights:\n"
            f"1. Visit:  https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd\n"
            f"2. Download:  OSTrack_ep0300.pth.tar (for {yaml_name})\n"
            f"3. Place it in: {os.path.dirname(checkpoint_path)}/\n"
            f"{'='*60}\n"
        )
    
    print(f"Loading checkpoint from: {params.checkpoint}")
    params.save_all_boxes = False
    params.debug = 0
    
    # Check CUDA availability and set device
    if torch.cuda.is_available():
        print(f"GPU detected:  {torch.cuda.get_device_name(0)}")
        params.device = 'cuda'
    else:
        print("WARNING: No GPU detected. OSTrack will run very slowly on CPU!")
        params.device = 'cpu'
    
    # Create OSTrack instance
    tracker = OSTrack(params, dataset_name='video')
    print("OSTrack initialized successfully!")
    
    return tracker


def init_tracker_with_bbox(tracker, frame, bbox):
    """
    Initialize OSTrack with bounding box
    
    Args:
        tracker: OSTrack instance
        frame: Current frame (BGR format from cv2)
        bbox: [x, y, w, h] bounding box
    """
    # OSTrack expects RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize tracker
    init_info = {'init_bbox': bbox}
    tracker.initialize(frame_rgb, init_info)
    
    return tracker


def track_frame(tracker, frame):
    """
    Track object in current frame
    
    Args: 
        tracker: OSTrack instance
        frame: Current frame (BGR format from cv2)
        
    Returns:
        bbox: [x, y, w, h] or None if tracking failed
    """
    try:
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Track
        out = tracker.track(frame_rgb)
        
        if 'target_bbox' in out: 
            bbox = out['target_bbox']
            # OSTrack returns [x, y, w, h]
            return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        else:
            return None
    except Exception as e:
        print(f"Tracking error: {e}")
        return None


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        "-v",
        default="../Dataset/Lapchole/Lapchole1.mp4",
        type=str,
        help="Path to video file"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="vitb_256_mae_ce_32x4_ep300",
        type=str,
        choices=["vitb_256_mae_ce_32x4_ep300", "vitb_384_mae_ce_32x4_ep300"],
        help="OSTrack model config"
    )
    args = parser.parse_args()

    # Initialize OSTrack
    print("\n" + "="*60)
    print("Initializing OSTrack Tracker")
    print("="*60)
    
    try:
        ostrack_instance = initialize_ostrack(args.model)
    except Exception as e:
        print(f"\nFailed to initialize OSTrack: {e}")
        sys.exit(1)
    
    print("="*60 + "\n")

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video:  {args.video_path}")

    cv2.namedWindow("Tracker")
    cv2.setMouseCallback("Tracker", mouse_handler)

    print("Controls:")
    print("  - SPACE: Pause/Resume")
    print("  - ESC:  Exit")
    print("  - When paused:  Draw bounding box, then press SPACE to start tracking\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_frame = frame. copy()

            if tracking_active:
                # Use OSTrack to track
                bbox = track_frame(tracker, current_frame)

                if bbox is not None: 
                    x, y, w, h = bbox
                    tracked_bboxes[frame_idx] = [x, y, w, h]

                    cv2.rectangle(
                        current_frame,
                        (x, y),
                        (x + w, y + h),
                        (255, 0, 0),
                        2
                    )
                    
                    # Add tracking info
                    cv2.putText(
                        current_frame,
                        f"Frame:  {frame_idx}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )
                else:
                    print(f"[INFO] Tracking lost at frame {frame_idx}")
                    tracking_active = False

        display = current_frame.copy()

        # -------- LIVE ANNOTATION PREVIEW --------
        if paused and drawing and start_point and current_mouse_pos:
            cv2.rectangle(
                display,
                start_point,
                current_mouse_pos,
                (0, 255, 0),
                2
            )
            cv2.putText(
                display,
                "Draw ROI and release mouse",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # -------- FINAL ANNOTATION --------
        elif paused and current_bbox is not None:
            x, y, w, h = current_bbox
            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
            cv2.putText(
                display,
                "Press SPACE to start tracking",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Tracker", display)

        key = cv2.waitKey(1 if paused else 30) & 0xFF

        if key == 27:  # ESC Key
            break

        elif key == ord(' ') and not no_more_pause:
            paused = not paused

            if not paused and current_bbox is not None and not tracking_active:
                annotations[frame_idx] = {
                    "type": "bbox",
                    "coords": current_bbox
                }

                # Initialize OSTrack with the selected bbox
                print(f"[INFO] Initializing tracker with bbox: {current_bbox}")
                tracker = init_tracker_with_bbox(
                    ostrack_instance, 
                    current_frame, 
                    current_bbox
                )

                tracking_active = True
                no_more_pause = True

                tracked_bboxes[frame_idx] = current_bbox
                current_bbox = None

                print(f"[INFO] OSTrack tracking started at frame {frame_idx}")

    cap.release()
    cv2.destroyAllWindows()

    # Save results
    os.makedirs("../Annotations", exist_ok=True)

    with open("../Annotations/initial_annotation.json", "w") as f:
        json. dump(annotations, f, indent=2)

    with open("../Annotations/tracked_boxes.json", "w") as f:
        json.dump(tracked_bboxes, f, indent=2)

    print(f"\n[INFO] Saved {len(tracked_bboxes)} tracked boxes")
    print("[INFO] Annotations and tracked boxes saved to ../Annotations/")