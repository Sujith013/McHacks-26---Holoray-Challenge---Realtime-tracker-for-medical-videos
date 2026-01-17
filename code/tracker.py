import cv2
import os
import json
import argparse

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



def create_tracker(tracker_type="CSRT"):
    if tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()
    elif tracker_type == "KCF":
        return cv2.TrackerKCF_create()
    elif tracker_type == "MOSSE":
        return cv2.TrackerMOSSE_create()
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        "-v",
        default="../Dataset/Lapchole/Lapchole1.mp4",
        type=str,
        help="Path to video file"
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    cv2.namedWindow("Tracker")
    cv2.setMouseCallback("Tracker", mouse_handler)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_frame = frame.copy()

            if tracking_active:
                success, bbox = tracker.update(current_frame)

                if success:
                    x, y, w, h = map(int, bbox)
                    tracked_bboxes[frame_idx] = [x, y, w, h]

                    cv2.rectangle(
                        current_frame,
                        (x, y),
                        (x + w, y + h),
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

                tracker = create_tracker("CSRT")
                tracker.init(current_frame, tuple(current_bbox))

                tracking_active = True
                no_more_pause = True

                tracked_bboxes[frame_idx] = current_bbox
                current_bbox = None

                print(f"[INFO] Tracking started at frame {frame_idx}")

    cap.release()
    cv2.destroyAllWindows()

    os.makedirs("../Annotations", exist_ok=True)

    with open("../Annotations/initial_annotation.json", "w") as f:
        json.dump(annotations, f, indent=2)

    with open("../Annotations/tracked_boxes.json", "w") as f:
        json.dump(tracked_bboxes, f, indent=2)

    print("Annotations and tracked boxes saved.")