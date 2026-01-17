import cv2
import os
import json
import argparse

paused = False
drawing = False
start_point = None
current_bbox = None
annotations = {}
frame_idx = 0
current_frame = None


# ---------------- MOUSE CALLBACK ----------------
def mouse_handler(event, x, y, flags, param):
    global drawing, start_point, current_bbox, current_frame

    if not paused:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = current_frame.copy()
        cv2.rectangle(temp, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Annotator", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x0, y0 = start_point
        current_bbox = [x0, y0, x - x0, y - y0]

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--video_path", '-v', default="../Dataset/Lapchole/Lapchole1.mp4", type=str, required=True, help="Name of the video file to annotate")
    args = argparse.parse_args()
    video_path = args.video_path

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError("Cannot open video")
    
    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", mouse_handler)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_frame = frame.copy()
    
        display = current_frame.copy()
    
        if paused and current_bbox is not None:
            x, y, w, h = current_bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        cv2.imshow("Annotator", display)
        key = cv2.waitKey(30) & 0xFF
    
        if key == 27:  # ESC
            break
        
        elif key == ord(' '):
            paused = not paused
    
            # When resuming, save annotation if exists
            if not paused and current_bbox is not None:
                annotations[frame_idx] = {
                    "type": "bbox",
                    "coords": current_bbox
                }
                print(f"Saved annotation at frame {frame_idx}: {current_bbox}")
                current_bbox = None
    
    
    cap.release()
    cv2.destroyAllWindows()
    
    # ---------------- SAVE ANNOTATIONS ----------------
    with open("annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)
    
    print("Annotations saved to annotations.json")