import cv2
import sys
import numpy as np

# --- CONFIGURATION ---
VIDEO_PATH = 'Lapchole3.mp4' 
TRACKER_TYPE = 'CSRT'
MATCH_THRESHOLD = 0.7 # Adjust based on how much the anatomy changes shape

def create_tracker():
    return cv2.TrackerCSRT_create()

def preprocess_frame(frame):
    # Standard medical image enhancement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def main():
    video = cv2.VideoCapture(VIDEO_PATH)
    if not video.isOpened():
        sys.exit()

    ret, frame = video.read()
    if not ret: sys.exit()

    processed_frame = preprocess_frame(frame)

    # 1. INITIAL SELECTION
    print("Draw box around anatomy and press SPACE/ENTER.")
    bbox = cv2.selectROI("HoloRay Tracker", processed_frame, False)
    cv2.destroyWindow("HoloRay Tracker")

    # 2. SAVE TEMPLATE FOR RE-ENTRY
    x, y, w, h = [int(v) for v in bbox]
    template = processed_frame[y:y+h, x:x+w]
    
    tracker = create_tracker()
    tracker.init(processed_frame, bbox)
    is_tracking = True

    while True:
        ret, frame = video.read()
        
        # --- VIDEO LOOP LOGIC ---
        if not ret:
            print("Looping video...")
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed_frame = preprocess_frame(frame)
        timer = cv2.getTickCount()

        if is_tracking:
            # NORMAL TRACKING MODE
            success, bbox = tracker.update(processed_frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "LOCKED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                is_tracking = False # Target lost
        
        if not is_tracking:
            # SEARCH MODE: Re-scanning for anatomy
            cv2.putText(frame, "SEARCHING FOR RE-ENTRY...", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Use Template Matching to find target anywhere in frame
            res = cv2.matchTemplate(processed_frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > MATCH_THRESHOLD:
                bbox = (max_loc[0], max_loc[1], w, h)
                tracker = create_tracker() # Re-create and re-init
                tracker.init(processed_frame, bbox)
                is_tracking = True
                print(f"Target Re-acquired! Conf: {max_val:.2f}")

        # Performance Display
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imshow("HoloRay Robust Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): # Manual recalibration
            bbox = cv2.selectROI("HoloRay Robust Tracker", processed_frame, False)
            x, y, w, h = [int(v) for v in bbox]
            template = processed_frame[y:y+h, x:x+w]
            tracker = create_tracker()
            tracker.init(processed_frame, bbox)
            is_tracking = True

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()