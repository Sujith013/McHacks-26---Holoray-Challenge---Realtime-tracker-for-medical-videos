import cv2
import os
import argparse

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError("Cannot open video")

    metadata = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }

    cap.release()
    return metadata


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--data_path", '-d', type=str, default="../Dataset", help="Path to the dataset directory")
    args = argparse.parse_args()
    data_path = args.data_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    for x in os.listdir(data_path):
        if x == ".DS_Store":
            continue

        for y in os.listdir(os.path.join(data_path, x)):
            if y == ".DS_Store":
                continue

            video_path = os.path.join(data_path, x, y)

            if os.path.exists(video_path):
                metadata = get_video_metadata(video_path)

                print(f"Video Metadata: {video_path}")
                for k, v in metadata.items():
                    print(f"{k}: {v}")
                print("\n")
            else:
                raise FileNotFoundError(f"Video file not found: {video_path}")