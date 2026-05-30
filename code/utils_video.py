from pathlib import Path
import cv2
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_videos(input_dir):
    input_dir = Path(input_dir)
    extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV"]

    unique_videos = {}
    for ext in extensions:
        for video_path in input_dir.glob(ext):
            # On Windows, patterns such as *.mp4 and *.MP4 can match the same file.
            # Deduplicate using the resolved lowercase path.
            key = str(video_path.resolve()).lower()
            unique_videos[key] = video_path

    return sorted(unique_videos.values(), key=lambda p: p.name.lower())


def resize_frame(frame, max_side=768):
    h, w = frame.shape[:2]
    largest_side = max(h, w)

    if largest_side > max_side:
        ratio = max_side / largest_side
        new_w = int(w * ratio)
        new_h = int(h * ratio)
    else:
        new_w = w
        new_h = h

    # Even dimensions avoid video codec issues.
    if new_w % 2 == 1:
        new_w -= 1
    if new_h % 2 == 1:
        new_h -= 1

    new_w = max(new_w, 2)
    new_h = max(new_h, 2)

    if new_w != w or new_h != h:
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return frame


def write_video_from_frames(frames_paths, output_path, fps=24):
    frames_paths = list(frames_paths)
    if len(frames_paths) == 0:
        raise ValueError("No frame available to write the video.")

    first_frame = cv2.imread(str(frames_paths[0]))
    if first_frame is None:
        raise ValueError(f"Cannot read the first frame: {frames_paths[0]}")

    h, w = first_frame.shape[:2]

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame_path in frames_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Skipped unreadable frame: {frame_path}")
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)

    writer.release()


def video_stem(video_path):
    return Path(video_path).stem


def load_bgr_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def find_video_ids(frames_color_dir="frames/color"):
    base = Path(frames_color_dir)
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir() and list(p.glob("*.png"))])
