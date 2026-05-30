from pathlib import Path
import argparse
import cv2
from tqdm import tqdm

from utils_video import ensure_dir, list_videos, resize_frame, write_video_from_frames


def prepare_video(video_path, output_dir, frames_dir, gray_dir, duration=5, fps=24, max_side=768, start_time=0.0):
    video_path = Path(video_path)
    video_id = video_path.stem

    output_video_path = Path(output_dir) / f"{video_id}_5s_{fps}fps.mp4"
    frames_video_dir = Path(frames_dir) / video_id
    gray_video_dir = Path(gray_dir) / video_id

    ensure_dir(output_video_path.parent)
    ensure_dir(frames_video_dir)
    ensure_dir(gray_video_dir)

    # If preprocessing is run again, remove old frames first.
    # Otherwise later scripts could mix old and new frames.
    for old_frame in frames_video_dir.glob("*.png"):
        old_frame.unlink()
    for old_frame in gray_video_dir.glob("*.png"):
        old_frame.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open : {video_path}")

    total_frames = int(duration * fps)
    saved_frame_paths = []

    print(f"\nPreparing {video_path.name}")
    print(f"→ duration = {duration}s | fps = {fps} | frames = {total_frames} | start = {start_time}s")

    for i in tqdm(range(total_frames)):
        timestamp_ms = (start_time + i / fps) * 1000.0
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

        ok, frame = cap.read()
        if not ok:
            print(f"Reading stopped at frame {i}.")
            break

        frame = resize_frame(frame, max_side=max_side)

        frame_path = frames_video_dir / f"frame_{i:05d}.png"
        gray_path = gray_video_dir / f"frame_{i:05d}.png"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(gray_path), gray_bgr)

        saved_frame_paths.append(frame_path)

    cap.release()

    write_video_from_frames(saved_frame_paths, output_video_path, fps=fps)
    print(f"Prepared video : {output_video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="input_videos")
    parser.add_argument("--output_dir", default="processed_videos")
    parser.add_argument("--frames_dir", default="frames/color")
    parser.add_argument("--gray_dir", default="frames/gray")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--start_time", type=float, default=1.0)
    args = parser.parse_args()

    videos = list_videos(args.input_dir)
    if len(videos) == 0:
        raise RuntimeError("No video found in input_videos/")

    print(f"Detected videos : {len(videos)}")
    for video_path in videos:
        print(f"- {video_path.name}")

    for video_path in videos:
        prepare_video(
            video_path=video_path,
            output_dir=args.output_dir,
            frames_dir=args.frames_dir,
            gray_dir=args.gray_dir,
            duration=args.duration,
            fps=args.fps,
            max_side=args.max_side,
            start_time=args.start_time,
        )


if __name__ == "__main__":
    main()
