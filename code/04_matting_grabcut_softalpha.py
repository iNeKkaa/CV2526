from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from utils_video import ensure_dir, write_video_from_frames


def grabcut_soft_alpha(frame_bgr, rect_ratio=(0.10, 0.05, 0.80, 0.90), iterations=5):
    h, w = frame_bgr.shape[:2]

    x_ratio, y_ratio, width_ratio, height_ratio = rect_ratio
    x = int(x_ratio * w)
    y = int(y_ratio * h)
    rw = int(width_ratio * w)
    rh = int(height_ratio * h)

    rect = (x, y, rw, rh)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(frame_bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    alpha = cv2.GaussianBlur(binary, (31, 31), 0)
    alpha_float = alpha.astype(np.float32) / 255.0

    return alpha_float


def composer_sur_fond(frame_bgr, alpha, background_color=(30, 144, 255)):
    bg = np.zeros_like(frame_bgr, dtype=np.float32)
    bg[:, :] = background_color[::-1]  # input RGB, OpenCV BGR

    alpha_3 = alpha[:, :, None]
    comp = frame_bgr.astype(np.float32) * alpha_3 + bg * (1.0 - alpha_3)
    return np.clip(comp, 0, 255).astype(np.uint8)


def traiter_video(video_id, input_frames_dir, output_dir, fps=24, rect_ratio=(0.10, 0.05, 0.80, 0.90), iterations=5):
    input_frames_dir = Path(input_frames_dir) / video_id

    output_frames_dir = Path(output_dir) / video_id / "composition_frames"
    output_alpha_dir = Path(output_dir) / video_id / "alpha_frames"

    output_video_path = Path(output_dir) / video_id / f"{video_id}_grabcut_composition.mp4"
    output_alpha_video_path = Path(output_dir) / video_id / f"{video_id}_grabcut_alpha.mp4"

    ensure_dir(output_frames_dir)
    ensure_dir(output_alpha_dir)

    frame_paths = sorted(input_frames_dir.glob("*.png"))
    if len(frame_paths) == 0:
        raise RuntimeError(f"No frame found for {video_id}")

    comp_paths = []
    alpha_paths = []

    print(f"\nMatting baseline GrabCut + soft alpha : {video_id}")

    for i, frame_path in enumerate(tqdm(frame_paths)):
        frame = cv2.imread(str(frame_path))

        alpha = grabcut_soft_alpha(frame, rect_ratio=rect_ratio, iterations=iterations)
        comp = composer_sur_fond(frame, alpha)

        alpha_gray = np.clip(alpha * 255, 0, 255).astype(np.uint8)
        alpha_bgr = cv2.cvtColor(alpha_gray, cv2.COLOR_GRAY2BGR)

        comp_path = output_frames_dir / f"frame_{i:05d}.png"
        alpha_path = output_alpha_dir / f"frame_{i:05d}.png"

        cv2.imwrite(str(comp_path), comp)
        cv2.imwrite(str(alpha_path), alpha_bgr)

        comp_paths.append(comp_path)
        alpha_paths.append(alpha_path)

    write_video_from_frames(comp_paths, output_video_path, fps=fps)
    write_video_from_frames(alpha_paths, output_alpha_video_path, fps=fps)

    print(f"Composition GrabCut : {output_video_path}")
    print(f"Alpha GrabCut : {output_alpha_video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--input_frames_dir", default="frames/color")
    parser.add_argument("--output_dir", default="outputs/matting/grabcut_softalpha")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--rect", nargs=4, type=float, default=[0.10, 0.05, 0.80, 0.90])
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    traiter_video(
        video_id=args.video_id,
        input_frames_dir=args.input_frames_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        rect_ratio=tuple(args.rect),
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
