from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from utils_video import ensure_dir, write_video_from_frames


def construire_table_chrominance(reference_bgr):
    ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB)
    L = ref_lab[:, :, 0]
    A = ref_lab[:, :, 1]
    B = ref_lab[:, :, 2]

    table_a = np.zeros(256, dtype=np.float32)
    table_b = np.zeros(256, dtype=np.float32)

    global_a = float(np.mean(A))
    global_b = float(np.mean(B))

    for valeur_l in range(256):
        masque = np.abs(L.astype(np.int16) - valeur_l) <= 2

        if np.any(masque):
            table_a[valeur_l] = np.mean(A[masque])
            table_b[valeur_l] = np.mean(B[masque])
        else:
            table_a[valeur_l] = global_a
            table_b[valeur_l] = global_b

    return table_a, table_b


def coloriser_frame(gray_bgr, table_a, table_b):
    lab = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    lab[:, :, 1] = table_a[L].astype(np.uint8)
    lab[:, :, 2] = table_b[L].astype(np.uint8)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def traiter_video(video_id, color_frames_dir, gray_frames_dir, output_dir, fps=24):
    color_frames_dir = Path(color_frames_dir) / video_id
    gray_frames_dir = Path(gray_frames_dir) / video_id
    output_frames_dir = Path(output_dir) / video_id / "frames"
    output_video_path = Path(output_dir) / video_id / f"{video_id}_colour_luminance_lookup.mp4"

    ensure_dir(output_frames_dir)

    color_frames = sorted(color_frames_dir.glob("*.png"))
    gray_frames = sorted(gray_frames_dir.glob("*.png"))

    if len(color_frames) == 0 or len(gray_frames) == 0:
        raise RuntimeError(f"Frames manquantes pour {video_id}")

    reference = cv2.imread(str(color_frames[0]))
    table_a, table_b = construire_table_chrominance(reference)

    output_paths = []

    print(f"\nColourisation luminance lookup : {video_id}")

    for i, frame_path in enumerate(tqdm(gray_frames)):
        gray_bgr = cv2.imread(str(frame_path))
        result = coloriser_frame(gray_bgr, table_a, table_b)

        out_path = output_frames_dir / f"frame_{i:05d}.png"
        cv2.imwrite(str(out_path), result)
        output_paths.append(out_path)

    write_video_from_frames(output_paths, output_video_path, fps=fps)
    print(f"Written video : {output_video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--color_frames_dir", default="frames/color")
    parser.add_argument("--gray_frames_dir", default="frames/gray")
    parser.add_argument("--output_dir", default="outputs/colourisation/luminance_lookup")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    traiter_video(
        video_id=args.video_id,
        color_frames_dir=args.color_frames_dir,
        gray_frames_dir=args.gray_frames_dir,
        output_dir=args.output_dir,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
