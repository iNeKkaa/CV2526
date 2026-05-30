from pathlib import Path
import argparse
import cv2
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from utils_video import ensure_dir, write_video_from_frames


def calculer_features_luminance(l_channel):
    l_float = l_channel.astype(np.float32)

    local_mean = cv2.GaussianBlur(l_float, (0, 0), sigmaX=3)
    local_sq_mean = cv2.GaussianBlur(l_float ** 2, (0, 0), sigmaX=3)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

    h, w = l_channel.shape
    yy, xx = np.mgrid[0:h, 0:w]

    features = np.stack(
        [
            l_float / 255.0,
            local_mean / 255.0,
            local_std / 255.0,
            yy.astype(np.float32) / max(h - 1, 1),
            xx.astype(np.float32) / max(w - 1, 1),
        ],
        axis=-1,
    )

    return features


def construire_kdtree_reference(reference_bgr, max_samples=30000):
    ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB)
    L = ref_lab[:, :, 0]
    A = ref_lab[:, :, 1]
    B = ref_lab[:, :, 2]

    features = calculer_features_luminance(L)
    h, w, c = features.shape

    features_flat = features.reshape(-1, c)
    chroma_flat = np.stack([A.reshape(-1), B.reshape(-1)], axis=1)

    n = features_flat.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=max_samples, replace=False)
        features_flat = features_flat[indices]
        chroma_flat = chroma_flat[indices]

    tree = cKDTree(features_flat)
    return tree, chroma_flat


def coloriser_frame_local(gray_bgr, tree, chroma_reference):
    lab = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    features = calculer_features_luminance(L)
    h, w, c = features.shape

    features_flat = features.reshape(-1, c)

    _, indices = tree.query(features_flat, k=1)
    chroma = chroma_reference[indices].reshape(h, w, 2)

    lab[:, :, 1] = chroma[:, :, 0].astype(np.uint8)
    lab[:, :, 2] = chroma[:, :, 1].astype(np.uint8)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def traiter_video(video_id, color_frames_dir, gray_frames_dir, output_dir, fps=24, max_samples=30000, reference_image=None):
    color_frames_dir = Path(color_frames_dir) / video_id
    gray_frames_dir = Path(gray_frames_dir) / video_id
    output_frames_dir = Path(output_dir) / video_id / "frames"
    output_video_path = Path(output_dir) / video_id / f"{video_id}_colour_local_exemplar.mp4"

    ensure_dir(output_frames_dir)

    color_frames = sorted(color_frames_dir.glob("*.png"))
    gray_frames = sorted(gray_frames_dir.glob("*.png"))

    if len(gray_frames) == 0:
        raise RuntimeError(f"Frames grayscale manquantes pour {video_id}")

    if reference_image is not None:
        reference = cv2.imread(str(reference_image))
        if reference is None:
            raise RuntimeError(f"Unreadable reference image : {reference_image}")
    else:
        if len(color_frames) == 0:
            raise RuntimeError(f"Frames couleur manquantes pour {video_id}")
        reference = cv2.imread(str(color_frames[0]))

    print(f"\nBuilding reference KDTree : {video_id}")
    tree, chroma_reference = construire_kdtree_reference(reference, max_samples=max_samples)

    output_paths = []

    print(f"Colourisation local exemplar : {video_id}")

    for i, frame_path in enumerate(tqdm(gray_frames)):
        gray_bgr = cv2.imread(str(frame_path))
        result = coloriser_frame_local(gray_bgr, tree, chroma_reference)

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
    parser.add_argument("--output_dir", default="outputs/colourisation/local_exemplar")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max_samples", type=int, default=30000)
    parser.add_argument("--reference_image", default=None)
    args = parser.parse_args()

    traiter_video(
        video_id=args.video_id,
        color_frames_dir=args.color_frames_dir,
        gray_frames_dir=args.gray_frames_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        max_samples=args.max_samples,
        reference_image=args.reference_image,
    )


if __name__ == "__main__":
    main()
