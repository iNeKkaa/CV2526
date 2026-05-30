from pathlib import Path
import argparse
import cv2
import matplotlib.pyplot as plt

from utils_video import ensure_dir


def lire_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def chercher_frame(base_dir, video_id, frame_index, subdir_name="frames"):
    path = Path(base_dir) / video_id / subdir_name / f"frame_{frame_index:05d}.png"
    if path.exists():
        return path
    return None


def chercher_frame_simple(base_dir, video_id, frame_index):
    path = Path(base_dir) / video_id / f"frame_{frame_index:05d}.png"
    if path.exists():
        return path
    return None


def faire_figure(video_id, frame_index, output_dir):
    items = []

    # Input frames
    p = chercher_frame_simple("frames/color", video_id, frame_index)
    if p:
        items.append(("Input", p))

    p = chercher_frame_simple("frames/gray", video_id, frame_index)
    if p:
        items.append(("Grayscale", p))

    # Colourisation
    for title, base in [
        ("Colour: OpenCV/Zhang", "outputs/colourisation/opencv_zhang"),
        ("Colour: Local exemplar", "outputs/colourisation/local_exemplar"),
        ("Colour: Luminance lookup", "outputs/colourisation/luminance_lookup"),
    ]:
        p = chercher_frame(base, video_id, frame_index, "frames")
        if p:
            items.append((title, p))

    # Matting
    for title, base, subdir in [
        ("Matting: RVM alpha", "outputs/matting/rvm", "alpha_frames"),
        ("Matting: RVM composite", "outputs/matting/rvm", "composition_frames"),
        ("Matting: BMv2 alpha", "outputs/matting/background_matting_v2", "alpha_frames"),
        ("Matting: BMv2 composite", "outputs/matting/background_matting_v2", "composition_frames"),
        ("Matting: GrabCut alpha", "outputs/matting/grabcut_softalpha", "alpha_frames"),
        ("Matting: GrabCut composite", "outputs/matting/grabcut_softalpha", "composition_frames"),
    ]:
        p = chercher_frame(base, video_id, frame_index, subdir)
        if p:
            items.append((title, p))

    # Editing
    for title, base in [
        ("Editing: InstructPix2Pix", "outputs/editing/instructpix2pix"),
        ("Editing: ControlNet Canny", "outputs/editing/controlnet_canny"),
    ]:
        p = chercher_frame(base, video_id, frame_index, "frames")
        if p:
            items.append((title, p))

    if len(items) == 0:
        raise RuntimeError("No image found to generate the figure.")

    n = len(items)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 3.2 * rows))

    for i, (title, path) in enumerate(items):
        img = lire_rgb(path)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis("off")

    plt.tight_layout()

    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    out_path = output_dir / f"comparison_{video_id}_frame_{frame_index:05d}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved figure : {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--frame_index", type=int, default=20)
    parser.add_argument("--output_dir", default="comparison_figures")
    args = parser.parse_args()

    faire_figure(args.video_id, args.frame_index, args.output_dir)


if __name__ == "__main__":
    main()
