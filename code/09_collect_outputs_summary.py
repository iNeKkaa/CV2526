from pathlib import Path
import argparse
import csv


def compter_fichiers(path, pattern):
    path = Path(path)
    if not path.exists():
        return 0
    return len(list(path.rglob(pattern)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_csv", default="outputs_summary.csv")
    args = parser.parse_args()

    rows = []
    roots = [
        "processed_videos",
        "frames/color",
        "frames/gray",
        "outputs/colourisation/opencv_zhang",
        "outputs/colourisation/local_exemplar",
        "outputs/colourisation/luminance_lookup",
        "outputs/matting/rvm",
        "outputs/matting/background_matting_v2",
        "outputs/matting/grabcut_softalpha",
        "outputs/editing/instructpix2pix",
        "outputs/editing/controlnet_canny",
        "comparison_figures",
    ]

    for root in roots:
        rows.append({
            "folder": root,
            "png_count": compter_fichiers(root, "*.png"),
            "mp4_count": compter_fichiers(root, "*.mp4"),
        })

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "png_count", "mp4_count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary : {args.output_csv}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
