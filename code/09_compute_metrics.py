from pathlib import Path
import argparse
import csv
import cv2
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "frames": 0,
            "fps": 0.0,
            "duration_sec": 0.0,
            "width": 0,
            "height": 0,
        }

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frames / fps if fps > 0 else 0.0

    return {
        "frames": frames,
        "fps": fps,
        "duration_sec": duration,
        "width": width,
        "height": height,
    }


def mse_psnr(frame_a, frame_b):
    a = frame_a.astype(np.float32)
    b = frame_b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))

    if mse == 0:
        return mse, 99.0

    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
    return mse, float(psnr)


def ssim_gray(frame_a, frame_b):
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(gray_a ** 2, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(gray_b ** 2, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)

    ssim_map = numerator / (denominator + 1e-8)
    return float(np.mean(ssim_map))


def compare_videos(reference_path, output_path, sample_every=1):
    cap_ref = cv2.VideoCapture(str(reference_path))
    cap_out = cv2.VideoCapture(str(output_path))

    psnr_values = []
    ssim_values = []
    mae_values = []

    index = 0

    while True:
        ok_ref, frame_ref = cap_ref.read()
        ok_out, frame_out = cap_out.read()

        if not ok_ref or not ok_out:
            break

        if index % sample_every == 0:
            if frame_out.shape[:2] != frame_ref.shape[:2]:
                frame_out = cv2.resize(frame_out, (frame_ref.shape[1], frame_ref.shape[0]))

            _, psnr = mse_psnr(frame_ref, frame_out)
            ssim = ssim_gray(frame_ref, frame_out)
            mae = float(np.mean(np.abs(frame_ref.astype(np.float32) - frame_out.astype(np.float32))))

            psnr_values.append(psnr)
            ssim_values.append(ssim)
            mae_values.append(mae)

        index += 1

    cap_ref.release()
    cap_out.release()

    if len(psnr_values) == 0:
        return None

    return {
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "mae": float(np.mean(mae_values)),
        "sampled_frames": len(psnr_values),
    }


def temporal_mad(video_path, sample_every=1):
    cap = cv2.VideoCapture(str(video_path))

    previous = None
    values = []
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if index % sample_every == 0:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            if previous is not None:
                diff = float(np.mean(np.abs(frame_gray - previous)))
                values.append(diff)

            previous = frame_gray

        index += 1

    cap.release()

    if len(values) == 0:
        return None

    return float(np.mean(values))


def alpha_statistics(alpha_video_path, sample_every=1):
    cap = cv2.VideoCapture(str(alpha_video_path))

    means = []
    stds = []
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if index % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            means.append(float(np.mean(gray)))
            stds.append(float(np.std(gray)))

        index += 1

    cap.release()

    if len(means) == 0:
        return None

    return {
        "alpha_mean": float(np.mean(means)),
        "alpha_std": float(np.mean(stds)),
    }


def find_processed_video(processed_dir, video_id):
    processed_dir = Path(processed_dir)
    candidates = list(processed_dir.glob(f"{video_id}*.mp4"))

    if len(candidates) == 0:
        return None

    return sorted(candidates)[0]


def detect_topic_method(output_path):
    parts = output_path.parts

    topic = "unknown"
    method = "unknown"
    output_type = "video"

    if "colourisation" in parts:
        topic = "colourisation"
        idx = parts.index("colourisation")
        method = parts[idx + 1] if idx + 1 < len(parts) else "unknown"

    elif "matting" in parts:
        topic = "matting"
        idx = parts.index("matting")
        method = parts[idx + 1] if idx + 1 < len(parts) else "unknown"

        name = output_path.name.lower()
        if "alpha" in name:
            output_type = "alpha"
        elif "composition" in name:
            output_type = "composition"

    elif "editing" in parts:
        topic = "editing"
        idx = parts.index("editing")
        method = parts[idx + 1] if idx + 1 < len(parts) else "unknown"

    return topic, method, output_type


def infer_video_id(output_path):
    parent_name = output_path.parent.name

    if parent_name in ["frames", "composition_frames", "alpha_frames", "canny_frames"]:
        return output_path.parent.parent.name

    return parent_name


def write_csv(rows, output_path):
    ensure_dir(Path(output_path).parent)

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--processed_dir", default="processed_videos")
    parser.add_argument("--metrics_dir", default="metrics")
    parser.add_argument("--sample_every", type=int, default=1)
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    processed_dir = Path(args.processed_dir)
    metrics_dir = Path(args.metrics_dir)

    ensure_dir(metrics_dir)

    output_videos = sorted(outputs_dir.rglob("*.mp4"))

    rows = []

    print(f"Found {len(output_videos)} output videos.")

    input_temporal_cache = {}

    for output_path in output_videos:
        topic, method, output_type = detect_topic_method(output_path)
        video_id = infer_video_id(output_path)

        original_path = find_processed_video(processed_dir, video_id)
        info = get_video_info(output_path)

        size_mb = output_path.stat().st_size / (1024 * 1024)

        row = {
            "topic": topic,
            "method": method,
            "video_id": video_id,
            "output_type": output_type,
            "file": str(output_path),
            "size_mb": round(size_mb, 4),
            "frames": info["frames"],
            "fps": round(info["fps"], 4),
            "duration_sec": round(info["duration_sec"], 4),
            "width": info["width"],
            "height": info["height"],
            "psnr_to_original": "",
            "ssim_to_original": "",
            "mae_to_original": "",
            "temporal_mad": "",
            "input_temporal_mad": "",
            "temporal_ratio": "",
            "alpha_mean": "",
            "alpha_std": "",
            "sampled_frames": "",
        }

        print(f"Processing: {topic} / {method} / {video_id} / {output_type}")

        output_temporal = temporal_mad(output_path, sample_every=args.sample_every)
        if output_temporal is not None:
            row["temporal_mad"] = round(output_temporal, 4)

        if original_path is not None:
            if video_id not in input_temporal_cache:
                input_temporal_cache[video_id] = temporal_mad(original_path, sample_every=args.sample_every)

            input_temporal = input_temporal_cache[video_id]
            if input_temporal is not None:
                row["input_temporal_mad"] = round(input_temporal, 4)

                if output_temporal is not None and input_temporal > 1e-8:
                    row["temporal_ratio"] = round(output_temporal / input_temporal, 4)

            if output_type != "alpha":
                comparison = compare_videos(original_path, output_path, sample_every=args.sample_every)
                if comparison is not None:
                    row["psnr_to_original"] = round(comparison["psnr"], 4)
                    row["ssim_to_original"] = round(comparison["ssim"], 4)
                    row["mae_to_original"] = round(comparison["mae"], 4)
                    row["sampled_frames"] = comparison["sampled_frames"]

        if output_type == "alpha":
            alpha_stats = alpha_statistics(output_path, sample_every=args.sample_every)
            if alpha_stats is not None:
                row["alpha_mean"] = round(alpha_stats["alpha_mean"], 4)
                row["alpha_std"] = round(alpha_stats["alpha_std"], 4)

        rows.append(row)

    summary_path = metrics_dir / "metrics_summary.csv"
    write_csv(rows, summary_path)

    print("\nDone.")
    print(f"Saved metrics to: {summary_path}")
    print("\nOpen it with:")
    print(f"notepad {summary_path}")


if __name__ == "__main__":
    main()