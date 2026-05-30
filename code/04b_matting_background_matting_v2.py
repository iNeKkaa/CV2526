from pathlib import Path
import argparse
import urllib.request
import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils_video import ensure_dir, write_video_from_frames


MODEL_URL = "https://sourceforge.net/projects/backgroundmattingv2.mirror/files/v1.0.0/torchscript_mobilenetv2_fp32.pth/download"
MODEL_PATH = Path("models/background_matting_v2/torchscript_mobilenetv2_fp32.pth")


def _download_progress(block_num, block_size, total_size):
    if total_size <= 0:
        downloaded_mb = block_num * block_size / (1024 * 1024)
        print(f"\rDownloaded {downloaded_mb:.1f} MB", end="")
        return

    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100)
    downloaded_mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    print(f"\rDownloaded {downloaded_mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%)", end="")


def download_if_missing(url, path):
    path = Path(path)
    ensure_dir(path.parent)

    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"Model already found: {path}")
        return path

    if path.exists():
        print(f"Removing incomplete model file: {path}")
        path.unlink()

    print("Downloading BackgroundMattingV2 TorchScript model...")
    print(f"Source: {url}")
    print(f"Destination: {path}")

    urllib.request.urlretrieve(url, path, reporthook=_download_progress)
    print()

    if not path.exists() or path.stat().st_size < 1_000_000:
        raise RuntimeError("Model download seems invalid or incomplete.")

    print(f"Downloaded: {path}")
    return path


def frame_to_tensor(frame_bgr, device, dtype):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)
    return tensor


def tensor_to_rgb_uint8(tensor):
    image = tensor[0].permute(1, 2, 0).detach().float().cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def estimate_background(frame_paths, max_frames=40):
    print(f"Estimating background with temporal median over {max_frames} frames...")

    selected = frame_paths[:max_frames]
    frames = []

    for path in selected:
        frame = cv2.imread(str(path))
        if frame is not None:
            frames.append(frame)

    if len(frames) == 0:
        raise RuntimeError("Could not estimate background: no readable frames.")

    stack = np.stack(frames, axis=0).astype(np.uint8)
    background = np.median(stack, axis=0).astype(np.uint8)
    return background


def compose_on_background(fgr_rgb, pha, background_rgb):
    pha_3 = pha[:, :, None]
    comp = fgr_rgb.astype(np.float32) * pha_3 + background_rgb.astype(np.float32) * (1.0 - pha_3)
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    return comp


def load_model(backbone_scale=0.25, refine_sample_pixels=80_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Device used: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available. BackgroundMattingV2 will run on CPU and can be slow.")

    model_path = download_if_missing(MODEL_URL, MODEL_PATH)

    print("Loading BackgroundMattingV2 TorchScript model...")
    model = torch.jit.load(str(model_path), map_location=device)

    # TorchScript model exposes these attributes for speed/quality control.
    model.backbone_scale = backbone_scale
    model.refine_mode = "sampling"
    model.refine_sample_pixels = refine_sample_pixels

    model = model.to(device).eval()
    return model, device, dtype


def process_video_background_matting(
    video_id,
    input_frames_dir,
    output_dir,
    fps=24,
    background_path=None,
    max_background_frames=40,
    backbone_scale=0.25,
    refine_sample_pixels=80_000,
):
    print("\nMethod: Matting 2/2 - BackgroundMattingV2 TorchScript")
    print(f"Video ID: {video_id}")

    input_frames_dir = Path(input_frames_dir) / video_id

    output_dir = Path(output_dir) / video_id
    output_frames_dir = output_dir / "composition_frames"
    output_alpha_dir = output_dir / "alpha_frames"

    output_video_path = output_dir / f"{video_id}_background_matting_v2_composition.mp4"
    output_alpha_video_path = output_dir / f"{video_id}_background_matting_v2_alpha.mp4"
    background_output_path = output_dir / "estimated_or_used_background.png"

    ensure_dir(output_frames_dir)
    ensure_dir(output_alpha_dir)

    frame_paths = sorted(input_frames_dir.glob("*.png"))
    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found for video_id={video_id} in {input_frames_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")
    h, w = first_frame.shape[:2]

    if background_path is not None:
        background_bgr = cv2.imread(str(background_path))
        if background_bgr is None:
            raise RuntimeError(f"Could not read background image: {background_path}")
        background_bgr = cv2.resize(background_bgr, (w, h))
    else:
        background_bgr = estimate_background(frame_paths, max_frames=max_background_frames)
        if background_bgr.shape[:2] != (h, w):
            background_bgr = cv2.resize(background_bgr, (w, h))

    cv2.imwrite(str(background_output_path), background_bgr)
    print(f"Saved background: {background_output_path}")

    model, device, dtype = load_model(
        backbone_scale=backbone_scale,
        refine_sample_pixels=refine_sample_pixels,
    )

    background_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
    background_tensor = frame_to_tensor(background_bgr, device=device, dtype=dtype)

    composition_paths = []
    alpha_paths = []

    print(f"Running BackgroundMattingV2 on {len(frame_paths)} frames...")

    with torch.no_grad():
        for i, frame_path in enumerate(tqdm(frame_paths)):
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                print(f"Skipping unreadable frame: {frame_path}")
                continue

            if frame_bgr.shape[:2] != background_bgr.shape[:2]:
                frame_bgr = cv2.resize(frame_bgr, (background_bgr.shape[1], background_bgr.shape[0]))

            src = frame_to_tensor(frame_bgr, device=device, dtype=dtype)

            # TorchScript BackgroundMattingV2 returns alpha matte and foreground.
            pha, fgr = model(src, background_tensor)[:2]

            fgr_rgb = tensor_to_rgb_uint8(fgr)
            pha_np = pha[0, 0].detach().float().cpu().numpy()
            pha_np = np.clip(pha_np, 0.0, 1.0)

            comp_rgb = compose_on_background(fgr_rgb, pha_np, background_rgb)
            comp_bgr = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR)

            alpha_gray = np.clip(pha_np * 255, 0, 255).astype(np.uint8)
            alpha_bgr = cv2.cvtColor(alpha_gray, cv2.COLOR_GRAY2BGR)

            comp_path = output_frames_dir / f"frame_{i:05d}.png"
            alpha_path = output_alpha_dir / f"frame_{i:05d}.png"

            cv2.imwrite(str(comp_path), comp_bgr)
            cv2.imwrite(str(alpha_path), alpha_bgr)

            composition_paths.append(comp_path)
            alpha_paths.append(alpha_path)

    write_video_from_frames(composition_paths, output_video_path, fps=fps)
    write_video_from_frames(alpha_paths, output_alpha_video_path, fps=fps)

    print(f"Written BackgroundMattingV2 composition video: {output_video_path}")
    print(f"Written BackgroundMattingV2 alpha video: {output_alpha_video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--input_frames_dir", default="frames/color")
    parser.add_argument("--output_dir", default="outputs/matting/background_matting_v2")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--background_path", default=None)
    parser.add_argument("--max_background_frames", type=int, default=40)
    parser.add_argument("--backbone_scale", type=float, default=0.25)
    parser.add_argument("--refine_sample_pixels", type=int, default=80_000)

    # Kept only for compatibility with the batch script.
    parser.add_argument("--model_type", default="mobilenetv2")

    args = parser.parse_args()

    process_video_background_matting(
        video_id=args.video_id,
        input_frames_dir=args.input_frames_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        background_path=args.background_path,
        max_background_frames=args.max_background_frames,
        backbone_scale=args.backbone_scale,
        refine_sample_pixels=args.refine_sample_pixels,
    )


if __name__ == "__main__":
    main()
