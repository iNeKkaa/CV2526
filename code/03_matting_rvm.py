from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils_video import ensure_dir, write_video_from_frames


def frame_to_tensor(frame_bgr, device, dtype):
    """Convert an OpenCV BGR frame to a PyTorch RGB tensor [1, 3, H, W]."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)
    return tensor


def tensor_to_rgb_uint8(tensor):
    """Convert a PyTorch RGB tensor [1, 3, H, W] to a uint8 RGB image."""
    image = tensor[0].permute(1, 2, 0).detach().float().cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def process_video_rvm(
    video_id,
    input_frames_dir,
    output_dir,
    fps=24,
    background_color=(30, 144, 255),
    model_type="mobilenetv3",
    downsample_ratio=0.25,
):
    """
    Run Robust Video Matting on one preprocessed video.

    downsample_ratio must be a real float. Passing None makes the current RVM
    torch.hub implementation call torch.nn.functional.interpolate without a
    size or scale_factor, which crashes on some PyTorch versions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("\nMethod: Matting 1/2 - Robust Video Matting")
    print(f"Video ID: {video_id}")
    print(f"Device used: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available. RVM will run on CPU and can be slow.")

    print(f"Loading RVM model: {model_type}")
    print(f"Downsample ratio: {downsample_ratio}")

    model = torch.hub.load(
        "PeterL1n/RobustVideoMatting",
        model_type,
        pretrained=True,
        trust_repo=True,
    )
    model = model.to(device=device, dtype=dtype).eval()

    input_frames_dir = Path(input_frames_dir) / video_id
    output_frames_dir = Path(output_dir) / video_id / "composition_frames"
    output_alpha_dir = Path(output_dir) / video_id / "alpha_frames"

    output_video_path = Path(output_dir) / video_id / f"{video_id}_rvm_composition.mp4"
    output_alpha_video_path = Path(output_dir) / video_id / f"{video_id}_rvm_alpha.mp4"

    ensure_dir(output_frames_dir)
    ensure_dir(output_alpha_dir)

    frame_paths = sorted(input_frames_dir.glob("*.png"))
    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found for video_id={video_id} in {input_frames_dir}")

    composition_paths = []
    alpha_paths = []

    recurrent_states = [None, None, None, None]
    background_rgb = np.array(background_color, dtype=np.float32).reshape(1, 1, 3)

    print(f"\nRunning RVM on {len(frame_paths)} frames...")

    with torch.no_grad():
        for i, frame_path in enumerate(tqdm(frame_paths)):
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                print(f"Skipping unreadable frame: {frame_path}")
                continue

            src = frame_to_tensor(frame_bgr, device=device, dtype=dtype)

            fgr, pha, *recurrent_states = model(
                src,
                *recurrent_states,
                downsample_ratio=downsample_ratio,
            )

            foreground_rgb = tensor_to_rgb_uint8(fgr)
            alpha = pha[0, 0].detach().float().cpu().numpy()
            alpha = np.clip(alpha, 0.0, 1.0)
            alpha_3 = alpha[:, :, None]

            composition_rgb = foreground_rgb.astype(np.float32) * alpha_3 + background_rgb * (1.0 - alpha_3)
            composition_rgb = np.clip(composition_rgb, 0, 255).astype(np.uint8)
            composition_bgr = cv2.cvtColor(composition_rgb, cv2.COLOR_RGB2BGR)

            alpha_gray = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
            alpha_bgr = cv2.cvtColor(alpha_gray, cv2.COLOR_GRAY2BGR)

            comp_path = output_frames_dir / f"frame_{i:05d}.png"
            alpha_path = output_alpha_dir / f"frame_{i:05d}.png"

            cv2.imwrite(str(comp_path), composition_bgr)
            cv2.imwrite(str(alpha_path), alpha_bgr)

            composition_paths.append(comp_path)
            alpha_paths.append(alpha_path)

    write_video_from_frames(composition_paths, output_video_path, fps=fps)
    write_video_from_frames(alpha_paths, output_alpha_video_path, fps=fps)

    print(f"Written RVM composition video: {output_video_path}")
    print(f"Written RVM alpha video: {output_alpha_video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--input_frames_dir", default="frames/color")
    parser.add_argument("--output_dir", default="outputs/matting/rvm")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--model_type", default="mobilenetv3", choices=["mobilenetv3", "resnet50"])
    parser.add_argument("--downsample_ratio", type=float, default=0.25)
    args = parser.parse_args()

    process_video_rvm(
        video_id=args.video_id,
        input_frames_dir=args.input_frames_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        model_type=args.model_type,
        downsample_ratio=args.downsample_ratio,
    )


if __name__ == "__main__":
    main()
