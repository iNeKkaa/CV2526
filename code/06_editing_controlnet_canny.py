from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

from utils_video import ensure_dir, write_video_from_frames


def rendre_multiple_de_8(image):
    w, h = image.size
    new_w = w - (w % 8)
    new_h = h - (h % 8)
    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h))
    return image


def creer_image_canny(image_pil, low_threshold=100, high_threshold=200):
    image_np = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = np.stack([edges, edges, edges], axis=2)

    return Image.fromarray(edges_rgb)


def charger_pipeline(controlnet_model, base_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("Device used :", device)
    if device.type == "cuda":
        print("GPU :", torch.cuda.get_device_name(0))

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=dtype,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device.type == "cuda":
        pipe.enable_attention_slicing()

    return pipe, device


def traiter_video(video_id, input_frames_dir, output_dir, prompt, negative_prompt,
                  fps=24, steps=20, guidance_scale=7.5, seed=42, limit=None,
                  controlnet_model="lllyasviel/sd-controlnet-canny",
                  base_model="runwayml/stable-diffusion-v1-5",
                  low_threshold=100, high_threshold=200):
    pipe, device = charger_pipeline(controlnet_model, base_model)

    input_frames_dir = Path(input_frames_dir) / video_id
    output_frames_dir = Path(output_dir) / video_id / "frames"
    output_canny_dir = Path(output_dir) / video_id / "canny_frames"
    output_video_path = Path(output_dir) / video_id / f"{video_id}_controlnet_canny.mp4"

    ensure_dir(output_frames_dir)
    ensure_dir(output_canny_dir)

    frame_paths = sorted(input_frames_dir.glob("*.png"))
    if limit is not None:
        frame_paths = frame_paths[:limit]

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frame found for {video_id}")

    output_paths = []

    print(f"\nEditing ControlNet Canny : {video_id}")
    print(f"Prompt : {prompt}")

    generator = torch.Generator(device=device).manual_seed(seed)

    for i, frame_path in enumerate(tqdm(frame_paths)):
        image = Image.open(frame_path).convert("RGB")
        image = rendre_multiple_de_8(image)

        canny_image = creer_image_canny(image, low_threshold=low_threshold, high_threshold=high_threshold)
        canny_path = output_canny_dir / f"frame_{i:05d}.png"
        canny_image.save(canny_path)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        out_path = output_frames_dir / f"frame_{i:05d}.png"
        result.save(out_path)
        output_paths.append(out_path)

    write_video_from_frames(output_paths, output_video_path, fps=fps)
    print(f"Written video : {output_video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--input_frames_dir", default="frames/color")
    parser.add_argument("--output_dir", default="outputs/editing/controlnet_canny")
    parser.add_argument("--prompt", default="cinematic realistic video frame, high quality")
    parser.add_argument("--negative_prompt", default="low quality, blurry, distorted, deformed")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--controlnet_model", default="lllyasviel/sd-controlnet-canny")
    parser.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--low_threshold", type=int, default=100)
    parser.add_argument("--high_threshold", type=int, default=200)
    args = parser.parse_args()

    traiter_video(
        video_id=args.video_id,
        input_frames_dir=args.input_frames_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        fps=args.fps,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        limit=args.limit,
        controlnet_model=args.controlnet_model,
        base_model=args.base_model,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )


if __name__ == "__main__":
    main()
