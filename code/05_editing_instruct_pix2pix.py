from pathlib import Path
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from utils_video import ensure_dir, write_video_from_frames


def charger_pipeline(model_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("Device used :", device)
    if device.type == "cuda":
        print("GPU :", torch.cuda.get_device_name(0))

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device.type == "cuda":
        pipe.enable_attention_slicing()

    return pipe, device


def traiter_video(video_id, input_frames_dir, output_dir, prompt, fps=24, model_id="timbrooks/instruct-pix2pix",
                  steps=20, text_guidance=7.5, image_guidance=1.5, seed=42, limit=None):
    pipe, device = charger_pipeline(model_id)

    input_frames_dir = Path(input_frames_dir) / video_id
    output_frames_dir = Path(output_dir) / video_id / "frames"
    output_video_path = Path(output_dir) / video_id / f"{video_id}_instructpix2pix.mp4"

    ensure_dir(output_frames_dir)

    frame_paths = sorted(input_frames_dir.glob("*.png"))
    if limit is not None:
        frame_paths = frame_paths[:limit]

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frame found for {video_id}")

    output_paths = []

    print(f"\nEditing InstructPix2Pix : {video_id}")
    print(f"Prompt : {prompt}")

    generator = torch.Generator(device=device).manual_seed(seed)

    for i, frame_path in enumerate(tqdm(frame_paths)):
        image = Image.open(frame_path).convert("RGB")

        result = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=steps,
            guidance_scale=text_guidance,
            image_guidance_scale=image_guidance,
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
    parser.add_argument("--output_dir", default="outputs/editing/instructpix2pix")
    parser.add_argument("--prompt", default="make the scene look cinematic, realistic, high quality")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--model_id", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--text_guidance", type=float, default=7.5)
    parser.add_argument("--image_guidance", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    traiter_video(
        video_id=args.video_id,
        input_frames_dir=args.input_frames_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        fps=args.fps,
        model_id=args.model_id,
        steps=args.steps,
        text_guidance=args.text_guidance,
        image_guidance=args.image_guidance,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
