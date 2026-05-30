from pathlib import Path
import argparse
import subprocess
import sys

from utils_video import find_video_ids


def run(cmd, stage_name=None):
    print("\n" + "=" * 100)
    if stage_name:
        print(f"STAGE: {stage_name}")
    print("RUN:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=["preprocess"],
                        help="Stages: preprocess, colour, colour_backup, matting_light, matting_rvm, matting_bmv2, editing_test, editing_full, figures")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--start_time", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=8, help="Frame limit for editing_test")
    args = parser.parse_args()

    py = sys.executable

    if "preprocess" in args.stages:
        run([py, "code/00_prepare_videos.py", "--fps", str(args.fps), "--duration", str(args.duration),
             "--max_side", str(args.max_side), "--start_time", str(args.start_time)],
            stage_name="Preprocessing: trim to 5s, 24 fps, resize, extract frames")

    video_ids = find_video_ids("frames/color")
    if len(video_ids) == 0:
        raise RuntimeError("No video_id found. Run preprocessing first.")

    print("Detected video IDs :", video_ids)

    for video_id in video_ids:
        if "colour" in args.stages:
            run([py, "code/01_colourisation_opencv_zhang.py", "--video_id", video_id, "--fps", str(args.fps)],
                stage_name=f"Colourisation method 1/2: OpenCV Zhang learning-based model on {video_id}")
            run([py, "code/02_colourisation_local_exemplar.py", "--video_id", video_id, "--fps", str(args.fps)],
                stage_name=f"Colourisation method 2/2: local exemplar transfer on {video_id}")

        if "colour_backup" in args.stages:
            run([py, "code/02b_colourisation_luminance_lookup.py", "--video_id", video_id, "--fps", str(args.fps)],
                stage_name=f"Colourisation backup: luminance lookup on {video_id}")

        if "matting_light" in args.stages:
            run([py, "code/04_matting_grabcut_softalpha.py", "--video_id", video_id, "--fps", str(args.fps)],
                stage_name=f"Matting baseline: GrabCut soft alpha on {video_id}")

        if "matting_rvm" in args.stages:
            run([py, "code/03_matting_rvm.py", "--video_id", video_id, "--fps", str(args.fps), "--model_type", "mobilenetv3", "--downsample_ratio", "0.25"],
                stage_name=f"Matting method 1/2: Robust Video Matting RVM on {video_id}")

        if "matting_bmv2" in args.stages:
            run([py, "code/04b_matting_background_matting_v2.py", "--video_id", video_id, "--fps", str(args.fps), "--model_type", "mobilenetv2"],
                stage_name=f"Matting method 2/2: BackgroundMattingV2 on {video_id}")

        if "editing_test" in args.stages:
            run([py, "code/05_editing_instruct_pix2pix.py", "--video_id", video_id, "--fps", str(args.fps),
                 "--limit", str(args.limit), "--prompt", "make the scene look cinematic"],
                stage_name=f"Editing method 1/2 test: InstructPix2Pix on {video_id}")
            run([py, "code/06_editing_controlnet_canny.py", "--video_id", video_id, "--fps", str(args.fps),
                 "--limit", str(args.limit), "--prompt", "cinematic realistic video frame, high quality"],
                stage_name=f"Editing method 2/2 test: ControlNet Canny on {video_id}")

        if "editing_full" in args.stages:
            run([py, "code/05_editing_instruct_pix2pix.py", "--video_id", video_id, "--fps", str(args.fps),
                 "--prompt", "make the scene look cinematic"],
                stage_name=f"Editing method 1/2 full: InstructPix2Pix on {video_id}")
            run([py, "code/06_editing_controlnet_canny.py", "--video_id", video_id, "--fps", str(args.fps),
                 "--prompt", "cinematic realistic video frame, high quality"],
                stage_name=f"Editing method 2/2 full: ControlNet Canny on {video_id}")

        if "figures" in args.stages:
            for idx in [10, 40, 80]:
                run([py, "code/07_make_comparison_figures.py", "--video_id", video_id, "--frame_index", str(idx)],
                    stage_name=f"Comparison figure for {video_id}, frame {idx}")


if __name__ == "__main__":
    main()
