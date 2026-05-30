# CV Seminars Project — Colourisation, Matting and Editing

This project skeleton (v8, RTX 3090-ready with fixed BackgroundMattingV2 TorchScript loading) is designed for the Computer Vision seminars project.
It processes the same 3 short videos through several methods from Seminars 2–4.

The project is intentionally built as an experimental pipeline: the goal is not to obtain perfect videos, but to compare methods and discuss their strengths, weaknesses, artefacts, temporal coherence and computation cost.

---

### v8 notes

This version includes the latest fixes used during debugging:

- RVM uses an explicit `downsample_ratio=0.25`.
- BackgroundMattingV2 uses the TorchScript model instead of `torch.hub`, because the official repository does not expose a `hubconf.py`.
- Video detection is deduplicated on Windows.
- The default cut is still from 1s to 6s, 24 fps.
- Extra batch scripts are provided for editing: `run_editing_test_windows.bat` and `run_editing_full_windows.bat`.


## 0. Important preprocessing choice

By default, the scripts cut the videos as follows:

```text
start time = 1 second
duration   = 5 seconds
fps        = 24
max side   = 768 pixels
```

So the extracted clip corresponds to the interval:

```text
from 1s to 6s of the original video
```

This was chosen to skip the first moment where the camera may still be moving or the subject may not yet be visible. You can change it with `--start_time` if needed.

---

## 1. Methods included

### Seminar 2 — Colourisation

Main methods:

1. **OpenCV/Zhang learning-based colourisation**  
   Automatic deep-learning colourisation from grayscale frames.

2. **Local exemplar-based chrominance transfer**  
   Transfers chrominance from a reference frame using local luminance/statistical features.

Backup/baseline:

3. **Luminance lookup colour transfer**  
   Simple and fast baseline, useful if the deep model download fails.

### Seminar 3 — Matting

Main methods to count for the project:

1. **Robust Video Matting (RVM)**  
   Video-oriented recurrent method, especially adapted to human video matting.

2. **Background Matting V2 / background-matting family**  
   Background-based matting method. If no clean background image is provided, the script estimates one using the temporal median of the video frames.

Extra baseline:

3. **GrabCut + soft alpha**  
   Classical segmentation baseline. It is useful for comparison, but it should be presented as a baseline rather than as one of the two main seminar methods.

### Seminar 4 — Editing

Main methods:

1. **InstructPix2Pix**  
   Instruction-based image editing applied frame-by-frame.

2. **ControlNet Canny**  
   Diffusion editing guided by Canny edge maps, used to preserve more of the original structure.

---

## 2. Folder structure

Put your original videos in:

```text
input_videos/
```

Recommended names:

```text
input_videos/video1_simple.mp4
input_videos/video2_street.mp4
input_videos/video3_object.mp4
```

The scripts automatically create:

```text
processed_videos/       # 5s, 24 fps, resized videos
frames/color/           # extracted colour frames
frames/gray/            # extracted grayscale frames
outputs/                # method outputs
comparison_figures/     # before/after figures for slides
```

---

## 3. Installation on Windows with RTX 3090

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install PyTorch with CUDA:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

Check GPU detection:

```powershell
python code/verify_gpu.py
```

You want to see something like:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

---

## 4. Preprocess the videos

From the project root:

```powershell
python code/00_prepare_videos.py --fps 24 --duration 5 --max_side 768 --start_time 1
```

Or simply run:

```powershell
run_preprocess_windows.bat
```

This cuts from 1s to 6s, samples at 24 fps, resizes the largest side to 768 px, extracts colour and grayscale frames, and reconstructs clean processed videos.

If you want another crop:

```powershell
python code/00_prepare_videos.py --fps 24 --duration 5 --max_side 768 --start_time 0
```

---

## 5. Run methods manually

After preprocessing, check the names of the folders in `frames/color/`. Those are the `video_id` values.

Example for `video1_simple`:

### Colourisation

```powershell
python code/01_colourisation_opencv_zhang.py --video_id video1_simple --fps 24
python code/02_colourisation_local_exemplar.py --video_id video1_simple --fps 24
python code/02b_colourisation_luminance_lookup.py --video_id video1_simple --fps 24
```

### Matting

Main methods:

```powershell
python code/03_matting_rvm.py --video_id video1_simple --fps 24 --model_type mobilenetv3
python code/04b_matting_background_matting_v2.py --video_id video1_simple --fps 24 --model_type mobilenetv2
```

If you have a clean background image for a video, you can give it directly:

```powershell
python code/04b_matting_background_matting_v2.py --video_id video1_simple --fps 24 --background_path backgrounds/video1_clean_background.png
```

If no background image is provided, the script estimates one by median over the frames. This is useful, but also interesting to discuss: it works better with a stable camera and moving foreground, and worse with a moving camera or strongly changing background.

Optional baseline:

```powershell
python code/04_matting_grabcut_softalpha.py --video_id video1_simple --fps 24
```

For the street video, if the person is not centered, adjust the GrabCut rectangle:

```powershell
python code/04_matting_grabcut_softalpha.py --video_id video2_street --fps 24 --rect 0.15 0.05 0.70 0.90
```

### Editing

First test on a few frames:

```powershell
python code/05_editing_instruct_pix2pix.py --video_id video1_simple --fps 24 --limit 8 --prompt "make the scene look cinematic"
python code/06_editing_controlnet_canny.py --video_id video1_simple --fps 24 --limit 8 --prompt "cinematic realistic video frame, high quality"
```

Then run the full 5-second video:

```powershell
python code/05_editing_instruct_pix2pix.py --video_id video1_simple --fps 24 --prompt "make the scene look cinematic"
python code/06_editing_controlnet_canny.py --video_id video1_simple --fps 24 --prompt "cinematic realistic video frame, high quality"
```

---

## 6. Batch scripts

### Preprocessing only

```powershell
run_preprocess_windows.bat
```

### Light test without heavy diffusion

```powershell
run_light_batch_windows.bat
```

This assumes preprocessing has already been run and then executes:

```text
backup colourisation + GrabCut baseline + figures
```

### GPU editing test

```powershell
run_gpu_test_windows.bat
```

This assumes preprocessing has already been run and then executes:

```text
GPU check + InstructPix2Pix test + ControlNet Canny test on a few frames
```

### Main seminar methods without full editing

```powershell
run_main_methods_windows.bat
```

This assumes preprocessing has already been run and then executes:

```text
two colourisation methods + RVM + BackgroundMattingV2 + figures
```

It does **not** run preprocessing again. If you want to preprocess and then run the main methods in one command, use:

```powershell
run_all_from_scratch_windows.bat
```

### Full heavy editing run

```powershell
run_full_gpu_windows.bat
```

This assumes preprocessing has already been run and then executes:

```text
full InstructPix2Pix + full ControlNet
```

Warning: the full editing run can be long even on a 3090 because it processes many frames with diffusion.

---

## 7. Batch command details

You can also call the batch runner directly:

```powershell
python code/08_run_batch.py --stages colour matting_rvm matting_bmv2 figures --fps 24 --duration 5 --max_side 768 --start_time 1
```

For heavy GPU testing on only 8 frames:

```powershell
python code/08_run_batch.py --stages matting_rvm matting_bmv2 editing_test --fps 24 --limit 8
```

Full heavy editing:

```powershell
python code/08_run_batch.py --stages editing_full --fps 24
```

---

## 8. Create comparison figures for slides

```powershell
python code/07_make_comparison_figures.py --video_id video1_simple --frame_index 20
python code/07_make_comparison_figures.py --video_id video2_street --frame_index 40
python code/07_make_comparison_figures.py --video_id video3_object --frame_index 60
```

---

## 9. Notes for the final presentation

The final slides should focus on:

- Why the 3 videos are different and useful.
- Which methods were used for each topic.
- Any adaptations made: 1s crop, resizing, fps, frame extraction, prompts, reference frame, estimated background.
- Visual comparison between methods.
- Main errors and limitations:
  - colour bleeding,
  - wrong semantic colours,
  - matting holes,
  - rough alpha boundaries,
  - background estimation errors,
  - flickering,
  - diffusion frame inconsistency,
  - loss of identity or geometry,
  - computation time.
- Why some methods work better on simple background, complex street background, or non-human objects.

Suggested interpretation for the matting part:

```text
RVM does not require a clean background and is designed for video matting, so it should be more stable on human videos.
BackgroundMattingV2 uses background information, so it can be strong when the background is known or well estimated, but it is more sensitive to camera motion and imperfect background estimation.
GrabCut is kept only as a classical baseline to show the gap between simple segmentation and modern alpha matting.
```

## Cleaning the virtual environment

At the end of the project, if you want to free disk space, you can delete the local Python virtual environment with:

```bat
delete_venv_windows.bat
```

This only removes the `.venv/` folder inside the project. It does not delete input videos, outputs, slides, code, or generated figures. If you need to run the project again later, simply run `setup_windows.bat` again to recreate the environment.

Note: the largest files may also be model caches downloaded by PyTorch / Hugging Face outside the project folder. Do not delete them unless you know you do not need them anymore.

## v7 fix notes

This version fixes the RVM script naming consistency:

- `03_matting_rvm.py` now imports `ensure_dir` and `write_video_from_frames`, which are the actual utility function names used in `utils_video.py`.
- RVM now uses `downsample_ratio = 0.25` by default instead of `None`, avoiding a PyTorch `interpolate` error.
- `08_run_batch.py` passes `--downsample_ratio 0.25` explicitly when running RVM.
- Running on a laptop without CUDA is normal: the script will use CPU and print a warning. On the desktop with the RTX 3090, it should print `Device used: cuda` if PyTorch CUDA is correctly installed.

### Editing on the RTX 3090

First run a short editing test on 8 frames per video:

```powershell
run_editing_test_windows.bat
```

If it works and CUDA is detected, run the full editing stage:

```powershell
run_editing_full_windows.bat
```

The full editing stage is expensive because InstructPix2Pix and ControlNet are applied frame by frame.

