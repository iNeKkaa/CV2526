# CV Seminars Project — Colourisation, Matting and Editing

This project skeleton is designed for the Computer Vision seminars project. It was tested on an NVIDIA RTX 3090 GPU, but it should normally work with any CUDA-compatible NVIDIA GPU, provided that the correct PyTorch CUDA version is installed.
It processes the same 3 short videos through several methods from Seminars 2–4.

The project is intentionally built as an experimental pipeline: the goal is not to obtain perfect videos, but to compare methods and discuss their strengths, weaknesses, artefacts, temporal coherence and computation cost.

---

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

## 3. Hardware note

The experiments were run on a desktop machine equipped with an **NVIDIA RTX 3090 GPU**.

However, the project should normally work with any **CUDA-compatible NVIDIA GPU**, provided that the correct PyTorch CUDA version is installed. Execution time will vary depending on the GPU.

The project can also run on CPU, but the heavy deep-learning stages, especially **RVM**, **BackgroundMattingV2**, **InstructPix2Pix** and **ControlNet Canny**, will be much slower.

---

## 4. Large outputs and final videos

The repository is intended to contain the **code, scripts, README, requirements and selected comparison figures**.

The following folders can become very large and should normally **not** be committed directly to GitHub:

```text
.venv/
models/
frames/
processed_videos/
outputs/
input_videos/
```

For the final rendered videos, the recommended solution is to upload them separately, for example to:

```text
Google Drive / OneDrive / GitHub Releases / Git LFS
```

Then add the download link in this README or in the project submission.

Recommended GitHub strategy:

```text
GitHub repository:
- source code
- scripts
- README
- requirements.txt
- small comparison figures
- optional short compressed demo clips

External link:
- full final videos
- large generated outputs
- model files if needed
```

This keeps the GitHub repository clean and easy to clone, while still making the final videos available for evaluation.


---

## 5. Installation on Windows with a CUDA GPU

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
GPU: NVIDIA CUDA-compatible GPU
```

---

## 6. Preprocess the videos

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

## 7. Run methods manually

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

## 8. Batch scripts

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

## 9. Batch command details

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

## 10. Create comparison figures for slides

```powershell
python code/07_make_comparison_figures.py --video_id video1_simple --frame_index 20
python code/07_make_comparison_figures.py --video_id video2_street --frame_index 40
python code/07_make_comparison_figures.py --video_id video3_object --frame_index 60
```

---

## 11. Cleaning the virtual environment

At the end of the project, if you want to free disk space, you can delete the local Python virtual environment with:

```bat
delete_venv_windows.bat
```

This only removes the `.venv/` folder inside the project. It does not delete input videos, outputs, slides, code, or generated figures. If you need to run the project again later, simply run `setup_windows.bat` again to recreate the environment.

Note: the largest files may also be model caches downloaded by PyTorch / Hugging Face outside the project folder. Do not delete them unless you know you do not need them anymore.

### Editing on GPU

First run a short editing test on 8 frames per video:

```powershell
run_editing_test_windows.bat
```

If it works and CUDA is detected, run the full editing stage:

```powershell
run_editing_full_windows.bat
```

The full editing stage is expensive because InstructPix2Pix and ControlNet are applied frame by frame.

