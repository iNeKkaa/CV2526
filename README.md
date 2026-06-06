# CV Seminars Project — Colorization, Video Matting and Video Editing

This repository contains the code used for our Computer Vision Seminars project.

The goal of the project was to apply and compare several methods related to the seminar topics on the same short videos:

* **Colorization**
* **Video matting**
* **Video editing**

The project focuses not only on visual quality, but also on the analysis of artefacts, limitations, temporal consistency and method behaviour on realistic short videos.

---

## Authors

* Antoine Metz — u271676 @iNeKkaa
* Corentin Le Bris — u271785

---

## Project overview

We used three short input videos with different levels of difficulty:

1. **Video 1 — simple human subject**
   A human subject with a relatively simple background.

2. **Video 2 — street scene**
   A walking person with a more complex background and camera perspective.

3. **Video 3 — dog / non-human moving subject**
   A non-human moving subject, useful to test generalisation beyond human-centred videos.

All videos were processed with the same setup:

```text
start time = 1 second
duration   = 5 seconds
fps        = 24
frames     = 120 frames per video
max side   = 768 pixels
```

---

## Methods included

### 1. Colorization

Two main colorization methods were tested:

* **OpenCV / Zhang learning-based colorization**
  A pretrained neural network predicts plausible chrominance values from grayscale frames. This method was used as a lightweight and reproducible learning-based baseline, after dependency and compatibility issues with the ChromaGAN implementation.

* **Local exemplar-based color transfer**
  Colors are transferred from a reference frame using luminance and local image features.

The purpose of this comparison was to contrast a learning-based method, which predicts colors from learned visual priors, with a reference-based method, which transfers colors from a selected color example.

---

### 2. Video matting

Two main matting methods were tested:

* **Robust Video Matting (RVM)**
  A video-oriented deep learning matting method, especially adapted to human subjects.

* **BackgroundMattingV2**
  A background-based deep learning matting method. Since no clean background capture was available, the background was estimated automatically using a temporal median over the video frames.

For matting, both the **alpha matte videos** and the **composite videos** were generated. The alpha matte shows the technical quality of the foreground extraction, while the composite shows the practical visual result.

---

### 3. Video editing

Two diffusion-based editing methods were tested:

* **InstructPix2Pix**
  Instruction-based image editing applied frame by frame.

* **ControlNet Canny**
  Edge-guided diffusion editing using Canny maps as structural conditioning.

Both methods were applied independently to each frame and then reconstructed into videos. The main limitation observed in this part is that these methods do not explicitly enforce temporal consistency.

---

## Repository content

```text
code/                  Python scripts for preprocessing, methods and figures
comparison_figures/    Selected comparison figures used for the presentation
metrics/               Computed quantitative indicators
slides/                Final presentation slides
requirements.txt       Python dependencies
*.bat                  Windows scripts for easier execution
README.md              Project documentation
```

Large generated folders are not intended to be stored directly in the repository:

```text
.venv/
models/
frames/
processed_videos/
outputs/
input_videos/
```

---

## Final videos and large outputs

The final rendered videos and large generated outputs are provided separately through the GitHub Release associated with this repository:

```text
https://github.com/iNeKkaa/CV2526/releases/latest
```

This avoids storing very large videos directly in the Git history while keeping the repository easy to clone.

---

## Hardware note

The experiments were run on a desktop machine equipped with an **NVIDIA RTX 3090 GPU**.

The project should normally work with any **CUDA-compatible NVIDIA GPU**, provided that the correct PyTorch CUDA version is installed. Execution time may vary depending on the GPU.

The project can also run on CPU, but the heavy deep-learning stages, especially RVM, BackgroundMattingV2, InstructPix2Pix and ControlNet Canny, will be significantly slower.

---

## Installation

Create a Python virtual environment:

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

Check that the GPU is detected:

```powershell
python code/verify_gpu.py
```

Expected result:

```text
CUDA available: True
GPU: NVIDIA CUDA-compatible GPU
```

---

## Running the project

### 1. Preprocess videos

```powershell
run_preprocess_windows.bat
```

This cuts the videos from 1s to 6s, converts them to 24 fps, resizes them and extracts frames.

Equivalent command:

```powershell
python code/00_prepare_videos.py --fps 24 --duration 5 --max_side 768 --start_time 1
```

---

### 2. Run the main project methods

```powershell
run_main_methods_windows.bat
```

This runs the main methods used in the presentation:

```text
Colorization:
- OpenCV / Zhang
- Local exemplar color transfer

Matting:
- RVM
- BackgroundMattingV2
```

---

### 3. Run editing methods

For a short test on a few frames:

```powershell
run_editing_test_windows.bat
```

For the full editing stage:

```powershell
run_editing_full_windows.bat
```

The full editing stage is expensive because diffusion models are applied frame by frame.

---

### 4. Run everything from scratch

```powershell
run_all_from_scratch_windows.bat
```

This preprocesses the videos and runs the main pipeline.

---

## Metrics and figures

Comparison figures can be generated with:

```powershell
python code/07_make_comparison_figures.py --video_id video1_simple --frame_index 20
python code/07_make_comparison_figures.py --video_id video2_street --frame_index 40
python code/07_make_comparison_figures.py --video_id video3_object --frame_index 60
```

The project uses simple indicators to support the analysis:

* **PSNR and SSIM** for colorization;
* **alpha statistics** for matting;
* **visual temporal consistency analysis** for editing.

These indicators are used as support for the discussion, but they are not interpreted as perfect measures of perceptual quality.

In particular, SSIM is limited for colorization in our setting, because it mainly confirms that the image structure is preserved and does not fully evaluate color realism.

---

## Main observations

### Colorization

Colorization produced the most stable results.

The structure of the video was mostly preserved, because the task mainly modifies color information rather than geometry.

The local exemplar method was closer to the original colors according to PSNR, but it depends strongly on the selected reference frame.

The Zhang model produced plausible colors, but sometimes with visible color shifts. This is expected because it predicts likely colors from learned priors rather than transferring the original colors from a reference frame.

---

### Matting

Matting was more challenging because it requires accurate foreground extraction and soft alpha boundaries.

RVM was generally more robust on human subjects. This is consistent with the fact that RVM is designed mainly for human video matting.

BackgroundMattingV2 was more sensitive to the quality of the estimated background. Since no clean background image was available, the background was estimated with a temporal median, which can introduce errors if the foreground remains visible or if the scene is too complex.

---

### Editing

Editing produced the most visually unstable results.

Some individual frames were visually interesting, but reconstructed videos showed flickering, subject deformation, identity drift, background drift and hallucinated details.

The main reason is that the editing methods were applied frame by frame, without temporal memory.

InstructPix2Pix generally preserved the global structure better because it uses the full input image. ControlNet Canny used edge maps as structural guidance, but the edge map does not preserve colors, textures, clothes, identity or exact background appearance. These details are regenerated by the diffusion model, which explains the stronger visual drift.

---

## Conclusion

This project shows that applying image-based methods to videos is not enough to obtain stable video results.

Colorization was relatively coherent, because the methods mostly modified chrominance while preserving the original structure. Matting depended strongly on the subject and on the background assumptions. Editing revealed the strongest temporal consistency issues, because the diffusion-based editing methods were applied independently to each frame.

The main lesson is that video processing is not only about producing good individual frames. It also requires preserving **temporal coherence** across time.

Possible improvements include:

* using real video diffusion models;
* adding optical-flow-based temporal smoothing;
* using a clean background image for BackgroundMattingV2;
* performing more systematic parameter tuning;
* testing more videos and prompts.
