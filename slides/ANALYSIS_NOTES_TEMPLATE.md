# Analysis notes template

## Video 1 — human, simple background

Expected observations:
- Easier matting because foreground/background separation is clearer.
- RVM should produce smoother alpha boundaries than GrabCut. BackgroundMattingV2 can also work well if the estimated background is close to the true background.
- Editing is easier because the scene is visually stable.
- Colourisation may be relatively stable if lighting does not change.

## Video 2 — human walking in a street

Expected observations:
- Harder due to complex background, moving subject, shadows, pedestrians or cars.
- Matting can fail around legs, hair, clothes, and background objects with similar colour.
- Diffusion-based editing can create flickering because each frame is edited almost independently.
- ControlNet should better preserve structure than InstructPix2Pix, but may still hallucinate details.

## Video 3 — non-human moving object

Expected observations:
- Human-oriented matting models may be less reliable.
- Classical segmentation baseline may sometimes work if the object is well separated.
- Editing can change object identity or geometry if the prompt is too strong.
- This video is useful to test generalization beyond human subjects.

## Colourisation comparison

Compare:
- OpenCV/Zhang learning-based colourisation: semantic but may produce desaturated or wrong colours.
- Local exemplar-based transfer: uses reference chrominance but can copy wrong colours when luminance/texture matches are ambiguous.
- Luminance lookup baseline: simple, fast, but can produce global colour bleeding.

## Matting comparison

Compare:
- RVM: temporally recurrent, better for human video matting, smoother and more stable.
- BackgroundMattingV2: background-based matting, strong when a clean/estimated background is good, but sensitive to camera motion or poor background estimation.
- GrabCut soft alpha: simple baseline, sensitive to bounding box and background complexity, less temporally consistent. It should be presented as an extra baseline, not as the main second seminar method.

## Editing comparison

Compare:
- InstructPix2Pix: follows text instruction but may alter structure or identity.
- ControlNet Canny: better structure preservation thanks to edge constraint, but output can be less flexible and still flicker.

