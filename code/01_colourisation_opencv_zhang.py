from pathlib import Path
import argparse
import urllib.request
import urllib.error
import cv2
import numpy as np
from tqdm import tqdm

from utils_video import ensure_dir, write_video_from_frames


MODEL_DIR = Path("models/colorization")

# The original richzhang GitHub repository has moved paths across branches over time.
# The first URL uses the OpenVINO model mirror, which currently hosts the three files.
# Fallback URLs are kept for robustness.
PROTOTXT_URLS = [
    "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_deploy_v2.prototxt",
    "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
    "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt",
    "https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt",
]

POINTS_URLS = [
    "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/pts_in_hull.npy",
    "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy",
    "https://raw.githubusercontent.com/richzhang/colorization/master/resources/pts_in_hull.npy",
    "https://raw.githubusercontent.com/richzhang/colorization/master/colorization/resources/pts_in_hull.npy",
]

CAFFEMODEL_URLS = [
    "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel",
    "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel",
]


def download_if_missing(urls, path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        print(f"Reusing existing model file: {path}")
        return path

    ensure_dir(path.parent)

    last_error = None
    for url in urls:
        try:
            print(f"Downloading model file: {path.name}")
            print(f"Source: {url}")
            urllib.request.urlretrieve(url, path)
            if path.exists() and path.stat().st_size > 0:
                print(f"Downloaded: {path}")
                return path
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            print(f"Download failed from this source: {error}")
            if path.exists() and path.stat().st_size == 0:
                path.unlink()

    raise RuntimeError(
        f"Could not download {path.name}. Last error: {last_error}. "
        "Download the file manually into models/colorization/ or try again later."
    )


def load_colorization_model():
    prototxt = download_if_missing(PROTOTXT_URLS, MODEL_DIR / "colorization_deploy_v2.prototxt")
    points = download_if_missing(POINTS_URLS, MODEL_DIR / "pts_in_hull.npy")
    caffemodel = download_if_missing(CAFFEMODEL_URLS, MODEL_DIR / "colorization_release_v2.caffemodel")

    print("Loading OpenCV Zhang colorization model...")
    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    pts = np.load(str(points))

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)

    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net


def colorize_frame(gray_bgr, net):
    image = gray_bgr.astype("float32") / 255.0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    resized = cv2.resize(image, (224, 224))
    lab_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    L_resized = lab_resized[:, :, 0]
    L_resized -= 50

    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    result_lab = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    result_bgr = np.clip(result_bgr, 0, 1)
    return (255 * result_bgr).astype("uint8")


def process_video(video_id, gray_frames_dir, output_dir, fps=24):
    print(f"\nMethod: Colourisation 1/2 - OpenCV/Zhang learning-based model")
    print(f"Video ID: {video_id}")
    net = load_colorization_model()

    gray_frames_dir = Path(gray_frames_dir) / video_id
    output_frames_dir = Path(output_dir) / video_id / "frames"
    output_video_path = Path(output_dir) / video_id / f"{video_id}_opencv_zhang_colourisation.mp4"

    ensure_dir(output_frames_dir)

    frame_paths = sorted(gray_frames_dir.glob("*.png"))
    if len(frame_paths) == 0:
        raise RuntimeError(f"No grayscale frame found for {video_id}")

    output_paths = []

    for i, frame_path in enumerate(tqdm(frame_paths)):
        gray_bgr = cv2.imread(str(frame_path))
        result = colorize_frame(gray_bgr, net)

        out_path = output_frames_dir / f"frame_{i:05d}.png"
        cv2.imwrite(str(out_path), result)
        output_paths.append(out_path)

    write_video_from_frames(output_paths, output_video_path, fps=fps)
    print(f"Written video: {output_video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--gray_frames_dir", default="frames/gray")
    parser.add_argument("--output_dir", default="outputs/colourisation/opencv_zhang")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    process_video(
        video_id=args.video_id,
        gray_frames_dir=args.gray_frames_dir,
        output_dir=args.output_dir,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
