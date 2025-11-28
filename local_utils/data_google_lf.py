import json
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

OUT_RESIZE_TO_1080P = True  # set False if you want 2560x1440 instead of 1920x1080


def visualize_camera_positions(models_json_path, plane="xy"):
    models_json_path = Path(models_json_path)

    with models_json_path.open("r") as f:
        data = json.load(f)

    if isinstance(data, list):
        views = data
    elif isinstance(data, dict):
        views = data.get("views", list(data.values()))
    else:
        raise ValueError("Unexpected JSON structure in models.json")

    positions = []
    names = []

    for v in views:
        if "position" not in v:
            continue
        positions.append(v["position"])  # [x, y, z]
        names.append(v.get("name", "unknown"))

    if not positions:
        raise ValueError("No camera positions found")

    xs3 = [p[0] for p in positions]
    ys3 = [p[1] for p in positions]
    zs3 = [p[2] for p in positions]

    if plane == "xy":
        xs, ys = xs3, ys3
        xlabel, ylabel = "X", "Y"
    elif plane == "xz":
        xs, ys = xs3, zs3
        xlabel, ylabel = "X", "Z"
    elif plane == "yz":
        xs, ys = ys3, zs3
        xlabel, ylabel = "Y", "Z"
    else:
        raise ValueError("plane must be one of 'xy', 'xz', 'yz'")

    plt.figure(figsize=(7, 6))
    plt.scatter(xs, ys, s=20)

    for x, y, name in zip(xs, ys, names):
        plt.text(x, y, name.split("_")[1], fontsize=7)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Camera Positions ({plane.upper()} projection)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(models_json_path.parent / "models_2d.png")

def load_models(models_path):
    with open(models_path, "r") as f:
        models = json.load(f)
    # Convert to dict by camera name for easy lookup
    return {view["name"]: view for view in models}


def build_fisheye_maps(view, balance=0.0):
    w = int(view["width"])
    h = int(view["height"])

    f = view["focal_length"]
    cx, cy = view["principal_point"]

    # Intrinsics matrix for the original fisheye
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)

    # OpenCV fisheye model expects up to 4 coefficients
    radial = view["radial_distortion"]
    if len(radial) < 4:
        radial = list(radial) + [0.0] * (4 - len(radial))
    D = np.array(radial[:4], dtype=np.float64)

    # We’ll undistort into the same resolution first
    DIM = (w, h)

    # New camera matrix for undistorted image
    # R = identity (no rectification rotation)
    R = np.eye(3)
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, DIM, R, None, balance=balance
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, K_new, DIM, cv2.CV_32FC1
    )

    return map1, map2, DIM


def crop_to_16x9(image):
    h, w = image.shape[:2]
    # Use full width, crop height
    target_h = int(w * 9 / 16)
    if target_h > h:
        # Fallback: use full height and crop width (shouldn't happen here, but just in case)
        target_h = h
        target_w = int(h * 16 / 9)
        x0 = (w - target_w) // 2
        return image[:, x0:x0 + target_w]
    else:
        y0 = (h - target_h) // 2
        return image[y0:y0 + target_h, :]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def process_scene(scene_dir: Path):
    print(f"Processing scene: {scene_dir.name}")

    models_path = scene_dir / "models.json"
    if not models_path.exists():
        print(f"  No models.json in {scene_dir}, skipping.")
        return

    models = load_models(models_path)

    # Pre-build undistortion maps for each camera
    maps = {}
    for cam_name, view in models.items():
        print(f"  Building maps for {cam_name}")
        maps[cam_name] = build_fisheye_maps(view)

    # Process videos
    for video_path in sorted(scene_dir.glob("*.mp4")):
        cam_name = video_path.stem  # assumes 'camera_0001.mp4' -> 'camera_0001'
        if cam_name not in maps:
            print(f"  Warning: no calibration for {cam_name}, skipping video {video_path}")
            continue

        map1, map2, (w, h) = maps[cam_name]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Could not open {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  {video_path.name}: fps={fps}, size=({w}x{h})")

        # Dummy frame to get output size
        ret, frame = cap.read()
        if not ret:
            print(f"  Empty video {video_path}, skipping.")
            cap.release()
            continue

        # Rewind to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Undistort + crop on one frame to compute writer size
        undist = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        cropped = crop_to_16x9(undist)

        if OUT_RESIZE_TO_1080P:
            out_w, out_h = 1920, 1080
        else:
            out_h, out_w = cropped.shape[:2]  # note: shape is (h, w, c)

        out_path = scene_dir.parent / f"{scene_dir.name}_rect"
        out_path.mkdir(exist_ok=True)
        out_file = out_path / video_path.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file), fourcc, fps, (out_w, out_h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            undist = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            cropped = crop_to_16x9(undist)
            if OUT_RESIZE_TO_1080P:
                cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(cropped)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"    {cam_name}: processed {frame_idx} frames...")

        cap.release()
        writer.release()
        print(f"  Saved {out_path}")


def main():
    DATA_ROOT = Path("/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/google_lightfield")  # folder with all scene zips extracted

    for scene_dir in sorted(DATA_ROOT.iterdir()):
        if scene_dir.is_dir():
            process_scene(scene_dir)

    glf_path = Path("/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/google_lightfield")
    for dir in glf_path.iterdir():
        if dir.is_dir():
            if (dir / "models.json").exists():
                visualize_camera_positions(dir / "models.json")


if __name__ == "__main__":
    main()
