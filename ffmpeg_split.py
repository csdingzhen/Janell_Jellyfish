import argparse
import os
import subprocess
import math
import cv2
import numpy as np

# Configuration constants (change these directly for debugging)
INPUT_DIR = r"E:\Janelle_babyJs"  # folder containing all input videos
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')
OUTPUT_DIR = r"E:\Janelle_babyJs\output"
PADDING = 1.2
CODEC = "libx264"  # e.g. libx264 or h264_nvenc
HWACCEL = None  # e.g. "cuda" or None
SCALE = None  # e.g. (640, 640) or None to use detected size
DEBUG = True
SELECT_GRID = True
SELECT_ROIS = False
DURATION = None  # seconds, or None to process full video
FFMPEG_BIN = None  # set to full path to ffmpeg.exe if not on PATH, e.g. r"C:\ffmpeg\bin\ffmpeg.exe"


def detect_wells(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=20, maxRadius=200)

    centers = []
    radii = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda x: -x[2])[:9]
        for (x, y, r) in circles:
            centers.append((x, y))
            radii.append(r)

    if len(centers) < 9:
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            circularity = (4 * math.pi * area) / (cv2.arcLength(cnt, True) ** 2 + 1e-6)
            if 0.4 < circularity <= 1.3:
                cand.append((int(x), int(y), int(r)))
        cand = sorted(cand, key=lambda x: -x[2])[:9]
        centers = [(c[0], c[1]) for c in cand]
        radii = [c[2] for c in cand]

    if len(centers) != 9:
        h, w = frame.shape[:2]
        grid_w = w // 3
        grid_h = h // 3
        centers = []
        radii = []
        for ry in range(3):
            for rx in range(3):
                cx = rx * grid_w + grid_w // 2
                cy = ry * grid_h + grid_h // 2
                centers.append((cx, cy))
                radii.append(min(grid_w, grid_h) // 3)

    pts = np.array(centers)
    order = np.argsort(pts[:, 1])
    rows = [order[i * 3:(i + 1) * 3] for i in range(3)]
    sorted_centers = []
    sorted_radii = []
    for r_idx in range(3):
        row_idxs = rows[r_idx]
        row_pts = pts[row_idxs]
        xs = row_pts[:, 0]
        sorted_row_order = row_idxs[np.argsort(xs)]
        for idx in sorted_row_order:
            sorted_centers.append(tuple(pts[idx]))
            sorted_radii.append(radii[idx])

    return sorted_centers, sorted_radii


def compute_crops(frame, centers, radii, padding=1.2):
    crops = []
    sizes = []
    h, w = frame.shape[:2]
    for (cx, cy), r in zip(centers, radii):
        half = int(r * padding)
        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        x2 = min(w, int(cx + half))
        y2 = min(h, int(cy + half))
        ww = x2 - x1
        hh = y2 - y1
        s = max(ww, hh)
        cx2 = int((x1 + x2) / 2)
        cy2 = int((y1 + y2) / 2)
        x1 = max(0, cx2 - s // 2)
        y1 = max(0, cy2 - s // 2)
        x2 = min(w, x1 + s)
        y2 = min(h, y1 + s)
        x1 = max(0, x2 - s)
        y1 = max(0, y2 - s)
        crops.append((x1, y1, x2 - x1, y2 - y1))
        sizes.append((s, s))
    return crops, sizes


def build_filter_complex(crops, out_size):
    # crops: list of (x,y,w,h)
    # out_size: (w,h) or None for no scale
    filters = []
    for i, (x, y, w, h) in enumerate(crops):
        chain = f"[0:v]crop={w}:{h}:{x}:{y}"
        if out_size is not None:
            chain += f",scale={out_size[0]}:{out_size[1]}"
        chain += f"[v{i}]"
        filters.append(chain)
    return ";".join(filters)


def run_ffmpeg(input_path, output_dir, crops, out_size, codec, hwaccel=None, extra_args=None, duration=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    filter_complex = build_filter_complex(crops, out_size)
    # Locate ffmpeg executable: prefer configured FFMPEG_BIN, else look on PATH
    import shutil
    ffmpeg_exec = FFMPEG_BIN if globals().get('FFMPEG_BIN') else shutil.which("ffmpeg")
    if not ffmpeg_exec:
        raise RuntimeError("ffmpeg not found: install ffmpeg and add it to PATH, or set FFMPEG_BIN in the script to the full path to ffmpeg.exe")
    cmd = [ffmpeg_exec, "-y"]
    if hwaccel:
        # hwaccel is passed as -hwaccel <name> (ffmpeg will use if available)
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", input_path]
    if duration:
        # limit processing time (seconds)
        cmd += ["-t", str(duration)]
    cmd += ["-filter_complex", filter_complex]

    # add mapping and codec for each output; each well goes into its own subfolder
    base = os.path.splitext(os.path.basename(input_path))[0]
    for i in range(len(crops)):
        well_dir = os.path.join(output_dir, f"well_{i+1}")
        os.makedirs(well_dir, exist_ok=True)
        out_path = os.path.join(well_dir, f"{base}.mp4")
        cmd += ["-map", f"[v{i}]", "-c:v", codec, out_path]

    if extra_args:
        # append any extra args (user-provided)
        cmd += extra_args

    print("Running ffmpeg command:")
    print(" ".join(cmd if len(" ".join(cmd)) < 2000 else cmd[:6] + ['...']))
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Split a 3x3 petri-dish video into 9 videos using ffmpeg")
    p.add_argument("input", nargs='?', default=INPUT_VIDEO, help="Input video path (optional). Change INPUT_VIDEO constant in the file to set a default.")
    p.add_argument("-o", "--out", default="output_ffmpeg", help="Output directory")
    p.add_argument("--padding", type=float, default=1.2, help="Crop padding multiplier")
    p.add_argument("--codec", default="libx264", help="ffmpeg codec for outputs (e.g. libx264, h264_nvenc)")
    p.add_argument("--hwaccel", default=None, help="ffmpeg hwaccel option (e.g. cuda, vaapi)")
    p.add_argument("--scale", type=int, nargs=2, metavar=("W", "H"), help="Scale outputs to W H; if omitted uses detected crop size")
    p.add_argument("--debug", action="store_true", help="Save detection debug image")
    p.add_argument("--select-grid", action="store_true", help="Manually draw one ROI that will be subdivided into a 3x3 grid")
    p.add_argument("--select-rois", action="store_true", help="Manually select multiple ROIs (via mouse). Use ESC/Enter when done")
    p.add_argument("--duration", type=float, default=None, help="Limit processed duration in seconds (for testing)")
    return p.parse_args()


def main():
    # Use module-level constants instead of CLI args
    out_dir = OUTPUT_DIR
    padding = PADDING
    codec = CODEC
    hwaccel = HWACCEL
    scale = SCALE
    debug = DEBUG
    select_grid = SELECT_GRID
    select_rois = SELECT_ROIS
    duration = DURATION

    # Collect all video files from INPUT_DIR, sorted for consistent ordering
    video_files = sorted(
        os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )
    if not video_files:
        raise RuntimeError(f"No video files found in: {INPUT_DIR}")
    print(f"Found {len(video_files)} video(s) to process in: {INPUT_DIR}")

    # Determine crops from the first video (same camera setup assumed for all)
    cap = cv2.VideoCapture(video_files[0])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_files[0]}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame from: {video_files[0]}")

    # Selection options: manual or automatic (done once, reused for all videos)
    crops = None
    sizes = None
    if select_grid:
        # let user draw a single ROI to subdivide into 3x3
        print("Draw a single ROI on the window and press ENTER/SPACE. Press ESC to cancel.")
        roi = cv2.selectROI("Select grid region", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select grid region")
        if roi is None or roi == (0, 0, 0, 0):
            raise RuntimeError("No ROI selected")
        x, y, w, h = [int(v) for v in roi]
        # subdivide into 3x3
        cell_w = w // 3
        cell_h = h // 3
        crops = []
        sizes = []
        for ry in range(3):
            for rx in range(3):
                cx = x + rx * cell_w
                cy = y + ry * cell_h
                # last column/row take remainder
                cw = cell_w if rx < 2 else (w - cell_w * 2)
                ch = cell_h if ry < 2 else (h - cell_h * 2)
                crops.append((cx, cy, cw, ch))
                sizes.append((cw, ch))
        print(f"Created {len(crops)} crops from selected grid")
    elif select_rois:
        print("Select multiple ROIs. Finish selection with ENTER/ESC.")
        rois = cv2.selectROIs("Select ROIs", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROIs")
        if len(rois) == 0:
            raise RuntimeError("No ROIs selected")
        crops = []
        sizes = []
        for (x, y, w, h) in rois:
            crops.append((int(x), int(y), int(w), int(h)))
            sizes.append((int(w), int(h)))
        print(f"Selected {len(crops)} ROIs")
    else:
        centers, radii = detect_wells(frame)
        crops, sizes = compute_crops(frame, centers, radii, padding=padding)

    # unify output size
    if scale:
        out_size = (scale[0], scale[1])
    else:
        max_s = max([s for s, _ in sizes]) if sizes else min(frame.shape[:2])
        out_size = (max_s, max_s)

    # libx264 (and most codecs) require width and height to be divisible by 2;
    # round each dimension up to the nearest even number to avoid encoder errors
    out_size = (out_size[0] + out_size[0] % 2, out_size[1] + out_size[1] % 2)

    if debug:
        vis = frame.copy()
        if crops is not None:
            for i, (x, y, w, h) in enumerate(crops):
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis, str(i + 1), (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            for i, ((cx, cy), r) in enumerate(zip(centers, radii)):
                cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                cv2.putText(vis, str(i + 1), (int(cx) - 10, int(cy) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        dbg_path = os.path.join(out_dir, os.path.splitext(os.path.basename(video_files[0]))[0] + "_detection_ffmpeg.png")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(dbg_path, vis)
        print("Saved debug image:", dbg_path)

    # Process each video with the same crops
    for idx, input_path in enumerate(video_files):
        print(f"\n[{idx+1}/{len(video_files)}] Processing: {os.path.basename(input_path)}")
        run_ffmpeg(input_path, out_dir, crops, out_size, codec, hwaccel=hwaccel, duration=duration)

    print(f"\nDone. {len(video_files)} video(s) split into {len(crops)} wells each. Outputs in: {out_dir}")


if __name__ == '__main__':
    main()
