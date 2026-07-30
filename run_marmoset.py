#!/usr/bin/env python3
"""
run_marmoset.py — live MARMOSET facial-keypoint tool (OAK camera or video file).

This is the marmoset twin of run_fer.py. It runs the trained DeepLabCut model
(24 facial keypoints) through DLC-Live and shows the keypoints live, plus the
derived expression measures (eye/mouth opening etc.). It saves recordings in
exactly the same layout as run_fer.py, so the analysis/visualisation scripts
work unchanged.

The marmoset engine needs a Python env with dlclive + torch + opencv + depthai
(numpy<2). This launcher runs directly if the current Python already has those,
otherwise it re-execs into a venv it can find (see below) — no machine-specific
path baked in. Set up the env once per PC (see SETUP.md).

Examples:
    # live from the OAK camera
    python run_marmoset.py --source oak

    # try it on an existing marmoset video (no camera needed)
    python run_marmoset.py --source "path\\to\\clip_Cam1_1.mp4"

    # headless 15-second recording test
    python run_marmoset.py --source oak --headless --auto-record --duration 15
"""

from __future__ import annotations

import os
import subprocess
import sys

# --- Run in an environment that has the marmoset deps (dlclive/torch/cv2) -----
# If the current Python already has them, just run. Otherwise re-exec into a venv
# we can find. PORTABLE: no hard-coded machine path. Resolution order:
#   $MARMOSET_PY  ->  ./venv or ./.venv next to this file  ->  (legacy D:\dlc3).
# If none works, print a clear setup hint (see SETUP.md).
def _have_deps() -> bool:
    import importlib.util as u
    return all(u.find_spec(m) is not None for m in ("dlclive", "cv2", "numpy", "torch"))


if not _have_deps():
    _here = os.path.dirname(os.path.abspath(__file__))
    _cands = [
        os.environ.get("MARMOSET_PY"),
        os.path.join(_here, "venv", "Scripts", "python.exe"),
        os.path.join(_here, ".venv", "Scripts", "python.exe"),
        os.path.join(_here, "venv", "bin", "python"),          # non-Windows
        r"D:\dlc3\venv\Scripts\python.exe",                     # legacy fallback
    ]
    _self = os.path.normcase(os.path.abspath(sys.executable))
    for _c in _cands:
        if _c and os.path.exists(_c) and os.path.normcase(os.path.abspath(_c)) != _self:
            sys.stderr.write(f"[run_marmoset] switching to env: {_c}\n")
            sys.exit(subprocess.run([_c, os.path.abspath(__file__), *sys.argv[1:]]).returncode)
    sys.exit("[run_marmoset] marmoset deps (dlclive/torch/cv2) not found in this Python.\n"
             "  Set up the env (see SETUP.md), then either run it with that env's python,\n"
             "  set MARMOSET_PY to its python.exe, or place the venv in ./venv next to this file.")
# -----------------------------------------------------------------------------

import argparse

from fer import app
from fer.source import make_source


def main() -> None:
    p = argparse.ArgumentParser(description="Real-time marmoset facial-keypoint capture")
    p.add_argument("--source", default="oak",
                   help="'oak', 'webcam', or a path to a video file")
    p.add_argument("--model", default=None,
                   help="path to the exported DLC .pt model "
                        "(default: the v2 shuffle-3 model)")
    p.add_argument("--device", default="cuda", help="cuda | cpu")
    p.add_argument("--pcutoff", type=float, default=0.5,
                   help="min keypoint confidence to count as a detection")
    p.add_argument("--out",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "recordings"),
                   help="output directory (default: <project>/recordings)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--headless", action="store_true", help="no display window")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--duration", type=float, default=None,
                   help="stop after N seconds")
    p.add_argument("--no-landmarks", action="store_true",
                   help="do not save the 24 keypoints to frames.csv")
    # recording controls (same semantics as run_fer.py)
    p.add_argument("--record", choices=["data", "video", "both", "none"],
                   default="both",
                   help="what a recording saves (change live with the buttons)")
    p.add_argument("--auto-record", action="store_true",
                   help="start recording immediately (needed in --headless)")
    p.add_argument("--record-width", type=int, default=1920,
                   help="H.265 archive width (OAK)")
    p.add_argument("--record-height", type=int, default=1080)
    # acquisition parameters recorded into each session's metadata
    p.add_argument("--working-distance", type=float, default=None,
                   help="camera-to-face distance in cm (recorded in metadata)")
    p.add_argument("--physio-rate", type=float, default=None,
                   help="physio recorder sample rate in Hz (recorded in metadata)")
    p.add_argument("--note", default="",
                   help="free text: WHY these parameters were chosen")
    args = p.parse_args()

    from fer.marmoset_backend import MarmosetBackend, DEFAULT_MODEL
    from fer.recorder import Recorder

    model_path = args.model or DEFAULT_MODEL
    if not os.path.exists(model_path):
        sys.exit(f"[run_marmoset] model not found:\n  {model_path}\n"
                 f"Pass --model <path-to-.pt> if it lives elsewhere.")

    print(f"[run_marmoset] loading model:\n  {model_path}")
    backend = MarmosetBackend(model_path=model_path, device=args.device,
                              pcutoff=args.pcutoff)
    source = make_source(args.source, width=args.width, height=args.height,
                         fps=args.fps, record=args.record != "none",
                         record_size=(args.record_width, args.record_height),
                         record_fps=args.fps)
    recorder = Recorder(args.out, backend, source,
                        save_landmarks=not args.no_landmarks,
                        params={"working_distance_cm": args.working_distance,
                                "camera_fps": args.fps,
                                "physio_sample_rate_hz": args.physio_rate,
                                "species": "marmoset",
                                "model_path": model_path,
                                "parameter_notes": args.note})
    recorder.want_data = args.record in ("data", "both")
    recorder.want_video = args.record in ("video", "both")

    print(f"source={source.name} backend={backend.name} "
          f"encoder={'yes' if source.has_encoder else 'no'} -> {args.out}")
    stats = app.run(source, backend, recorder, show=not args.headless,
                    max_frames=args.max_frames, duration_s=args.duration,
                    auto_record=args.auto_record)
    print("done:", stats)


if __name__ == "__main__":
    main()
