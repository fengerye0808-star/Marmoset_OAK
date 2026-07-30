# Marmoset facial-keypoint tool — setup on a new PC (Python 3.12)

This folder is **self-contained**: the trained model is bundled in `model/`, and the
tool auto-finds both its environment and the model — **no path editing needed**.
It is marmoset-only: the human py-feat tool, the HR/combined tool, the web dashboard,
and the human dataset are intentionally NOT included.

```
marmoset_tool/
  run_marmoset.py      ← launcher
  fer/                 ← the tool (marmoset modules only)
  model/               ← bundled trained model (marmoset_v2_shuffle3.pt)
  analysis/            ← research scripts (fUSi/airpuff/expression) — reference, optional
  requirements.txt
  SETUP.md             ← this file
```

## Prerequisites
- **Python 3.12** (64-bit).
- **NVIDIA GPU + current driver** (strongly recommended; runs on CPU but slowly).
- For live capture: an **OAK-D Pro PoE** camera + a PoE switch/injector.

## Install (once)
Run these **inside this folder**. Creating the venv as `./venv` lets the launcher
find it automatically.

```bat
:: 1) create the environment (named 'venv' so the tool auto-detects it)
py -3.12 -m venv venv

:: 2) install torch with the CUDA build that MATCHES your GPU/driver.
::    Check https://pytorch.org for the right index. Examples:
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
::    (older driver? use cu121 etc.  no NVIDIA GPU? use the default CPU wheel — slow)

:: 3) install the rest
venv\Scripts\python -m pip install -r requirements.txt
```

Verify:
```bat
venv\Scripts\python -c "import torch, dlclive, cv2, depthai; print('cuda:', torch.cuda.is_available())"
```
Expect `cuda: True`.

## Run
```bat
:: live from the OAK camera (a window opens; r=record, q=quit)
venv\Scripts\python run_marmoset.py --source oak

:: or on an existing video, no camera:
venv\Scripts\python run_marmoset.py --source "path\to\clip.mp4"

:: record with metadata:
venv\Scripts\python run_marmoset.py --source oak --working-distance 35 --note "test"
```
Because the venv is `./venv` and the model is in `./model`, you can even launch with a
bare `py run_marmoset.py --source oak` — it auto-relaunches into `./venv`.

## Notes / gotchas
- **torch CUDA build must match the driver.** This is the single most common setup snag.
  Match the `cuXXX` index to your GPU/driver; don't install a plain-PyPI torch on Windows.
- **Python 3.12 compatibility:** the tool's own code is 3.12-clean (verified). `torch`,
  `deeplabcut-live`, and `depthai` all ship 3.12 wheels. If any dependency lacks a 3.12
  wheel on your setup, that's the only likely install snag.
- **GUI window:** `deeplabcut-live` pulls `opencv-python-headless`; `requirements.txt` also
  installs `opencv-python` (GUI) so `cv2.imshow` works. If the window ever fails to open,
  `pip uninstall opencv-python-headless` and keep `opencv-python`. (Or just use `--headless`.)
- **Recording:** webcam/file recording uses OpenCV (no extra deps). OAK H.265 recording is
  muxed to `.mp4` with **ffmpeg** — put `ffmpeg` on PATH if you record from the OAK
  (otherwise the raw `.h265` is kept and still playable).
- **A different model:** drop any exported DLC `.pt` into `model/`, or pass `--model <path>`,
  or set the `MARMOSET_MODEL` env var. The env can also be pointed at via `MARMOSET_PY`.
- **Retraining** (not needed to run) requires the full `deeplabcut` package + the original
  DLC project (15.5 GB) — deliberately excluded here.
