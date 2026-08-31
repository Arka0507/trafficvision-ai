# Run TrafficVision AI in Antigravity

These steps are written for Windows, where Antigravity is most commonly used.

## 1. Install prerequisites once

- Python 3.10, 3.11, or 3.12 from [python.org](https://www.python.org/downloads/). During installation, enable **Add Python to PATH**.
- Git from [git-scm.com](https://git-scm.com/downloads) if you want to push to GitHub.
- FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) is recommended for browser-compatible H.264 output. Add its `bin` folder to PATH.

## 2. Open the project

1. Extract `trafficvision-ai.zip`.
2. In Antigravity, choose **Open Folder** and select the extracted `trafficvision-ai` folder.
3. Open **Terminal → New Terminal**.

## 3. Create the environment

In the Antigravity terminal:

```bat
.\setup_windows.bat
```

This creates `.venv` and installs the detector, tracker, classifier, API, and video libraries. It can take several minutes.

If Windows blocks the batch file, use these equivalent commands:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, run once in that terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again.

## 4. Start the application

```bat
.\start_windows.bat
```

Open [http://localhost:8000](http://localhost:8000) if it does not open automatically.

Do not close the terminal while the application is running. Stop it with `Ctrl+C`.

## 5. Analyze a video

1. Drop the video onto the upload area.
2. Keep **640 px · Balanced** for the first run.
3. Keep car make/model on if you want the second-stage classifier. Turn it off for maximum speed.
4. Set the camera FOV if known. Phone and CCTV cameras are often roughly 55–90°, but use the actual specification when possible.
5. Click **Analyze video**.
6. Download the annotated video, track CSV, frame CSV, and JSON summary.

The first run downloads two model files. This is normal; later jobs reuse the local cache.

## Optional NVIDIA GPU acceleration

1. Install a current NVIDIA driver.
2. Activate the project environment.
3. Use the official selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to install the PyTorch build matching your driver/CUDA setup.
4. In the web request, the default `auto` device uses CUDA when available. The command-line runner accepts `--device 0`.

Verify GPU detection:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Put the project on GitHub

Create an empty GitHub repository, then run from the project root:

```bash
git init
git add .
git commit -m "Build end-to-end TrafficVision AI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/trafficvision-ai.git
git push -u origin main
```

Model weights, uploaded videos, generated jobs, `.env`, and `.venv` are already excluded by `.gitignore`.

## Common fixes

### `python` or `py` is not recognized

Reinstall Python and enable **Add Python to PATH**, then restart Antigravity.

### Output video does not play in the browser

Install FFmpeg and confirm this command works:

```powershell
ffmpeg -version
```

Restart the server and process the video again.

### Processing is slow

- Use 512 or 640 inference resolution.
- Turn off car make/model recognition.
- Use a shorter video for testing.
- Enable a compatible NVIDIA GPU build of PyTorch.

### Car model is `Uncertain`

That is intentional when the crop is too small, confidence is low, or predictions across frames disagree. The classifier only knows its 196 Stanford Cars categories. Review [docs/CAR_MODEL_LIMITATIONS.md](docs/CAR_MODEL_LIMITATIONS.md).

### Speed is wrong or unstable

Use a fixed camera and complete road-plane calibration with measured ground coordinates. Default values are estimates, not radar measurements. Review [docs/CALIBRATION.md](docs/CALIBRATION.md).
