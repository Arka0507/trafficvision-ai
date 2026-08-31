# GitHub and deployment

## 1. Push the source to GitHub

The repository is ready for Git. Generated video, model weights, environment files, and virtual environments are ignored.

```bash
git init
git add .
git commit -m "Initial TrafficVision AI release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/trafficvision-ai.git
git push -u origin main
```

GitHub Actions runs compilation, static checks, and unit tests on pushes and pull requests.

## 2. Choose a deployment target

GitHub Pages is unsuitable because the application needs Python/PyTorch inference, FFmpeg, background processing, and writable storage.

Use a container service with:

- at least 4 CPU cores for practical CPU inference;
- 8 GB RAM recommended;
- persistent disk for `data/jobs` and the model cache;
- upload limits above your expected video size;
- an NVIDIA GPU for higher throughput when available.

Container platforms such as Railway, Render, Fly.io, AWS, Azure, Google Cloud, or a GPU provider can run the included Dockerfile. Compare their current limits and pricing before selecting one.

## 3. Build and test the image locally

```bash
docker compose up --build
```

Open `http://localhost:8000`, upload a short video, and verify all downloads.

## 4. Runtime configuration

| Variable | Default | Meaning |
| --- | ---: | --- |
| `MAX_UPLOAD_MB` | `2048` | Maximum accepted upload size |
| `MAX_PROCESSING_WORKERS` | `1` | Simultaneous video jobs in one server process |
| `TRAFFICVISION_DATA_DIR` | `./data` | Uploaded and generated job storage |
| `CORS_ORIGINS` | local URLs | Allowed external frontend origins |

Run one Uvicorn process for the included in-memory job manager. `MAX_PROCESSING_WORKERS=1` avoids exhausting CPU/GPU memory. For horizontal scaling, replace the local executor/job dictionary with a durable queue such as Redis + Celery/RQ and put artifacts in object storage.

## 5. Production concerns

- Put TLS in front of the application.
- Configure reverse-proxy body-size limits above `MAX_UPLOAD_MB`.
- Mount persistent storage at `/app/data` and preserve the model cache.
- Add authentication before accepting public uploads.
- Add retention cleanup for old videos and reports.
- Monitor disk, RAM, GPU memory, queue depth, job duration, and failures.
- Treat uploaded videos as sensitive data.
- Scan custom model weights and never load arbitrary user-supplied pickle checkpoints.

The current vehicle loader uses `torch.load(..., weights_only=True)` to avoid executing arbitrary pickled code from the checkpoint.

## 6. GPU image

The supplied Dockerfile intentionally uses CPU PyTorch for broad compatibility. For NVIDIA deployment:

1. Start from an NVIDIA CUDA runtime compatible with your host driver.
2. Install the matching CUDA PyTorch build using the official PyTorch instructions.
3. Keep the rest of `requirements.txt` unchanged.
4. Ensure the container runtime exposes the GPU.
5. Verify `torch.cuda.is_available()` before production traffic.

The processing options use `device=auto` by default and choose CUDA when it is available to the vehicle classifier; Ultralytics likewise uses its automatic device selection when `auto` is selected.
