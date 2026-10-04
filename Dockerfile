# One-Prompt AI Video -- containerized for any Docker host with an attached
# NVIDIA GPU (needs the host to have the NVIDIA driver + nvidia-container-
# toolkit installed; `docker run --gpus all ...`). Not needed for Kaggle/Colab
# -- there, run app.py directly in a notebook cell instead (see the README).
#
# A slim Python base is enough: modern `pip install torch` wheels bundle
# their own CUDA runtime libraries, so this image doesn't need an
# nvidia/cuda-flavored base -- only the host needs the actual GPU driver.
FROM python:3.12-slim

# ffmpeg: imageio-ffmpeg bundles its own static binary that the app uses
# directly, but having the system package too is a cheap safety net.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860

# GRADIO_SHARE defaults to false here -- map the port yourself with
# `-p 7860:7860` rather than relying on Gradio's own share tunnel, since a
# real deployment target controls its own ingress/HTTPS.
ENV GRADIO_SHARE=false

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

CMD ["python", "app.py"]
