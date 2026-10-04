# AI Video Generation (Free-Tier GPU Edition)

Write **one sentence** describing the video you want. A small local language
model expands it into a shot list, generates each scene, stitches them
together with crossfade transitions, and adds sound effects matched to each
scene's content — all on a free Colab/Kaggle GPU, no local GPU and no API
key required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alirezayaned/AI-Video-Generation/blob/main/notebooks/cogvideox_text_to_video.ipynb)
![License](https://img.shields.io/badge/license-mixed-lightgrey)

## Sample output

<!-- Add a generated GIF here once you've run the notebook, e.g.: -->
<!-- ![sample output](outputs/samples/sample_01.gif) -->

## What this actually does (and doesn't)

No open video model that fits a free GPU generates one continuous minute-long
shot — [CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b) natively
produces ~6 seconds (49 frames) per generation, and pushing that much further
degrades quality sharply since it's outside what the model was trained on.
This project works around that by:

1. Generating a **sequence of short scenes** from a list of prompts you define.
2. Stitching them together with **crossfade transitions** (not hard cuts) using ffmpeg.
3. Generating a **matching sound effect per scene** with AudioLDM2 (birdsong
   for a forest shot, waves for a beach shot — not one generic soundtrack
   over everything), and muxing each one in before the final stitch.

The result is a longer, more natural-feeling montage — not a single unbroken
AI-generated shot, which isn't something free-tier hardware (or most paid
hardware) can currently do.

## Why these models

| Stage | Model | Footprint | License |
|---|---|---|---|
| Scene planning (one prompt → shot list) | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | ~6GB **system RAM**, CPU only — doesn't touch the GPU | Apache 2.0 |
| Video | [CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b) | ~11GB VRAM (with CPU offload + VAE tiling — loaded naively it wants ~33GB) | Apache 2.0 |
| Ambient sound effects | [AudioLDM2](https://huggingface.co/cvssp/audioldm2) | ~a few GB VRAM, short clips | **CC-BY-NC-SA-4.0 (non-commercial)** |
| Narration (optional, replaces ambient sound) | [Kokoro-82M](https://github.com/hexgrad/kokoro) | ~82M params, runs with room to spare | Apache 2.0 |

All three fit a free Colab/Kaggle T4 (~15GB VRAM) **one at a time** — the
notebook frees each model from GPU memory before loading the next.

**License reminder:** AudioLDM2 is non-commercial. If you need this project
to stay Apache 2.0 end-to-end, use the narration step instead of the
ambient-sound step (it replaces it, not adds to it).

## Quickstart

**The only file you need is `notebooks/cogvideox_text_to_video.ipynb`.** It's
fully self-contained — every function it uses is defined in its own cells, so
it doesn't import anything else from this repo. You don't need to clone the
repo, link GitHub, or attach any other file on Kaggle.

**On Colab:** click the **Open in Colab** badge above.

**On Kaggle:**
1. Go to kaggle.com/code → **New Notebook**.
2. **File → Import Notebook → Upload** and select `cogvideox_text_to_video.ipynb` (download it from the repo first, or drag it in directly).
3. In the right sidebar: **Settings → Accelerator → GPU T4 x2**.
4. Edit `user_theme` (section 4) to describe your video in one sentence — **set `TARGET_SCENES` to 2-3 first** to confirm everything works before running the full-length version.
5. **Run All.**

That's the entire setup on either platform — no datasets to attach, no utility
scripts to add.

### Optional: running a single clip as a standalone script instead

`src/generate.py` covers the single-clip part of this pipeline (one prompt in,
one silent video out) as a CLI, for running outside a notebook. It does not
include the multi-scene stitching or sound generation — those currently only
exist in the notebook. This is entirely optional — skip it if the notebook
covers what you need.

```bash
pip install -r requirements.txt
python src/generate.py --prompt "a cat surfing on a tiny wave" --output outputs/surf_cat.mp4
```

CPU offload + VAE tiling are on by default (needed for a free-tier GPU). If
you're running on a large GPU (32GB+) and want faster generation instead, add
`--fast` to skip offloading.

## Interactive UI (optional)

The whole pipeline (one-sentence prompt → scene planning → per-scene video +
matching sound → crossfade stitching) is also wrapped in a small Gradio app
instead of running notebook cells one at a time.

**Easiest: run it from the notebook.** Section 11 of
`cogvideox_text_to_video.ipynb` writes this same `app.py` to Kaggle's working
directory and runs it with a public share link (`https://....gradio.live`,
live only as long as that cell keeps running, ~72h max) — no separate files
needed, same self-contained-notebook approach as the rest of this project.

**For other targets later:** `app/` has the standalone version plus a
`Dockerfile`, for running this anywhere with an NVIDIA GPU attached (a cloud
GPU host, your own machine with `nvidia-container-toolkit`, etc.):

```bash
cd app
docker build -t one-prompt-video .
docker run --gpus all -p 7860:7860 \
  -v huggingface_cache:/root/.cache/huggingface \
  one-prompt-video
```

Then open `http://localhost:7860` (or the host's address, if run remotely).
The volume mount persists downloaded model weights across container
restarts — without it, each restart re-downloads CogVideoX-2b + AudioLDM2 +
Qwen2.5 from scratch. `GRADIO_SHARE=true` as an env var switches it to
Gradio's own share-link mode instead of binding `0.0.0.0:7860`, same as the
Kaggle path above — useful if the host doesn't have a port you can expose
directly.

## Repo structure

```
.
├── notebooks/
│   └── cogvideox_text_to_video.ipynb   # ← The only file you need for Kaggle/Colab. Self-contained.
├── app/
│   ├── app.py                          # Standalone Gradio UI (same pipeline, for Docker/other targets)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
├── src/
│   └── generate.py                     # Optional: single-clip CLI version (no stitching/sound/UI)
├── outputs/
│   └── samples/                        # Optional: a few curated example clips/GIFs (generated files are gitignored)
├── requirements.txt                    # Optional: only needed for src/generate.py; the notebook installs its own deps in-cell
└── README.md
```

If you want the absolute minimal version of this repo, it's safe to delete
`src/`, `app/`, `outputs/`, and `requirements.txt` entirely — the notebook
has no dependency on any of them.

## Troubleshooting

- **`AttributeError: 'GPT2Model' object has no attribute '_update_model_kwargs_for_generation'`
  when generating ambient sound:** a real, verified upstream incompatibility, not
  a bug in this project's code. `transformers>=4.52.0` stopped automatically
  giving every model class (including AudioLDM2's bare `GPT2Model` language
  model component) generation-mixin methods that AudioLDM2's diffusers pipeline
  code depends on unconditionally. Pinning old versions to dodge this turns out
  to conflict with modern Gradio (which needs a newer `huggingface-hub` than
  the old transformers/diffusers combo supports), so instead the AudioLDM2
  loading cell replaces that one method (`generate_language_model`) with a
  self-contained version that doesn't depend on transformers internals at all
  -- works regardless of which transformers version is installed, no pin
  needed. If you're calling AudioLDM2 somewhere this patch isn't applied,
  copy it from that cell before instantiating the pipeline.
- **Hit a CUDA out-of-memory error?** Restart the kernel/runtime before trying
  again — don't just re-run cells. GPU memory from a crashed `.to()` call
  doesn't reliably get freed within the same session, so a second attempt in
  the same kernel can fail even when the actual memory requirement would
  otherwise fit. The notebook includes a memory-check cell near the top that
  flags this if it happens.
- **Muxing audio onto video:** the notebook uses `ffmpeg` directly (via
  `imageio_ffmpeg`) for combining video and audio, rather than moviepy's
  `write_videofile()` — that function has a confirmed, unresolved bug in
  moviepy 1.0.3 ([Zulko/moviepy#2158](https://github.com/Zulko/moviepy/issues/2158))
  where it can lose track of the frame rate entirely. This project doesn't
  depend on moviepy at all.

## Hardware & time notes

- Free-tier GPUs aren't guaranteed on demand and sessions are capped —
  Colab allows up to ~12 hours per session with idle timeouts, and Kaggle
  gives roughly 30 GPU-hours per week with a 12-hour session cap.
- Each scene is a 49-frame clip (~6.125 seconds at 8fps) at 720×480 — CogVideoX-2b's
  native trained length. Total video length = `N scenes × 6.125s − (N−1) × crossfade_overlap`;
  the default 10 scenes with a 0.75s crossfade comes out to ~54 seconds.
- **Generating many clips takes a while.** Each clip is several minutes on a
  free T4, so a full 10-scene run can take 30-60+ minutes. Start small.

## License

This project's code is MIT-licensed (add a `LICENSE` file if you want this
explicit). The models it calls have their own, different licenses — see the
table above. The pipeline as a whole is only as permissive as its most
restrictive model (currently AudioLDM2, non-commercial) unless you swap that
step out.
