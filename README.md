# AI Video Generation (Free-Tier GPU Edition)

A multi-scene text-to-video pipeline — several AI-generated clips stitched
together with crossfade transitions, each with sound effects generated to
match that scene's content — built entirely on free Colab/Kaggle GPUs, no
local GPU required.

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
4. Edit the scene list (section 4) — **start with 2-3 scenes first** to confirm everything works before running the full set.
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

## Repo structure

```
.
├── notebooks/
│   └── cogvideox_text_to_video.ipynb   # ← The only file you need. Multi-scene + sound, self-contained.
├── src/
│   └── generate.py                     # Optional: single-clip CLI version (no stitching/sound)
├── outputs/
│   └── samples/                        # Optional: a few curated example clips/GIFs (generated files are gitignored)
├── requirements.txt                    # Optional: only needed for src/generate.py; the notebook installs its own deps in-cell
└── README.md
```

If you want the absolute minimal version of this repo, it's safe to delete
`src/`, `outputs/`, and `requirements.txt` entirely — the notebook has no
dependency on any of them.

## Troubleshooting

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
