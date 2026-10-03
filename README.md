# AI Video Generation (Free-Tier GPU Edition)

Text-to-video generation using [CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b),
an open-source (Apache 2.0) video diffusion model — set up to run entirely on
**free GPU runtimes** (Google Colab / Kaggle Notebooks), no local GPU required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alirezayaned/AI-Video-Generation/blob/main/notebooks/cogvideox_text_to_video.ipynb)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

## Sample output

<!-- Add a generated GIF here once you've run the notebook, e.g.: -->
<!-- ![sample output](outputs/samples/sample_01.gif) -->

## Why this model

Video diffusion models are usually VRAM-hungry (many need 16–80GB+). CogVideoX-2b
is one of the lightest options available — it runs in as little as ~4GB of VRAM
with fp16 precision, which comfortably fits a free Colab/Kaggle T4 GPU (16GB VRAM).

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
4. **Run All.**

That's the entire setup on either platform — no datasets to attach, no utility
scripts to add.

### Optional: running as a standalone script instead

If you'd rather run generation outside a notebook (e.g. on your own GPU
machine), `src/generate.py` has the same pipeline as a CLI. This is entirely
optional — skip it if the notebook covers what you need.

```bash
pip install -r requirements.txt
python src/generate.py --prompt "a cat surfing on a tiny wave" --output outputs/surf_cat.mp4
```

Add `--low-vram` if you're running on a GPU smaller than 16GB — it trades
generation speed for a much smaller memory footprint via CPU offload and VAE
tiling.

## Repo structure

```
.
├── notebooks/
│   └── cogvideox_text_to_video.ipynb   # ← The only file you need. Self-contained, run on Colab/Kaggle.
├── src/
│   └── generate.py                     # Optional: same pipeline as a CLI script, for running outside a notebook
├── outputs/
│   └── samples/                        # Optional: a few curated example clips/GIFs (generated files are gitignored)
├── requirements.txt                    # Optional: only needed for src/generate.py; the notebook installs its own deps in-cell
└── README.md
```

If you want the absolute minimal version of this repo, it's safe to delete
`src/`, `outputs/`, and `requirements.txt` entirely — the notebook has no
dependency on any of them.

## Hardware notes

- Free-tier GPUs aren't guaranteed on demand and sessions are capped —
  Colab allows up to ~12 hours per session with idle timeouts, and Kaggle
  gives roughly 30 GPU-hours per week with a 12-hour session cap. Plan batch
  generations accordingly rather than re-running cells one at a time.
- CogVideoX-2b generates 49-frame clips (~6 seconds at 8fps) at 720×480,
  which is the resolution/length it was trained on.

## Next steps / ideas

- Swap in [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5)
  via the [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) runner for higher
  quality output — still fits in ~6GB VRAM.
- Wrap `generate_video()` in a [Gradio](https://www.gradio.app/) UI and deploy
  to a free [Hugging Face Space](https://huggingface.co/spaces) for a live, shareable demo.
- Add image-to-video generation using `CogVideoXImageToVideoPipeline`.

## License

This project's code is MIT-licensed (add a `LICENSE` file if you want this
explicit). CogVideoX-2b itself is released under Apache 2.0 by THUDM.
