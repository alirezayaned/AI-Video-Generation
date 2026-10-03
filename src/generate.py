"""
Text-to-video generation with CogVideoX-2b.

A reusable, scriptable version of the notebook in notebooks/ — same pipeline,
callable from the command line so it isn't locked inside a notebook.

Usage:
    python src/generate.py --prompt "a cat surfing on a tiny wave" --output outputs/surf_cat.mp4

Requires a CUDA GPU. CogVideoX-2b needs ~33GB of VRAM with no optimization at
all, so this script enables CPU offload + VAE tiling by default, which brings
that down to ~11GB -- small enough to fit a free Colab/Kaggle T4 (~15GB VRAM).
Pass --fast if you have a larger GPU (32GB+) and want to skip offloading for
quicker generation. See the repo README for setup on Colab/Kaggle.
"""

import argparse
import os

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

MODEL_ID = "THUDM/CogVideoX-2b"


def load_pipeline(fast: bool = False) -> CogVideoXPipeline:
    """Load the CogVideoX-2b pipeline.

    Args:
        fast: if True, skips CPU offload/tiling and loads the whole model
            onto the GPU directly. Needs ~33GB VRAM -- only use this on a
            large GPU (e.g. A100/L4 40GB+). Default (False) uses CPU offload
            + VAE tiling (~11GB VRAM), which is what a free-tier T4 needs.
    """
    pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)

    if fast:
        pipe = pipe.to("cuda")
    else:
        # enable_model_cpu_offload() manages device placement itself --
        # don't also call pipe.to("cuda") on top of it.
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()

    return pipe


def generate_video(
    pipe: CogVideoXPipeline,
    prompt: str,
    output_path: str = "outputs/output.mp4",
    num_frames: int = 49,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = 42,
    fps: int = 8,
) -> str:
    """Generate a short video clip from a text prompt and save it to disk."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    frames = pipe(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).frames[0]

    export_to_video(frames, output_path, fps=fps)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a video clip from a text prompt with CogVideoX-2b.")
    parser.add_argument("--prompt", required=True, help="Text prompt describing the video to generate.")
    parser.add_argument("--output", default="outputs/output.mp4", help="Path to save the generated .mp4 file.")
    parser.add_argument("--num-frames", type=int, default=49, help="Number of frames to generate (default: 49, ~6s at 8fps).")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps (default: 50).")
    parser.add_argument("--guidance-scale", type=float, default=6.0, help="Classifier-free guidance scale (default: 6.0).")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the saved video (default: 8).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--fast", action="store_true", help="Skip CPU offload/tiling (needs ~33GB VRAM; only for large GPUs).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. This script needs a GPU runtime — "
            "on Colab: Runtime > Change runtime type > GPU; "
            "on Kaggle: Notebook Settings > Accelerator > GPU."
        )

    pipe = load_pipeline(fast=args.fast)
    output_path = generate_video(
        pipe,
        prompt=args.prompt,
        output_path=args.output,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        fps=args.fps,
    )
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
