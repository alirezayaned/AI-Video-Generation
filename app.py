"""
One-Prompt AI Video -- standalone Gradio app.

Write one sentence describing a video. A small CPU-only planner model expands
it into a few short scenes; each scene gets an AI-generated video clip plus a
matching ambient sound effect; the clips are stitched together with crossfade
transitions.

This is the full-pipeline version (not scaled down), intended to run where
GPU time isn't tightly metered -- a Kaggle or Colab GPU session, or a
Docker container with a GPU attached. It does NOT depend on Hugging Face
Spaces' `spaces` package or ZeroGPU.

Two ways to run this:

1. Kaggle/Colab: write this file in a notebook cell (`%%writefile app.py`),
   then run it with GRADIO_SHARE=true so it prints a public share link:
       !GRADIO_SHARE=true python app.py
   The link stays live only as long as this process keeps running (i.e. the
   notebook cell stays "busy") and expires after ~72 hours per Gradio's own
   share-link limit, whichever comes first.

2. Docker (see the companion Dockerfile): the container binds to 0.0.0.0:7860
   and expects you to map that port yourself (`-p 7860:7860`) rather than
   using Gradio's own share tunnel.
"""

import json
import os
import re
import subprocess
import tempfile

import gradio as gr
import imageio_ffmpeg
import soundfile as sf
import torch
from diffusers import AudioLDM2Pipeline, CogVideoXPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config -- full-pipeline settings (not scaled down for a metered free tier).
# Matches the companion notebook's defaults.
# ---------------------------------------------------------------------------
MAX_SCENES = 15          # a safety cap, not a quota workaround -- prevents an
                         # accidental request from running for hours
DEFAULT_SCENES = 10      # ~54s total with the defaults below
NUM_FRAMES = 49          # ~6.125s per scene at 8fps -- CogVideoX-2b's native trained length
FPS = 8
VIDEO_STEPS = 40
AUDIO_STEPS = 20
CROSSFADE_SECONDS = 0.75
CLIP_DURATION = NUM_FRAMES / FPS

NEGATIVE_AUDIO_PROMPT = "low quality, average quality, noisy, music"

FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()

# ---------------------------------------------------------------------------
# Load models once at startup.
# ---------------------------------------------------------------------------
PLANNER_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
planner_tokenizer = AutoTokenizer.from_pretrained(PLANNER_MODEL_ID)
planner_model = AutoModelForCausalLM.from_pretrained(PLANNER_MODEL_ID, torch_dtype=torch.float32)
# Deliberately NOT moved to "cuda" -- the planner runs on CPU and never
# competes with the video/audio models for the single GPU's VRAM.

VIDEO_MODEL_ID = "THUDM/CogVideoX-2b"
video_pipe = CogVideoXPipeline.from_pretrained(VIDEO_MODEL_ID, torch_dtype=torch.float16)
video_pipe.enable_model_cpu_offload()
video_pipe.vae.enable_tiling()
# Required on a single free-tier GPU (T4, ~15GB VRAM): CogVideoX-2b needs
# ~33GB loaded naively, ~19GB with cpu offload alone -- both exceed a T4.
# Offload + tiling together bring it to ~11GB.

# Newer transformers versions (4.52.0+) stopped giving every model class
# generation-mixin methods automatically, which breaks AudioLDM2's own
# pipeline code (it calls one of those methods unconditionally on its bare
# GPT2Model language_model component -- AttributeError otherwise). This
# replaces that one method with a self-contained version that doesn't
# depend on any transformers internals, so it works regardless of which
# transformers version ends up installed -- no need to pin old versions.
def _patched_generate_language_model(self, inputs_embeds=None, max_new_tokens=8, attention_mask=None, **model_kwargs):
    # The real call sites in this pipeline always pass a meaningful
    # attention_mask (reflecting padding from the text encoders, even for a
    # single prompt) -- it has to be threaded through and extended each step,
    # not dropped, or output quality degrades silently instead of crashing.
    max_new_tokens = max_new_tokens if max_new_tokens is not None else self.language_model.config.max_new_tokens
    for _ in range(max_new_tokens):
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        next_hidden_states = output.hidden_states[-1]
        inputs_embeds = torch.cat([inputs_embeds, next_hidden_states[:, -1:, :]], dim=1)
        if attention_mask is not None:
            new_col = attention_mask.new_ones((attention_mask.shape[0], 1))
            attention_mask = torch.cat([attention_mask, new_col], dim=1)
    return inputs_embeds[:, -max_new_tokens:, :]


AudioLDM2Pipeline.generate_language_model = _patched_generate_language_model

AUDIO_MODEL_ID = "cvssp/audioldm2"
audio_pipe = AudioLDM2Pipeline.from_pretrained(AUDIO_MODEL_ID, torch_dtype=torch.float16)
audio_pipe.enable_model_cpu_offload()
# Also offloaded, even though AudioLDM2 alone is small enough to fit
# comfortably on its own: with CogVideoX ALSO resident on the same GPU,
# offloading both means only whichever model is actively generating at a
# given moment touches GPU memory, rather than the sum of both models' peaks.
audio_pipe.scheduler = DPMSolverMultistepScheduler.from_config(audio_pipe.scheduler.config)


# ---------------------------------------------------------------------------
# Scene planning (CPU only)
# ---------------------------------------------------------------------------
def parse_scene_json(raw_text):
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list")
    cleaned = []
    for item in data:
        if isinstance(item, dict) and "video_prompt" in item and "sound_prompt" in item:
            cleaned.append({
                "video_prompt": str(item["video_prompt"]),
                "sound_prompt": str(item["sound_prompt"]),
            })
    if not cleaned:
        raise ValueError("No valid {video_prompt, sound_prompt} objects found")
    return cleaned


def plan_scenes(user_theme, target_scenes, max_attempts=3):
    system_prompt = (
        "You are a shot planner for a short AI-generated video montage. Given a single "
        "theme from the user, break it into a sequence of distinct visual scenes that "
        "flow naturally. Respond with ONLY a JSON array -- no markdown code fences, no "
        "commentary before or after. Each element must be an object with exactly two "
        "string keys: \"video_prompt\" (a vivid, visually specific description for a "
        "single short text-to-video shot: subject, setting, lighting, camera framing) "
        "and \"sound_prompt\" (the real-world ambient sound or sound effects audible in "
        f"that scene -- not music). Produce exactly {target_scenes} scene objects."
    )

    raw = ""
    for _ in range(max_attempts):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_theme},
        ]
        text = planner_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = planner_tokenizer([text], return_tensors="pt")

        generated_ids = planner_model.generate(
            **model_inputs,
            max_new_tokens=1500,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
        generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
        raw = planner_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        try:
            return parse_scene_json(raw)
        except Exception:
            continue

    raise RuntimeError("The scene planner couldn't produce valid output -- try rephrasing your prompt.")


# ---------------------------------------------------------------------------
# Per-scene generation (video + matching sound)
# ---------------------------------------------------------------------------
def generate_scene(video_prompt, sound_prompt, seed, work_dir):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    frames = video_pipe(
        prompt=video_prompt,
        num_frames=NUM_FRAMES,
        num_inference_steps=VIDEO_STEPS,
        guidance_scale=6.0,
        generator=generator,
    ).frames[0]

    video_path = os.path.join(work_dir, f"scene_{seed}_video.mp4")
    export_to_video(frames, video_path, fps=FPS)

    audio_generator = torch.Generator("cuda").manual_seed(seed + 1)
    audio = audio_pipe(
        sound_prompt,
        negative_prompt=NEGATIVE_AUDIO_PROMPT,
        audio_length_in_s=CLIP_DURATION + 0.5,
        num_inference_steps=AUDIO_STEPS,
        generator=audio_generator,
    ).audios[0]

    audio_path = os.path.join(work_dir, f"scene_{seed}_audio.wav")
    sf.write(audio_path, audio, 16000)

    return video_path, audio_path


# ---------------------------------------------------------------------------
# CPU-only muxing / crossfade stitching
# ---------------------------------------------------------------------------
def mux_audio_video(video_path, audio_path, output_path):
    subprocess.run(
        [
            FFMPEG_BINARY, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-ar", "44100",
            "-shortest",
            output_path,
        ],
        check=True, capture_output=True,
    )


def concatenate_with_crossfade(clip_paths, clip_durations, output_path, transition_duration=0.75):
    n = len(clip_paths)
    inputs = []
    for p in clip_paths:
        inputs += ["-i", p]

    filter_parts = []
    prev_v, prev_a = "0:v", "0:a"
    running_duration = clip_durations[0]

    for i in range(1, n):
        offset = running_duration - transition_duration
        v_out, a_out = f"v{i}", f"a{i}"
        filter_parts.append(
            f"[{prev_v}][{i}:v]xfade=transition=fade:duration={transition_duration}:offset={offset:.3f}[{v_out}]"
        )
        filter_parts.append(f"[{prev_a}][{i}:a]acrossfade=d={transition_duration}[{a_out}]")
        prev_v, prev_a = v_out, a_out
        running_duration = running_duration + clip_durations[i] - transition_duration

    cmd = [
        FFMPEG_BINARY, "-y", *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
        "-c:v", "libx264", "-c:a", "aac",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Orchestration -- drives the UI, yields progress. Each request gets its own
# temp directory so concurrent requests (if ever more than one) never collide.
# ---------------------------------------------------------------------------
def run_pipeline(theme, num_scenes):
    if not theme or not theme.strip():
        yield "Please describe your video first.", None, None
        return

    num_scenes = max(1, min(MAX_SCENES, int(num_scenes)))
    work_dir = tempfile.mkdtemp(prefix="onepromptvideo_")

    yield f"Planning {num_scenes} scene(s)...", None, None
    try:
        scenes = plan_scenes(theme.strip(), target_scenes=num_scenes)
    except Exception as e:
        yield f"Scene planning failed: {e}", None, None
        return

    scene_lines = "\n".join(f"{i + 1}. {s['video_prompt']}" for i, s in enumerate(scenes))
    yield f"Planned {len(scenes)} scene(s):\n\n{scene_lines}\n\nGenerating scene 1/{len(scenes)}...", None, None

    clip_paths = []
    durations = []
    for i, scene in enumerate(scenes):
        try:
            video_path, audio_path = generate_scene(
                scene["video_prompt"], scene["sound_prompt"], seed=1000 + i, work_dir=work_dir
            )
        except Exception as e:
            yield f"Scene {i + 1} failed: {e}", None, None
            return

        with_sound_path = os.path.join(work_dir, f"scene_{i:02d}_with_sound.mp4")
        mux_audio_video(video_path, audio_path, with_sound_path)
        clip_paths.append(with_sound_path)
        durations.append(CLIP_DURATION)

        status = f"Planned {len(scenes)} scene(s):\n\n{scene_lines}\n\nScene {i + 1}/{len(scenes)} done."
        if i + 1 < len(scenes):
            status += f" Generating scene {i + 2}/{len(scenes)}..."
        yield status, with_sound_path, None

    if len(clip_paths) == 1:
        final_path = clip_paths[0]
    else:
        yield f"{len(scenes)} scene(s) generated. Stitching with crossfade...", clip_paths[-1], None
        final_path = os.path.join(work_dir, "final.mp4")
        concatenate_with_crossfade(clip_paths, durations, final_path, transition_duration=CROSSFADE_SECONDS)

    yield "Done!", clip_paths[-1], final_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="One-Prompt AI Video") as demo:
    gr.Markdown(
        "# One-Prompt AI Video\n"
        "Describe a video in one sentence. A small local model plans a few short "
        "scenes, each gets its own AI-generated clip and a matching ambient sound "
        "effect, then they're stitched together with a crossfade.\n\n"
        f"Default is {DEFAULT_SCENES} scenes (~{DEFAULT_SCENES * CLIP_DURATION - (DEFAULT_SCENES - 1) * CROSSFADE_SECONDS:.0f}s total) "
        "-- each scene takes a few minutes to generate on a free GPU, so a full run "
        "can take a while. Try 2-3 scenes first to confirm everything works.\n\n"
        "*Sound effects are generated with AudioLDM2, which is CC-BY-NC-SA-4.0 "
        "(non-commercial) -- keep that in mind for any commercial use of outputs.*"
    )

    with gr.Row():
        theme_input = gr.Textbox(
            label="Describe your video",
            placeholder="e.g. a quiet morning in a mountain forest, from sunrise to midday",
            scale=3,
        )
        scenes_input = gr.Slider(label="Scenes", minimum=1, maximum=MAX_SCENES, step=1, value=DEFAULT_SCENES, scale=1)

    generate_btn = gr.Button("Generate", variant="primary")
    status_output = gr.Markdown()

    with gr.Row():
        preview_output = gr.Video(label="Latest scene")
        final_output = gr.Video(label="Final video")

    generate_btn.click(
        run_pipeline,
        inputs=[theme_input, scenes_input],
        outputs=[status_output, preview_output, final_output],
    )

if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    demo.queue().launch(share=share, server_name="0.0.0.0")
