model_name = "AnimateDiff"
file_name = 'Outputs/' + model_name + '.gif'

import torch
import gradio as gr
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def generate_animation(prompt):
    device = "cuda"
    dtype = torch.float16

    step = 4  # Options: [1,2,4,8]
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    base = "emilianJR/epiCRealism"  # Choose to your favorite base model.

    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
    pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

    output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=step)
    export_to_gif(output.frames[0], file_name)

    return file_name

interface = gr.Interface(
    fn=generate_animation,
    inputs="text",
    outputs="image",
    title = model_name,
    description="Enter a prompt and watch the animation!",
    examples=[
        ["A girl smiling"]
    ]
)

interface.launch()
