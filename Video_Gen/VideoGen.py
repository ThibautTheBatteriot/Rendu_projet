#pip install diffusers["torch"] transformers
#pip install diffusers["flax"] transformers
#pip install accelerate
#pip install git+https://github.com/huggingface/diffusers
#pip install peft
#pip install gradio


#source : https://huggingface.co/docs/diffusers/installation

#from diffusers import DiffusionPipeline

#pipeline = DiffusionPipeline.from_pretrained("wangfuyun/AnimateLCM")

import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
pipe.set_adapters(["lcm-lora"], [0.8])

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="un chat qui respire dans un inhalateur tout en poursuivant une souris, 1080p, high resolution",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    guidance_scale=2.0,
    num_inference_steps=6,
    generator=torch.Generator("cuda").manual_seed(0),
)
frames = output.frames[0]
export_to_gif(frames, "Outputs/VidGen.gif")

