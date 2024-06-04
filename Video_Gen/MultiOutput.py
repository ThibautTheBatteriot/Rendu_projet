# Import des bibliothèques nécessaires
#pip install torch diffusers gradio safetensors accelerate huggingface_hub transformers peft
model_name = "AnimateLCM"
file_name = 'Outputs/' + model_name + '.gif'

import torch
import gradio as gr
from diffusers import I2VGenXLPipeline, AnimateDiffPipeline, LCMScheduler, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


#####   AnimateLCM Model   ######
# Initialisation du modèle et du pipeline
adapter_LCM = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
pipe_ALCM = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter_LCM, torch_dtype=torch.float16)
pipe_ALCM.scheduler = LCMScheduler.from_config(pipe_ALCM.scheduler.config, beta_schedule="linear")

pipe_ALCM.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
pipe_ALCM.set_adapters(["lcm-lora"], [0.8])

pipe_ALCM.enable_vae_slicing()
pipe_ALCM.enable_model_cpu_offload()

output_titles = []

def generate_animations(prompt):
    files = []
    outputs = []

    #####   AnimateDiff Output   ######
    model_name = "AnimateDiff"
    file_name = 'Outputs/' + model_name + '.gif'

    device = "cuda"
    dtype = torch.float16

    step = 4  # Options: [1,2,4,8]
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    base = "emilianJR/epiCRealism"  # Choose to your favorite base model.

    adapter_ADiff = MotionAdapter().to(device, dtype)
    adapter_ADiff.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
    pipe_ADiff = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter_ADiff, torch_dtype=dtype).to(device)
    pipe_ADiff.scheduler = EulerDiscreteScheduler.from_config(pipe_ADiff.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

    output = pipe_ADiff(prompt=prompt, guidance_scale=1.0, num_inference_steps=step)
    export_to_gif(output.frames[0], file_name)

    outputs.append(model_name)
    outputs.append(file_name)

    #####   AnimateLCM Output   ######
    model_name = "AnimateLCM"
    file_name = 'Outputs/' + model_name + '.gif'

    output = pipe_ALCM(
        prompt=prompt,
        negative_prompt="bad quality, worse quality, low resolution",
        num_frames=32,
        guidance_scale=2.0,
        num_inference_steps=6,
        generator=torch.Generator("cuda").manual_seed(0),
    )
    frames = output.frames[0]
    export_to_gif(frames, file_name)  # Exporter l'animation au format GIF

    outputs.append(model_name)
    outputs.append(file_name)

    return outputs
  
            
# Création de l'interface utilisateur avec Gradio
interface = gr.Interface(
    fn=generate_animations,
    inputs="text",
    outputs=["text", "image" ,"text", "image"],  # Le modèle renvoie un fichier GIF, donc nous définissons le type de sortie comme "image"
    title='Multiple Outputs',
    description="Enter a prompt and watch the animation!",
    examples=
        ["A cat breathing into an inhaler while chasing a mouse",
         'A young woman smelling a sunflower',
         'diamond-made fishs falling on a pink and green beach with waves in it'],
)

# Lancement de l'interface utilisateur
interface.launch()
