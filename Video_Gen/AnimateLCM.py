# Import des bibliothèques nécessaires
model_name = "AnimateLCM"
file_name = 'Outputs/' + model_name + '.gif'



import torch
import gradio as gr
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

# Initialisation du modèle et du pipeline
adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
pipe.set_adapters(["lcm-lora"], [0.8])

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# Définition de la fonction pour l'interface
def generate_animation(prompt):
    output = pipe(
        prompt=prompt,
        negative_prompt="bad quality, worse quality, low resolution",
        num_frames=32,
        guidance_scale=2.0,
        num_inference_steps=6,
        generator=torch.Generator("cuda").manual_seed(0),
    )
    frames = output.frames[0]
    export_to_gif(frames, file_name)  # Exporter l'animation au format GIF
    return file_name  # Renvoyer le nom du fichier GIF généré

# Création de l'interface utilisateur avec Gradio
interface = gr.Interface(
    fn=generate_animation,
    inputs="text",
    outputs="image",  # Le modèle renvoie un fichier GIF, donc nous définissons le type de sortie comme "image"
    title=model_name,
    description="Enter a prompt and watch the animation!",
    examples=[
        ["A cat breathing into an inhaler while chasing a mouse"]
    ]
)

# Lancement de l'interface utilisateur
interface.launch()
