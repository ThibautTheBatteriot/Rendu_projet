#https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
#pip install diffusers --upgrade
#pip install invisible_watermark transformers accelerate safetensors

model_name = "Stable_Diffusion"

from datetime import datetime
from diffusers import DiffusionPipeline
import torch
import gradio as gr

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
#base.to("cuda")  #not #Using GPU VRAM
base.enable_model_cpu_offload() #Using GPU VRAM

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
#refiner.to("cuda") #not #Using GPU VRAM
refiner.enable_model_cpu_offload() #Using GPU VRAM

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

# Définition de la fonction pour l'interface
def generate_animation(prompt):

    # run both experts
    StableDiff_output_1 = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    StableDiff_output_2 = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=StableDiff_output_1,
    ).images[0]

    now = datetime.now()
    formatted_datetime = now.strftime("%Y%m%d-%H%M%S")

    file_name = 'Outputs/'+ formatted_datetime+ '-' + model_name
    file_name_output_1 = file_name + '_output_base.jpg'
    file_name_output_2 = file_name + '_output_refiner.jpg'

    #StableDiff_output_1[0].save(file_name_output_1)
    StableDiff_output_2.save(file_name_output_2)

    return file_name_output_1, file_name_output_2  # Renvoyer le nom du fichier généré

# Création de l'interface utilisateur avec Gradio
interface = gr.Interface(
    fn=generate_animation,
    inputs="text",
    outputs=["image","image"],  # Le modèle renvoie un fichier GIF, donc nous définissons le type de sortie comme "image"
    title=model_name,
    description="Enter a prompt",

)

# Lancement de l'interface utilisateur
interface.launch(share=True)
