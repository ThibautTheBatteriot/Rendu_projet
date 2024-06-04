model_name = "i2vgen-xl"


# Import des bibliothèques nécessaires
from datetime import datetime
import PIL
from PIL import Image
import numpy as np

import torch
import gradio as gr
from diffusers import I2VGenXLPipeline, AnimateDiffPipeline, LCMScheduler, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import load_image, export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from diffusers import DiffusionPipeline

repo_id = "ali-vilab/i2vgen-xl" 
pipeline = I2VGenXLPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16").to("cuda")


# Définition de la fonction pour l'interface
def generate_animation(prompt, image_raw):

    image = Image.fromarray(image_raw)
    image_pil = image.convert("RGB")

    I2VGen_output = pipeline(
        prompt=prompt,
        image=image_pil,
        generator=torch.Generator("cuda").manual_seed(8888),
    )

    frames = I2VGen_output.frames[0]

    now = datetime.now()
    formatted_datetime = now.strftime("%Y%m%d-%H%M%S")

    file_name = 'Outputs/'+ formatted_datetime+ '-' + model_name
    file_name_input  = file_name+'_input.jpg'
    file_name_output = file_name + '_output.gif'

    image_pil.save(file_name_input)
    export_to_gif(frames, file_name_output)  # Exporter l'animation au format GIF

    return file_name_output, file_name_input  # Renvoyer le nom du fichier GIF généré

# Création de l'interface utilisateur avec Gradio
interface = gr.Interface(
    fn=generate_animation,
    inputs=["text", "image"],
    outputs=["image", "image"],  # Le modèle renvoie un fichier GIF, donc nous définissons le type de sortie comme "image"
    title=model_name,
    description="Enter a prompt and watch the animation!",

)

# Lancement de l'interface utilisateur
interface.launch()
