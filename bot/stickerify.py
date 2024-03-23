import torch
from diffusers import (AutoencoderKL, DDPMScheduler,
                       StableDiffusionInstructPix2PixPipeline,
                       UNet2DConditionModel)
from diffusers.utils import load_image
from PIL import Image

model_id = "Alexator26/instructPix2PixCartoon_700imgs_train"

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, safety_checker=None).to("mps")


def load_unet(pipeline, idx):
    unet = UNet2DConditionModel.from_pretrained(
            f'checkpoint-{idx}',
            subfolder="unet", torch_dtype=torch.float16
        )
    pipeline.unet = unet
    return pipeline.to("mps")


def stickerify(original_image):
    original_image = original_image.resize((512, 512))
    generator = torch.Generator(device='mps').manual_seed(42)
    edited_image = pipeline("Refashion the photo into a sticker.",
                            image=original_image,
                            num_inference_steps=20,
                            image_guidance_scale=1.5,
                            guidance_scale=7,
                            generator=generator).images[0]
    return edited_image


pipeline = load_unet(pipeline, 4860)