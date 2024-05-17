import torch

import os
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionInstructPix2PixPipeline, LCMScheduler

import face_detection

model_flowers_id = "misshimichka/pix2pix_people_flowers_v2"
model_cat_id = "misshimichka/pix2pix_cat_ears"
model_clown_id = "misshimichka/pix2pix_clown_faces"
model_butterfly_id = "misshimichka/pix2pix_butterflies"
model_pink_id = "misshimichka/pix2pix_pink_hair"
model_id = "misshimichka/instructPix2PixCartoon_4860_ckpt"

models = {
    "default": model_id,
    "flowers": model_flowers_id,
    "cat": model_cat_id,
    "butterfly": model_butterfly_id,
    "clown": model_clown_id,
    "pink": model_pink_id
}


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def crop_image(im):
    if isinstance(im, Image.Image):
        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    elif isinstance(im, str) and os.path.exists(im):
        im = cv2.imread(im)
        im = cv2.resize(im, (512, 512))
        im = im[:, :, ::-1]
    else:
        raise Exception("Can't handle img")

    detector = face_detection.build_detector(
        "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    detections = detector.detect(im)

    assert detections.shape[0] == 1
    x_min, y_min, x_max, y_max, _ = [int(i) + 1 for i in detections.tolist()[0]]
    y_min = max(0, y_min - 100)
    y_max = min(512, y_max + 100)
    x_min = max(0, x_min - 100)
    x_max = min(x_max + 100, 512)
    cropped_img = im[y_min:y_max, x_min:x_max]

    im_pil = Image.fromarray(cropped_img)
    img = im_pil.resize((512, 512))
    return img


def generate(original_image, mode, chat_id):
    print("Loading style...")
    print(mode)

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        models[mode],
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True,
        cache_dir=mode
    )

    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

    pipeline.load_lora_weights(
        pretrained_model_name_or_path_or_dict="latent-consistency/lcm-lora-sdv1-5",
        weight_name="pytorch_lora_weights.safetensors",
        cache_dir="lcm",
        local_files_only=True
    )

    pipeline.generator = torch.Generator(device='cuda:0').manual_seed(42)

    pipeline.load_ip_adapter(
        pretrained_model_name_or_path_or_dict="h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin",
        local_files_only=True,
        cache_dir="adapter"
    )
    pipeline.set_ip_adapter_scale(1)

    pipeline = pipeline.to("cuda")

    print("Generating image...")

    cropped_image = crop_image(original_image)

    if not cropped_image:
        return None

    edited_image = pipeline(
        prompt="Refashion the photo into a sticker.",
        image=cropped_image,
        ip_adapter_image=cropped_image,
        num_inference_steps=4,
        image_guidance_scale=1,
        guidance_scale=2,
    )

    torch.cuda.empty_cache()

    for idx, img in enumerate(edited_image):
        img.save(f"result{idx}_{chat_id}.webp", "webp")

    return image_grid(edited_image, 3, 1)
