import io

from PIL import Image
import os
import logging
import cv2
import face_detection
import numpy as np
from collections import deque
import uuid

from aiogram.types import update, FSInputFile
from aiogram.enums import ParseMode
from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import CommandStart, Command
from aiogram import F

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, LCMScheduler

import modal

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

stub = modal.App("hse-project-1k")
image = modal.Image.debian_slim().apt_install("git", "ffmpeg", "libsm6", "libxext6").pip_install(
    "aiogram", "pillow", "torch", "torchvision"
)
image = image.pip_install(
    "git+https://github.com/huggingface/diffusers",
    "git+https://github.com/hukkelas/DSFD-Pytorch-Inference",
    "transformers", "peft", "opencv-python"
)


def load_weights():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "h94/IP-Adapter",
        cache_dir="adapter"
    )
    snapshot_download(
        repo_id="latent-consistency/lcm-lora-sdv1-5",
        cache_dir="lcm"
    )

    for key in models:
        snapshot_download(
            models[key],
            cache_dir=f"{key}"
        )

    face_detection.build_detector(
        "DSFDDetector",
        confidence_threshold=.5,
        nms_iou_threshold=.3
    )


image = image.run_function(load_weights)

stub.image = image

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


async def webhook(request):
    dispatcher = request.app.state.dispatcher
    bot = request.app.state.bot
    upd = update.Update.model_validate(await request.json(), context={"bot": bot})

    await dispatcher.feed_update(bot=bot, update=upd)
    return JSONResponse({'status': 'ok'})


web_app = Starlette(debug=True, routes=[
    Route("/", webhook, methods=["POST"])
])

logging.basicConfig(level=logging.INFO)

def load_pipeline(mode):
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

    return pipeline


def crop_image(im):
    if isinstance(im, Image.Image):
        im = im.resize((512, 512))
        im = np.array(im)
    elif isinstance(im, str) and os.path.exists(im):
        im = cv2.imread(im)
        im = cv2.resize(im, (512, 512))
        im = im[:, :, ::-1]
    else:
        raise Exception("Can't handle img")

    detector = face_detection.build_detector(
        "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    detections = detector.detect(im)

    if detections.shape[0] != 1:
        return None

    x_min, y_min, x_max, y_max, _ = [int(i) + 1 for i in detections.tolist()[0]]
    y_min = max(0, y_min - 50)
    y_max = min(512, y_max + 50)
    x_min = max(0, x_min - 50)
    x_max = min(x_max + 50, 512)
    cropped_img = im[y_min:y_max, x_min:x_max]

    img = Image.fromarray(cropped_img)
    img = img.resize((512, 512))
    return img


@stub.function(image=image, gpu="T4")
def generate(original_image, mode):
    print("Loading style...")
    print(mode)

    pipeline = load_pipeline(mode).to("cuda")

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
    ).images[0]

    return edited_image


callback_data_storage = {}


def get_styles_markup(photo_id):
    default_btn = types.InlineKeyboardButton(text="Default 🤫🧏‍", callback_data=f"default:{photo_id}")
    flowers_btn = types.InlineKeyboardButton(text="Flowers 🌸🌺", callback_data=f"flowers:{photo_id}")
    cat_btn = types.InlineKeyboardButton(text="Cat ears 🐈🐱", callback_data=f"cat:{photo_id}")
    butterfly_btn = types.InlineKeyboardButton(text="Butterflies 🦋🌈", callback_data=f"butterfly:{photo_id}")
    clown_btn = types.InlineKeyboardButton(text="Clown 🤡🤣", callback_data=f"clown:{photo_id}")
    pink_btn = types.InlineKeyboardButton(text="Pink hair 🩷✨", callback_data=f"pink:{photo_id}")
    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[[default_btn, flowers_btn],
                         [cat_btn, butterfly_btn],
                         [clown_btn, pink_btn]]
    )
    return markup


async def handle_start(message: types.Message):
    await message.reply("Welcome to Stickerify bot! 🥶\nSend me a photo and I will create your own sticker 👠")


async def handle_photo(message: types.Message):
    photo_id = message.photo[-1].file_id
    unique_id = str(uuid.uuid4())
    callback_data_storage[unique_id] = photo_id
    await message.reply("Choose your sticker style:", reply_markup=get_styles_markup(unique_id))


async def process_stickerify_callback(callback_query: types.CallbackQuery):
    chat_id = callback_query.from_user.id
    sticker_style, unique_id = callback_query.data.split(":")
    file_id = callback_data_storage.get(unique_id)
    if not file_id:
        await web_app.state.bot.send_message(chat_id, "We couldn't find your photo. Please send it again.")
        return
    try:
        file = await web_app.state.bot.get_file(file_id)
        file_path = file.file_path
        contents = await web_app.state.bot.download_file(file_path)

        im = Image.open(contents)

        await web_app.state.bot.send_message(chat_id, "Started generating your sticker! 👨‍🔬")
        stickerified_image = generate.remote(im, sticker_style)
        if not stickerified_image:
            await web_app.state.bot.send_message(chat_id, "Unfortunately, we couldn't find a human face on your "
                                                          "photo, or there were too many of them 😰 Please, "
                                                          "send another photo.")
            return

        stickerified_image.save(f"result{file_id}.webp", "webp")

        await web_app.state.bot.send_sticker(
            chat_id=chat_id,
            sticker=FSInputFile(f"result{file_id}.webp"),
            emoji="🎁",
        )

    except Exception as e:
        print(e)
        await web_app.state.bot.send_message(chat_id, f"Sorry, an error occurred.")


def setup_handlers(router: Router):
    router.message.register(handle_start, CommandStart())
    router.message.register(handle_photo, F.content_type.in_({'photo'}))
    router.callback_query.register(process_stickerify_callback)


@stub.function(image=image, secrets=[modal.Secret.from_name("hse-bot-token")])
@modal.asgi_app()
def run_app():
    router = Router()

    setup_handlers(router)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    bot = Bot(os.environ["hse_bot"], parse_mode=ParseMode.HTML)

    web_app.state.dispatcher = dispatcher
    web_app.state.bot = bot

    return web_app
