from PIL import Image
from io import BytesIO
import os
import logging
import cv2
import face_detection
import numpy as np
import asyncio
from collections import deque

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import CommandStart, Command
from aiogram.types import FSInputFile
from aiogram.enums import ParseMode

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, LCMScheduler

photo_storage = {}

model_ids = {
    "default": "misshimichka/instructPix2PixCartoon_4860_ckpt",
    "flowers": "misshimichka/pix2pix_people_flowers_v2",
    "cat": "misshimichka/pix2pix_cat_ears",
    "butterfly": "misshimichka/pix2pix_butterflies",
    "clown": "misshimichka/pix2pix_clown_faces",
    "pink": "misshimichka/pix2pix_pink_hair"
}

def crop_image(image):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, str) and os.path.exists(image):
        image = cv2.imread(image)
        image = cv2.resize(image, (512, 512))
        image = image[:, :, ::-1]
    else:
        raise Exception("Can't handle img")

    detector = face_detection.build_detector(
        "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    detections = detector.detect(image)

    if detections.shape[0] != 1:
        return None

    x_min, y_min, x_max, y_max, _ = [int(i) + 1 for i in detections.tolist()[0]]
    y_min, y_max = max(0, y_min - 100), min(512, y_max + 100)
    x_min, x_max = max(0, x_min - 100), min(x_max + 100, 512)
    cropped_img = image[y_min:y_max, x_min:x_max]

    return Image.fromarray(cropped_img).resize((512, 512))

def generate_image(original_image, mode):
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_ids[mode],
        torch_dtype=torch.float16,
        safety_checker=None
    )

    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

    pipeline.load_lora_weights(
        "latent-consistency/lcm-lora-sdv1-5",
        weight_name="pytorch_lora_weights.safetensors"
    )

    pipeline.generator = torch.Generator(device='cuda:0').manual_seed(42)

    pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin"
    )
    pipeline.set_ip_adapter_scale(1)
    pipeline = pipeline.to("cuda")

    cropped_image = crop_image(original_image)
    if not cropped_image:
        return None

    return pipeline(
        prompt="Refashion the photo into a sticker.",
        image=cropped_image,
        ip_adapter_image=cropped_image,
        num_inference_steps=4,
        image_guidance_scale=1,
        guidance_scale=2
    ).images[0]

def get_styles_markup():
    buttons = [
        [types.InlineKeyboardButton(text="Default 🤫🧏‍", callback_data="default"),
         types.InlineKeyboardButton(text="Flowers 🌸🌺", callback_data="flowers")],
        [types.InlineKeyboardButton(text="Cat ears 🐈🐱", callback_data="cat"),
         types.InlineKeyboardButton(text="Butterflies 🦋🌈", callback_data="butterfly")],
        [types.InlineKeyboardButton(text="Clown 🤡🤣", callback_data="clown"),
         types.InlineKeyboardButton(text="Pink hair 🩷✨", callback_data="pink")]
    ]
    return types.InlineKeyboardMarkup(inline_keyboard=buttons)

async def handle_start(message: types.Message):
    await message.reply("Welcome to Stickerify bot! 🥶\nSend me a photo and I will create your own sticker 👠")

async def handle_photo(message: types.Message):
    chat_id = message.chat.id
    if chat_id not in photo_storage:
        photo_storage[chat_id] = deque()
    photo_storage[chat_id].append(message.photo[-1].file_id)
    await message.reply("Choose your sticker style:", reply_markup=get_styles_markup())

async def handle_debug(message: types.Message):
    print(photo_storage)

async def process_stickerify_callback(callback_query: types.CallbackQuery):
    chat_id = callback_query.from_user.id
    sticker_style = callback_query.data
    if chat_id in photo_storage and photo_storage[chat_id]:
        file_id = photo_storage[chat_id].popleft()
        try:
            file = await bot.get_file(file_id)
            contents = await bot.download_file(file.file_path)
            img = Image.open(BytesIO(contents.getvalue()))

            await bot.send_message(chat_id, "Started generating your sticker! 👨‍🔬")
            stickerified_image = generate_image(img, sticker_style)
            if not stickerified_image:
                await bot.send_message(chat_id, "Unfortunately, we couldn't find a human face on your photo, or there were too many of them 😰 Please, send another photo.")
                return

            stickerified_image.save(f"result{chat_id}.webp", "webp")
            await bot.send_sticker(chat_id=chat_id, sticker=FSInputFile(f"result{chat_id}.webp"), emoji="🎁")

        except Exception as e:
            print(e)
            await bot.send_message(chat_id, "Sorry, an error occurred.")
    else:
        await bot.send_message(chat_id, "We couldn't find your photo. Please send it again.")

def setup_handlers(router: Router):
    router.message.register(handle_start, CommandStart())
    router.message.register(handle_photo, types.Message.content_type.in_({'photo'}))
    router.message.register(handle_debug, Command("debug"))
    router.callback_query.register(process_stickerify_callback)

bot = Bot("YOUR_TOKEN_HERE", parse_mode=ParseMode.HTML)

async def main():
    router = Router()
    setup_handlers(router)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    await dispatcher.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

