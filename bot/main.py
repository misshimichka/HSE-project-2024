from PIL import Image
from io import BytesIO
import os
import logging
import cv2
import face_detection
import numpy as np
from collections import deque

from aiogram.types import update, FSInputFile, BufferedInputFile
from aiogram.enums import ParseMode
from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import CommandStart, Command
from aiogram import F

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, LCMScheduler, UNet2DConditionModel
from transformers import AutoModelForImageSegmentation

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
stub.image = image


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

photo_storage = {}

model_flowers_id = "misshimichka/pix2pix_people_flowers_v2"
model_cat_id = "misshimichka/pix2pix_cat_ears"
model_clown_id = "misshimichka/pix2pix_clown_faces"
model_butterfly_id = "misshimichka/pix2pix_butterflies"
model_pink_id = "misshimichka/pix2pix_pink_hair"
model_id = "misshimichka/instructPix2PixCartoon_4860_ckpt"


@stub.cls(gpu="T4")
class Model:
    def __init__(self):
        self.bg_remover = None
        self.detector = None
        self.pipeline = None
        self.models = {
            "default": model_id,
            "flowers": model_flowers_id,
            "cat": model_cat_id,
            "butterfly": model_butterfly_id,
            "clown": model_clown_id,
            "pink": model_pink_id
        }

    @modal.build()
    def download_model_to_folder(self):
        from huggingface_hub import snapshot_download

        snapshot_download(
            "h94/IP-Adapter",
            cache_dir="adapter"
        )
        snapshot_download(
            repo_id="latent-consistency/lcm-lora-sdv1-5",
            cache_dir="lcm"
        )

        for key in self.models:
            snapshot_download(
                self.models[key],
                cache_dir=f"{key}"
            )

        self.detector = face_detection.build_detector(
            "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

    @modal.enter()
    def setup(self):
        print("Loading model...")

        self.detector = face_detection.build_detector(
            "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True,
            cache_dir="default"
        )

        self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config)

        self.pipeline.load_lora_weights(
            pretrained_model_name_or_path_or_dict="latent-consistency/lcm-lora-sdv1-5",
            weight_name="pytorch_lora_weights.safetensors",
            cache_dir="lcm",
            local_files_only=True)

    def crop_img(self, im):
        if isinstance(im, Image.Image):
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        elif isinstance(im, str) and os.path.exists(im):
            im = cv2.imread(im)
            im = cv2.resize(im, (512, 512))
            im = im[:, :, ::-1]
        else:
            raise Exception("Can't handle img")

        detections = self.detector.detect(im)

        # assert detections.shape[0] == 1
        if detections.shape[0] != 1:
            return None

        xmin, ymin, xmax, ymax, _ = [int(i) + 1 for i in detections.tolist()[0]]
        ymin = max(0, ymin - 100)
        ymax = min(512, ymax + 100)
        xmin = max(0, xmin - 100)
        xmax = min(xmax + 100, 512)
        cropped_img = im[ymin:ymax, xmin:xmax]

        im_pil = Image.fromarray(cropped_img)
        img = im_pil.resize((512, 512))
        return img

    @modal.method()
    def generate(self, original_image, mode):
        print("Loading style...")
        print(mode)

        self.pipeline.generator = torch.Generator(device='cuda:0').manual_seed(42)

        self.pipeline.unet = UNet2DConditionModel.from_pretrained(
            self.models[mode],
            subfolder="unet",
            torch_dtype=torch.float16,
            local_files_only=True,
            cache_dir=mode
        )

        self.pipeline.load_ip_adapter(
            pretrained_model_name_or_path_or_dict="h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin",
            local_files_only=True,
            cache_dir="adapter"
        )
        self.pipeline.set_ip_adapter_scale(1)

        self.pipeline = self.pipeline.to("cuda")

        print("Generating image...")

        cropped_image = self.crop_img(original_image)

        if not cropped_image:
            return None

        edited_image = self.pipeline(
            prompt="Refashion the photo into a sticker.",
            image=cropped_image,
            ip_adapter_image=cropped_image,
            num_inference_steps=4,
            image_guidance_scale=1,
            guidance_scale=2,
        ).images[0]

        return edited_image


def get_styles_markup():
    default_btn = types.InlineKeyboardButton(text="Default 🤫🧏‍", callback_data="default")
    flowers_btn = types.InlineKeyboardButton(text="Flowers 🌸🌺", callback_data="flowers")
    cat_btn = types.InlineKeyboardButton(text="Cat ears 🐈🐱", callback_data="cat")
    butterfly_btn = types.InlineKeyboardButton(text="Butterflies 🦋🌈", callback_data="butterfly")
    clown_btn = types.InlineKeyboardButton(text="Clown 🤡🤣", callback_data="clown")
    pink_btn = types.InlineKeyboardButton(text="Pink hair 🩷✨", callback_data="pink")
    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[[default_btn, flowers_btn],
                         [cat_btn, butterfly_btn],
                         [clown_btn, pink_btn]]
    )
    return markup


async def handle_start(message: types.Message):
    await message.reply("Welcome to Stickerify bot! 🥶\nSend me a photo and I will create your own sticker 👠")


async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    file_id = photo.file_id
    chat_id = message.chat.id
    if chat_id not in photo_storage.keys():
        photo_storage[chat_id] = deque()
    photo_storage[chat_id].append(file_id)

    await message.reply("Choose your sticker style:", reply_markup=get_styles_markup())


async def handle_debug(message: types.Message):
    print(photo_storage)


async def process_stickerify_callback(callback_query: types.CallbackQuery):
    chat_id = callback_query.from_user.id
    sticker_style = callback_query.data
    if chat_id in photo_storage.keys() and len(photo_storage[chat_id]) > 0:
        file_id = photo_storage[chat_id].popleft()
        try:
            file = await web_app.state.bot.get_file(file_id)
            file_path = file.file_path
            contents = await web_app.state.bot.download_file(file_path)

            img = Image.open(BytesIO(contents.getvalue()))

            await web_app.state.bot.send_message(chat_id, "Started generating your sticker! 👨‍🔬")
            stickerified_image = web_app.state.model.generate.remote(img, sticker_style)
            if not stickerified_image:
                await web_app.state.bot.send_message(chat_id, "Unfortunately, we couldn't find a human face on your "
                                                              "photo, or there were too many of them 😰 Please, "
                                                              "send another photo.")
                return

            stickerified_image.save(f"result{chat_id}.webp", "webp")

            await web_app.state.bot.send_sticker(
                chat_id=chat_id,
                sticker=FSInputFile(f"result{chat_id}.webp"),
                emoji="🎁",
            )

        except Exception as e:
            print(e)
            await web_app.state.bot.send_message(chat_id, f"Sorry, an error occurred.")

    else:
        await web_app.state.bot.send_message(chat_id, "We couldn't find your photo. Please send it again.")


def setup_handlers(router: Router):
    router.message.register(handle_start, CommandStart())
    router.message.register(handle_photo, F.content_type.in_({'photo'}))
    router.message.register(handle_debug, Command("debug"))
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
    web_app.state.model = Model()

    return web_app
