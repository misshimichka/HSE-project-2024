from PIL import Image
from io import BytesIO
import os
import logging

from aiogram.types import update, FSInputFile
from aiogram.enums import ParseMode
from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import CommandStart
from aiogram import F

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel

import modal

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

stub = modal.Stub("hse-project-1k")
image = modal.Image.debian_slim().apt_install("git", "ffmpeg", "libsm6", "libxext6").pip_install(
    "aiogram", "pillow", "torch", "torchvision"
)
image = image.pip_install(
    "git+https://github.com/huggingface/diffusers",
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


@stub.cls(gpu="T4")
class Model:
    def __init__(self):
        self.detector = None
        self.pipeline = None
        self.models = self.models = {
            "flowers": model_flowers_id,
            "cat": model_cat_id,
            "butterfly": model_butterfly_id,
            "clown": model_clown_id
        }

    @modal.build()  # add another step to the image build
    def download_model_to_folder(self):
        from huggingface_hub import snapshot_download, hf_hub_download

        snapshot_download(
            "misshimichka/instructPix2PixCartoon_4860_ckpt",
            cache_dir="pix2pix"
        )
        snapshot_download(
            "h94/IP-Adapter"
        )
        snapshot_download(
            repo_id="latent-consistency/lcm-lora-sdv1-5"
        )

        for key in self.models:
            snapshot_download(
                self.models[key],
                cache_dir=f"{key}"
            )

    @modal.enter()
    def setup(self):
        self.models = {
            "flowers": model_flowers_id,
            "cat": model_cat_id,
            "butterfly": model_butterfly_id,
            "clown": model_clown_id
        }

        print("Loading model...")

    @modal.method()
    def generate(self, original_image, mode):
        print("Loading style...")

        if mode == "default":
            self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "misshimichka/instructPix2PixCartoon_4860_ckpt",
                torch_dtype=torch.float16,
                safety_checker=None,
                local_files_only=True,
                cache_dir="pix2pix"
            )
        else:
            self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.models[mode],
                torch_dtype=torch.float16,
                safety_checker=None,
                local_files_only=True,
                cache_dir=mode
            )

        self.pipeline.load_ip_adapter(
            pretrained_model_name_or_path_or_dict="h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin",
            local_files_only=True
        )
        self.pipeline.set_ip_adapter_scale(1)

        self.pipeline.load_lora_weights(
            pretrained_model_name_or_path_or_dict="latent-consistency/lcm-lora-sdv1-5",
            weight_name="pytorch_lora_weights.safetensors",
            local_files_only=True)

        print("Generating image...")

        self.pipeline = self.pipeline.to("cuda")

        edited_image = self.pipeline(
            prompt="Refashion the photo into a sticker.",
            image=original_image,
            ip_adapter_image=original_image,
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
    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[[default_btn, flowers_btn], [cat_btn, butterfly_btn], [clown_btn]])
    return markup


async def handle_start(message: types.Message):
    await message.reply("Welcome to Stickerify bot! 🥶\nSend me a photo and I will create your own sticker 👠")


async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    file_id = photo.file_id
    chat_id = message.chat.id
    photo_storage[chat_id] = file_id

    await message.reply("Choose your sticker style:", reply_markup=get_styles_markup())


async def process_stickerify_callback(callback_query: types.CallbackQuery):
    chat_id = callback_query.from_user.id
    sticker_style = callback_query.data
    if chat_id in photo_storage:
        file_id = photo_storage[chat_id]
        try:
            file = await web_app.state.bot.get_file(file_id)
            file_path = file.file_path
            contents = await web_app.state.bot.download_file(file_path)

            await web_app.state.bot.send_message(chat_id, "Started generating your sticker! 👨‍🔬")

            img = Image.open(BytesIO(contents.getvalue()))
            stickerified_image = web_app.state.model.generate.remote(img, sticker_style)

            stickerified_image.save(f"{chat_id}_result.jpeg", "JPEG")

            await web_app.state.bot.send_photo(chat_id=chat_id, photo=FSInputFile(path=f"{chat_id}_result.jpeg"),
                                               caption="Here is your sticker! 🎁")

            del photo_storage[chat_id]

        except Exception as e:
            await web_app.state.bot.send_message(chat_id, f"Sorry, an error occurred.")

    else:
        await web_app.state.bot.send_message(chat_id, "I couldn't find your photo. Please send it again.")


def setup_handlers(router: Router):
    router.message.register(handle_start, CommandStart())
    router.message.register(handle_photo, F.content_type.in_({'photo'}))
    router.callback_query.register(process_stickerify_callback)


@stub.function(image=image, secrets=[modal.Secret.from_name("hse_bot")])
@modal.asgi_app()
def run_app():
    router = Router()

    setup_handlers(router)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    bot = Bot(os.environ["hse_bot_token"], parse_mode=ParseMode.HTML)

    web_app.state.dispatcher = dispatcher
    web_app.state.bot = bot
    web_app.state.model = Model()

    return web_app
