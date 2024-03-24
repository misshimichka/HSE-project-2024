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
from diffusers import (StableDiffusionInstructPix2PixPipeline, DiffusionPipeline)

import modal

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

stub = modal.Stub("hse-project-1k")
image = modal.Image.debian_slim().pip_install("aiogram", "pillow", "diffusers", "torch", "transformers")
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

pipelines_dict = {
    "default": 0,
    "flowers": 1
}

photo_storage = {}

model_id = "misshimichka/instructPix2PixCartoon_4860_ckpt"
model_flowers_id = "misshimichka/pix2pix_people_flowers_v2"


@stub.cls(gpu="T4")
class Model:
    def __enter__(self):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading model...")

        default_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")

        self.pipelines = [default_pipeline,
                          DiffusionPipeline.from_pretrained(model_flowers_id,
                                                            torch_dtype=torch.float16,
                                                            safety_checker=None).to("cuda")
                          ]

    @modal.method()
    def generate(self, original_image, mode):
        print("Generating image...")

        original_image = original_image.resize((512, 512))
        generator = torch.Generator(device="cuda").manual_seed(42)
        assert mode in pipelines_dict.keys()
        edited_image = self.pipelines[pipelines_dict[mode]](
            "Refashion the photo into a sticker.",
            image=original_image,
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=7,
            generator=generator).images[0]
        return edited_image


def get_styles_markup():
    default_btn = types.InlineKeyboardButton(text="Default 🤫🧏‍", callback_data="default")
    flowers_btn = types.InlineKeyboardButton(text="Flowers 🌸🌺", callback_data="flowers")
    markup = types.InlineKeyboardMarkup(inline_keyboard=[[default_btn, flowers_btn]])
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
