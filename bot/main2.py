import sys
import logging
import asyncio
from collections import deque
import uuid

from aiogram.types import FSInputFile
from aiogram.enums import ParseMode
from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import CommandStart, Command
from aiogram.filters.callback_data import CallbackData
from aiogram import F

from load_weights import load_weights
from generate_sticker import *
from generate_animation import *
from config import API_TOKEN
from markups import get_selection_markup, get_animations_markup, get_styles_markup
from callbacks import StyleCallbackData, ChoiceCallbackData, AnimationCallbackData


photo_storage = {}
animation_storage = {}
photo_uuid_dict = {}

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


async def process_choice_callback(callback_query: types.CallbackQuery, callback_data: ChoiceCallbackData):
    chat_id = callback_query.from_user.id
    choice = callback_data.choice
    photo_uuid = callback_data.photo_uuid
    try:
        await bot.send_sticker(
            chat_id=chat_id,
            sticker=FSInputFile(f"result_{choice - 1}_{photo_uuid}.webp"),
            emoji="🎁",
            reply_markup=get_animations_markup(photo_uuid, choice)
        )
    except Exception as e:
        await bot.send_message(chat_id, "Sorry, an error occurred")
        print(e)


async def process_animation_callback(callback_query: types.CallbackQuery, callback_data: AnimationCallbackData):
    chat_id = callback_query.from_user.id
    animation_style = callback_data.animation_style
    photo_uuid = callback_data.photo_uuid
    choice = callback_data.choice
    await process_animation(chat_id, animation_style, photo_uuid, choice)


async def handle_start(message: types.Message):
    await message.reply(
        "Welcome to Stickerify bot! 🥶\n"
        "Send me a photo and I will create your own sticker 👠"
    )


async def handle_photo(message: types.Message):
    chat_id = message.chat.id
    photo_id = message.photo[-1].file_id
    photo_uuid = str(uuid.uuid4())
    photo_uuid_dict[photo_uuid] = photo_id
    if chat_id not in photo_storage.keys():
        photo_storage[chat_id] = set()
    photo_storage[chat_id].add(photo_uuid)

    await message.reply("Choose your sticker style:", reply_markup=get_styles_markup(photo_uuid))


async def handle_debug(message: types.Message):
    print(photo_storage)


async def process_animation(chat_id, style, photo_uuid, choice):
    try:
        await bot.send_message(chat_id, "Started generating your animation!")
        generate_animation(
            image_path=f"result_{choice - 1}_{photo_uuid}.webp",
            style=style,
            output_path=f"animated_result_{choice - 1}_{photo_uuid}.mp4"
        )

        await bot.send_sticker(
            chat_id=chat_id,
            sticker=FSInputFile(f"animated_result_{choice - 1}_{photo_uuid}.webm")
        )

    except Exception as e:
        print(e)
        await bot.send_message(chat_id, f"Sorry, an error occurred.\n{e}")


async def process_sticker(chat_id, style, photo_uuid):
    if chat_id in photo_storage.keys() and photo_uuid in photo_storage[chat_id]:
        photo_storage[chat_id].remove(photo_uuid)
        file_id = photo_uuid_dict.get(photo_uuid)
        try:
            file = await bot.get_file(file_id)
            file_path = file.file_path
            contents = await bot.download_file(file_path)

            img = Image.open(contents)

            await bot.send_message(chat_id, "Started generating your sticker! 👨‍🔬")
            stickerified_images = generate(img, style, photo_uuid)
            if not stickerified_images:
                await bot.send_message(chat_id, "Unfortunately, we couldn't find a human face on your "
                                                "photo, or there were too many of them 😰 Please, "
                                                "send another photo.")
                return

            stickerified_images.save(f"{photo_uuid}_result.jpeg")
            await bot.send_photo(chat_id,
                                 photo=FSInputFile(path=f"{photo_uuid}_result.jpeg"),
                                 caption="Choose the generation, you like!",
                                 reply_markup=get_selection_markup(photo_uuid)
                                 )

        except Exception as e:
            print(e)
            await bot.send_message(chat_id, f"Sorry, an error occurred.\n{e}")

    else:
        await bot.send_message(chat_id, "We couldn't find your photo. Please send it again.")


async def process_stickerify_callback(callback_query: types.CallbackQuery, callback_data: StyleCallbackData):
    chat_id = callback_query.from_user.id
    style = callback_data.style
    photo_uuid = callback_data.photo_uuid

    if style not in models.keys():
        await process_animation(chat_id, style)
    else:
        await process_sticker(chat_id, style, photo_uuid)


def setup_handlers(router: Router):
    router.message.register(handle_start, CommandStart())
    router.message.register(handle_photo, F.content_type.in_({'photo'}))
    router.message.register(handle_debug, Command("debug"))
    router.callback_query.register(process_stickerify_callback, StyleCallbackData.filter())
    router.callback_query.register(process_choice_callback, ChoiceCallbackData.filter())
    router.callback_query.register(process_animation_callback, AnimationCallbackData.filter())


bot = Bot(API_TOKEN, parse_mode=ParseMode.HTML)


async def main():
    load_weights()
    router = Router()
    setup_handlers(router)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
