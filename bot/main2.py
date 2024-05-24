import sys
import logging
import asyncio
from collections import deque

from aiogram.types import FSInputFile
from aiogram.enums import ParseMode
from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import CommandStart, Command
from aiogram import F

from load_weights import load_weights
from generate_sticker import *
from generate_animation import *

photo_storage = {}
animation_storage = {}

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


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


def get_animations_markup():
    wow_btn = types.InlineKeyboardButton(text="Wow", callback_data="wow")
    sigma_btn = types.InlineKeyboardButton(text="Sigma", callback_data="sigma")

    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[[wow_btn, sigma_btn]]
    )

    return markup


async def handle_selection(message: types.Message):
    index = int(message.text) - 1
    chat_id = message.chat.id

    animation_storage[chat_id] = index

    await bot.send_sticker(
        chat_id=chat_id,
        sticker=FSInputFile(f"result{index}_{chat_id}.webp"),
        emoji="🎁",
    )

    await message.reply("Choose your animation:", reply_markup=get_animations_markup())

    await bot.send_message(
        chat_id, "Started generating your animation!"
    )


async def handle_start(message: types.Message):
    await message.reply(
        "Welcome to Stickerify bot! 🥶\n"
        "Send me a photo and I will create your own sticker 👠"
    )


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


async def process_animation(chat_id, style):
    try:
        generate_animation(
            image_path=f"result{animation_storage[chat_id]}_{chat_id}.webp",
            style=style,
            output_path=f"{chat_id}.mp4"
        )

        await bot.send_video(
            chat_id=chat_id,
            video=FSInputFile(f"{chat_id}.mp4")
        )
    except Exception as e:
        print(e)
        await bot.send_message(chat_id, f"Sorry, an error occurred.\n{e}")


async def process_sticker(chat_id, style):
    if chat_id in photo_storage.keys() and len(photo_storage[chat_id]) > 0:
        file_id = photo_storage[chat_id].popleft()
        try:
            file = await bot.get_file(file_id)
            file_path = file.file_path
            contents = await bot.download_file(file_path)

            img = Image.open(contents)

            await bot.send_message(chat_id, "Started generating your sticker! 👨‍🔬")
            stickerified_images = generate(img, style, chat_id)
            if not stickerified_images:
                await bot.send_message(chat_id, "Unfortunately, we couldn't find a human face on your "
                                                "photo, or there were too many of them 😰 Please, "
                                                "send another photo.")
                return

            stickerified_images.save(f"{chat_id}_result.jpeg")
            await bot.send_photo(chat_id,
                                 photo=FSInputFile(path=f"{chat_id}_result.jpeg"),
                                 caption="Type number from 1 to 4 to pick up sticker."
                                 )

        except Exception as e:
            print(e)
            await bot.send_message(chat_id, f"Sorry, an error occurred.\n{e}")

    else:
        await bot.send_message(chat_id, "We couldn't find your photo. Please send it again.")


async def process_stickerify_callback(callback_query: types.CallbackQuery):
    chat_id = callback_query.from_user.id
    sticker_style = callback_query.data

    if sticker_style not in models.keys():
        await process_animation(chat_id, sticker_style)
    else:
        await process_sticker(chat_id, sticker_style)


def setup_handlers(router: Router):
    router.message.register(handle_selection, F.text.lower().in_(['1', '2', '3', '4']))
    router.message.register(handle_start, CommandStart())
    router.message.register(handle_photo, F.content_type.in_({'photo'}))
    router.message.register(handle_debug, Command("debug"))
    router.callback_query.register(process_stickerify_callback)


bot = Bot("YOUR_TOKEN_HERE", parse_mode=ParseMode.HTML)


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
