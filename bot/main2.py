import sys
import logging
import asyncio
from collections import deque
from argparse import Namespace

from aiogram.types import FSInputFile
from aiogram.enums import ParseMode
from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import CommandStart, Command
from aiogram import F

from load_weights import *
from generate_sticker import *
from generate_animation import *

photo_storage = {}

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

opt = Namespace(
    config='config/vox-256.yaml',
    checkpoint='/kaggle/input/checkpoint/00000099-checkpoint.pth.tar',
    source_image='img.jpg',
    relative=True,
    adapt_scale=True,
    generator='Unet_Generator_keypoint_aware',
    kp_num=15,
    mb_channel=512,
    mb_spatial=32,
    mbunit='ExpendMemoryUnit',
    memsize=1,
    find_best_frame=False,
    best_frame=None,
    cpu=False
)

generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)


def get_styles_markup():
    default_btn = types.InlineKeyboardButton(text="Default 🤫🧏‍", callback_data="default")
    flowers_btn = types.InlineKeyboardButton(text="Flowers 🌸🌺", callback_data="flowers")
    cat_btn = types.InlineKeyboardButton(text="Cat ears 🐈🐱", callback_data="cat")
    butterfly_btn = types.InlineKeyboardButton(text="Butterflies 🦋🌈", callback_data="butterfly")
    clown_btn = types.InlineKeyboardButton(text="Clown 🤡🤣", callback_data="clown")
    pink_btn = types.InlineKeyboardButton(text="Pink hair 🩷✨", callback_data="pink")
    animate_btn = types.InlineKeyboardButton(text="Animate 🐈", callback_data="animate")
    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[[default_btn, flowers_btn],
                         [cat_btn, butterfly_btn],
                         [clown_btn, pink_btn],
                         [animate_btn]]
    )
    return markup


async def handle_selection(message: types.Message):
    index = int(message.text) - 1
    chat_id = message.chat.id
    await bot.send_sticker(
        chat_id=chat_id,
        sticker=FSInputFile(f"result{index}_{chat_id}.webp"),
        emoji="🎁",
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


async def process_stickerify_callback(callback_query: types.CallbackQuery):
    chat_id = callback_query.from_user.id
    sticker_style = callback_query.data

    if chat_id in photo_storage.keys() and len(photo_storage[chat_id]) > 0:
        file_id = photo_storage[chat_id].popleft()
        try:
            file = await bot.get_file(file_id)
            file_path = file.file_path
            contents = await bot.download_file(file_path)

            img = Image.open(contents)
            img.save(f"{chat_id}.jpg")

            if sticker_style != 'animate':
                await bot.send_message(chat_id, "Started generating your sticker! 👨‍🔬")
                stickerified_images = generate(img, sticker_style, chat_id)
                if not stickerified_images:
                    await bot.send_message(chat_id, "Unfortunately, we couldn't find a human face on your "
                                                    "photo, or there were too many of them 😰 Please, "
                                                    "send another photo.")
                    return

                stickerified_images.save(f"{chat_id}_result.jpeg")
                await bot.send_photo(chat_id, photo=FSInputFile(path=f"{chat_id}_result.jpeg"),
                                     caption="Type number from 1 to 9 to pick up sticker.")
            else:
                await bot.send_message(chat_id, "Started animating your sticker! 👨‍🔬")

                generate_animation(f"{chat_id}.jpg", 'wow-grey.mp4', f"{chat_id}.webm")

                await bot.send_sticker(
                    chat_id=chat_id,
                    sticker=FSInputFile(f"{chat_id}.webm"),
                    emoji="🎁"
                )

        except Exception as e:
            print(e)
            await bot.send_message(chat_id, f"Sorry, an error occurred.\n{e}")

    else:
        await bot.send_message(chat_id, "We couldn't find your photo. Please send it again.")


def setup_handlers(router: Router):
    router.message.register(handle_start, CommandStart())
    router.message.register(handle_photo, F.content_type.in_({'photo'}))
    router.message.register(handle_debug, Command("debug"))
    router.callback_query.register(process_stickerify_callback)


bot = Bot("7110638399:AAEA5ksrQVE31EYAOQf4n6LjYK87X_yvm-8", parse_mode=ParseMode.HTML)


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
