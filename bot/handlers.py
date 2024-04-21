import aiogram
from aiogram import types, Dispatcher
from stickerify import stickerify, pipelines_dict
from PIL import Image
from io import BytesIO
from collections import deque

photo_storage = {}


def get_styles_markup():
    markup = types.InlineKeyboardMarkup()
    default_btn = types.InlineKeyboardButton("Default 🤫🧏‍", callback_data="default")
    flowers_btn = types.InlineKeyboardButton("Flowers 🌸🌺", callback_data="flowers")
    markup.add(default_btn, flowers_btn)
    return markup


async def handle_start(message: types.Message):
    await message.reply("Welcome to Stickerify bot! 🥶\nSend me a photo and I will create your own sticker 👠")


async def handle_photo(message: types.Message, bot):
    photo = message.photo[-1]
    file_id = photo.file_id
    chat_id = message.chat.id
    if chat_id not in photo_storage.keys():
        photo_storage[chat_id] = deque()
    photo_storage[chat_id].append(file_id)

    await message.reply("Choose your sticker style:", reply_markup=get_styles_markup())


async def process_stickerify_callback(callback_query: types.CallbackQuery, bot):
    chat_id = callback_query.from_user.id
    sticker_style = callback_query.data
    if chat_id in photo_storage.keys() and len(photo_storage[chat_id]) > 0:
        file_id = photo_storage[chat_id].popleft()
        try:
            file = await bot.get_file(file_id)
            file_path = file.file_path
            contents = await bot.download_file(file_path)

            await bot.send_message(chat_id, "Started generating your sticker! 👨‍🔬")

            image = Image.open(BytesIO(contents.getvalue()))
            stickerified_image = stickerify(image, sticker_style)

            bio = BytesIO()
            bio.name = "image.jpeg"
            stickerified_image.save(bio, "JPEG")
            bio.seek(0)

            await bot.send_photo(chat_id, photo=bio, caption="Here is your sticker! 🎁")

        except Exception as e:
            await bot.send_message(chat_id, f"Sorry, an error occurred.")

    else:
        await bot.send_message(chat_id, "I couldn't find your photo. Please send it again.")


def setup_handlers(dp: Dispatcher, bot):
    dp.register_message_handler(handle_start, commands=['start'])
    dp.register_message_handler(lambda message: handle_photo(message, bot), content_types=['photo'])
    dp.register_callback_query_handler(lambda query: process_stickerify_callback(query, bot),
                                       lambda query: query.data in pipelines_dict.keys())