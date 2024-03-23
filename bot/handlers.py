import aiogram
from aiogram import types, Dispatcher

user_sticker_packs = {}


async def handle_start(message: types.Message):
    user_id = message.from_user.id
    if user_id not in user_sticker_packs:
        sticker_pack_name = f"user_{user_id}_by_your_bot_name"

        # Create a sticker pack via telegram api

        user_sticker_packs[user_id] = sticker_pack_name
        await message.reply(
            "Welcome! Your personal sticker pack has been created. Send me a picture to add it to your sticker pack.")
    else:
        await message.reply("Welcome back! Send me a picture to add to your sticker pack.")


async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    if user_id in user_sticker_packs:
        sticker_pack_name = user_sticker_packs[user_id]

        # Add a photo to sticker pack via telegram api

        await message.reply("Picture added to your sticker pack!")
    else:
        await message.reply("Please start by sending /start to create your sticker pack.")


def setup_handlers(dp: Dispatcher):
    dp.register_message_handler(handle_start, commands=['start'])
    dp.register_message_handler(handle_photo, content_types=['photo'])
