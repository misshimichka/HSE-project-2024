import aiogram
from aiogram import types, Dispatcher
from stickerify import stickerify
from PIL import Image
from io import BytesIO

user_sticker_packs = {}


async def handle_start(message: types.Message):
    user_id = message.from_user.id
    await message.reply("Welcome to Stickerify bot!")


async def handle_photo(message: types.Message, bot):
    photo = message.photo[-1]

    file = await bot.get_file(photo.file_id)
    file_path = file.file_path
    contents = await bot.download_file(file_path)

    image = Image.open(BytesIO(contents.getvalue()))
    stickerified_image = stickerify(image)

    bio = BytesIO()
    bio.name = 'image.jpeg'
    stickerified_image.save(bio, 'JPEG')
    bio.seek(0)

    # Send the processed image back to the user
    await bot.send_photo(chat_id=message.chat.id, photo=bio, caption="Here is your sticker")

def setup_handlers(dp: Dispatcher, bot):
    dp.register_message_handler(handle_start, commands=['start'])
    dp.register_message_handler(lambda message: handle_photo(message, bot), content_types=['photo'])
