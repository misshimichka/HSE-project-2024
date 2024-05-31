import io

from aiogram.types import BufferedInputFile, FSInputFile, update
from aiogram import Router, types
from aiogram.filters import CommandStart, Command
from aiogram import F

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from callbacks import StyleCallbackData, ChoiceCallbackData, AnimationCallbackData
from markups import get_choice_markup, get_animations_markup, get_styles_markup
from generate_animation import *
from generate_sticker import *
from common import *

import uuid

with image.imports():
    from skimage import img_as_ubyte


async def webhook(request):
    dispatcher = request.app.state.dispatcher
    bot = request.app.state.bot
    upd = update.Update.model_validate(await request.json(), context={"bot-modal": bot})

    await dispatcher.feed_update(bot=bot, update=upd)
    return JSONResponse({'status': 'ok'})


web_app = Starlette(debug=True, routes=[
    Route("/", webhook, methods=["POST"])
])

photo_storage = {}
photo_uuid_dict = {}
sticker_storage = {}


async def generate_and_send_sticker(chat_id, style, photo_uuid):
    if chat_id in photo_storage.keys() and photo_uuid in photo_storage[chat_id]:
        photo_storage[chat_id].remove(photo_uuid)
        file_id = photo_uuid_dict.get(photo_uuid)
        try:
            file = await web_app.state.bot.get_file(file_id)
            file_path = file.file_path
            contents = await web_app.state.bot.download_file(file_path)

            img = Image.open(contents)

            await web_app.state.bot.send_message(chat_id, "Started generating your sticker! 👨‍🔬")

            grid, stickerified_images = generate.remote(img, style, photo_uuid)

            if not grid:
                await web_app.state.bot.send_message(
                    chat_id=chat_id,
                    text="Unfortunately, we couldn't find a human face on your photo, "
                         "or there were too many of them 😰 Please, send another photo."
                )
                return

            buffer = io.BytesIO()
            grid.save(buffer, format='webp')
            sticker_storage[photo_uuid] = stickerified_images

            await web_app.state.bot.send_photo(
                chat_id=chat_id,
                photo=BufferedInputFile(buffer.getvalue(), filename=f"grid_{photo_uuid}.webp"),
                caption="Choose the generation you like!",
                reply_markup=get_choice_markup(photo_uuid)
            )

        except Exception as e:
            print(e)
            await web_app.state.bot.send_message(
                chat_id=chat_id,
                text=f"Sorry, an error occurred."
            )

    else:
        await web_app.state.bot.send_message(
            chat_id=chat_id,
            text="We couldn't find your photo. Please send it again."
        )


async def generate_and_send_animation(chat_id, style, choice, photo_uuid):
    try:
        await web_app.state.bot.send_message(
            chat_id=chat_id,
            text="Started generating your animation!"
        )

        animation, fps = generate_animation(
            img=sticker_storage[photo_uuid][choice - 1],
            style=style
        )

        imageio.mimsave(f"animation_{photo_uuid}_{choice}.mp4", [img_as_ubyte(p) for p in animation], fps=fps)

        with imageio.imopen(f"animation_{photo_uuid}_{choice}.webm", "w", plugin="pyav") as out_file:
            out_file.init_video_stream("vp9", fps=fps)

            for frame in imageio.imiter(f"animation_{photo_uuid}_{choice}.mp4", plugin="pyav"):
                out_file.write_frame(frame)

        await web_app.state.bot.send_sticker(
            chat_id=chat_id,
            sticker=FSInputFile(f"animation_{photo_uuid}_{choice}.webm")
        )

    except Exception as e:
        print(e)
        await web_app.state.bot.send_message(
            chat_id=chat_id,
            text=f"Sorry, an error occurred."
        )


async def handle_start(message: types.Message):
    await message.reply(
        "Welcome to Stickerify bot-modal 77! 🥶\n"
        "Send me a photo and I will create your own sticker 👠"
    )


async def handle_debug(message: types.Message):
    print(photo_storage)
    print(sticker_storage)
    print(photo_uuid_dict)


async def handle_photo(message: types.Message):
    chat_id = message.chat.id
    photo_id = message.photo[-1].file_id
    photo_uuid = str(uuid.uuid4())
    photo_uuid_dict[photo_uuid] = photo_id
    if chat_id not in photo_storage.keys():
        photo_storage[chat_id] = set()
    photo_storage[chat_id].add(photo_uuid)

    await message.reply("Choose your sticker style:", reply_markup=get_styles_markup(photo_uuid))


async def handle_animation_callback(callback_query: types.CallbackQuery, callback_data: AnimationCallbackData):
    chat_id = callback_query.from_user.id
    animation_style = callback_data.animation_style
    choice = callback_data.choice
    photo_uuid = callback_data.photo_uuid

    await generate_and_send_animation(chat_id, animation_style, choice, photo_uuid)


async def handle_style_callback(callback_query: types.CallbackQuery, callback_data: StyleCallbackData):
    chat_id = callback_query.from_user.id
    style = callback_data.style
    photo_uuid = callback_data.photo_uuid

    await generate_and_send_sticker(chat_id, style, photo_uuid)


async def handle_choice_callback(callback_query: types.CallbackQuery, callback_data: ChoiceCallbackData):
    chat_id = callback_query.from_user.id
    choice = callback_data.choice
    photo_uuid = callback_data.photo_uuid

    buffer = io.BytesIO()
    sticker_storage[photo_uuid][choice - 1].save(buffer, format='webp')

    try:
        await web_app.state.bot.send_sticker(
            chat_id=chat_id,
            sticker=BufferedInputFile(buffer.getvalue(), filename=f"result_{photo_uuid}_{choice}.webp"),
            emoji="🎁",
            reply_markup=get_animations_markup(photo_uuid, choice)
        )
    except Exception as e:
        await web_app.state.bot.send_message(
            chat_id=chat_id,
            text="Sorry, an error occurred"
        )
        print(e)


def setup_handlers(router: Router):
    router.message.register(handle_start, CommandStart())
    router.message.register(handle_photo, F.content_type.in_({'photo'}))
    router.message.register(handle_debug, Command("debug"))
    router.callback_query.register(handle_style_callback, StyleCallbackData.filter())
    router.callback_query.register(handle_choice_callback, ChoiceCallbackData.filter())
    router.callback_query.register(handle_animation_callback, AnimationCallbackData.filter())
