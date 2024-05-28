import os
import sys
import logging

from aiogram.enums import ParseMode
from aiogram import Bot, Dispatcher, Router

from common import *
from handlers import setup_handlers, web_app


logging.basicConfig(level=logging.INFO)

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


@app.function(image=image, secrets=[modal.Secret.from_name("hse-bot-modal-token")])
@modal.asgi_app()
def run_app():
    router = Router()

    setup_handlers(router)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    bot = Bot(os.environ["hse_bot"], parse_mode=ParseMode.HTML)

    web_app.state.dispatcher = dispatcher
    web_app.state.bot = bot

    return web_app
