from aiogram import Bot, Dispatcher, executor
import logging
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from config import API_TOKEN
from handlers import setup_handlers

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())


def main():
    setup_handlers(dp, bot)
    executor.start_polling(dp, skip_updates=True)


if __name__ == '__main__':
    main()
