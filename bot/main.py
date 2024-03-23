from aiogram import Bot, Dispatcher, executor
import logging
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from config import API_TOKEN

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())


def main():
    setup_handlers(dp)
    executor.start_polling(dp, skip_updates=True)


if __name__ == '__main__':
    main()
