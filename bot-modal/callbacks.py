from aiogram.filters.callback_data import CallbackData


class StyleCallbackData(CallbackData, prefix="style"):
    style: str
    photo_uuid: str


class ChoiceCallbackData(CallbackData, prefix="choice"):
    choice: int
    photo_uuid: str


class AnimationCallbackData(CallbackData, prefix="animation"):
    animation_style: str
    photo_uuid: str
    choice: int