from aiogram import types

from bot.callbacks import StyleCallbackData, ChoiceCallbackData, AnimationCallbackData


def get_styles_markup(photo_uuid: str):
    default_btn = types.InlineKeyboardButton(text="Default 🤫🧏‍", callback_data=StyleCallbackData(
        style="default",
        photo_uuid=photo_uuid).pack())
    flowers_btn = types.InlineKeyboardButton(text="Flowers 🌸🌺", callback_data=StyleCallbackData(
        style="flowers",
        photo_uuid=photo_uuid).pack())
    cat_btn = types.InlineKeyboardButton(text="Cat ears 🐈🐱", callback_data=StyleCallbackData(
        style="cat",
        photo_uuid=photo_uuid).pack())
    butterfly_btn = types.InlineKeyboardButton(text="Butterflies 🦋🌈", callback_data=StyleCallbackData(
        style="butterfly",
        photo_uuid=photo_uuid).pack())
    clown_btn = types.InlineKeyboardButton(text="Clown 🤡🤣", callback_data=StyleCallbackData(
        style="clown",
        photo_uuid=photo_uuid).pack())
    pink_btn = types.InlineKeyboardButton(text="Pink hair 🩷✨", callback_data=StyleCallbackData(
        style="pink",
        photo_uuid=photo_uuid).pack())
    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[[default_btn, flowers_btn],
                         [cat_btn, butterfly_btn],
                         [clown_btn, pink_btn]]
    )
    return markup


def get_selection_markup(photo_uuid: str):
    choice_1 = types.InlineKeyboardButton(text="1️⃣",
                                          callback_data=ChoiceCallbackData(choice=1, photo_uuid=photo_uuid).pack())
    choice_2 = types.InlineKeyboardButton(text="2️⃣",
                                          callback_data=ChoiceCallbackData(choice=2, photo_uuid=photo_uuid).pack())
    choice_3 = types.InlineKeyboardButton(text="3️⃣",
                                          callback_data=ChoiceCallbackData(choice=3, photo_uuid=photo_uuid).pack())
    choice_4 = types.InlineKeyboardButton(text="4️⃣",
                                          callback_data=ChoiceCallbackData(choice=4, photo_uuid=photo_uuid).pack())
    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[[choice_1, choice_2],
                         [choice_3, choice_4]]
    )
    return markup


def get_animations_markup(photo_uuid: str, choice: int):
    wow_btn = types.InlineKeyboardButton(text="Wow😲",
                                         callback_data=AnimationCallbackData(animation_style="wow",
                                                                             photo_uuid=photo_uuid,
                                                                             choice=choice).pack())
    sigma_btn = types.InlineKeyboardButton(text="Sigma💪",
                                           callback_data=AnimationCallbackData(animation_style="sigma",
                                                                               photo_uuid=photo_uuid,
                                                                               choice=choice).pack())
    rock_btn = types.InlineKeyboardButton(text="Rock🏋️‍♂️",
                                          callback_data=AnimationCallbackData(animation_style="rock",
                                                                              photo_uuid=photo_uuid,
                                                                              choice=choice).pack())
    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[[wow_btn], [sigma_btn], [rock_btn]]
    )
    return markup