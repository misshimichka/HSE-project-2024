## Проект по созданию датасета стикеров и обучению модели для генерации стикеров

### Обзор
Основная задача - обучение модели для трансформации изображений в стилизованные стикеры и добавление к ним анимаций.

### Датасеты

1. **Общий датасет объектов** ([Ссылка на Kaggle](https://www.kaggle.com/datasets/dmitrykutsenko/synthetic-stickers-dataset))
   - Исходные изображения различных объектов без лиц.
   - Пары изображений: оригинал и его стилизованная версия в виде стикера.
   - Работа с датасетом велась в течение 1.5-2 месяцев.

2. **Датасет лиц** ([Превью датасета](https://huggingface.co/datasets/Alexator26/479_stickers_improved_v2))
   - Содержит пары изображений: фотографии лиц и соответствующие стикеры.
   - Разработка и очистка датасета заняли около 2 недель.

### Ноутбуки и их функции

#### Для датасета объектов:

- `prompt-collecting-with-api.ipynb` - сбор промптов через API ChatGPT.
- `prompt-collecting.ipynb` - сбор промптов через парсинг диалогов с ChatGPT.
- `prompt-filtration.ipynb` - фильтрация промптов, удаление дубликатов.
- `synthetic-dataset-generation.ipynb` - генерация датасета с использованием базы данных Firebase.
- `calc_metrics.ipynb` - расчет метрик сходства стикеров и исходных изображений.

#### Для датасета лиц:

- `generate-faces.ipynb` - генерация стикеров из фотографий лиц.
- `drop-similar.ipynb` - удаление похожих фотографий лиц.
- `faces-clean-notebook.ipynb` - очистка датасета и расчет метрик.

### Обучение модели

- `train_pix2pix.ipynb` - обучение модели pix2pix для трансформации лиц в стикеры.
- Процесс дообучения стилей, интеграция существующих стилей в базовую модель.

### Улучшение модели и визуализация результатов

- `utils/improved_inference.ipynb` - улучшения инференса модели.
- `animations/base_inference.ipynb` - создание анимаций стикеров.

### Визуальные примеры работы модели

- Исходное фото пользователя: `img1.jpg`
- Стикер старой модели: `img2.jpg`
- Сетка стикеров новой модели: `img3.jpg`

### Внешние ресурсы

- [Датасет розовые волосы](https://huggingface.co/datasets/misshimichka/pink_hair_dataset_cleared)
- [Датасет цветы](https://huggingface.co/datasets/misshimichka/flower_faces_dataset_v3)
- [Датасет клоунское лицо](https://huggingface.co/datasets/misshimichka/clown_faces_dataset_cleared)
- [Датасет бабочек](https://huggingface.co/datasets/misshimichka/butterfly_faces_dataset_v1)
- [Датасет уши кошки](https://huggingface.co/datasets/misshimichka/cat_faces_dataset_cleared)
- [Датасет объекты базовый стиль](https://huggingface.co/datasets/Alexator26/479_stickers_improved_v2)
- [Датасет объекты базовый стиль v1](https://huggingface.co/datasets/Alexator26/703_stickers_improved_v1)
- [Датасет лица базовый стиль](https://huggingface.co/datasets/misshimichka/face_stickers_cleared)

---

### Демо

Попробуйте нашего бота в Telegram для демонстрации возможностей генерации стикеров: [Telegram Bot](https://t.me/hse_project_test_bot)
