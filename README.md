# SmartCook

SmartCook - учебный AI-ассистент для подбора рецептов с учетом ингредиентов, аллергенов и калорийности.

## Что умеет проект

- искать рецепты, ингредиенты и аллергены в графе знаний;
- подбирать рецепты по условиям (калории, прием пищи, исключения);
- обрабатывать запросы через NLP (`/nlp ...`);
- распознавать и объяснять подключенные датасеты из проекта;
- анализировать фото блюда (Computer Vision + OCR) и предлагать рецепт.

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download ru_core_news_sm
streamlit run src/main.py
```

## NLP: что добавлено

В `src/nlp.py` добавлен каталог датасетов и их алиасов. Теперь NLP распознает:

- `RecipeNLG Dataset`
- `Food-11 Image Classification Dataset`

Поддерживаются запросы:

- `/nlp покажи датасеты`
- `/nlp что такое RecipeNLG Dataset`
- `/nlp инфо про food-11`

Также остались кулинарные сценарии: рецепты, ингредиенты, аллергены, калории, порции, прием пищи.

## Week 7: Computer Vision (OCR + классификация изображений)

В проект добавлен обработчик изображений в `src/vision.py`:

- классификация фото по Food-11 через OpenCV (HSV histogram + cosine similarity);
- извлечение текста с изображения через EasyOCR (если установлен);
- подбор рецепта из графа знаний по результату классификации и OCR-подсказкам.

В UI (`streamlit run src/main.py`) есть блок:

- `Фото блюда: распознавание и рецепт`
- загрузка файла JPG/PNG/WEBP;
- кнопка `Определить блюдо и предложить рецепт`.

Если Food-11 не загружен локально, система все равно работает в fallback-режиме и предлагает рецепт из базы.

Для обучения CNN-классификатора на Food-11 (ResNet18):

```bash
.venv/bin/python scripts/train_food11_cnn.py --epochs 3
```

После обучения создается чекпоинт:

```text
artifacts/food11_resnet18.pt
```

Далее `src/vision.py` использует:
- Food-11 для классификации изображения;
- RecipeNLG для подбора рецепта по предсказанному классу.

## Подключение датасетов

### 1. RecipeNLG

Источник: <https://www.kaggle.com/datasets/saldenisov/recipenlg>

Рекомендуемая локальная папка:

```text
data/external/recipenlg/
```

### 2. Food-11 Image Classification

Источник: <https://www.kaggle.com/datasets/imbikramsaha/food11>

Рекомендуемая локальная папка:

```text
data/external/food11/
```

Важно: даже если файлы датасетов не скачаны локально, NLP уже знает их названия, алиасы, ссылки и умеет отвечать по ним в чате.

Примечание по CV: для реальной классификации по фото нужно положить изображения Food-11 в `data/external/food11/`.

## Примеры запросов в чате

- `покажи рецепты`
- `рецепты с курицей до 500 ккал`
- `без глютена`
- `/nlp Подбери ужин на двоих без молока`
- `/nlp покажи датасеты`
- `/nlp где скачать food11`
- загрузить фото блюда в блоке Computer Vision и нажать `Определить блюдо и предложить рецепт`

## Структура проекта

```text
src/
  main.py             # Streamlit UI
  logic.py            # Правила обработки запросов
  nlp.py              # NLP-анализ и извлечение сущностей
  knowledge_graph.py  # Граф рецептов/ингредиентов/аллергенов
data/
  raw/rules.json      # Правила фильтрации
```

## Технологии

- Python
- Streamlit
- spaCy
- NetworkX
- OpenCV
- EasyOCR
- Pandas / NumPy
