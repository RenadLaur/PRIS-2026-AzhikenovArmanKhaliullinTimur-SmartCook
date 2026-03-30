# SmartCook

SmartCook - учебный AI-ассистент для подбора рецептов с учетом ингредиентов, аллергенов и калорийности.

## Что умеет проект

- искать рецепты в `RecipeNLG` и изображения в `Food-11`;
- подбирать рецепты по условиям (калории, прием пищи, исключения);
- обрабатывать запросы через NLP (`/nlp ...`);
- распознавать и объяснять подключенные датасеты из проекта;
- анализировать фото блюда (Computer Vision + OCR) и предлагать рецепт.
- искать похожие блюда через гибридные рекомендации (cosine similarity + fuzzy search).

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

## Week 8: Гибридные методы и рекомендации

В проект добавлен модуль `src/recommender.py`:

- векторизация рецептов по названию, ингредиентам, аллергенам и meal-tag;
- поиск и ранжирование рецептов из `RecipeNLG Dataset`;
- `cosine similarity` между запросом и рецептами;
- нечеткий поиск по формулировке (`fuzzy` на базе `SequenceMatcher`);
- гибридный score: `правила + cosine + fuzzy`.

В `src/pipeline.py` реализован единый текстовый пайплайн:

```text
Вход -> NLP -> правила/фильтры -> гибридное ранжирование -> решение
```

Примеры запросов:

- `похожие блюда на плов`
- `что похоже на омлет`
- `что похоже на блюдо с рисом и курицей`
- `подбери ужин с курицей до 500 ккал`
- `рецепты без молока`

В ответе для рекомендаций показывается причина выбора и hybrid score.

## Week 9-10: Интеграция модулей

В проекте добавлен модуль `src/pipeline.py`, который объединяет:

- текстовый вход -> `NLP` -> фильтры/правила -> рекомендации;
- вход по изображению -> `CV/OCR` -> правила выбора -> рецепт.

Это убирает разрозненную логику из UI и делает обработку запросов единым пайплайном.

## Week 11: API и Backend

Для Week 11 проект переведен на более модульную backend-архитектуру:

- `src/app_service.py` - сервисный слой, общий для UI и API;
- `src/api.py` - REST API на `FastAPI`;
- `src/api_schemas.py` - схемы запросов/ответов API;
- `src/main.py` - только Streamlit UI, без прямого вызова бизнес-логики.

Теперь структура разделена на:

- `UI`: `src/main.py`
- `Service/Backend`: `src/app_service.py`
- `API contracts`: `src/api_schemas.py`
- `Domain logic`: `src/logic.py`, `src/pipeline.py`, `src/recommender.py`, `src/nlp.py`, `src/vision.py`

### Запуск Streamlit UI

```bash
streamlit run src/main.py
```

### Запуск REST API

```bash
.venv/bin/uvicorn src.api:app --reload
```

После запуска доступны endpoints:

- `GET /health`
- `GET /status`
- `POST /chat`
- `POST /image/analyze`

Пример запроса к API:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"похожие на плов"}'
```

## Week 12: Пользовательский интерфейс (UI)

Интерфейс Streamlit переработан под Data App сценарий:

- вкладки `Чат`, `Фото блюда`, `Аналитика`, `API/Backend`;
- формы для отправки текстовых запросов и загрузки изображения;
- кнопки быстрых сценариев;
- таблицы и графики для датасетов и top-кандидатов CV;
- визуализация backend/API статусов прямо в приложении.

Что появилось в UI:

- быстрые кнопки запросов;
- форма отправки текста;
- форма загрузки изображения;
- таблица рецептов;
- bar charts по калориям, meal-type и аллергенам;
- таблица endpoints и JSON-статус backend.

## Week 13: Тестирование и подготовка к деплою

Для Week 13 в проект добавлены QA и reproducibility-проверки:

- unit-тесты для ключевых правил и backend-сценариев;
- smoke-тест API;
- фиксированные runtime-зависимости в `requirements.txt`;
- скрипт проверки окружения;
- единый скрипт QA-проверки перед деплоем.

### Unit tests

Тесты находятся в `tests/`:

- `tests/test_logic.py`
- `tests/test_pipeline.py`
- `tests/test_app_service.py`

Запуск:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### Проверка окружения

Проверка импортов и ключевых библиотек:

```bash
.venv/bin/python scripts/verify_environment.py
```

### Полный QA-прогон

```bash
bash scripts/run_qa_checks.sh
```

Этот сценарий проверяет:

- окружение;
- unit-тесты;
- compile check проекта.

## Week 7: Computer Vision (OCR + классификация изображений)

В проект добавлен обработчик изображений в `src/vision.py`:

- классификация фото по Food-11 через OpenCV (HSV histogram + cosine similarity);
- извлечение текста с изображения через EasyOCR (если установлен);
- подбор рецепта только из `RecipeNLG` по результату классификации и OCR-подсказкам.

В UI (`streamlit run src/main.py`) есть блок:

- `Фото блюда: распознавание и рецепт`
- загрузка файла JPG/PNG/WEBP;
- кнопка `Определить блюдо и предложить рецепт`.

Если Food-11 не загружен локально, система не сможет нормально классифицировать фото. Для текстовых рецептов нужен `RecipeNLG`, для CV-классификации нужен `Food-11`.

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
- `src/pipeline.py` для финального шага `CV -> rules -> decision`.

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
- `похожие блюда на плов`
- `что похоже на курицу с рисом`
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
  recommender.py      # Dataset-first поиск по RecipeNLG
  vision.py           # Food-11 + EasyOCR + RecipeNLG
data/
  raw/rules.json      # Правила фильтрации
```

## Технологии

- Python
- Streamlit
- spaCy
- OpenCV
- EasyOCR
- Pandas / NumPy
