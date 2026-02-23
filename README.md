# SmartCook

SmartCook - учебный AI-ассистент для подбора рецептов с учетом ингредиентов, аллергенов и калорийности.

## Что умеет проект

- искать рецепты, ингредиенты и аллергены в графе знаний;
- подбирать рецепты по условиям (калории, прием пищи, исключения);
- обрабатывать запросы через NLP (`/nlp ...`);
- распознавать и объяснять подключенные датасеты из проекта.

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

## Примеры запросов в чате

- `покажи рецепты`
- `рецепты с курицей до 500 ккал`
- `без глютена`
- `/nlp Подбери ужин на двоих без молока`
- `/nlp покажи датасеты`
- `/nlp где скачать food11`

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
- Pandas / NumPy
