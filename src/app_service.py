from functools import lru_cache
from pathlib import Path
import re

try:
    from .logic import process_text_interaction
    from .nlp import analyze_cooking_request, get_spacy_status, warmup_spacy_model
    from .pipeline import run_image_pipeline
    from .recommender import (
        get_translation_status,
        get_recipenlg_preview,
        get_search_index_status,
        recipenlg_csv_path,
        translate_to_en,
    )
    from .vision import get_vision_status
except ImportError:
    from logic import process_text_interaction
    from nlp import analyze_cooking_request, get_spacy_status, warmup_spacy_model
    from pipeline import run_image_pipeline
    from recommender import (
        get_translation_status,
        get_recipenlg_preview,
        get_search_index_status,
        recipenlg_csv_path,
        translate_to_en,
    )
    from vision import get_vision_status


def compact_text(text):
    return " ".join(str(text or "").split())


DEFAULT_ASSISTANT_MESSAGE = (
    "Привет! Я бот SmartCook. Напишите запрос о блюде или рецепте на русском, либо загрузите фото блюда."
)


def get_sample_queries():
    return [
        "похожие на плов",
        "подбери ужин с курицей",
        "подбери завтрак с яйцом",
        "рецепты без молока",
        "что похоже на омлет",
        "/nlp покажи датасеты",
    ]


@lru_cache(maxsize=1)
def get_nlp_runtime():
    return warmup_spacy_model()


def initial_chat_messages():
    return [{"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}]


def initial_chat_context():
    return {
        "last_recipe_title": "",
        "query_recipe_history": {},
    }


def _error_response(query, message):
    return {
        "ok": False,
        "query": str(query or "").strip(),
        "response": message,
    }


def _is_followup_query(query_text):
    lowered = str(query_text or "").strip().lower()
    return lowered.startswith(("другой", "другая", "другое", "другую", "еще", "ещё"))


def _history_key(query_text):
    return re.sub(r"\s+", " ", str(query_text or "").strip().lower())


def _semantic_query_bucket(query_text):
    parsed = analyze_cooking_request(query_text)
    parts = []
    service_lemmas = {
        "рецепт", "подобрать", "подбери", "приготовить", "посоветовать",
        "показать", "сделать", "блюдо", "еда", "что", "еще", "ещё", "другой",
    }

    meal_type = parsed.get("meal_type")
    filters = parsed.get("filters", {})
    include_ingredients = filters.get("include_ingredients", [])
    exclude_ingredients = filters.get("exclude_ingredients", [])
    exclude_allergens = filters.get("exclude_allergens", [])
    if meal_type and not include_ingredients and not exclude_ingredients and not exclude_allergens:
        return f"meal:{meal_type}"

    if meal_type:
        parts.append(f"meal:{meal_type}")

    for ingredient in include_ingredients:
        parts.append(f"ing:{ingredient.lower()}")
    for ingredient in exclude_ingredients:
        parts.append(f"ex_ing:{ingredient.lower()}")
    for allergen in exclude_allergens:
        parts.append(f"ex_allergen:{allergen.lower()}")

    lemmas = [
        lemma for lemma in parsed.get("lemmas", [])
        if lemma and lemma not in service_lemmas and len(lemma) >= 3
    ]
    if len(lemmas) == 1 and not include_ingredients and not exclude_ingredients and not exclude_allergens:
        return f"lemma:{lemmas[0]}"
    if lemmas:
        parts.extend(f"lemma:{lemma}" for lemma in lemmas[:3])

    native_tokens = [
        token
        for token in re.split(r"[^a-zA-Zа-яА-Я0-9]+", str(query_text or "").lower())
        if len(token) >= 3 and token not in service_lemmas
    ]
    if native_tokens:
        parts.extend(f"topic:{token}" for token in native_tokens[:3])

    if not parts:
        return _history_key(query_text)
    return "|".join(dict.fromkeys(parts))


def _build_chat_context(query_text, chat_state):
    if not isinstance(chat_state, dict):
        return None, None

    query_key = _semantic_query_bucket(query_text)
    context = {}
    previous_titles = chat_state.get("query_recipe_history", {}).get(query_key, [])
    if previous_titles:
        context["exclude_titles"] = list(previous_titles)

    last_recipe_title = str(chat_state.get("last_recipe_title", "")).strip()
    if _is_followup_query(query_text) and last_recipe_title:
        context["exclude_titles"] = list(context.get("exclude_titles", [])) + [
            last_recipe_title,
            translate_to_en(last_recipe_title),
        ]

    return (context or None), query_key


def _update_chat_state(chat_state, query_key, recipe_title):
    if not isinstance(chat_state, dict):
        return ""

    recipe_title = str(recipe_title or "").strip()
    if not recipe_title:
        return ""

    chat_state["last_recipe_title"] = recipe_title
    query_recipe_history = dict(chat_state.get("query_recipe_history", {}))
    query_titles = list(query_recipe_history.get(query_key, []))
    for candidate in (recipe_title, translate_to_en(recipe_title)):
        normalized_candidate = candidate.strip()
        if normalized_candidate and normalized_candidate not in query_titles:
            query_titles.append(normalized_candidate)
    query_recipe_history[query_key] = query_titles[-40:]
    chat_state["query_recipe_history"] = query_recipe_history
    return recipe_title

def get_runtime_status():
    dataset_inventory = get_dataset_inventory()
    search_index_status = get_search_index_status()

    return {
        "datasets": {
            "recipenlg_ready": bool(recipenlg_csv_path()),
            "food11_ready": bool(get_vision_status().get("food11_ready")),
            "inventory": dataset_inventory,
            "search_index": search_index_status,
        },
        "nlp": {
            "status": get_spacy_status(),
            "runtime": get_nlp_runtime(),
            "translation": get_translation_status(),
        },
        "vision": get_vision_status(),
    }


def handle_chat_message(text, context=None):
    clean_text = str(text or "").strip()
    if not clean_text:
        return _error_response("", "Пустой запрос.")

    chat_context = context
    query_key = None
    if isinstance(context, dict) and any(key in context for key in ["query_recipe_history", "last_recipe_title"]):
        chat_context, query_key = _build_chat_context(clean_text, context)

    try:
        interaction = process_text_interaction(clean_text, None, context=chat_context)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return _error_response(clean_text, f"Ошибка обработки запроса: {exc}")

    response = interaction.get("response", "")
    recipe_title = interaction.get("recipe_title", "")
    if query_key and recipe_title:
        recipe_title = _update_chat_state(context, query_key, recipe_title)
    return {
        "ok": True,
        "query": clean_text,
        "response": compact_text(response),
        "recipe_title": recipe_title,
        "query_bucket": query_key or "",
    }


def handle_image_message(image_bytes):
    if not image_bytes:
        return {
            "ok": False,
            "response": "Пустое изображение.",
            "vision_result": {"error": "Пустое изображение."},
        }

    try:
        result = run_image_pipeline(image_bytes, None)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return {
            "ok": False,
            "response": f"Ошибка анализа изображения: {exc}",
            "vision_result": {"error": f"Ошибка анализа изображения: {exc}"},
        }

    result["ok"] = not bool(result.get("vision_result", {}).get("error"))
    return result


def get_dashboard_data():
    dataset_inventory = get_dataset_inventory()
    external_dataset_data = get_external_dataset_data()

    return {
        "summary": {
            "recipenlg_rows": dataset_inventory["recipenlg_rows"],
            "food11_train_images": dataset_inventory["food11_train"]["images"],
            "food11_test_images": dataset_inventory["food11_test"]["images"],
            "food11_classes": dataset_inventory["food11_train"]["classes"],
        },
        "dataset_inventory": dataset_inventory,
        "external_dataset_data": external_dataset_data,
    }


def get_api_catalog():
    return [
        {"method": "GET", "path": "/health", "purpose": "Проверка доступности API"},
        {"method": "GET", "path": "/status", "purpose": "Статус NLP/CV, переводчика и датасетов"},
        {"method": "POST", "path": "/chat", "purpose": "Обработка текстового запроса"},
        {"method": "POST", "path": "/image/analyze", "purpose": "Анализ изображения блюда"},
    ]


@lru_cache(maxsize=1)
def get_dataset_inventory():
    recipenlg_path = recipenlg_csv_path()
    recipenlg_rows = 0
    if recipenlg_path:
        with open(recipenlg_path, "r", encoding="utf-8", errors="ignore") as file:
            recipenlg_rows = max(sum(1 for _ in file) - 1, 0)

    vision_status = get_vision_status()
    food11_root = Path(vision_status["food11_path"] or "")
    split_candidates = {
        "train": [food11_root / "train", food11_root / "food11" / "train"],
        "test": [food11_root / "test", food11_root / "food11" / "test"],
    }

    def resolve_split(split_name):
        for candidate in split_candidates[split_name]:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    train_dir = resolve_split("train")
    test_dir = resolve_split("test")

    def split_stats(split_dir):
        if split_dir is None:
            return {"classes": 0, "images": 0}
        class_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
        image_count = 0
        for class_dir in class_dirs:
            image_count += sum(1 for path in class_dir.iterdir() if path.is_file())
        return {"classes": len(class_dirs), "images": image_count}

    return {
        "recipenlg_rows": recipenlg_rows,
        "food11_train": split_stats(train_dir),
        "food11_test": split_stats(test_dir),
    }


@lru_cache(maxsize=1)
def get_external_dataset_data():
    vision_status = get_vision_status()
    food11_root = Path(vision_status["food11_path"] or "")
    food11_train_dir = None
    for candidate in [food11_root / "train", food11_root / "food11" / "train"]:
        if candidate.exists() and candidate.is_dir():
            food11_train_dir = candidate
            break

    food11_class_stats = []
    if food11_train_dir is not None:
        for class_dir in sorted(food11_train_dir.iterdir()):
            if class_dir.is_dir():
                image_count = sum(1 for path in class_dir.iterdir() if path.is_file())
                food11_class_stats.append(
                    {
                        "class": class_dir.name,
                        "train_images": image_count,
                    }
                )

    return {
        "recipenlg_preview": get_recipenlg_preview(limit=10),
        "food11_class_stats": food11_class_stats,
    }
