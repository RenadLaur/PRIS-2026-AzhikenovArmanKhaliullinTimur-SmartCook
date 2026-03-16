from functools import lru_cache

try:
    from .knowledge_graph import load_graph
    from .logic import process_text_message
    from .nlp import get_spacy_status, warmup_spacy_model
    from .pipeline import run_image_pipeline
    from .recommender import (
        graph_lists,
        infer_meal_tags,
        recipe_allergens,
        recipe_calories,
        recipe_ingredients,
    )
    from .vision import get_vision_status
except ImportError:
    from knowledge_graph import load_graph
    from logic import process_text_message
    from nlp import get_spacy_status, warmup_spacy_model
    from pipeline import run_image_pipeline
    from recommender import graph_lists, infer_meal_tags, recipe_allergens, recipe_calories, recipe_ingredients
    from vision import get_vision_status


def compact_text(text):
    return " ".join(str(text or "").split())


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
def get_data_source():
    return load_graph()


@lru_cache(maxsize=1)
def get_nlp_runtime():
    return warmup_spacy_model()


def get_runtime_status():
    graph_ok = True
    graph_error = ""
    try:
        get_data_source()
    except Exception as exc:  # pragma: no cover - runtime safeguard
        graph_ok = False
        graph_error = str(exc)

    return {
        "graph": {
            "ok": graph_ok,
            "error": graph_error,
        },
        "nlp": {
            "status": get_spacy_status(),
            "runtime": get_nlp_runtime(),
        },
        "vision": get_vision_status(),
    }


def handle_chat_message(text):
    clean_text = str(text or "").strip()
    if not clean_text:
        return {
            "ok": False,
            "query": "",
            "response": "Пустой запрос.",
        }

    try:
        graph = get_data_source()
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return {
            "ok": False,
            "query": clean_text,
            "response": f"Ошибка загрузки базы знаний: {exc}",
        }

    try:
        response = process_text_message(clean_text, graph)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return {
            "ok": False,
            "query": clean_text,
            "response": f"Ошибка обработки запроса: {exc}",
        }

    return {
        "ok": True,
        "query": clean_text,
        "response": compact_text(response),
    }


def handle_image_message(image_bytes):
    if not image_bytes:
        return {
            "ok": False,
            "response": "Пустое изображение.",
            "vision_result": {"error": "Пустое изображение."},
        }

    try:
        graph = get_data_source()
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return {
            "ok": False,
            "response": f"Ошибка загрузки базы знаний: {exc}",
            "vision_result": {"error": f"Ошибка загрузки базы знаний: {exc}"},
        }

    try:
        result = run_image_pipeline(image_bytes, graph)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return {
            "ok": False,
            "response": f"Ошибка анализа изображения: {exc}",
            "vision_result": {"error": f"Ошибка анализа изображения: {exc}"},
        }

    result["ok"] = not bool(result.get("vision_result", {}).get("error"))
    return result


def get_dashboard_data():
    graph = get_data_source()
    recipes, ingredients, allergens = graph_lists(graph)

    recipe_rows = []
    allergen_counter = {}
    meal_counter = {}

    for recipe_name in recipes:
        calories = recipe_calories(graph, recipe_name)
        recipe_ingredient_list = recipe_ingredients(graph, recipe_name)
        recipe_allergen_list = recipe_allergens(graph, recipe_name)
        meal_tags = infer_meal_tags(recipe_name, recipe_ingredient_list)

        recipe_rows.append(
            {
                "recipe": recipe_name,
                "calories": calories,
                "ingredients_count": len(recipe_ingredient_list),
                "allergens_count": len(recipe_allergen_list),
                "meal_tags": ", ".join(meal_tags),
                "ingredients": ", ".join(recipe_ingredient_list),
                "allergens": ", ".join(recipe_allergen_list) if recipe_allergen_list else "нет",
            }
        )

        for allergen in recipe_allergen_list:
            allergen_counter[allergen] = allergen_counter.get(allergen, 0) + 1

        for tag in meal_tags:
            meal_counter[tag] = meal_counter.get(tag, 0) + 1

    allergen_rows = [
        {"allergen": allergen, "recipes_count": count}
        for allergen, count in sorted(allergen_counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    meal_rows = [
        {"meal_type": meal_type, "recipes_count": count}
        for meal_type, count in sorted(meal_counter.items(), key=lambda item: (-item[1], item[0]))
    ]

    top_calories = sorted(recipe_rows, key=lambda item: item["calories"], reverse=True)[:10]

    return {
        "summary": {
            "recipes_count": len(recipes),
            "ingredients_count": len(ingredients),
            "allergens_count": len(allergens),
        },
        "recipes": recipe_rows,
        "top_calories": top_calories,
        "allergen_stats": allergen_rows,
        "meal_stats": meal_rows,
    }


def get_api_catalog():
    return [
        {"method": "GET", "path": "/health", "purpose": "Проверка доступности API"},
        {"method": "GET", "path": "/status", "purpose": "Статус NLP/CV и графа"},
        {"method": "POST", "path": "/chat", "purpose": "Обработка текстового запроса"},
        {"method": "POST", "path": "/image/analyze", "purpose": "Анализ изображения блюда"},
    ]
