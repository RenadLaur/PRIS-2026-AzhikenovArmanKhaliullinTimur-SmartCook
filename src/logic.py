import json
import os
import re
from difflib import get_close_matches

try:
    from .nlp import analyze_text_message
    from .pipeline import run_text_pipeline
    from .recommender import graph_lists, join_items, recipe_allergens, recipe_calories, recipe_ingredients
except ImportError:
    from nlp import analyze_text_message
    from pipeline import run_text_pipeline
    from recommender import graph_lists, join_items, recipe_allergens, recipe_calories, recipe_ingredients

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH = os.path.join(BASE_DIR, "data", "raw", "rules.json")


def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def check_rules(data):
    rules = load_rules()

    if rules["critical_rules"]["must_have_allergy_info"] and not data["has_allergy_info"]:
        return "⛔️ Критическая ошибка: Не указаны аллергии пользователя"

    min_cal = rules["thresholds"]["min_calories"]
    max_cal = rules["thresholds"]["max_calories"]

    if data["calories"] < min_cal:
        return "❌ Отказ: Калорийность ниже допустимого порога"

    if data["calories"] > max_cal:
        return "❌ Отказ: Калорийность выше допустимого порога"

    for ingredient in data["ingredients"]:
        if ingredient in rules["lists"]["blacklist"]:
            return f"⚠️ Предупреждение: Найден запрещенный ингредиент ({ingredient})"

    whitelist = rules["lists"]["whitelist"]
    if whitelist and not any(item in whitelist for item in data["ingredients"]):
        return "⚠️ Предупреждение: Нет рекомендованных ингредиентов"

    return f"✅ Успех: Блюдо соответствует сценарию '{rules['scenario_name']}'"


def _format_dict_value(value):
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}: {val}" for key, val in value.items())
    return str(value)


def _normalize(text):
    return str(text).strip().lower().replace("ё", "е")


def _split_tokens(text):
    return [token for token in re.split(r"[^a-zA-Zа-яА-Я0-9]+", _normalize(text)) if token]


def _is_nlp_request(query):
    query = str(query or "").strip()
    return query.startswith("/nlp") or query.startswith("nlp:")


def _extract_nlp_payload(raw_text):
    text = str(raw_text or "").strip()
    lowered = text.lower()

    prefixes = [
        "/nlp",
        "nlp:",
        "разбери запрос:",
        "проанализируй запрос:",
        "анализ запроса:",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip(" :\n\t")
    return text


def _find_graph_matches(graph, query):
    node_map = {_normalize(node): node for node in graph.nodes}
    if query in node_map:
        return [node_map[query]]

    partial_matches = [
        node for node in graph.nodes if query in _normalize(node) or _normalize(node) in query
    ]
    if partial_matches:
        return sorted(partial_matches)

    close = get_close_matches(query, list(node_map.keys()), n=5, cutoff=0.65)
    return [node_map[item] for item in close]


def _describe_node(graph, node_name):
    node_type = graph.nodes[node_name].get("type", "unknown")
    neighbors = list(graph.neighbors(node_name))

    if node_type == "recipe":
        response = [
            f"Рецепт: {node_name} ({recipe_calories(graph, node_name)} ккал)",
            f"Ингредиенты: {join_items(recipe_ingredients(graph, node_name)) or 'нет данных'}",
            f"Аллергены: {join_items(recipe_allergens(graph, node_name)) or 'не обнаружены'}",
        ]
        return "\n".join(response)

    if node_type == "ingredient":
        recipes = [node for node in neighbors if graph.nodes[node].get("type") == "recipe"]
        allergens = [node for node in neighbors if graph.nodes[node].get("type") == "allergen"]
        response = [
            f"Ингредиент: {node_name}",
            f"Используется в рецептах: {join_items(recipes) if recipes else 'нет данных'}",
            f"Связанные аллергены: {join_items(allergens) if allergens else 'не обнаружены'}",
        ]
        return "\n".join(response)

    if node_type == "allergen":
        ingredients = [node for node in neighbors if graph.nodes[node].get("type") == "ingredient"]
        response = [
            f"Аллерген: {node_name}",
            f"Источники (ингредиенты): {join_items(ingredients) if ingredients else 'нет данных'}",
        ]
        return "\n".join(response)

    return f"Я нашел '{node_name}' в базе. Соседи: {join_items(neighbors)}"


def process_text_message(text, data_source):
    if text is None:
        return "Я не знаю такого термина"

    query = _normalize(text)
    if not query:
        return "Я не знаю такого термина"

    if _is_nlp_request(query):
        payload = _extract_nlp_payload(text)
        if not payload.strip():
            return (
                "После `/nlp` передайте текст запроса.\n"
                "Пример: `/nlp Подбери ужин без глютена до 500 ккал с курицей на 2 порции`"
            )
        try:
            return analyze_text_message(payload, data_source)
        except RuntimeError as exc:
            return f"Ошибка NLP: {exc}"

    if any(word in query for word in ["привет", "здравствуй", "добрый день", "hello"]):
        return "Привет! Я готов помочь. Напиши название объекта."

    if query in {"помощь", "help", "что ты умеешь", "команды"}:
        return (
            "Я умею:\n"
            "- искать рецепты, ингредиенты и аллергены;\n"
            "- подбирать рецепты по правилам и ограничениям;\n"
            "- искать похожие блюда по cosine/fuzzy similarity;\n"
            "- обрабатывать запросы через NLP (`/nlp ...`);\n"
            "- анализировать фото блюда через CV/OCR."
        )

    if hasattr(data_source, "nodes") and hasattr(data_source, "neighbors"):
        pipeline_result = run_text_pipeline(text, data_source)
        if pipeline_result.get("handled"):
            return pipeline_result.get("response") or "Не удалось обработать запрос."

        recipes, ingredients, allergens = graph_lists(data_source)
        if query in {"покажи рецепты", "рецепты", "список рецептов"}:
            return f"Доступные рецепты: {join_items(recipes)}"
        if query in {"покажи ингредиенты", "ингредиенты", "список ингредиентов"}:
            return f"Доступные ингредиенты: {join_items(ingredients)}"
        if query in {"покажи аллергены", "аллергены", "список аллергенов"}:
            return f"Доступные аллергены: {join_items(allergens)}"

        matches = _find_graph_matches(data_source, query)
        if len(matches) == 1:
            return _describe_node(data_source, matches[0])
        if len(matches) > 1:
            preview = ", ".join(matches[:6])
            suffix = "" if len(matches) <= 6 else f" и еще {len(matches) - 6}"
            return f"Нашел несколько похожих терминов: {preview}{suffix}. Уточните запрос."

    if isinstance(data_source, dict):
        for key, value in data_source.items():
            if _normalize(key) == query:
                return f"Найдено по ключу '{key}': {_format_dict_value(value)}"

            if isinstance(value, str) and query in _normalize(value):
                return f"Найдено в '{key}': {value}"

            if isinstance(value, list):
                list_values = [str(item) for item in value]
                if query in [_normalize(item) for item in list_values]:
                    return f"Найдено в '{key}': {', '.join(list_values)}"

    return (
        "Я не знаю такого термина.\n"
        "Попробуйте: 'похожие блюда на плов', 'рецепты с курицей', 'без глютена', "
        "'до 500 ккал' или точное название блюда."
    )
