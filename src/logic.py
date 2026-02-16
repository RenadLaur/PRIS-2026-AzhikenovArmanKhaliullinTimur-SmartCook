import json
import os
import re
from difflib import get_close_matches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH = os.path.join(BASE_DIR, "data", "raw", "rules.json")


def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_rules(data):
    """
    Принимает словарь данных (data), возвращает строковый вердикт.
    """
    rules = load_rules()

    # --- 1. HARD FILTERS (Критические проверки) ---
    if rules["critical_rules"]["must_have_allergy_info"] and not data["has_allergy_info"]:
        return "⛔️ Критическая ошибка: Не указаны аллергии пользователя"

    # --- 2. БИЗНЕС-ЛОГИКА (Сравнение) ---
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
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def _normalize(text):
    return str(text).strip().lower().replace("ё", "е")


def _split_tokens(text):
    return [token for token in re.split(r"[^a-zA-Zа-яА-Я0-9]+", _normalize(text)) if token]


def _join_items(items):
    return ", ".join(sorted({str(item) for item in items}))


def _graph_lists(graph):
    recipes = sorted(
        node for node in graph.nodes if graph.nodes[node].get("type") == "recipe"
    )
    ingredients = sorted(
        node for node in graph.nodes if graph.nodes[node].get("type") == "ingredient"
    )
    allergens = sorted(
        node for node in graph.nodes if graph.nodes[node].get("type") == "allergen"
    )
    return recipes, ingredients, allergens


def _recipe_calories(graph, recipe_name):
    data = graph.nodes[recipe_name].get("data")
    calories = getattr(data, "calories", 0) if data is not None else 0
    return int(calories or 0)


def _format_recipe_list(graph, recipes, limit=8):
    if not recipes:
        return "нет подходящих рецептов"
    shown = recipes[:limit]
    rendered = [f"{name} ({_recipe_calories(graph, name)} ккал)" for name in shown]
    if len(recipes) > limit:
        rendered.append(f"... и еще {len(recipes) - limit}")
    return ", ".join(rendered)


def _find_graph_matches(graph, query):
    node_map = {_normalize(node): node for node in graph.nodes}
    if query in node_map:
        return [node_map[query]]

    partial_matches = [
        node
        for node in graph.nodes
        if query in _normalize(node) or _normalize(node) in query
    ]
    if partial_matches:
        return sorted(partial_matches)

    close = get_close_matches(query, list(node_map.keys()), n=5, cutoff=0.65)
    return [node_map[item] for item in close]


def _extract_entity_from_query(query, entities):
    normalized_map = {_normalize(entity): entity for entity in entities}

    for normalized_entity, entity in normalized_map.items():
        if normalized_entity in query:
            return entity

    tokens = _split_tokens(query)

    for token in tokens:
        if len(token) < 4:
            continue
        for normalized_entity, entity in normalized_map.items():
            if " " in normalized_entity:
                continue
            prefix_len = min(4, len(token), len(normalized_entity))
            if token[:prefix_len] == normalized_entity[:prefix_len]:
                return entity

    for token in tokens:
        if len(token) < 4:
            continue
        close = get_close_matches(token, list(normalized_map.keys()), n=1, cutoff=0.8)
        if close:
            return normalized_map[close[0]]

    return None


def _recipes_with_ingredient(graph, ingredient_name):
    if ingredient_name not in graph:
        return []
    return sorted(
        node
        for node in graph.neighbors(ingredient_name)
        if graph.nodes[node].get("type") == "recipe"
    )


def _risky_and_safe_recipes(graph, allergen_name):
    risky_recipes = set()
    for ingredient in graph.neighbors(allergen_name):
        if graph.nodes[ingredient].get("type") != "ingredient":
            continue
        for recipe in graph.neighbors(ingredient):
            if graph.nodes[recipe].get("type") == "recipe":
                risky_recipes.add(recipe)

    all_recipes = {
        node for node in graph.nodes if graph.nodes[node].get("type") == "recipe"
    }
    safe_recipes = sorted(all_recipes - risky_recipes)
    return sorted(risky_recipes), safe_recipes


def _filter_recipes_by_calories(graph, max_cal=None, min_cal=None):
    matched = []
    for node in graph.nodes:
        if graph.nodes[node].get("type") != "recipe":
            continue
        calories = _recipe_calories(graph, node)
        if max_cal is not None and calories > max_cal:
            continue
        if min_cal is not None and calories < min_cal:
            continue
        matched.append(node)
    return sorted(matched, key=lambda recipe: _recipe_calories(graph, recipe))


def _describe_node(graph, node_name):
    node_type = graph.nodes[node_name].get("type", "unknown")
    neighbors = list(graph.neighbors(node_name))

    if node_type == "recipe":
        ingredients = [node for node in neighbors if graph.nodes[node].get("type") == "ingredient"]
        allergens = sorted(
            {
                allergen
                for ingredient in ingredients
                for allergen in graph.neighbors(ingredient)
                if graph.nodes[allergen].get("type") == "allergen"
            }
        )
        response = [
            f"Рецепт: {node_name} ({_recipe_calories(graph, node_name)} ккал)",
            f"Ингредиенты: {_join_items(ingredients) if ingredients else 'нет данных'}",
            f"Аллергены: {_join_items(allergens) if allergens else 'не обнаружены'}",
        ]
        return "\n".join(response)

    if node_type == "ingredient":
        recipes = [node for node in neighbors if graph.nodes[node].get("type") == "recipe"]
        allergens = [node for node in neighbors if graph.nodes[node].get("type") == "allergen"]
        response = [
            f"Ингредиент: {node_name}",
            f"Используется в рецептах: {_join_items(recipes) if recipes else 'нет данных'}",
            f"Связанные аллергены: {_join_items(allergens) if allergens else 'не обнаружены'}",
        ]
        return "\n".join(response)

    if node_type == "allergen":
        ingredients = [node for node in neighbors if graph.nodes[node].get("type") == "ingredient"]
        risky_recipes, _ = _risky_and_safe_recipes(graph, node_name)
        response = [
            f"Аллерген: {node_name}",
            f"Источники (ингредиенты): {_join_items(ingredients) if ingredients else 'нет данных'}",
            f"Рецепты с риском: {_join_items(risky_recipes) if risky_recipes else 'не обнаружены'}",
        ]
        return "\n".join(response)

    return f"Я нашел '{node_name}' в базе. Соседи: {_join_items(neighbors)}"


def process_text_message(text, data_source):
    """
    Принимает текст пользователя, ищет термин в графе или словаре
    и возвращает ответ в виде строки.
    """
    if text is None:
        return "Я не знаю такого термина"

    query = _normalize(text)
    if not query:
        return "Я не знаю такого термина"

    if any(word in query for word in ["привет", "здравствуй", "добрый день", "hello"]):
        return "Привет! Я готов помочь. Напиши название объекта."

    if query in {"помощь", "help", "что ты умеешь", "команды"}:
        return (
            "Я умею:\n"
            "- искать рецепты, ингредиенты и аллергены;\n"
            "- подбирать безопасные рецепты (например: 'без глютена');\n"
            "- находить рецепты по ингредиенту ('рецепты с курицей');\n"
            "- фильтровать по калорийности ('до 500 ккал')."
        )

    if hasattr(data_source, "nodes") and hasattr(data_source, "neighbors"):
        recipes, ingredients, allergens = _graph_lists(data_source)

        if "покажи рецепты" in query or query in {"рецепты", "список рецептов"}:
            return f"Доступные рецепты: {_join_items(recipes)}"
        if "покажи ингредиенты" in query or query in {"ингредиенты", "список ингредиентов"}:
            return f"Доступные ингредиенты: {_join_items(ingredients)}"
        if "покажи аллергены" in query or query in {"аллергены", "список аллергенов"}:
            return f"Доступные аллергены: {_join_items(allergens)}"

        if any(
            phrase in query
            for phrase in ["что приготовить", "посоветуй", "подбери рецепт", "рекоменд"]
        ):
            low_cal = _filter_recipes_by_calories(data_source, max_cal=500)
            if not low_cal:
                low_cal = _filter_recipes_by_calories(data_source)
            return (
                "Могу предложить несколько вариантов:\n"
                f"{_format_recipe_list(data_source, low_cal, limit=5)}"
            )

        if any(phrase in query for phrase in ["без ", "аллерг", "исключи", "не переношу"]):
            allergen_name = _extract_entity_from_query(query, allergens)
            if allergen_name:
                risky_recipes, safe_recipes = _risky_and_safe_recipes(data_source, allergen_name)
                safe_text = _format_recipe_list(data_source, safe_recipes, limit=8)
                risky_text = _format_recipe_list(data_source, risky_recipes, limit=6)
                return (
                    f"При аллергии на '{allergen_name}' безопаснее выбирать:\n"
                    f"{safe_text}\n"
                    f"Под риск попадают: {risky_text}"
                )

        wants_recipes_by_ingredient = (
            "рецепт" in query and (" с " in f" {query} " or "из " in query)
        ) or "ингредиент" in query
        if wants_recipes_by_ingredient:
            ingredient_name = _extract_entity_from_query(query, ingredients)
            if ingredient_name:
                ingredient_recipes = _recipes_with_ingredient(data_source, ingredient_name)
                if ingredient_recipes:
                    return (
                        f"Рецепты с ингредиентом '{ingredient_name}': "
                        f"{_format_recipe_list(data_source, ingredient_recipes, limit=8)}"
                    )
                return f"По ингредиенту '{ingredient_name}' рецепты не найдены."

        calorie_keywords = ["ккал", "калори", "низкокал", "высококал"]
        if any(word in query for word in calorie_keywords):
            max_match = re.search(r"(до|меньше|не более)\s*(\d{2,4})", query)
            min_match = re.search(r"(от|больше|не менее)\s*(\d{2,4})", query)

            max_cal = int(max_match.group(2)) if max_match else None
            min_cal = int(min_match.group(2)) if min_match else None

            if max_cal is None and "низкокал" in query:
                max_cal = 450
            if min_cal is None and "высококал" in query:
                min_cal = 600

            cal_filtered = _filter_recipes_by_calories(
                data_source, max_cal=max_cal, min_cal=min_cal
            )
            if cal_filtered:
                rule = []
                if min_cal is not None:
                    rule.append(f"от {min_cal} ккал")
                if max_cal is not None:
                    rule.append(f"до {max_cal} ккал")
                rule_text = " и ".join(rule) if rule else "по калорийности"
                return (
                    f"Подобрал рецепты {rule_text}:\n"
                    f"{_format_recipe_list(data_source, cal_filtered, limit=8)}"
                )
            return "По заданному диапазону калорий рецепты не найдены."

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
        "Попробуйте: 'покажи рецепты', 'рецепты с курицей', 'без глютена', "
        "'до 500 ккал' или точное название блюда."
    )
