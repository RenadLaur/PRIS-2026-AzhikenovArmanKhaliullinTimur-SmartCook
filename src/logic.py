import json
import os
import re
from difflib import get_close_matches

try:
    from .nlp import analyze_cooking_request, analyze_text_message, get_known_datasets
except ImportError:
    from nlp import analyze_cooking_request, analyze_text_message, get_known_datasets

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


def _join_items(items):
    return ", ".join(sorted({str(item) for item in items}))


def _dataset_catalog():
    return get_known_datasets()


def _dataset_names():
    return sorted(dataset.get("name", "") for dataset in _dataset_catalog() if dataset.get("name"))


def _dataset_by_name(dataset_name):
    normalized = _normalize(dataset_name)
    for dataset in _dataset_catalog():
        if _normalize(dataset.get("name", "")) == normalized:
            return dataset
    return None


def _describe_dataset(dataset_name):
    dataset = _dataset_by_name(dataset_name)
    if dataset is None:
        return f"Датасет '{dataset_name}' не найден в каталоге."

    return (
        f"Датасет: {dataset['name']}\n"
        f"Описание: {dataset.get('description', 'нет описания')}\n"
        f"Источник: {dataset.get('url', 'не указан')}\n"
        f"Локальный путь: {dataset.get('local_path', 'не задан')}"
    )


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


def _extract_calorie_constraints(query):
    max_match = re.search(r"(до|меньше|не более)\s*(\d{2,4})", query)
    min_match = re.search(r"(от|больше|не менее)\s*(\d{2,4})", query)

    max_cal = int(max_match.group(2)) if max_match else None
    min_cal = int(min_match.group(2)) if min_match else None

    if max_cal is None and "низкокал" in query:
        max_cal = 450
    if min_cal is None and "высококал" in query:
        min_cal = 600
    return min_cal, max_cal


def _extract_meal_type(query):
    if "завтрак" in query:
        return "завтрак"
    if "обед" in query:
        return "обед"
    if "ужин" in query:
        return "ужин"
    return None


def _recipe_matches_meal(recipe_name, meal_type):
    name = _normalize(recipe_name)
    if meal_type == "завтрак":
        hints = ["омлет", "панкейк", "сырник", "шакшука", "йогурт", "каша"]
    elif meal_type == "обед":
        hints = ["суп", "плов", "лагман", "бешбармак", "манты", "рамен", "паста"]
    elif meal_type == "ужин":
        hints = ["лосось", "курица", "говядина", "креветки", "тофу"]
    else:
        return True
    return any(hint in name for hint in hints)


def _recipe_ingredients(graph, recipe_name):
    return sorted(
        node
        for node in graph.neighbors(recipe_name)
        if graph.nodes[node].get("type") == "ingredient"
    )


def _recipe_allergens(graph, recipe_name):
    allergens = set()
    for ingredient in _recipe_ingredients(graph, recipe_name):
        for neighbor in graph.neighbors(ingredient):
            if graph.nodes[neighbor].get("type") == "allergen":
                allergens.add(neighbor)
    return sorted(allergens)


def _recipes_with_required_ingredients(graph, recipes, required_ingredients):
    if not required_ingredients:
        return recipes

    required = {_normalize(item) for item in required_ingredients}
    matched = []
    for recipe in recipes:
        recipe_ingredients = {_normalize(item) for item in _recipe_ingredients(graph, recipe)}
        if required.issubset(recipe_ingredients):
            matched.append(recipe)
    return matched


def _recipes_without_excluded_ingredients(graph, recipes, excluded_ingredients):
    if not excluded_ingredients:
        return recipes

    excluded = {_normalize(item) for item in excluded_ingredients}
    filtered = []
    for recipe in recipes:
        recipe_ingredients = {_normalize(item) for item in _recipe_ingredients(graph, recipe)}
        if recipe_ingredients.intersection(excluded):
            continue
        filtered.append(recipe)
    return filtered


def _exclude_recipes_by_requested_allergens(graph, recipes, allergens):
    if not allergens:
        return recipes

    risky = set()
    for allergen in allergens:
        risky_recipes, _ = _risky_and_safe_recipes(graph, allergen)
        risky.update(risky_recipes)
    return [recipe for recipe in recipes if recipe not in risky]


def _build_recipe_answer(graph, recipe_name):
    calories = _recipe_calories(graph, recipe_name)
    ingredients = _recipe_ingredients(graph, recipe_name)
    allergens = _recipe_allergens(graph, recipe_name)
    return (
        f"Рецепт: {recipe_name} ({calories} ккал). "
        f"Ингредиенты: {', '.join(ingredients) if ingredients else 'нет данных'}. "
        f"Аллергены: {', '.join(allergens) if allergens else 'не обнаружены'}."
    )


def _score_recipe(graph, recipe_name, include_ingredients, meal_type, target_calories):
    score = 0.0
    recipe_ingredients = {_normalize(item) for item in _recipe_ingredients(graph, recipe_name)}

    for ingredient in include_ingredients:
        if _normalize(ingredient) in recipe_ingredients:
            score += 6.0

    if meal_type and _recipe_matches_meal(recipe_name, meal_type):
        score += 2.0

    if target_calories is not None:
        calories = _recipe_calories(graph, recipe_name)
        score -= abs(calories - target_calories) / 120.0

    return score


def _rank_recipes(graph, candidates, include_ingredients, meal_type, target_calories):
    return sorted(
        candidates,
        key=lambda recipe: (
            _score_recipe(graph, recipe, include_ingredients, meal_type, target_calories),
            -_recipe_calories(graph, recipe),
        ),
        reverse=True,
    )


def _pick_recipe(candidates, meal_type, include_ingredients, target_calories, graph):
    if not candidates:
        return None

    ranked = _rank_recipes(graph, candidates, include_ingredients, meal_type, target_calories)

    if meal_type == "завтрак":
        priority = ["омлет", "шакшука", "сырник", "панкейк", "йогурт"]
        for hint in priority:
            for recipe in ranked:
                if hint in _normalize(recipe):
                    return recipe
    return ranked[0]


def _recommend_recipe_from_query(graph, raw_text, query):
    soft_markers = [
        "рецепт",
        "напиши",
        "приготов",
        "что приготовить",
        "посоветуй",
        "подбери",
        "ккал",
        "калори",
        "завтрак",
        "обед",
        "ужин",
        "без ",
        "аллерг",
        "ингредиент",
        "список",
        "покажи",
        "датасет",
        "dataset",
    ]
    if not any(marker in query for marker in soft_markers):
        return None

    try:
        parsed = analyze_cooking_request(raw_text, graph)
    except RuntimeError as exc:
        return f"Ошибка NLP: {exc}"

    mode = parsed.get("query_mode", "generic")
    entities = parsed.get("entities", {})
    filters = parsed.get("filters", {})
    constraints = parsed.get("constraints", {})
    meal_type = parsed.get("meal_type")
    include_ingredients = filters.get("include_ingredients", [])
    exclude_ingredients = filters.get("exclude_ingredients", [])
    exclude_allergens = filters.get("exclude_allergens", [])
    min_cal = constraints.get("min_calories")
    max_cal = constraints.get("max_calories")
    target_cal = constraints.get("target_calories")
    limit = max(1, min(20, int(filters.get("max_results", 8))))
    all_recipes, all_ingredients, all_allergens = _graph_lists(graph)
    all_datasets = _dataset_names()

    if mode == "list_datasets":
        return f"Подключенные датасеты: {_join_items(all_datasets)}"
    if mode == "dataset_detail" and entities.get("datasets"):
        return _describe_dataset(entities["datasets"][0])

    if mode == "list_allergens":
        return f"Доступные аллергены: {_join_items(all_allergens)}"
    if mode == "list_ingredients":
        return f"Доступные ингредиенты: {_join_items(all_ingredients)}"

    if mode == "recipe_detail" and entities.get("recipes"):
        return _describe_node(graph, entities["recipes"][0])
    if mode == "ingredient_detail" and entities.get("ingredients"):
        ingredient_name = entities["ingredients"][0]
        ingredient_recipes = _recipes_with_ingredient(graph, ingredient_name)
        if ingredient_recipes:
            return (
                f"Ингредиент '{ingredient_name}' встречается в: "
                f"{_format_recipe_list(graph, ingredient_recipes, limit=limit)}"
            )
    if mode == "allergen_detail" and entities.get("allergens"):
        return _describe_node(graph, entities["allergens"][0])

    candidates = _filter_recipes_by_calories(graph, max_cal=max_cal, min_cal=min_cal)
    if meal_type:
        candidates = [recipe for recipe in candidates if _recipe_matches_meal(recipe, meal_type)]
    candidates = _recipes_with_required_ingredients(graph, candidates, include_ingredients)
    candidates = _recipes_without_excluded_ingredients(graph, candidates, exclude_ingredients)
    candidates = _exclude_recipes_by_requested_allergens(graph, candidates, exclude_allergens)
    candidates = _rank_recipes(graph, candidates, include_ingredients, meal_type, target_cal)

    is_recipe_intent = (
        mode == "recipe_recommendation"
        or any(
            marker in query
            for marker in ["рецепт", "напиши", "приготов", "что приготовить", "посоветуй"]
        )
    )
    wants_list = mode in {"list_recipes", "list_ingredients"} or "рецепты" in query

    if not candidates:
        parts = []
        if meal_type:
            parts.append(meal_type)
        if include_ingredients:
            parts.append(f"с ингредиентами: {', '.join(include_ingredients)}")
        if exclude_ingredients:
            parts.append(f"без ингредиентов: {', '.join(exclude_ingredients)}")
        if exclude_allergens:
            parts.append(f"без аллергенов: {', '.join(exclude_allergens)}")
        if min_cal is not None:
            parts.append(f"от {min_cal} ккал")
        if max_cal is not None:
            parts.append(f"до {max_cal} ккал")
        suffix = "; ".join(parts) if parts else "по вашим условиям"
        return f"Не нашел рецепт ({suffix}). Попробуйте смягчить ограничения."

    if is_recipe_intent and not wants_list:
        chosen = _pick_recipe(candidates, meal_type, include_ingredients, target_cal, graph)
        return _build_recipe_answer(graph, chosen)

    if mode == "list_recipes" and min_cal is None and max_cal is None and not include_ingredients:
        return f"Доступные рецепты: {_join_items(all_recipes)}"

    return f"Подобрал варианты: {_format_recipe_list(graph, candidates, limit=limit)}"


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
            "- подбирать безопасные рецепты (например: 'без глютена');\n"
            "- находить рецепты по ингредиенту ('рецепты с курицей');\n"
            "- фильтровать по калорийности ('до 500 ккал');\n"
            "- показывать подключенные датасеты ('покажи датасеты')."
        )

    if hasattr(data_source, "nodes") and hasattr(data_source, "neighbors"):
        recipes, ingredients, allergens = _graph_lists(data_source)

        recipe_answer = _recommend_recipe_from_query(data_source, text, query)
        if recipe_answer:
            return recipe_answer

        if query in {"покажи рецепты", "рецепты", "список рецептов"}:
            return f"Доступные рецепты: {_join_items(recipes)}"
        if query in {"покажи ингредиенты", "ингредиенты", "список ингредиентов"}:
            return f"Доступные ингредиенты: {_join_items(ingredients)}"
        if query in {"покажи аллергены", "аллергены", "список аллергенов"}:
            return f"Доступные аллергены: {_join_items(allergens)}"
        if query in {"покажи датасеты", "датасеты", "список датасетов"}:
            return f"Подключенные датасеты: {_join_items(_dataset_names())}"

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
            min_cal, max_cal = _extract_calorie_constraints(query)

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
