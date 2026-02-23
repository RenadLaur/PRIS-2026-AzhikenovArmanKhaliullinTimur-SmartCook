from collections import Counter
from difflib import get_close_matches
from functools import lru_cache
import importlib.util
import re
import sys


RESULT_LIMIT_DEFAULT = 8

SERVINGS_WORDS = {
    "одного": 1,
    "одну": 1,
    "один": 1,
    "одной": 1,
    "двоих": 2,
    "двух": 2,
    "троих": 3,
    "трех": 3,
    "четверых": 4,
    "четырех": 4,
}

ALLERGEN_ALIASES = {
    "молоч": "Молоко",
    "лактоз": "Молоко",
    "глютен": "Глютен",
    "пшен": "Глютен",
    "орех": "Орехи",
    "арахис": "Орехи",
    "яйц": "Яйца",
    "рыб": "Рыба",
    "морепродукт": "Морепродукты",
    "соя": "Соя",
    "кунжут": "Кунжут",
}

MEAL_ALIASES = {
    "завтрак": "завтрак",
    "утро": "завтрак",
    "обед": "обед",
    "ужин": "ужин",
    "вечер": "ужин",
    "перекус": "перекус",
}


def _normalize(text):
    return str(text).strip().lower().replace("ё", "е")


def _tokenize(text):
    return [token for token in re.split(r"[^a-zA-Zа-яА-Я0-9]+", _normalize(text)) if token]


def _dedupe_keep_order(items):
    seen = set()
    result = []
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _extract_matches(query, candidates, cutoff=0.83):
    if not candidates:
        return []

    normalized = {_normalize(item): item for item in candidates}
    query_norm = _normalize(query)

    direct = []
    for key, value in normalized.items():
        if key in query_norm:
            direct.append(value)
    if direct:
        return _dedupe_keep_order(direct)

    close = []
    for token in _tokenize(query_norm):
        if len(token) < 3:
            continue
        matched = get_close_matches(token, list(normalized.keys()), n=1, cutoff=cutoff)
        if matched:
            close.append(normalized[matched[0]])
    return _dedupe_keep_order(close)


def _extract_calorie_constraints(text):
    query = _normalize(text)

    range_match = re.search(r"(\d{2,4})\s*[-–]\s*(\d{2,4})\s*ккал", query)
    if range_match:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        return min(low, high), max(low, high), int((low + high) / 2)

    max_match = re.search(r"(до|меньше|не более)\s*(\d{2,4})", query)
    min_match = re.search(r"(от|больше|не менее)\s*(\d{2,4})", query)
    around_match = re.search(r"(около|примерно|~)\s*(\d{2,4})\s*ккал", query)

    max_cal = int(max_match.group(2)) if max_match else None
    min_cal = int(min_match.group(2)) if min_match else None
    target = int(around_match.group(2)) if around_match else None

    if target is not None and min_cal is None and max_cal is None:
        min_cal = max(0, target - 100)
        max_cal = target + 100

    if max_cal is None and "низкокал" in query:
        max_cal = 450
    if min_cal is None and "высококал" in query:
        min_cal = 600

    if target is None and min_cal is not None and max_cal is not None:
        target = int((min_cal + max_cal) / 2)

    return min_cal, max_cal, target


def _extract_servings(text):
    query = _normalize(text)

    digit_match = re.search(r"(\d{1,2})\s*(порц|чел|персон|people|servings?)", query)
    if digit_match:
        return int(digit_match.group(1))

    word_match = re.search(r"на\s+(одного|одну|один|одной|двоих|двух|троих|трех|четверых|четырех)", query)
    if word_match:
        return SERVINGS_WORDS.get(word_match.group(1))

    return None


def _detect_meal_type(text):
    query = _normalize(text)
    for key, meal in MEAL_ALIASES.items():
        if key in query:
            return meal
    return None


def _detect_diet_tags(text):
    query = _normalize(text)
    tags = []
    pairs = [
        ("веган", "vegan"),
        ("вегетариан", "vegetarian"),
        ("безглютен", "gluten_free"),
        ("пп", "healthy"),
        ("диет", "diet"),
        ("без сахара", "no_sugar"),
        ("без молока", "dairy_free"),
        ("кето", "keto"),
        ("белков", "high_protein"),
    ]
    for needle, tag in pairs:
        if needle in query:
            tags.append(tag)
    return tags


def _extract_result_limit(text):
    query = _normalize(text)

    patterns = [
        r"(?:покажи|дай|подбери|предложи|выведи)\s*(\d{1,2})\s*(?:рецепт|вариант|блюд)",
        r"топ\s*(\d{1,2})",
        r"(\d{1,2})\s*(?:рецепт|вариант|блюд)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            value = int(match.group(1))
            return max(1, min(20, value))

    return RESULT_LIMIT_DEFAULT


def _classify_cooking_intent(text):
    query = _normalize(text)

    if any(token in query for token in ["без ", "аллерг", "не переношу", "исключи", "убери"]):
        return "Фильтрация по аллергенам"
    if any(token in query for token in ["ккал", "калори", "низкокал", "высококал"]):
        return "Фильтрация по калорийности"
    if any(token in query for token in ["что приготовить", "посоветуй", "подбери", "напиши рецепт"]):
        return "Рекомендация рецепта"
    if any(token in query for token in ["рецепт", "блюдо", "ингредиент"]):
        return "Поиск рецепта по параметрам"
    return "Свободный кулинарный запрос"


def _detect_query_mode(text, entities):
    query = _normalize(text)

    if any(word in query for word in ["покажи", "список", "какие", "перечисли"]):
        if "аллерген" in query:
            return "list_allergens"
        if "ингредиент" in query:
            return "list_ingredients"
        if "рецепт" in query:
            return "list_recipes"

    if entities.get("recipes") and any(word in query for word in ["состав", "калор", "что в", "инфо", "подроб"]):
        return "recipe_detail"

    if entities.get("allergens") and any(word in query for word in ["что такое", "опас", "аллерген", "чем опас"]):
        return "allergen_detail"

    if entities.get("ingredients") and any(word in query for word in ["где используется", "с чем", "в каких рецептах", "из чего"]):
        return "ingredient_detail"

    if any(word in query for word in ["что приготовить", "подбери", "посоветуй", "рецепт", "блюдо", "приготов"]):
        return "recipe_recommendation"

    if any(word in query for word in ["ккал", "калори"]):
        return "nutrition_filter"

    return "generic"


@lru_cache(maxsize=1)
def _load_spacy():
    try:
        import spacy  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise RuntimeError(
            "spaCy не установлен в текущем окружении. Установите его и перезапустите приложение."
        )

    try:
        return spacy.load("ru_core_news_sm"), "ru_core_news_sm"
    except OSError as exc:
        raise RuntimeError(
            "Модель `ru_core_news_sm` не найдена. Установите: "
            "`python -m spacy download ru_core_news_sm`"
        ) from exc


def get_spacy_status():
    spacy_installed = importlib.util.find_spec("spacy") is not None
    ru_model = importlib.util.find_spec("ru_core_news_sm") is not None
    model_found = ru_model

    return {
        "spacy_installed": spacy_installed,
        "model_found": model_found,
        "models": {
            "ru_core_news_sm": ru_model,
        },
        "python_executable": sys.executable,
    }


def _graph_catalogs(data_source):
    if data_source is None or not hasattr(data_source, "nodes"):
        return [], [], []

    ingredients = [
        node for node in data_source.nodes if data_source.nodes[node].get("type") == "ingredient"
    ]
    allergens = [
        node for node in data_source.nodes if data_source.nodes[node].get("type") == "allergen"
    ]
    recipes = [node for node in data_source.nodes if data_source.nodes[node].get("type") == "recipe"]
    return ingredients, allergens, recipes


def _extract_graph_entities(text, ingredients, allergens, recipes):
    return {
        "ingredients": _extract_matches(text, ingredients),
        "allergens": _extract_matches(text, allergens),
        "recipes": _extract_matches(text, recipes),
    }


def _extract_with_spacy(text):
    nlp, model_name = _load_spacy()

    doc = nlp(text)
    lemmas = []
    named_entities = []

    for token in doc:
        if not token.is_alpha:
            continue
        lemma = (token.lemma_ or token.text).lower().strip()
        if len(lemma) >= 3:
            lemmas.append(lemma)

    for ent in doc.ents:
        named_entities.append(f"{ent.text} ({ent.label_})")

    lemma_counter = Counter(lemmas)
    return {
        "engine": f"spacy:{model_name}",
        "lemmas": [lemma for lemma, _ in lemma_counter.most_common(15)],
        "named_entities": _dedupe_keep_order(named_entities),
    }


def _extract_negative_segments(text):
    query = _normalize(text)
    markers = ["без ", "кроме ", "исключи ", "убери ", "не добавляй "]
    segments = []

    for marker in markers:
        start = 0
        while True:
            idx = query.find(marker, start)
            if idx == -1:
                break
            fragment = query[idx + len(marker) :]
            for delimiter in [".", "?", "!", " на ", " до ", " от ", " для "]:
                cut_idx = fragment.find(delimiter)
                if cut_idx != -1:
                    fragment = fragment[:cut_idx]
            parts = re.split(r",| и |/|;", fragment)
            for part in parts:
                cleaned = part.strip()
                if len(cleaned) >= 3:
                    segments.append(cleaned)
            start = idx + len(marker)

    return _dedupe_keep_order(segments)


def _resolve_allergen_aliases(text, allergen_catalog):
    if not allergen_catalog:
        return []

    normalized_catalog = {_normalize(item): item for item in allergen_catalog}
    query = _normalize(text)
    resolved = []

    for alias, canonical in ALLERGEN_ALIASES.items():
        if alias in query:
            canonical_key = _normalize(canonical)
            if canonical_key in normalized_catalog:
                resolved.append(normalized_catalog[canonical_key])
    return _dedupe_keep_order(resolved)


def _extract_negative_entities(text, ingredients, allergens):
    negative_segments = _extract_negative_segments(text)
    excluded_ingredients = []
    excluded_allergens = []

    for segment in negative_segments:
        excluded_ingredients.extend(_extract_matches(segment, ingredients, cutoff=0.8))
        excluded_allergens.extend(_extract_matches(segment, allergens, cutoff=0.8))

    excluded_allergens.extend(_resolve_allergen_aliases(" ".join(negative_segments), allergens))
    return _dedupe_keep_order(excluded_ingredients), _dedupe_keep_order(excluded_allergens)


def analyze_cooking_request(text, data_source=None):
    text = str(text or "").strip()
    if not text:
        return {
            "engine": "none",
            "query_mode": "generic",
            "intent": "Пустой запрос",
            "entities": {"ingredients": [], "allergens": [], "recipes": []},
            "filters": {
                "include_ingredients": [],
                "exclude_ingredients": [],
                "exclude_allergens": [],
                "max_results": RESULT_LIMIT_DEFAULT,
            },
            "constraints": {
                "min_calories": None,
                "max_calories": None,
                "target_calories": None,
                "servings": None,
            },
            "meal_type": None,
            "diet_tags": [],
            "lemmas": [],
            "named_entities": [],
            "warnings": [],
        }

    spacy_result = _extract_with_spacy(text)

    min_cal, max_cal, target_cal = _extract_calorie_constraints(text)
    servings = _extract_servings(text)
    meal_type = _detect_meal_type(text)
    diet_tags = _detect_diet_tags(text)
    result_limit = _extract_result_limit(text)

    ingredients_catalog, allergens_catalog, recipes_catalog = _graph_catalogs(data_source)

    search_text = text
    if spacy_result["lemmas"]:
        search_text = f"{text} {' '.join(spacy_result['lemmas'])}"

    mentioned_entities = _extract_graph_entities(
        search_text, ingredients_catalog, allergens_catalog, recipes_catalog
    )
    excluded_ingredients, excluded_allergens = _extract_negative_entities(
        text, ingredients_catalog, allergens_catalog
    )

    positive_ingredients = [
        item for item in mentioned_entities.get("ingredients", []) if item not in excluded_ingredients
    ]
    positive_allergens = [
        item for item in mentioned_entities.get("allergens", []) if item not in excluded_allergens
    ]

    entities = {
        "ingredients": positive_ingredients,
        "allergens": positive_allergens,
        "recipes": mentioned_entities.get("recipes", []),
    }

    query_mode = _detect_query_mode(text, entities)
    intent = _classify_cooking_intent(text)

    warnings = []
    overlap = set(positive_ingredients) & set(excluded_ingredients)
    if overlap:
        warnings.append(
            "Есть противоречие: ингредиенты одновременно запрошены и исключены "
            f"({', '.join(sorted(overlap))})."
        )

    return {
        "engine": spacy_result["engine"],
        "query_mode": query_mode,
        "intent": intent,
        "entities": entities,
        "filters": {
            "include_ingredients": positive_ingredients,
            "exclude_ingredients": excluded_ingredients,
            "exclude_allergens": excluded_allergens,
            "max_results": result_limit,
        },
        "constraints": {
            "min_calories": min_cal,
            "max_calories": max_cal,
            "target_calories": target_cal,
            "servings": servings,
        },
        "meal_type": meal_type,
        "diet_tags": diet_tags,
        "lemmas": spacy_result["lemmas"],
        "named_entities": spacy_result["named_entities"],
        "warnings": warnings,
    }


def _line(title, values):
    if values:
        return f"- {title}: {', '.join(values)}"
    return f"- {title}: не найдено"


def analyze_text_message(text, data_source=None):
    result = analyze_cooking_request(text, data_source)
    entities = result["entities"]
    filters = result["filters"]
    constraints = result["constraints"]
    lines = [
        f"Тип запроса: {result['intent']}",
        f"Режим запроса: {result['query_mode']}",
        f"NLP движок: {result['engine']}",
        "",
        "Распознанные кулинарные сущности:",
        _line("Рецепты", entities.get("recipes", [])),
        _line("Ингредиенты", entities.get("ingredients", [])),
        _line("Аллергены", entities.get("allergens", [])),
        "",
        "Извлеченные фильтры:",
        _line("Исключить ингредиенты", filters.get("exclude_ingredients", [])),
        _line("Исключить аллергены", filters.get("exclude_allergens", [])),
        f"- Максимум результатов: {filters.get('max_results', RESULT_LIMIT_DEFAULT)}",
        "",
        "Ограничения и параметры:",
        f"- Калории (мин): {constraints['min_calories'] if constraints['min_calories'] is not None else 'не указано'}",
        f"- Калории (макс): {constraints['max_calories'] if constraints['max_calories'] is not None else 'не указано'}",
        f"- Калории (цель): {constraints['target_calories'] if constraints['target_calories'] is not None else 'не указано'}",
        f"- Порции: {constraints['servings'] if constraints['servings'] is not None else 'не указано'}",
        f"- Прием пищи: {result['meal_type'] or 'не указан'}",
        _line("Диетические теги", result["diet_tags"]),
        "",
        f"Леммы (топ): {', '.join(result['lemmas']) if result['lemmas'] else 'не найдено'}",
    ]

    if result["named_entities"]:
        lines.append(f"Доп. spaCy сущности: {', '.join(result['named_entities'])}")
    if result["warnings"]:
        lines.extend(["", "Примечания:"] + [f"- {warning}" for warning in result["warnings"]])

    return "\n".join(lines)