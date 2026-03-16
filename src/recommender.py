from collections import Counter
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from pathlib import Path
import ast
import math
import re

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional runtime import
    pd = None

try:
    from deep_translator import GoogleTranslator
except ImportError:  # pragma: no cover - optional runtime import
    GoogleTranslator = None

try:
    from .nlp import get_known_datasets
except ImportError:
    from nlp import get_known_datasets


MEAL_HINTS = {
    "завтрак": ["омлет", "шакшука", "сырник", "панкейк", "йогурт", "каша", "omelette", "pancake", "yogurt", "oatmeal", "egg"],
    "обед": ["суп", "плов", "лагман", "бешбармак", "манты", "рамен", "паста", "soup", "rice", "pasta", "noodle", "lunch"],
    "ужин": ["лосось", "курица", "говядина", "креветки", "тофу", "салат", "salmon", "chicken", "beef", "shrimp", "tofu", "salad", "dinner"],
    "перекус": ["батончик", "йогурт", "хумус", "сэндвич", "bar", "yogurt", "hummus", "sandwich", "snack"],
}
RECIPE_NLG_MAX_SCAN_CHUNKS = 8
RECIPE_NLG_CHUNK_SIZE = 50000
RECIPE_NLG_MAX_CANDIDATES = 120
TRANSLATE_CHUNK_LIMIT = 4500
STOP_TOKENS = {
    "что",
    "похоже",
    "похожие",
    "аналог",
    "блюдо",
    "рецепт",
    "рецепты",
    "подбери",
    "посоветуй",
    "приготовить",
    "приготовь",
    "мне",
    "для",
    "это",
}
DATASET_TOKEN_ALIASES = {
    "кур": ["chicken"],
    "рис": ["rice"],
    "яй": ["egg", "omelette"],
    "сыр": ["cheese"],
    "мол": ["milk"],
    "говяд": ["beef"],
    "баран": ["lamb"],
    "рыб": ["fish"],
    "лосос": ["salmon"],
    "тун": ["tuna"],
    "крев": ["shrimp"],
    "пиц": ["pizza"],
    "суш": ["sushi"],
    "бургер": ["burger", "hamburger"],
    "суп": ["soup"],
    "лапш": ["noodle"],
    "паст": ["pasta"],
    "карто": ["potato", "fries"],
    "обед": ["lunch", "soup", "rice", "pasta"],
    "ужин": ["dinner", "chicken", "beef", "fish"],
    "завтр": ["breakfast", "egg", "omelette", "pancake"],
    "десерт": ["dessert", "cake", "cookie"],
}


def normalize(text):
    return str(text).strip().lower().replace("ё", "е")


def tokenize(text):
    return [token for token in re.split(r"[^a-zA-Zа-яА-Я0-9]+", normalize(text)) if token]


def join_items(items):
    return ", ".join(sorted({str(item) for item in items}))


def graph_lists(graph):
    recipes = sorted(node for node in graph.nodes if graph.nodes[node].get("type") == "recipe")
    ingredients = sorted(
        node for node in graph.nodes if graph.nodes[node].get("type") == "ingredient"
    )
    allergens = sorted(node for node in graph.nodes if graph.nodes[node].get("type") == "allergen")
    return recipes, ingredients, allergens


def recipe_calories(graph, recipe_name):
    data = graph.nodes[recipe_name].get("data")
    calories = getattr(data, "calories", 0) if data is not None else 0
    return int(calories or 0)


def recipe_ingredients(graph, recipe_name):
    return sorted(
        node for node in graph.neighbors(recipe_name) if graph.nodes[node].get("type") == "ingredient"
    )


def recipe_allergens(graph, recipe_name):
    allergens = set()
    for ingredient in recipe_ingredients(graph, recipe_name):
        for neighbor in graph.neighbors(ingredient):
            if graph.nodes[neighbor].get("type") == "allergen":
                allergens.add(neighbor)
    return sorted(allergens)


def format_recipe_answer(graph, recipe_name):
    calories = recipe_calories(graph, recipe_name)
    ingredients = recipe_ingredients(graph, recipe_name)
    allergens = recipe_allergens(graph, recipe_name)
    return (
        f"Рецепт: {recipe_name} ({calories} ккал). "
        f"Ингредиенты: {', '.join(ingredients) if ingredients else 'нет данных'}. "
        f"Аллергены: {', '.join(allergens) if allergens else 'не обнаружены'}."
    )


def infer_meal_tags(recipe_name, ingredients=None):
    text = normalize(recipe_name)
    if ingredients:
        text = f"{text} {' '.join(normalize(item) for item in ingredients)}"

    tags = []
    for meal_type, hints in MEAL_HINTS.items():
        if any(hint in text for hint in hints):
            tags.append(meal_type)
    return tags or ["универсально"]


def recipe_document(graph, recipe_name):
    ingredients = recipe_ingredients(graph, recipe_name)
    allergens = recipe_allergens(graph, recipe_name)
    meal_tags = infer_meal_tags(recipe_name, ingredients)
    parts = [
        recipe_name,
        " ".join(ingredients),
        " ".join(allergens),
        " ".join(meal_tags),
        f"{recipe_calories(graph, recipe_name)} ккал",
    ]
    return " ".join(part for part in parts if part.strip())


def _dataset_by_name(name):
    target = normalize(name)
    for dataset in get_known_datasets():
        if normalize(dataset.get("name", "")) == target:
            return dataset
    return None


def recipenlg_csv_path():
    dataset = _dataset_by_name("RecipeNLG Dataset")
    if not dataset:
        return None

    root = Path(str(dataset.get("local_path", "")).strip())
    candidates = [root / "dataset" / "full_dataset.csv", root / "full_dataset.csv"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def recipenlg_ready():
    return recipenlg_csv_path() is not None and pd is not None


def _parse_list_like(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    parts = [item.strip(" '\"") for item in text.strip("[]").split(",")]
    return [item for item in parts if item]


@lru_cache(maxsize=1)
def _get_ru_translator():
    if GoogleTranslator is None:
        return None
    try:
        return GoogleTranslator(source="auto", target="ru")
    except Exception:  # pragma: no cover - network/runtime safeguard
        return None


@lru_cache(maxsize=1)
def _get_en_translator():
    if GoogleTranslator is None:
        return None
    try:
        return GoogleTranslator(source="auto", target="en")
    except Exception:  # pragma: no cover - network/runtime safeguard
        return None


@lru_cache(maxsize=512)
def translate_to_ru(text):
    text = str(text or "").strip()
    if not text:
        return ""

    translator = _get_ru_translator()
    if translator is None:
        return text

    if len(text) <= TRANSLATE_CHUNK_LIMIT:
        try:
            return translator.translate(text)
        except Exception:  # pragma: no cover - network/runtime safeguard
            return text

    return text


@lru_cache(maxsize=512)
def translate_to_en(text):
    text = str(text or "").strip()
    if not text:
        return ""

    translator = _get_en_translator()
    if translator is None:
        return text

    try:
        return translator.translate(text)
    except Exception:  # pragma: no cover - network/runtime safeguard
        return text


def _dataset_query_tokens(query_text, include_ingredients=None):
    include_ingredients = include_ingredients or []
    translated_query = translate_to_en(query_text)
    translated_ingredients = [translate_to_en(entry) for entry in include_ingredients]
    tokens = tokenize(query_text)
    tokens.extend(tokenize(translated_query))
    tokens.extend(tokenize(" ".join(include_ingredients)))
    tokens.extend(tokenize(" ".join(translated_ingredients)))
    unique = []
    for token in tokens:
        if len(token) < 3 or token in STOP_TOKENS:
            continue
        if token not in unique:
            unique.append(token)
        for stem, aliases in DATASET_TOKEN_ALIASES.items():
            if token.startswith(stem):
                for alias in aliases:
                    if alias not in unique:
                        unique.append(alias)
    return unique[:8]


def _dataset_recipe_document(item):
    parts = [
        item.get("title", ""),
        " ".join(item.get("ingredients", [])),
        " ".join(item.get("ner", [])),
        " ".join(item.get("directions", [])[:2]),
    ]
    return " ".join(part for part in parts if str(part).strip())


def _expand_dataset_alias_tokens(values):
    expanded = set()
    for value in values:
        for token in tokenize(value):
            expanded.add(token)
            for stem, aliases in DATASET_TOKEN_ALIASES.items():
                if token.startswith(stem):
                    expanded.update(normalize(alias) for alias in aliases)
    return expanded


def _dataset_recipe_matches_meal(item, meal_type):
    if not meal_type:
        return True
    text = normalize(_dataset_recipe_document(item))
    hints = MEAL_HINTS.get(meal_type, [])
    return any(hint in text for hint in hints)


def _dataset_rule_score(item, include_ingredients=None, exclude_ingredients=None, meal_type=None):
    include_ingredients = include_ingredients or []
    exclude_ingredients = exclude_ingredients or []
    ingredient_set = _expand_dataset_alias_tokens(item.get("ingredients", []))
    signals = []

    if include_ingredients:
        include_set = _expand_dataset_alias_tokens(include_ingredients)
        matched = len(include_set & ingredient_set)
        signals.append(matched / max(len(include_set), 1))

    if exclude_ingredients:
        exclude_set = _expand_dataset_alias_tokens(exclude_ingredients)
        signals.append(0.0 if ingredient_set.intersection(exclude_set) else 1.0)

    if meal_type:
        signals.append(1.0 if _dataset_recipe_matches_meal(item, meal_type) else 0.0)

    return sum(signals) / len(signals) if signals else 0.0


def _dataset_match_reason(item, include_ingredients, meal_type, cosine_score, fuzzy_score):
    reasons = []
    ingredient_set = _expand_dataset_alias_tokens(item.get("ingredients", []))

    if include_ingredients:
        matched = [
            entry for entry in include_ingredients if _expand_dataset_alias_tokens([entry]).intersection(ingredient_set)
        ]
        if matched:
            reasons.append(f"совпали ингредиенты: {', '.join(matched)}")

    if meal_type and _dataset_recipe_matches_meal(item, meal_type):
        reasons.append(f"подходит под прием пищи: {meal_type}")

    if cosine_score >= 0.3:
        reasons.append("высокая близость по описанию")
    if fuzzy_score >= 0.45:
        reasons.append("похоже по названию")

    if not reasons:
        reasons.append("лучший датасетный вариант")
    return "; ".join(reasons)


@lru_cache(maxsize=64)
def _search_recipenlg_candidates_cached(query_text, include_ingredients_key, limit):
    if pd is None:
        return []

    dataset_path = recipenlg_csv_path()
    if dataset_path is None:
        return []

    include_ingredients = list(include_ingredients_key)
    query_tokens = _dataset_query_tokens(query_text, include_ingredients)
    if not query_tokens:
        return []

    results = []
    seen_titles = set()

    try:
        reader = pd.read_csv(
            dataset_path,
            usecols=["title", "ingredients", "directions", "NER", "source"],
            chunksize=RECIPE_NLG_CHUNK_SIZE,
            on_bad_lines="skip",
        )
    except Exception:
        return []

    for chunk_idx, chunk in enumerate(reader):
        if chunk_idx >= RECIPE_NLG_MAX_SCAN_CHUNKS:
            break

        title_series = chunk["title"].astype(str)
        ingredients_series = chunk["ingredients"].astype(str)
        ner_series = chunk["NER"].astype(str)
        combined = (
            title_series.str.lower() + " " + ingredients_series.str.lower() + " " + ner_series.str.lower()
        )

        mask = False
        for token in query_tokens:
            mask = mask | combined.str.contains(re.escape(token), case=False, na=False)

        subset = chunk[mask]
        if subset.empty:
            continue

        for _, row in subset.head(40).iterrows():
            title = str(row.get("title", "")).strip()
            key = normalize(title)
            if not key or key in seen_titles:
                continue
            seen_titles.add(key)
            results.append(
                {
                    "title": title,
                    "ingredients": _parse_list_like(row.get("ingredients", "")),
                    "directions": _parse_list_like(row.get("directions", "")),
                    "ner": _parse_list_like(row.get("NER", "")),
                    "source": str(row.get("source", "")).strip(),
                }
            )
            if len(results) >= limit:
                return results

    return results


def search_recipenlg_candidates(query_text, include_ingredients=None, limit=RECIPE_NLG_MAX_CANDIDATES):
    include_ingredients = include_ingredients or []
    include_key = tuple(sorted(normalize(item) for item in include_ingredients))
    return _search_recipenlg_candidates_cached(str(query_text or ""), include_key, int(limit))


def rank_recipenlg_candidates(
    query_text,
    include_ingredients=None,
    exclude_ingredients=None,
    meal_type=None,
    limit=8,
):
    include_ingredients = include_ingredients or []
    exclude_ingredients = exclude_ingredients or []
    candidates = search_recipenlg_candidates(query_text, include_ingredients=include_ingredients)
    if not candidates:
        return []

    dataset_query_text = " ".join(_dataset_query_tokens(query_text, include_ingredients))
    query_vector = vectorize_text(dataset_query_text)
    query_normalized = normalize(dataset_query_text)
    exclude_set = _expand_dataset_alias_tokens(exclude_ingredients)
    ranked = []

    for item in candidates:
        ingredient_set = _expand_dataset_alias_tokens(item.get("ingredients", []))
        if exclude_set and ingredient_set.intersection(exclude_set):
            continue
        if meal_type and not _dataset_recipe_matches_meal(item, meal_type):
            continue

        document = _dataset_recipe_document(item)
        recipe_vector = vectorize_text(document)
        cosine_score = cosine_similarity(query_vector, recipe_vector)
        fuzzy_score = max(
            fuzzy_similarity(query_normalized, item.get("title", "")),
            fuzzy_similarity(query_normalized, " ".join(item.get("ingredients", []))),
        )
        rule_score = _dataset_rule_score(
            item,
            include_ingredients=include_ingredients,
            exclude_ingredients=exclude_ingredients,
            meal_type=meal_type,
        )
        total_score = (0.5 * cosine_score) + (0.2 * fuzzy_score) + (0.3 * rule_score)
        ranked.append(
            {
                "title": item.get("title", ""),
                "ingredients": item.get("ingredients", []),
                "directions": item.get("directions", []),
                "source": item.get("source", ""),
                "total_score": round(total_score, 4),
                "cosine_similarity": round(cosine_score, 4),
                "fuzzy_score": round(fuzzy_score, 4),
                "rule_score": round(rule_score, 4),
                "match_reason": _dataset_match_reason(
                    item,
                    include_ingredients,
                    meal_type,
                    cosine_score,
                    fuzzy_score,
                ),
            }
        )

    ranked.sort(
        key=lambda item: (item["total_score"], item["cosine_similarity"], item["fuzzy_score"]),
        reverse=True,
    )
    top_ranked = ranked[:limit]
    for item in top_ranked:
        item["title_ru"] = translate_to_ru(item.get("title", ""))
        item["ingredients_ru"] = [translate_to_ru(entry) for entry in item.get("ingredients", [])[:10]]
        item["directions_ru"] = [translate_to_ru(entry) for entry in item.get("directions", [])[:3]]
    return top_ranked


def vectorize_text(text):
    return Counter(tokenize(text))


def cosine_similarity(left_vector, right_vector):
    if not left_vector or not right_vector:
        return 0.0

    common = set(left_vector) & set(right_vector)
    numerator = sum(left_vector[token] * right_vector[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left_vector.values()))
    right_norm = math.sqrt(sum(value * value for value in right_vector.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def fuzzy_similarity(left_text, right_text):
    left = normalize(left_text)
    right = normalize(right_text)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def recipe_matches_meal(recipe_name, meal_type):
    if not meal_type:
        return True
    return meal_type in infer_meal_tags(recipe_name)


def filter_recipe_candidates(
    graph,
    include_ingredients=None,
    exclude_ingredients=None,
    exclude_allergens=None,
    meal_type=None,
    min_calories=None,
    max_calories=None,
):
    include_ingredients = include_ingredients or []
    exclude_ingredients = exclude_ingredients or []
    exclude_allergens = exclude_allergens or []

    include_set = {normalize(item) for item in include_ingredients}
    exclude_set = {normalize(item) for item in exclude_ingredients}
    exclude_allergen_set = {normalize(item) for item in exclude_allergens}

    recipes, _, _ = graph_lists(graph)
    filtered = []

    for recipe_name in recipes:
        calories = recipe_calories(graph, recipe_name)
        if min_calories is not None and calories < min_calories:
            continue
        if max_calories is not None and calories > max_calories:
            continue
        if meal_type and not recipe_matches_meal(recipe_name, meal_type):
            continue

        ingredients = {normalize(item) for item in recipe_ingredients(graph, recipe_name)}
        allergens = {normalize(item) for item in recipe_allergens(graph, recipe_name)}

        if include_set and not include_set.issubset(ingredients):
            continue
        if exclude_set and ingredients.intersection(exclude_set):
            continue
        if exclude_allergen_set and allergens.intersection(exclude_allergen_set):
            continue

        filtered.append(recipe_name)

    return filtered


def find_recipe_by_name(graph, text):
    recipes, _, _ = graph_lists(graph)
    normalized = {normalize(recipe): recipe for recipe in recipes}
    query = normalize(text)
    fragments = [query]

    pattern_fragments = [
        r"похож(?:ие|ий|ая)?\s+(?:блюда\s+)?на\s+(.+)",
        r"что\s+похоже\s+на\s+(.+)",
        r"аналог\s+(.+)",
        r"similar\s+to\s+(.+)",
    ]
    for pattern in pattern_fragments:
        match = re.search(pattern, query)
        if match:
            fragments.append(match.group(1).strip())

    for fragment in fragments:
        if fragment in normalized:
            return normalized[fragment]

    for fragment in fragments:
        for normalized_recipe, recipe_name in normalized.items():
            if normalized_recipe in fragment or fragment in normalized_recipe:
                return recipe_name

    for fragment in fragments:
        close = get_close_matches(fragment, list(normalized.keys()), n=1, cutoff=0.55)
        if close:
            return normalized[close[0]]

    for fragment in fragments:
        tokens = [token for token in tokenize(fragment) if len(token) >= 4]
        for token in tokens:
            for normalized_recipe, recipe_name in normalized.items():
                if token in normalized_recipe:
                    return recipe_name
    return None


def _symbolic_score(graph, recipe_name, include_ingredients, meal_type, target_calories, query_text):
    signals = []
    ingredients = {normalize(item) for item in recipe_ingredients(graph, recipe_name)}
    query_tokens = set(tokenize(query_text))

    if include_ingredients:
        include_set = {normalize(item) for item in include_ingredients}
        matched = len(include_set & ingredients)
        signals.append(matched / max(len(include_set), 1))
    else:
        overlap = len(query_tokens & ingredients)
        signals.append(min(1.0, overlap / 2.0))

    if meal_type:
        signals.append(1.0 if recipe_matches_meal(recipe_name, meal_type) else 0.0)

    if target_calories is not None:
        distance = abs(recipe_calories(graph, recipe_name) - target_calories)
        signals.append(max(0.0, 1.0 - (distance / max(target_calories, 250))))

    return sum(signals) / len(signals) if signals else 0.0


def _match_reason(graph, recipe_name, include_ingredients, meal_type, cosine_score, fuzzy_score):
    reasons = []
    recipe_ingredient_list = recipe_ingredients(graph, recipe_name)
    recipe_ingredient_set = {normalize(item) for item in recipe_ingredient_list}

    if include_ingredients:
        matched = [item for item in include_ingredients if normalize(item) in recipe_ingredient_set]
        if matched:
            reasons.append(f"совпали ингредиенты: {', '.join(matched)}")

    if meal_type and recipe_matches_meal(recipe_name, meal_type):
        reasons.append(f"подходит под прием пищи: {meal_type}")

    if cosine_score >= 0.45:
        reasons.append("высокая семантическая близость по описанию")
    if fuzzy_score >= 0.6:
        reasons.append("похоже по названию/формулировке")

    if not reasons:
        reasons.append("лучший доступный вариант после фильтрации")
    return "; ".join(reasons)


def rank_recipe_candidates(
    graph,
    candidates,
    query_text,
    include_ingredients=None,
    meal_type=None,
    target_calories=None,
    limit=8,
):
    include_ingredients = include_ingredients or []
    query_vector = vectorize_text(query_text)
    query_normalized = normalize(query_text)
    ranked = []

    for recipe_name in candidates:
        document = recipe_document(graph, recipe_name)
        recipe_vector = vectorize_text(document)
        cosine_score = cosine_similarity(query_vector, recipe_vector)
        fuzzy_score = max(
            fuzzy_similarity(query_normalized, recipe_name),
            fuzzy_similarity(query_normalized, " ".join(recipe_ingredients(graph, recipe_name))),
        )
        symbolic_score = _symbolic_score(
            graph,
            recipe_name,
            include_ingredients,
            meal_type,
            target_calories,
            query_text,
        )

        total_score = (0.5 * cosine_score) + (0.2 * fuzzy_score) + (0.3 * symbolic_score)
        ranked.append(
            {
                "recipe": recipe_name,
                "total_score": round(total_score, 4),
                "cosine_similarity": round(cosine_score, 4),
                "fuzzy_score": round(fuzzy_score, 4),
                "rule_score": round(symbolic_score, 4),
                "calories": recipe_calories(graph, recipe_name),
                "ingredients": recipe_ingredients(graph, recipe_name),
                "match_reason": _match_reason(
                    graph,
                    recipe_name,
                    include_ingredients,
                    meal_type,
                    cosine_score,
                    fuzzy_score,
                ),
            }
        )

    ranked.sort(
        key=lambda item: (item["total_score"], item["cosine_similarity"], -item["calories"]),
        reverse=True,
    )
    return ranked[:limit]


def rank_similar_recipes(graph, seed_recipe, limit=5):
    if seed_recipe not in graph:
        return []

    recipes, _, _ = graph_lists(graph)
    seed_document = recipe_document(graph, seed_recipe)
    seed_vector = vectorize_text(seed_document)
    ranked = []

    for recipe_name in recipes:
        if recipe_name == seed_recipe:
            continue

        candidate_document = recipe_document(graph, recipe_name)
        candidate_vector = vectorize_text(candidate_document)
        cosine_score = cosine_similarity(seed_vector, candidate_vector)
        fuzzy_score = fuzzy_similarity(seed_recipe, recipe_name)
        shared_ingredients = sorted(
            set(recipe_ingredients(graph, seed_recipe)) & set(recipe_ingredients(graph, recipe_name))
        )
        hybrid_score = (0.7 * cosine_score) + (0.2 * fuzzy_score) + (
            0.1 * min(1.0, len(shared_ingredients) / 3.0)
        )
        ranked.append(
            {
                "recipe": recipe_name,
                "total_score": round(hybrid_score, 4),
                "cosine_similarity": round(cosine_score, 4),
                "fuzzy_score": round(fuzzy_score, 4),
                "shared_ingredients": shared_ingredients,
                "calories": recipe_calories(graph, recipe_name),
            }
        )

    ranked.sort(
        key=lambda item: (item["total_score"], item["cosine_similarity"], -item["calories"]),
        reverse=True,
    )
    return ranked[:limit]


def format_ranked_recipe_list(ranked_candidates, limit=8):
    if not ranked_candidates:
        return "нет подходящих рецептов"

    shown = ranked_candidates[:limit]
    rendered = [
        f"{item['recipe']} ({item['calories']} ккал, score: {item['total_score']})"
        for item in shown
    ]
    if len(ranked_candidates) > limit:
        rendered.append(f"... и еще {len(ranked_candidates) - limit}")
    return ", ".join(rendered)
