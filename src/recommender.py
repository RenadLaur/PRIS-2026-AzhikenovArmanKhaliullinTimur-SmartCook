from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
import ast
import csv
import math
import re
import sqlite3
import time

try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator
except ImportError:  # pragma: no cover - optional runtime import
    GoogleTranslator = None
    MyMemoryTranslator = None

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
RECIPE_NLG_SEARCH_POOL = 320
RECIPE_NLG_INDEX_BATCH_SIZE = 5000
RECIPE_NLG_INDEX_SCHEMA_VERSION = "2"
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
    "recipe",
    "recipes",
    "dish",
    "food",
    "please",
    "with",
    "another",
    "other",
    "different",
    "choose",
    "pick",
    "find",
    "show",
    "give",
}
MEAL_QUERY_TOKENS = {"breakfast", "lunch", "dinner", "snack"}
MEAL_TAGS = {
    "завтрак": "mealbreakfast",
    "обед": "meallunch",
    "ужин": "mealdinner",
    "перекус": "mealsnack",
}
GENERIC_CATEGORY_HINTS = {
    "salad": {
        "native_tokens": {"салат", "салаты"},
        "dataset_tokens": ["salad"],
        "required_tokens": {"salad"},
        "exclude_title_tokens": {"dressing", "dip", "sauce"},
    },
    "soup": {
        "native_tokens": {"суп", "супы"},
        "dataset_tokens": ["soup", "broth", "chowder", "bisque"],
        "required_tokens": {"soup", "broth", "chowder", "bisque", "borscht"},
        "exclude_title_tokens": set(),
    },
    "pizza": {
        "native_tokens": {"пицца", "пиццы"},
        "dataset_tokens": ["pizza"],
        "required_tokens": {"pizza"},
        "exclude_title_tokens": set(),
    },
    "burger": {
        "native_tokens": {"бургер", "бургеры", "гамбургер"},
        "dataset_tokens": ["burger", "hamburger"],
        "required_tokens": {"burger", "hamburger"},
        "exclude_title_tokens": set(),
    },
    "sushi": {
        "native_tokens": {"суши"},
        "dataset_tokens": ["sushi"],
        "required_tokens": {"sushi"},
        "exclude_title_tokens": set(),
    },
    "pasta": {
        "native_tokens": {"паста", "макароны"},
        "dataset_tokens": ["pasta", "spaghetti", "macaroni"],
        "required_tokens": {"pasta", "spaghetti", "macaroni"},
        "exclude_title_tokens": set(),
    },
    "omelette": {
        "native_tokens": {"омлет"},
        "dataset_tokens": ["omelette", "omelet"],
        "required_tokens": {"omelette", "omelet"},
        "exclude_title_tokens": set(),
    },
    "dessert": {
        "native_tokens": {"десерт", "десерты", "сладкое"},
        "dataset_tokens": ["dessert", "cake", "cookie", "brownie", "pie", "pudding"],
        "required_tokens": {"dessert", "cake", "cookie", "brownie", "pie", "pudding"},
        "exclude_title_tokens": set(),
    },
}
DATASET_TOKEN_ALIASES = {
    "кур": ["chicken"],
    "куриц": ["chicken"],
    "рис": ["rice"],
    "плов": ["pilaf", "rice"],
    "борщ": ["borscht", "borsch", "borshch"],
    "свекл": ["beet", "beetroot", "borscht"],
    "яй": ["egg", "omelette"],
    "сыр": ["cheese"],
    "мол": ["milk"],
    "сал": ["salad"],
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
TRANSLATION_FILLER_PATTERNS = [
    r"\bчто\s+похоже\s+на\b",
    r"\bпохожие?\s+на\b",
    r"\bаналог\b",
    r"\bрецепт\b",
    r"\bрецепты\b",
    r"\bподбери\b",
    r"\bпосоветуй\b",
    r"\bчто\s+приготовить\b",
    r"\bприготовить\b",
]
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RECIPE_NLG_INDEX_PATH = ARTIFACTS_DIR / "recipenlg_search.sqlite3"
_TRANSLATION_RUNTIME = {
    "ru": {"provider": None, "last_error": None},
    "en": {"provider": None, "last_error": None},
}
_SEARCH_INDEX_RUNTIME = {"backend": "sqlite_fts5", "last_error": None}


def normalize(text):
    return str(text).strip().lower().replace("ё", "е")


def tokenize(text):
    return [token for token in re.split(r"[^a-zA-Zа-яА-Я0-9]+", normalize(text)) if token]


def join_items(items):
    return ", ".join(sorted({str(item) for item in items}))


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
    return recipenlg_csv_path() is not None


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
    if GoogleTranslator is not None:
        try:
            translator = GoogleTranslator(source="en", target="ru", timeout=4)
            _TRANSLATION_RUNTIME["ru"]["provider"] = "GoogleTranslator"
            _TRANSLATION_RUNTIME["ru"]["last_error"] = None
            return translator
        except Exception:  # pragma: no cover - network/runtime safeguard
            _TRANSLATION_RUNTIME["ru"]["last_error"] = "GoogleTranslator init failed"
    if MyMemoryTranslator is not None:
        try:
            translator = MyMemoryTranslator(source="en-GB", target="ru-RU", timeout=4)
            _TRANSLATION_RUNTIME["ru"]["provider"] = "MyMemoryTranslator"
            _TRANSLATION_RUNTIME["ru"]["last_error"] = None
            return translator
        except Exception:  # pragma: no cover - network/runtime safeguard
            _TRANSLATION_RUNTIME["ru"]["last_error"] = "MyMemoryTranslator init failed"
    return None


@lru_cache(maxsize=1)
def _get_en_translator():
    if GoogleTranslator is not None:
        try:
            translator = GoogleTranslator(source="ru", target="en", timeout=4)
            _TRANSLATION_RUNTIME["en"]["provider"] = "GoogleTranslator"
            _TRANSLATION_RUNTIME["en"]["last_error"] = None
            return translator
        except Exception:  # pragma: no cover - network/runtime safeguard
            _TRANSLATION_RUNTIME["en"]["last_error"] = "GoogleTranslator init failed"
    if MyMemoryTranslator is not None:
        try:
            translator = MyMemoryTranslator(source="ru-RU", target="en-GB", timeout=4)
            _TRANSLATION_RUNTIME["en"]["provider"] = "MyMemoryTranslator"
            _TRANSLATION_RUNTIME["en"]["last_error"] = None
            return translator
        except Exception:  # pragma: no cover - network/runtime safeguard
            _TRANSLATION_RUNTIME["en"]["last_error"] = "MyMemoryTranslator init failed"
    return None


def _contains_cyrillic(text):
    return bool(re.search(r"[а-яА-Я]", str(text or "")))


@lru_cache(maxsize=512)
def translate_to_ru(text):
    text = str(text or "").strip()
    if not text:
        return ""
    if _contains_cyrillic(text):
        return text

    translator = _get_ru_translator()
    if translator is None:
        return text

    if len(text) <= TRANSLATE_CHUNK_LIMIT:
        try:
            return translator.translate(text)
        except Exception:  # pragma: no cover - network/runtime safeguard
            _TRANSLATION_RUNTIME["ru"]["last_error"] = "translation request failed"
            return text

    return text


@lru_cache(maxsize=512)
def translate_to_en(text):
    text = str(text or "").strip()
    if not text:
        return ""
    if not _contains_cyrillic(text):
        return text

    translator = _get_en_translator()
    if translator is None:
        return text

    try:
        return translator.translate(text)
    except Exception:  # pragma: no cover - network/runtime safeguard
        _TRANSLATION_RUNTIME["en"]["last_error"] = "translation request failed"
        return text


def _strip_translation_fillers(text):
    cleaned = normalize(text)
    for pattern in TRANSLATION_FILLER_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _has_meaningful_tokens(tokens):
    return any(len(token) >= 3 and token not in STOP_TOKENS for token in tokens)


def _is_generic_category_request(original_tokens, translated_tokens, meal_type, include_ingredients):
    if meal_type or include_ingredients:
        return None

    meaningful_tokens = {
        token for token in translated_tokens if len(token) >= 3 and token not in STOP_TOKENS and token not in MEAL_QUERY_TOKENS
    }
    original_set = {token for token in original_tokens if len(token) >= 3 and token not in STOP_TOKENS}

    if not meaningful_tokens and not original_set:
        return None

    for category, config in GENERIC_CATEGORY_HINTS.items():
        if meaningful_tokens and meaningful_tokens.issubset(set(config["dataset_tokens"])):
            return category
        if original_set and original_set.issubset(set(config["native_tokens"])):
            return category
    return None


def _category_search_tokens(category_key):
    return list(GENERIC_CATEGORY_HINTS.get(category_key, {}).get("dataset_tokens", []))


def _meal_search_tokens(meal_type):
    hints = MEAL_HINTS.get(meal_type, [])
    tokens = []
    for hint in hints:
        translated = translate_to_en(str(hint))
        tokens.extend(tokenize(translated))
    unique = []
    for token in tokens:
        if len(token) < 3 or token in STOP_TOKENS:
            continue
        if token not in unique:
            unique.append(token)
    return unique[:8]


def _dataset_query_profile(query_text, include_ingredients=None, meal_type=None):
    include_ingredients = include_ingredients or []
    cleaned_query = _strip_translation_fillers(query_text)
    translated_query = translate_to_en(cleaned_query or query_text)
    translated_ingredients = [translate_to_en(entry) for entry in include_ingredients]
    original_tokens = tokenize(cleaned_query or query_text)
    translated_tokens = tokenize(translated_query)
    category_key = _is_generic_category_request(
        original_tokens,
        translated_tokens,
        meal_type,
        include_ingredients,
    )

    primary_tokens = list(translated_tokens)
    primary_tokens.extend(tokenize(" ".join(translated_ingredients)))
    alias_tokens = []
    for token in original_tokens + primary_tokens:
        for stem, aliases in DATASET_TOKEN_ALIASES.items():
            if token.startswith(stem):
                alias_tokens.extend(normalize(alias) for alias in aliases)
    meaningful_primary = [token for token in primary_tokens if len(token) >= 3 and token not in STOP_TOKENS]
    has_non_meal_tokens = any(token not in MEAL_QUERY_TOKENS for token in meaningful_primary)
    if has_non_meal_tokens:
        primary_tokens = [token for token in primary_tokens if token not in MEAL_QUERY_TOKENS]

    fallback_tokens = []
    if not _has_meaningful_tokens(primary_tokens):
        original_tokens = tokenize(cleaned_query or query_text)
        ingredient_tokens = tokenize(" ".join(include_ingredients))
        fallback_tokens.extend(original_tokens)
        fallback_tokens.extend(ingredient_tokens)
        for token in original_tokens + ingredient_tokens:
            for stem, aliases in DATASET_TOKEN_ALIASES.items():
                if token.startswith(stem):
                    fallback_tokens.extend(aliases)

    unique = []
    for token in primary_tokens + alias_tokens + fallback_tokens:
        if len(token) < 3 or token in STOP_TOKENS:
            continue
        if token not in unique:
            unique.append(token)

    search_tokens = list(unique[:8])
    if category_key and not search_tokens:
        search_tokens = _category_search_tokens(category_key)
    if meal_type and not search_tokens:
        search_tokens = _meal_search_tokens(meal_type)

    return {
        "query_text": str(query_text or ""),
        "cleaned_query": cleaned_query,
        "translated_query": translated_query,
        "query_tokens": list(unique[:8]),
        "search_tokens": search_tokens,
        "category_key": category_key,
        "meal_type": meal_type,
    }


def _dataset_query_tokens(query_text, include_ingredients=None, meal_type=None):
    return _dataset_query_profile(
        query_text,
        include_ingredients=include_ingredients,
        meal_type=meal_type,
    )["query_tokens"]


@lru_cache(maxsize=1)
def get_recipenlg_preview(limit=10):
    dataset_path = recipenlg_csv_path()
    if dataset_path is None:
        return []

    preview = []
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as file:
        reader = csv.DictReader(file)
        for idx, row in enumerate(reader):
            title = str(row.get("title", "")).strip()
            ingredients = _parse_list_like(row.get("ingredients", ""))
            if title:
                preview.append(
                    {
                        "title": title,
                        "title_ru": translate_to_ru(title),
                        "ingredient_count": len(ingredients),
                    }
                )
            if idx + 1 >= limit:
                break
    return preview


def localize_recipenlg_item(item, with_details=False):
    localized = dict(item or {})
    localized["title_ru"] = translate_to_ru(localized.get("title", ""))
    if not with_details:
        return localized

    ingredients = localized.get("ingredients", [])[:10]
    directions = localized.get("directions", [])[:3]
    localized["ingredients_ru_text"] = (
        translate_to_ru(", ".join(str(entry) for entry in ingredients if str(entry).strip()))
        if ingredients
        else "нет данных"
    )
    localized["directions_ru_text"] = (
        translate_to_ru(" ".join(str(entry) for entry in directions if str(entry).strip()))
        if directions
        else "нет шагов"
    )
    return localized


def get_translation_status():
    return {
        "ru_translator_ready": _get_ru_translator() is not None,
        "en_translator_ready": _get_en_translator() is not None,
        "ru_provider": _TRANSLATION_RUNTIME["ru"]["provider"],
        "en_provider": _TRANSLATION_RUNTIME["en"]["provider"],
        "ru_last_error": _TRANSLATION_RUNTIME["ru"]["last_error"],
        "en_last_error": _TRANSLATION_RUNTIME["en"]["last_error"],
    }


def _compute_dataset_tags(title, ingredients_text, ner_text):
    text = normalize(" ".join([title, ingredients_text, ner_text]))
    tags = set()
    for category, config in GENERIC_CATEGORY_HINTS.items():
        if any(token in text for token in config["required_tokens"]):
            tags.add(category)
    for meal_type, tag in MEAL_TAGS.items():
        if any(normalize(hint) in text for hint in MEAL_HINTS.get(meal_type, [])):
            tags.add(tag)
    return " ".join(sorted(tags))


def _index_metadata(path):
    try:
        with sqlite3.connect(str(path)) as conn:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
    except sqlite3.Error:
        return {}
    return {str(key): str(value) for key, value in rows}


def _expected_index_metadata(dataset_path):
    stats = dataset_path.stat()
    return {
        "schema_version": RECIPE_NLG_INDEX_SCHEMA_VERSION,
        "source_path": str(dataset_path.resolve()),
        "source_size": str(stats.st_size),
        "source_mtime_ns": str(getattr(stats, "st_mtime_ns", int(stats.st_mtime * 1_000_000_000))),
    }


def get_search_index_status():
    dataset_path = recipenlg_csv_path()
    status = {
        "ready": False,
        "backend": _SEARCH_INDEX_RUNTIME["backend"],
        "path": str(RECIPE_NLG_INDEX_PATH),
        "last_error": _SEARCH_INDEX_RUNTIME["last_error"],
        "row_count": 0,
        "built_at": None,
        "needs_rebuild": False,
    }
    if dataset_path is None:
        status["last_error"] = "RecipeNLG CSV not found"
        return status

    if not RECIPE_NLG_INDEX_PATH.exists():
        status["needs_rebuild"] = True
        return status

    metadata = _index_metadata(RECIPE_NLG_INDEX_PATH)
    expected = _expected_index_metadata(dataset_path)
    if not metadata:
        status["needs_rebuild"] = True
        status["last_error"] = "search index metadata missing"
        return status

    status["row_count"] = int(metadata.get("row_count", "0") or "0")
    status["built_at"] = metadata.get("built_at")
    status["needs_rebuild"] = any(metadata.get(key) != value for key, value in expected.items())
    status["ready"] = not status["needs_rebuild"]
    return status


def _open_search_index():
    conn = sqlite3.connect(str(RECIPE_NLG_INDEX_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_recipenlg_search_index(force_rebuild=False):
    dataset_path = recipenlg_csv_path()
    if dataset_path is None:
        _SEARCH_INDEX_RUNTIME["last_error"] = "RecipeNLG CSV not found"
        return get_search_index_status()

    current_status = get_search_index_status()
    if current_status["ready"] and not force_rebuild:
        return current_status

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = RECIPE_NLG_INDEX_PATH.with_suffix(".tmp.sqlite3")
    if temp_path.exists():
        temp_path.unlink()
    if force_rebuild and RECIPE_NLG_INDEX_PATH.exists():
        RECIPE_NLG_INDEX_PATH.unlink()

    expected = _expected_index_metadata(dataset_path)
    started_at = str(int(time.time()))
    row_count = 0
    _SEARCH_INDEX_RUNTIME["last_error"] = None

    try:
        with sqlite3.connect(str(temp_path)) as conn:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute(
                """
                CREATE VIRTUAL TABLE recipenlg_fts USING fts5(
                    title,
                    ingredients_text,
                    directions_text,
                    ner_text,
                    category_tags,
                    source UNINDEXED,
                    tokenize='unicode61'
                )
                """
            )

            batch = []
            with open(dataset_path, "r", encoding="utf-8", errors="ignore") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    title = str(row.get("title", "")).strip()
                    ingredients_text = str(row.get("ingredients", "")).strip()
                    directions_text = str(row.get("directions", "")).strip()
                    ner_text = str(row.get("NER", "")).strip()
                    source = str(row.get("source", "")).strip()
                    if not title:
                        continue

                    batch.append(
                        (
                            title,
                            ingredients_text,
                            directions_text,
                            ner_text,
                            _compute_dataset_tags(title, ingredients_text, ner_text),
                            source,
                        )
                    )
                    if len(batch) >= RECIPE_NLG_INDEX_BATCH_SIZE:
                        conn.executemany(
                            """
                            INSERT INTO recipenlg_fts(
                                title, ingredients_text, directions_text, ner_text, category_tags, source
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            batch,
                        )
                        conn.commit()
                        row_count += len(batch)
                        batch = []

                if batch:
                    conn.executemany(
                        """
                        INSERT INTO recipenlg_fts(
                            title, ingredients_text, directions_text, ner_text, category_tags, source
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    conn.commit()
                    row_count += len(batch)

            metadata_rows = list(expected.items()) + [
                ("row_count", str(row_count)),
                ("built_at", started_at),
            ]
            conn.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", metadata_rows)
            conn.execute("INSERT INTO recipenlg_fts(recipenlg_fts) VALUES ('optimize')")
            conn.commit()

        temp_path.replace(RECIPE_NLG_INDEX_PATH)
    except (OSError, sqlite3.Error, csv.Error) as exc:
        _SEARCH_INDEX_RUNTIME["last_error"] = str(exc)
        if temp_path.exists():
            temp_path.unlink()
        return get_search_index_status()

    _SEARCH_INDEX_RUNTIME["last_error"] = None
    return get_search_index_status()


def _dataset_recipe_document(item):
    parts = [
        item.get("title", ""),
        " ".join(item.get("ingredients", [])),
        " ".join(item.get("ner", [])),
        " ".join(item.get("directions", [])[:2]),
    ]
    return " ".join(part for part in parts if str(part).strip())


def _item_matches_category(item, category_key):
    config = GENERIC_CATEGORY_HINTS.get(category_key)
    if not config:
        return True
    title_tokens = set(tokenize(item.get("title", "")))
    document_tokens = set(tokenize(_dataset_recipe_document(item)))
    required_tokens = set(config.get("required_tokens", set()))
    if not required_tokens.intersection(document_tokens):
        return False
    excluded = set(config.get("exclude_title_tokens", set()))
    if excluded.intersection(title_tokens):
        return False
    return True


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


def _title_phrase_score(item, query_text, query_tokens):
    title_norm = normalize(item.get("title", ""))
    query_norm = normalize(query_text)
    if not title_norm or not query_norm:
        return 0.0

    score = 0.0
    if title_norm == query_norm:
        score += 1.0
    if query_norm in title_norm:
        score += 0.8

    title_tokens = set(tokenize(title_norm))
    token_set = {token for token in query_tokens if len(token) >= 3}
    if token_set:
        overlap = len(token_set & title_tokens) / len(token_set)
        score += 0.7 * overlap
        if token_set.issubset(title_tokens):
            score += 0.5

    if "salad" in token_set and any(token in title_tokens for token in {"dressing", "dip", "sauce"}):
        score -= 0.6

    return min(score, 1.8)


def _dataset_keyword_score(item, query_tokens):
    if not query_tokens:
        return 0.0

    title_tokens = set(tokenize(item.get("title", "")))
    ingredient_tokens = _expand_dataset_alias_tokens(item.get("ingredients", []))
    document_tokens = title_tokens | ingredient_tokens
    overlap = len(set(query_tokens) & document_tokens)
    score = overlap / max(len(set(query_tokens)), 1)
    if "salad" in set(query_tokens) and any(token in title_tokens for token in {"dressing", "dip", "sauce"}):
        score = max(0.0, score - 0.5)
    return score


def _candidate_search_score(item, query_text, query_tokens):
    combined_norm = normalize(_dataset_recipe_document(item))
    token_set = {token for token in query_tokens if len(token) >= 3}
    overlap = len(token_set & set(tokenize(combined_norm))) / max(len(token_set), 1)
    return (2.5 * _title_phrase_score(item, query_text, query_tokens)) + overlap


@lru_cache(maxsize=64)
def _search_recipenlg_candidates_cached(query_text, include_ingredients_key, meal_type, limit):
    profile = _dataset_query_profile(
        query_text,
        include_ingredients=list(include_ingredients_key),
        meal_type=meal_type or None,
    )
    query_tokens = profile["query_tokens"]
    search_tokens = profile["search_tokens"]
    if not query_tokens and not search_tokens:
        return []

    index_status = ensure_recipenlg_search_index()
    if not index_status["ready"]:
        return []

    try:
        conn = _open_search_index()
    except sqlite3.Error as exc:
        _SEARCH_INDEX_RUNTIME["last_error"] = str(exc)
        return []

    fts_queries = []
    if profile["category_key"]:
        category_query = f"category_tags:{profile['category_key']}*"
        fts_queries.append(category_query)
    if meal_type and meal_type in MEAL_TAGS:
        fts_queries.append(f"category_tags:{MEAL_TAGS[meal_type]}*")
    safe_tokens = [token for token in search_tokens if re.fullmatch(r"[a-z0-9]+", token)]
    if len(safe_tokens) >= 2:
        fts_queries.append(" ".join(f"{token}*" for token in safe_tokens[:4]))
    if safe_tokens:
        fts_queries.append(" OR ".join(f"{token}*" for token in safe_tokens[:6]))

    results = []
    seen_titles = set()

    try:
        for fts_query in dict.fromkeys(query.strip() for query in fts_queries if query.strip()):
            rows = conn.execute(
                """
                SELECT title, ingredients_text, directions_text, ner_text, category_tags, source,
                       bm25(recipenlg_fts) AS rank
                FROM recipenlg_fts
                WHERE recipenlg_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, RECIPE_NLG_SEARCH_POOL),
            ).fetchall()
            for row in rows:
                title = str(row["title"]).strip()
                key = normalize(title)
                if not key or key in seen_titles:
                    continue
                item = {
                    "title": title,
                    "ingredients": _parse_list_like(row["ingredients_text"]),
                    "directions": _parse_list_like(row["directions_text"]),
                    "ner": _parse_list_like(row["ner_text"]),
                    "source": str(row["source"]).strip(),
                    "category_tags": tokenize(row["category_tags"]),
                    "_fts_rank": float(row["rank"]),
                }
                if profile["category_key"] and not _item_matches_category(item, profile["category_key"]):
                    continue
                item["_search_score"] = _candidate_search_score(
                    item,
                    profile["translated_query"] or profile["query_text"],
                    query_tokens or search_tokens,
                )
                results.append(item)
                seen_titles.add(key)
                if len(results) >= RECIPE_NLG_SEARCH_POOL:
                    break
            if len(results) >= RECIPE_NLG_SEARCH_POOL:
                break
    except sqlite3.Error as exc:
        _SEARCH_INDEX_RUNTIME["last_error"] = str(exc)
        return []
    finally:
        conn.close()

    results.sort(
        key=lambda item: (item.get("_search_score", 0.0), -item.get("_fts_rank", 0.0), item.get("title", "")),
        reverse=True,
    )
    return results[:limit]


def search_recipenlg_candidates(query_text, include_ingredients=None, limit=RECIPE_NLG_MAX_CANDIDATES):
    include_ingredients = include_ingredients or []
    include_key = tuple(sorted(normalize(item) for item in include_ingredients))
    return _search_recipenlg_candidates_cached(str(query_text or ""), include_key, "", int(limit))


def rank_recipenlg_candidates(
    query_text,
    include_ingredients=None,
    exclude_ingredients=None,
    exclude_titles=None,
    meal_type=None,
    limit=8,
):
    include_ingredients = include_ingredients or []
    exclude_ingredients = exclude_ingredients or []
    exclude_titles = exclude_titles or []
    candidates = _search_recipenlg_candidates_cached(
        str(query_text or ""),
        tuple(sorted(normalize(item) for item in include_ingredients)),
        meal_type or "",
        RECIPE_NLG_MAX_CANDIDATES,
    )
    if not candidates:
        return []

    profile = _dataset_query_profile(query_text, include_ingredients=include_ingredients, meal_type=meal_type)
    query_tokens = profile["query_tokens"] or profile["search_tokens"]
    dataset_query_text = " ".join(query_tokens)
    query_vector = vectorize_text(dataset_query_text)
    query_normalized = normalize(dataset_query_text)
    exclude_set = _expand_dataset_alias_tokens(exclude_ingredients)
    excluded_titles_normalized = {normalize(title) for title in exclude_titles}
    ranked = []

    for item in candidates:
        if normalize(item.get("title", "")) in excluded_titles_normalized:
            continue
        ingredient_set = _expand_dataset_alias_tokens(item.get("ingredients", []))
        if exclude_set and ingredient_set.intersection(exclude_set):
            continue
        if meal_type and not _dataset_recipe_matches_meal(item, meal_type):
            continue
        if profile["category_key"] and not _item_matches_category(item, profile["category_key"]):
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
        keyword_score = _dataset_keyword_score(item, query_tokens)
        title_score = _title_phrase_score(item, dataset_query_text, query_tokens)
        category_score = 1.0 if profile["category_key"] and _item_matches_category(item, profile["category_key"]) else 0.0
        total_score = (
            (0.18 * cosine_score)
            + (0.1 * fuzzy_score)
            + (0.2 * rule_score)
            + (0.22 * keyword_score)
            + (0.2 * title_score)
            + (0.1 * category_score)
        )
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
                "keyword_score": round(keyword_score, 4),
                "title_score": round(title_score, 4),
                "category_score": round(category_score, 4),
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
    return [localize_recipenlg_item(item, with_details=False) for item in ranked[:limit]]


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
