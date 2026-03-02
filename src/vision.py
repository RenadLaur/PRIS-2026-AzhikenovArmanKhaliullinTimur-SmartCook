from functools import lru_cache
from pathlib import Path
import ast
import importlib.util
import os
import re

import cv2
import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional runtime import
    pd = None

try:
    from .nlp import get_known_datasets
except ImportError:
    from nlp import get_known_datasets


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FOOD11_MODEL_PATH = "artifacts/food11_resnet18.pt"
RECIPE_NLG_MAX_SCAN_CHUNKS = 12
RECIPE_NLG_CHUNK_SIZE = 50000
TRANSLATE_CHUNK_LIMIT = 4500

# Used for RecipeNLG search (dataset-driven recipe suggestion)
FOOD11_TO_RECIPE_KEYWORDS = {
    "apple pie": ["apple pie", "pie"],
    "cheesecake": ["cheesecake", "cheese cake"],
    "chicken curry": ["chicken curry", "curry chicken"],
    "french fries": ["french fries", "fries", "fried potato"],
    "fried rice": ["fried rice", "rice"],
    "hamburger": ["hamburger", "burger"],
    "hot dog": ["hot dog", "sausage"],
    "ice cream": ["ice cream"],
    "omelette": ["omelette", "omelet"],
    "pizza": ["pizza"],
    "sushi": ["sushi", "roll"],
    "bread": ["bread"],
    "dairy product": ["milk", "cheese", "yogurt"],
    "dessert": ["dessert", "cake", "cookie"],
    "egg": ["egg"],
    "fried food": ["fried"],
    "meat": ["meat", "beef", "chicken", "pork"],
    "noodles-pasta": ["noodle", "pasta"],
    "rice": ["rice"],
    "seafood": ["seafood", "fish", "shrimp", "salmon"],
    "soup": ["soup"],
    "vegetable-fruit": ["salad", "vegetable", "fruit"],
}

# Fallback hints only (when RecipeNLG has no good matches)
FOOD11_INGREDIENT_HINTS = {
    "apple pie": ["мука", "сахар"],
    "cheesecake": ["творог", "сахар", "молоко"],
    "chicken curry": ["курица", "рис", "лук"],
    "french fries": ["картофель"],
    "fried rice": ["рис", "морковь", "лук"],
    "hamburger": ["говядина", "хлеб"],
    "hot dog": ["хлеб", "курица"],
    "ice cream": ["молоко", "сахар"],
    "omelette": ["яйцо", "молоко", "сыр"],
    "pizza": ["мука", "сыр", "помидор"],
    "sushi": ["рис", "лосось"],
    "bread": ["хлеб", "мука"],
    "dairy product": ["сыр", "молоко", "йогурт"],
    "dessert": ["сахар", "творог", "мед"],
    "egg": ["яйцо"],
    "fried food": ["картофель", "курица"],
    "meat": ["говядина", "баранина", "курица"],
    "noodles-pasta": ["лапша", "паста"],
    "rice": ["рис"],
    "seafood": ["лосось", "креветки", "тунец"],
    "soup": ["чечевица", "лук", "морковь"],
    "vegetable-fruit": ["помидор", "огурец", "морковь", "зелень"],
}


def _normalize(text):
    return str(text).strip().lower().replace("ё", "е")


def _dataset_by_name(name):
    target = _normalize(name)
    for dataset in get_known_datasets():
        if _normalize(dataset.get("name", "")) == target:
            return dataset
    return None


def _food11_root():
    dataset = _dataset_by_name("Food-11 Image Classification Dataset")
    if not dataset:
        return ""
    return str(dataset.get("local_path", "")).strip()


def _food11_train_dir():
    root = Path(_food11_root())
    candidates = [root / "train", root / "food11" / "train"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _recipenlg_csv_path():
    dataset = _dataset_by_name("RecipeNLG Dataset")
    if not dataset:
        return None
    root = Path(str(dataset.get("local_path", "")).strip())
    candidates = [
        root / "dataset" / "full_dataset.csv",
        root / "full_dataset.csv",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _decode_image(image_bytes):
    if not image_bytes:
        return None
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _extract_histogram(bgr_image):
    resized = cv2.resize(bgr_image, (256, 256))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten().astype(np.float32)
    return histogram


def _cosine_similarity(vec1, vec2):
    denominator = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-12
    return float(np.dot(vec1, vec2) / denominator)


def _is_image_file(filename):
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def _display_label(raw_label):
    return str(raw_label).strip().replace("_", " ")


def _collect_food11_class_images(train_dir, per_class_limit=40):
    class_images = {}
    if train_dir is None:
        return class_images

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = _display_label(class_dir.name)
        image_paths = [
            str(path)
            for path in sorted(class_dir.iterdir())
            if path.is_file() and _is_image_file(path.name)
        ]
        if image_paths:
            class_images[label] = image_paths[:per_class_limit]

    return class_images


@lru_cache(maxsize=1)
def _load_food11_cnn():
    model_path = Path(FOOD11_MODEL_PATH)
    if not model_path.exists():
        return None

    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
        return None

    import torch  # pylint: disable=import-outside-toplevel
    from torchvision import models  # pylint: disable=import-outside-toplevel

    checkpoint = torch.load(str(model_path), map_location="cpu")
    class_names = checkpoint.get("class_names")
    if not class_names:
        return None

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return {
        "model": model,
        "class_names": class_names,
        "input_size": int(checkpoint.get("input_size", 224)),
    }


def _predict_with_cnn(bgr_image):
    cnn = _load_food11_cnn()
    if not cnn:
        return None

    import torch  # pylint: disable=import-outside-toplevel
    from torchvision import transforms  # pylint: disable=import-outside-toplevel

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_image = transforms.ToPILImage()(rgb)
    input_size = cnn["input_size"]
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        logits = cnn["model"](tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_k = min(3, probs.shape[0])
    values, indices = torch.topk(probs, k=top_k)
    class_names = cnn["class_names"]
    top = [
        (_display_label(class_names[int(idx)]), round(float(val), 4))
        for val, idx in zip(values, indices)
    ]
    best_label = _display_label(class_names[int(indices[0])])
    confidence = float(values[0])

    return {
        "status": "cnn_ok",
        "label": _display_label(best_label),
        "confidence": round(confidence, 3),
        "top": top,
    }


@lru_cache(maxsize=1)
def _build_food11_prototypes():
    train_dir = _food11_train_dir()
    class_images = _collect_food11_class_images(train_dir)
    prototypes = {}

    for label, image_paths in class_images.items():
        vectors = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is not None:
                vectors.append(_extract_histogram(image))
        if vectors:
            prototypes[label] = np.mean(np.stack(vectors, axis=0), axis=0)

    return prototypes


def _predict_with_histogram(bgr_image):
    prototypes = _build_food11_prototypes()
    if not prototypes:
        return {
            "status": "dataset_unavailable",
            "label": None,
            "confidence": 0.0,
            "top": [],
        }

    vector = _extract_histogram(bgr_image)
    ranked = sorted(
        ((label, _cosine_similarity(vector, proto)) for label, proto in prototypes.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    top = [(label, round(float(score), 4)) for label, score in ranked[:3]]
    best_label, best_score = ranked[0]

    return {
        "status": "heuristic_ok",
        "label": best_label,
        "confidence": round(max(0.0, min(1.0, float(best_score))), 3),
        "top": top,
    }


def _predict_food11_label(bgr_image):
    cnn_prediction = _predict_with_cnn(bgr_image)
    if cnn_prediction is not None:
        return cnn_prediction
    return _predict_with_histogram(bgr_image)


@lru_cache(maxsize=1)
def _get_easyocr_reader():
    import easyocr  # pylint: disable=import-outside-toplevel

    return easyocr.Reader(["ru", "en"], gpu=False)


def _extract_ocr_text(bgr_image):
    if importlib.util.find_spec("easyocr") is None:
        return {"enabled": False, "text": "", "warning": "easyocr не установлен"}

    try:
        reader = _get_easyocr_reader()
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        chunks = reader.readtext(rgb, detail=0, paragraph=True)
        text = " ".join(str(chunk).strip() for chunk in chunks if str(chunk).strip())
        return {"enabled": True, "text": text, "warning": None}
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return {"enabled": True, "text": "", "warning": f"Ошибка OCR: {exc}"}


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
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [text]


def _looks_cyrillic(text):
    if not text:
        return False
    cyr = sum(1 for ch in text if "а" <= ch.lower() <= "я")
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    return cyr > latin


@lru_cache(maxsize=1)
def _get_ru_translator():
    if importlib.util.find_spec("deep_translator") is None:
        return None
    try:
        from deep_translator import GoogleTranslator  # pylint: disable=import-outside-toplevel

        return GoogleTranslator(source="en", target="ru")
    except Exception:  # pragma: no cover - runtime safeguard
        return None


def _translate_to_ru(text):
    text = str(text or "").strip()
    if not text or _looks_cyrillic(text):
        return text

    translator = _get_ru_translator()
    if translator is None:
        return text

    if len(text) <= TRANSLATE_CHUNK_LIMIT:
        try:
            return translator.translate(text)
        except Exception:  # pragma: no cover - network/runtime safeguard
            return text

    chunks = []
    current = []
    current_len = 0
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if current_len + len(part) + 1 > TRANSLATE_CHUNK_LIMIT and current:
            chunks.append(" ".join(current))
            current = [part]
            current_len = len(part)
        else:
            current.append(part)
            current_len += len(part) + 1
    if current:
        chunks.append(" ".join(current))

    translated_parts = []
    for chunk in chunks:
        try:
            translated_parts.append(translator.translate(chunk))
        except Exception:  # pragma: no cover - network/runtime safeguard
            translated_parts.append(chunk)
    return " ".join(translated_parts).strip()


def _recipe_keywords_for_label(label):
    normalized = _normalize(label)
    return FOOD11_TO_RECIPE_KEYWORDS.get(normalized, [normalized])


@lru_cache(maxsize=64)
def _search_recipenlg_by_keyword(keyword):
    if pd is None:
        return []
    dataset_path = _recipenlg_csv_path()
    if dataset_path is None:
        return []

    keyword = str(keyword).strip()
    if not keyword:
        return []

    results = []
    pattern = re.escape(keyword)

    try:
        reader = pd.read_csv(
            dataset_path,
            usecols=["title", "ingredients", "directions", "NER"],
            chunksize=RECIPE_NLG_CHUNK_SIZE,
            on_bad_lines="skip",
        )
    except Exception:
        return []

    for idx, chunk in enumerate(reader):
        if idx >= RECIPE_NLG_MAX_SCAN_CHUNKS:
            break

        title_series = chunk["title"].astype(str)
        ingredients_series = chunk["ingredients"].astype(str)
        ner_series = chunk["NER"].astype(str)

        mask = (
            title_series.str.contains(pattern, case=False, na=False)
            | ingredients_series.str.contains(pattern, case=False, na=False)
            | ner_series.str.contains(pattern, case=False, na=False)
        )
        subset = chunk[mask]
        if subset.empty:
            continue

        for _, row in subset.head(25).iterrows():
            results.append(
                {
                    "title": str(row.get("title", "")).strip(),
                    "ingredients": _parse_list_like(row.get("ingredients", "")),
                    "directions": _parse_list_like(row.get("directions", "")),
                    "ner": _parse_list_like(row.get("NER", "")),
                }
            )
        if len(results) >= 40:
            break

    unique = []
    seen = set()
    for item in results:
        key = _normalize(item.get("title", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:40]


def _pick_recipenlg_recipe(label, ocr_text=""):
    keywords = _recipe_keywords_for_label(label)
    candidates = []
    for keyword in keywords:
        candidates.extend(_search_recipenlg_by_keyword(keyword))

    if not candidates:
        return None

    ocr_tokens = {_normalize(token) for token in re.split(r"[^a-zA-Zа-яА-Я0-9]+", ocr_text) if token}

    def score(item):
        text_pool = " ".join(
            [item.get("title", "")] + item.get("ingredients", []) + item.get("ner", [])
        ).lower()
        token_overlap = sum(1 for token in ocr_tokens if token and token in text_pool)
        short_ingredients_bonus = -min(len(item.get("ingredients", [])), 15)
        return (token_overlap, short_ingredients_bonus)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def _format_recipenlg_recipe(recipe):
    if not recipe:
        return None, None
    title = recipe.get("title") or "RecipeNLG recipe"
    ingredients = recipe.get("ingredients", [])[:10]
    directions = recipe.get("directions", [])[:3]
    title_ru = _translate_to_ru(title)

    ingredients_ru = [_translate_to_ru(item) for item in ingredients]
    directions_ru = [_translate_to_ru(item) for item in directions]

    ingredients_text = ", ".join(ingredients_ru) if ingredients_ru else "нет данных"
    directions_text = " ".join(directions_ru) if directions_ru else "нет шагов"

    text = (
        f"**Рецепт: {title_ru}**\n\n"
        f"**Ингредиенты:** {ingredients_text}\n\n"
        f"**Шаги:** {directions_text}"
    )
    return title_ru, text


def _graph_recipe_nodes(graph):
    return sorted(node for node in graph.nodes if graph.nodes[node].get("type") == "recipe")


def _graph_ingredient_nodes(graph):
    return sorted(node for node in graph.nodes if graph.nodes[node].get("type") == "ingredient")


def _recipe_ingredients(graph, recipe_name):
    return sorted(
        node for node in graph.neighbors(recipe_name) if graph.nodes[node].get("type") == "ingredient"
    )


def _recipe_allergens(graph, recipe_name):
    allergens = set()
    for ingredient in _recipe_ingredients(graph, recipe_name):
        for neighbor in graph.neighbors(ingredient):
            if graph.nodes[neighbor].get("type") == "allergen":
                allergens.add(neighbor)
    return sorted(allergens)


def _recipe_calories(graph, recipe_name):
    data = graph.nodes[recipe_name].get("data")
    calories = getattr(data, "calories", 0) if data is not None else 0
    return int(calories or 0)


def _recipe_answer(graph, recipe_name):
    ingredients = _recipe_ingredients(graph, recipe_name)
    allergens = _recipe_allergens(graph, recipe_name)
    calories = _recipe_calories(graph, recipe_name)
    return (
        f"Рецепт: {recipe_name} ({calories} ккал). "
        f"Ингредиенты: {', '.join(ingredients) if ingredients else 'нет данных'}. "
        f"Аллергены: {', '.join(allergens) if allergens else 'не обнаружены'}."
    )


def _hints_from_label(label):
    return FOOD11_INGREDIENT_HINTS.get(_normalize(label), [])


def _hints_from_ocr_text(graph, ocr_text):
    if not ocr_text:
        return [], None

    text = _normalize(ocr_text)
    recipes = _graph_recipe_nodes(graph)
    for recipe in recipes:
        if _normalize(recipe) in text:
            return [], recipe

    ingredients = []
    for ingredient in _graph_ingredient_nodes(graph):
        if _normalize(ingredient) in text:
            ingredients.append(ingredient)
    return sorted(set(ingredients)), None


def _best_recipe_by_hints(graph, hints):
    recipes = _graph_recipe_nodes(graph)
    if not recipes:
        return None
    if not hints:
        return min(recipes, key=lambda recipe: _recipe_calories(graph, recipe))

    hint_set = {_normalize(item) for item in hints}
    scored = []
    for recipe in recipes:
        ingredients = {_normalize(item) for item in _recipe_ingredients(graph, recipe)}
        overlap = len(hint_set.intersection(ingredients))
        if overlap == 0:
            continue
        calories = _recipe_calories(graph, recipe)
        scored.append((overlap, -calories, recipe))

    if scored:
        scored.sort(reverse=True)
        return scored[0][2]
    return min(recipes, key=lambda recipe: _recipe_calories(graph, recipe))


def analyze_food_photo(image_bytes, graph):
    image = _decode_image(image_bytes)
    if image is None:
        return {"error": "Не удалось прочитать изображение. Загрузите корректный файл JPG/PNG."}

    prediction = _predict_food11_label(image)
    ocr_result = _extract_ocr_text(image)

    predicted_label = prediction.get("label")
    confidence = float(prediction.get("confidence", 0.0))
    top_candidates = prediction.get("top", [])

    # Primary recipe source: RecipeNLG dataset
    recipenlg_recipe = _pick_recipenlg_recipe(predicted_label or "", ocr_result.get("text", ""))
    recipe_name, recipe_text = _format_recipenlg_recipe(recipenlg_recipe)

    # Fallback to graph only when RecipeNLG did not return matches
    ingredient_hints = _hints_from_label(predicted_label or "")
    if recipe_text is None:
        ocr_hints, recipe_from_ocr = _hints_from_ocr_text(graph, ocr_result.get("text", ""))
        all_hints = sorted(set(ingredient_hints + ocr_hints))
        selected_recipe = recipe_from_ocr or _best_recipe_by_hints(graph, all_hints)
        recipe_name = selected_recipe
        recipe_text = _recipe_answer(graph, selected_recipe) if selected_recipe else "Рецепт не найден."
        ingredient_hints = all_hints

    classification_note = None
    if prediction.get("status") != "cnn_ok":
        classification_note = (
            "CNN-модель Food-11 не найдена, используется fallback-классификатор по OpenCV признакам. "
            "Для нормального качества запустите обучение CNN."
        )
    elif top_candidates and confidence < 0.45:
        classification_note = (
            "Низкая уверенность CNN-классификации. Проверьте top кандидаты или загрузите более четкое фото."
        )

    if not predicted_label:
        predicted_label = "не определено"

    return {
        "error": None,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 3),
        "dataset_status": prediction.get("status"),
        "dataset_root": _food11_root(),
        "top_candidates": top_candidates,
        "ocr_enabled": ocr_result.get("enabled", False),
        "ocr_text": ocr_result.get("text", ""),
        "ocr_warning": ocr_result.get("warning"),
        "ingredient_hints": ingredient_hints,
        "recipe_name": recipe_name,
        "recipe_text": recipe_text,
        "classification_note": classification_note,
    }


def get_vision_status():
    food11_root = _food11_root()
    train_dir = _food11_train_dir()
    recipenlg_csv = _recipenlg_csv_path()

    return {
        "food11_path": food11_root,
        "food11_ready": bool(train_dir),
        "easyocr_installed": importlib.util.find_spec("easyocr") is not None,
        "cnn_model_path": FOOD11_MODEL_PATH,
        "cnn_model_ready": Path(FOOD11_MODEL_PATH).exists(),
        "recipenlg_ready": recipenlg_csv is not None,
        "recipenlg_path": str(recipenlg_csv) if recipenlg_csv else "",
    }