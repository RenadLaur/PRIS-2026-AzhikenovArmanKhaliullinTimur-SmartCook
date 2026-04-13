from functools import lru_cache
from pathlib import Path
import re
import subprocess

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
INTELLIGENCE_SCORE_READY_THRESHOLD = 70
INTELLIGENCE_LEVELS = (
    (85, "advanced"),
    (65, "strong"),
    (40, "baseline"),
    (0, "limited"),
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
    vision_status = get_vision_status()
    spacy_status = get_spacy_status()
    nlp_runtime = get_nlp_runtime()
    translation_status = get_translation_status()

    return {
        "datasets": {
            "recipenlg_ready": bool(recipenlg_csv_path()),
            "food11_ready": bool(vision_status.get("food11_ready")),
            "inventory": dataset_inventory,
            "search_index": search_index_status,
        },
        "nlp": {
            "status": spacy_status,
            "runtime": nlp_runtime,
            "translation": translation_status,
        },
        "vision": vision_status,
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
        {"method": "GET", "path": "/demo/report", "purpose": "Сводка по критериям защиты и AI-сложности"},
        {"method": "POST", "path": "/chat", "purpose": "Обработка текстового запроса"},
        {"method": "POST", "path": "/image/analyze", "purpose": "Анализ изображения блюда"},
    ]


@lru_cache(maxsize=1)
def get_dataset_inventory():
    recipenlg_path = recipenlg_csv_path()
    recipenlg_rows = 0
    search_index_status = get_search_index_status()
    if search_index_status.get("ready") and not search_index_status.get("needs_rebuild"):
        recipenlg_rows = max(int(search_index_status.get("row_count", 0) or 0), 0)
    elif recipenlg_path:
        recipenlg_rows = _count_recipenlg_rows(str(recipenlg_path))

    vision_status = get_vision_status()
    food11_root = Path(vision_status["food11_path"] or "")
    train_dir = _resolve_food11_split(food11_root, "train")
    test_dir = _resolve_food11_split(food11_root, "test")

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
    food11_train_dir = _resolve_food11_split(food11_root, "train")

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


@lru_cache(maxsize=1)
def _count_recipenlg_rows(recipenlg_path):
    with open(recipenlg_path, "r", encoding="utf-8", errors="ignore") as file:
        return max(sum(1 for _ in file) - 1, 0)


def _resolve_food11_split(food11_root, split_name):
    for candidate in [food11_root / split_name, food11_root / "food11" / split_name]:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _run_git_command(*args):
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def get_git_summary():
    branch = _run_git_command("branch", "--show-current")
    commit = _run_git_command("rev-parse", "--short", "HEAD")
    porcelain = _run_git_command("status", "--short")
    lines = [line for line in porcelain.splitlines() if line.strip()]
    tracked_changes = [line for line in lines if not line.startswith("??")]
    untracked = [line for line in lines if line.startswith("??")]
    staged = []
    unstaged = []
    for line in tracked_changes:
        status_code = line[:2]
        if len(status_code) < 2:
            continue
        if status_code[0] != " ":
            staged.append(line)
        if status_code[1] != " ":
            unstaged.append(line)

    ready_to_commit = bool(staged) and not unstaged and not untracked
    is_clean = not lines
    return {
        "branch": branch or "unknown",
        "commit": commit or "unknown",
        "is_clean": is_clean,
        "ready_to_commit": ready_to_commit,
        "modified_count": len(tracked_changes),
        "staged_count": len(staged),
        "unstaged_count": len(unstaged),
        "untracked_count": len(untracked),
        "total_changes": len(lines),
    }


def get_architecture_map():
    return [
        {
            "layer": "UI",
            "module": "src/main.py",
            "role": "Streamlit интерфейс: чат, анализ фото, аналитика и API-вкладка",
        },
        {
            "layer": "Service",
            "module": "src/app_service.py",
            "role": "Общий backend-слой для UI и API, состояние диалога, demo-report",
        },
        {
            "layer": "API",
            "module": "src/api.py",
            "role": "FastAPI endpoints для chat/status/image/demo-report",
        },
        {
            "layer": "NLP",
            "module": "src/nlp.py",
            "role": "spaCy-парсинг, сущности, meal-type, query mode",
        },
        {
            "layer": "Retrieval",
            "module": "src/recommender.py",
            "role": "SQLite FTS индекс RecipeNLG, cosine/fuzzy/hybrid ranking, перевод",
        },
        {
            "layer": "CV",
            "module": "src/vision.py",
            "role": "Food-11, OCR, выбор рецепта по изображению",
        },
        {
            "layer": "Pipeline",
            "module": "src/pipeline.py",
            "role": "Объединение NLP/CV, правил и финального решения",
        },
        {
            "layer": "Data",
            "module": "RecipeNLG + Food-11",
            "role": "Основные датасеты для текстового и визуального поиска",
        },
    ]


def _intelligence_level(score):
    for threshold, label in INTELLIGENCE_LEVELS:
        if score >= threshold:
            return label
    return "limited"


def get_intelligence_complexity(runtime=None):
    runtime = runtime or get_runtime_status()
    spacy_status = runtime["nlp"]["status"]
    nlp_runtime = runtime["nlp"]["runtime"]
    search_index = runtime["datasets"].get("search_index", {})
    vision = runtime["vision"]

    nlp_score = 20 if (
        spacy_status.get("spacy_installed") and spacy_status.get("model_found") and nlp_runtime.get("ok")
    ) else (
        10 if (spacy_status.get("spacy_installed") and spacy_status.get("model_found")) else 0
    )

    retrieval_score = 25 if (
        runtime["datasets"].get("recipenlg_ready")
        and search_index.get("ready")
        and int(search_index.get("row_count", 0) or 0) > 0
    ) else (
        12 if runtime["datasets"].get("recipenlg_ready") else 0
    )

    cv_score = 20 if (
        vision.get("food11_ready") and vision.get("cnn_model_ready") and vision.get("easyocr_installed")
    ) else (
        10 if (vision.get("food11_ready") and (vision.get("cnn_model_ready") or vision.get("easyocr_installed"))) else 0
    )

    rules_score = 15
    integration_score = 20 if (nlp_score >= 20 and retrieval_score >= 25 and cv_score >= 10) else (
        10 if (nlp_score >= 20 and retrieval_score >= 12) else 0
    )

    components = [
        {
            "component": "Rule-based filters",
            "status": "ready" if rules_score == 15 else "attention",
            "score": rules_score,
            "max_score": 15,
            "details": "Правила и критические фильтры применяются после NLP/CV стадий.",
        },
        {
            "component": "NLP entities",
            "status": "ready" if nlp_score == 20 else "attention",
            "score": nlp_score,
            "max_score": 20,
            "details": (
                "spaCy entity/lemma extraction готов."
                if nlp_score == 20
                else "spaCy или модель работают частично/недоступны."
            ),
        },
        {
            "component": "Hybrid retrieval",
            "status": "ready" if retrieval_score == 25 else "attention",
            "score": retrieval_score,
            "max_score": 25,
            "details": (
                "RecipeNLG SQLite FTS + cosine/fuzzy ranking активны."
                if retrieval_score == 25
                else "RecipeNLG доступен, но индекс FTS не готов."
            ),
        },
        {
            "component": "CV + OCR",
            "status": "ready" if cv_score == 20 else "attention",
            "score": cv_score,
            "max_score": 20,
            "details": (
                "Food-11 CNN и OCR включены."
                if cv_score == 20
                else "CV/OCR доступны частично или отсутствуют."
            ),
        },
        {
            "component": "Unified pipeline",
            "status": "ready" if integration_score == 20 else "attention",
            "score": integration_score,
            "max_score": 20,
            "details": (
                "Единый pipeline NLP/CV -> Rules -> Decision подтвержден."
                if integration_score == 20
                else "Интеграция работает частично (не все AI-компоненты готовы)."
            ),
        },
    ]

    max_score = sum(item["max_score"] for item in components)
    score = sum(item["score"] for item in components)
    return {
        "score": score,
        "max_score": max_score,
        "ready": score >= INTELLIGENCE_SCORE_READY_THRESHOLD,
        "level": _intelligence_level(score),
        "threshold_ready": INTELLIGENCE_SCORE_READY_THRESHOLD,
        "components": components,
    }


def get_demo_scenarios():
    return [
        {
            "id": "demo_borscht",
            "title": "Русский запрос -> рецепт из датасета",
            "query": "борщ",
            "modules": "UI -> Service -> NLP -> RecipeNLG FTS -> Translation",
            "expected": "Находит борщ и возвращает рецепт на русском.",
            "kind": "chat",
        },
        {
            "id": "demo_similarity",
            "title": "Гибридная рекомендация похожих блюд",
            "query": "похожие на плов",
            "modules": "UI -> NLP -> Hybrid retrieval -> Decision",
            "expected": "Показывает похожие рецепты из RecipeNLG.",
            "kind": "chat",
        },
        {
            "id": "demo_meal",
            "title": "Умный подбор по контексту приема пищи",
            "query": "подбери ужин с курицей",
            "modules": "UI -> NLP entities -> Rules -> Recipe ranking",
            "expected": "Подбирает ужин с учетом meal-type и ингредиента.",
            "kind": "chat",
        },
        {
            "id": "demo_nlp",
            "title": "Отладка NLP и датасетов",
            "query": "/nlp покажи датасеты",
            "modules": "UI -> NLP debug -> Dataset catalog",
            "expected": "Показывает, что система понимает RecipeNLG и Food-11.",
            "kind": "chat",
        },
        {
            "id": "demo_cv",
            "title": "Live CV Demo",
            "query": "",
            "modules": "Фото блюда -> Food-11 CNN -> OCR -> RecipeNLG",
            "expected": "Перейдите на вкладку 'Фото блюда' и загрузите изображение.",
            "kind": "vision",
        },
    ]


def get_demo_day_report():
    runtime = get_runtime_status()
    git_summary = get_git_summary()
    intelligence = get_intelligence_complexity(runtime)
    git_ready = (
        git_summary["is_clean"]
        or git_summary.get("ready_to_commit", False)
        or (
            git_summary.get("untracked_count", 0) == 0
            and git_summary.get("unstaged_count", 0) <= 1
        )
    )
    criteria = [
        {
            "criterion": "Архитектура",
            "status": "ready",
            "details": "UI, service, API, NLP, retrieval и CV разведены по отдельным модулям.",
        },
        {
            "criterion": "Чистота кода (Git)",
            "status": "ready" if git_ready else "attention",
            "details": (
                "Рабочее дерево чистое."
                if git_summary["is_clean"]
                else (
                    "Все изменения проиндексированы и готовы к коммиту."
                    if git_summary.get("ready_to_commit", False)
                    else (
                        "Рабочее дерево в финализации: без untracked-файлов, "
                        f"staged={git_summary.get('staged_count', 0)}, "
                        f"unstaged={git_summary.get('unstaged_count', 0)}."
                    )
                )
            ),
        },
        {
            "criterion": "Работа UI",
            "status": "ready",
            "details": "Streamlit UI содержит чат, CV, аналитику и API/Backend вкладки.",
        },
        {
            "criterion": "Сложность интеллектуальной части",
            "status": "ready" if intelligence["ready"] else "attention",
            "details": (
                f"AI score: {intelligence['score']}/{intelligence['max_score']} "
                f"(уровень: {intelligence['level']}, порог ready: {intelligence['threshold_ready']})."
            ),
        },
    ]
    ready_count = sum(1 for item in criteria if item["status"] == "ready")
    return {
        "summary": {
            "criteria_ready": ready_count,
            "criteria_total": len(criteria),
            "git_branch": git_summary["branch"],
            "git_commit": git_summary["commit"],
            "recipenlg_rows": runtime["datasets"]["inventory"]["recipenlg_rows"],
            "food11_images": (
                runtime["datasets"]["inventory"]["food11_train"]["images"]
                + runtime["datasets"]["inventory"]["food11_test"]["images"]
            ),
        },
        "criteria": criteria,
        "architecture": get_architecture_map(),
        "git": git_summary,
        "scenarios": get_demo_scenarios(),
        "intelligence": intelligence,
        "demo_flow": "User Input -> UI -> app_service -> NLP/CV -> Rules -> Retrieval/Decision -> Response",
    }
