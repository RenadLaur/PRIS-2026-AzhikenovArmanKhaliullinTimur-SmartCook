import json
import os
import re

try:
    from .nlp import analyze_text_message
    from .pipeline import run_text_pipeline
except ImportError:
    from nlp import analyze_text_message
    from pipeline import run_text_pipeline

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


def _normalize(text):
    return str(text).strip().lower().replace("ё", "е")


def _split_tokens(text):
    return [token for token in re.split(r"[^a-zA-Zа-яА-Я0-9]+", _normalize(text)) if token]


def _is_nlp_request(query):
    query = str(query or "").strip()
    return query.startswith("/nlp") or query.startswith("nlp:")


def _is_debug_request(query):
    query = str(query or "").strip()
    return query.startswith("/debug") or query.startswith("debug:") or query.startswith("/explain")


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


def _extract_debug_payload(raw_text):
    text = str(raw_text or "").strip()
    lowered = text.lower()

    prefixes = [
        "/debug",
        "debug:",
        "/explain",
        "объясни выбор:",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip(" :\n\t")
    return text


def _pipeline_recipe_title(pipeline_result):
    decision = (pipeline_result or {}).get("stages", {}).get("decision", {})
    ranked = decision.get("ranked") or []
    if ranked and isinstance(ranked[0], dict):
        return str(ranked[0].get("title", "")).strip()
    return ""


def process_text_interaction(text, data_source, context=None):
    if text is None:
        return {"response": "Я не знаю такого термина", "recipe_title": ""}

    query = _normalize(text)
    if not query:
        return {"response": "Я не знаю такого термина", "recipe_title": ""}

    if _is_nlp_request(query):
        payload = _extract_nlp_payload(text)
        if not payload.strip():
            return {
                "response": (
                    "После `/nlp` передайте текст запроса.\n"
                    "Пример: `/nlp Подбери ужин без глютена до 500 ккал с курицей на 2 порции`"
                ),
                "recipe_title": "",
            }
        try:
            return {"response": analyze_text_message(payload, data_source), "recipe_title": ""}
        except RuntimeError as exc:
            return {"response": f"Ошибка NLP: {exc}", "recipe_title": ""}

    if _is_debug_request(query):
        payload = _extract_debug_payload(text)
        if not payload.strip():
            return {
                "response": (
                    "После `/debug` передайте запрос.\n"
                    "Пример: `/debug рецепт борща`"
                ),
                "recipe_title": "",
            }
        pipeline_result = run_text_pipeline(
            payload,
            data_source,
            debug=True,
            exclude_titles=(context or {}).get("exclude_titles", []),
        )
        if pipeline_result.get("handled"):
            return {
                "response": pipeline_result.get("response") or "Не удалось объяснить выбор.",
                "recipe_title": _pipeline_recipe_title(pipeline_result),
            }
        return {"response": "Не удалось собрать debug-пояснение для этого запроса.", "recipe_title": ""}

    if any(word in query for word in ["привет", "здравствуй", "добрый день", "hello"]):
        return {"response": "Привет! Напиши запрос о блюде, рецепте или загрузи фото блюда.", "recipe_title": ""}

    if query in {"помощь", "help", "что ты умеешь", "команды"}:
        return {
            "response": (
                "Я умею:\n"
                "- искать рецепты в RecipeNLG на русском;\n"
                "- подбирать похожие блюда по описанию и ингредиентам;\n"
                "- обрабатывать запросы через NLP (`/nlp ...`);\n"
                "- показывать техническое объяснение через `/debug ...`;\n"
                "- анализировать фото блюда через Food-11 + RecipeNLG."
            ),
            "recipe_title": "",
        }

    pipeline_result = run_text_pipeline(
        text,
        data_source,
        debug=False,
        exclude_titles=(context or {}).get("exclude_titles", []),
    )
    if pipeline_result.get("handled"):
        return {
            "response": pipeline_result.get("response") or "Не удалось обработать запрос.",
            "recipe_title": _pipeline_recipe_title(pipeline_result),
        }

    return {
        "response": (
            "Не удалось подобрать рецепт из подключенных датасетов.\n"
            "Попробуйте: 'похожие на плов', 'рецепт с курицей и рисом', "
            "'подбери ужин с курицей' или загрузите фото блюда."
        ),
        "recipe_title": "",
    }


def process_text_message(text, data_source, context=None):
    return process_text_interaction(text, data_source, context=context)["response"]
