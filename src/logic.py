import json
import os

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
