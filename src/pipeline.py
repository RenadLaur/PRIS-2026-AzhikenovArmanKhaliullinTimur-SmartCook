try:
    from .nlp import analyze_cooking_request, get_known_datasets
    from .recommender import (
        get_recipenlg_preview,
        join_items,
        localize_recipenlg_item,
        rank_recipenlg_candidates,
        recipenlg_ready,
    )
    from .vision import analyze_food_photo
except ImportError:
    from nlp import analyze_cooking_request, get_known_datasets
    from recommender import (
        get_recipenlg_preview,
        join_items,
        localize_recipenlg_item,
        rank_recipenlg_candidates,
        recipenlg_ready,
    )
    from vision import analyze_food_photo


def _normalize(text):
    return str(text).strip().lower().replace("ё", "е")


def _dataset_by_name(dataset_name):
    normalized = _normalize(dataset_name)
    for dataset in get_known_datasets():
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


def _build_no_results_message(meal_type, include_ingredients, exclude_ingredients, exclude_allergens, min_cal, max_cal):
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
    return (
        f"Не нашел рецепт в RecipeNLG ({suffix}). "
        "Попробуйте уточнить блюдо, ингредиенты или убрать часть ограничений."
    )


def _is_similarity_request(query_text, parsed):
    mode = parsed.get("query_mode")
    query = _normalize(query_text)
    if mode == "similarity_search":
        return True
    return any(marker in query for marker in ["похож", "похожие", "аналог", "что похоже", "similar"])


def _format_dataset_similarity_response(query_text, ranked, debug=False):
    if not ranked:
        return f"По описанию '{query_text}' не нашел похожих рецептов в RecipeNLG."

    lines = [f"Похожие рецепты из RecipeNLG по запросу '{query_text}':"]
    for idx, item in enumerate(ranked, start=1):
        if debug:
            lines.append(
                f"{idx}. {item['title_ru']} (score: {item['total_score']}, cosine: {item['cosine_similarity']}, "
                f"fuzzy: {item['fuzzy_score']}) - {item['match_reason']}"
            )
        else:
            lines.append(f"{idx}. {item['title_ru']}")
    return "\n".join(lines)


def _format_dataset_recipe_response(item, debug=False):
    localized = localize_recipenlg_item(item, with_details=True)
    ingredients_text = localized.get("ingredients_ru_text", "нет данных")
    steps_text = localized.get("directions_ru_text", "нет шагов")
    response = (
        f"**Рецепт: {localized['title_ru']}**\n\n"
        f"**Ингредиенты:** {ingredients_text}\n\n"
        f"**Шаги:** {steps_text}"
    )
    if debug:
        response += (
            "\n\n"
            f"Почему выбран: {localized['match_reason']}. "
            f"Hybrid score: {localized['total_score']} "
            f"(cosine: {localized['cosine_similarity']}, fuzzy: {localized['fuzzy_score']}, "
            f"rules: {localized['rule_score']}, keyword: {localized.get('keyword_score', 0.0)})."
        )
    return response


def _format_dataset_preview_response():
    preview = get_recipenlg_preview(limit=10)
    if not preview:
        return "RecipeNLG подключен, но preview рецептов недоступен."

    lines = ["В RecipeNLG подключено более 2 млн рецептов. Примеры:"]
    for idx, item in enumerate(preview, start=1):
        lines.append(f"{idx}. {item['title_ru']} (ингредиентов: {item['ingredient_count']})")
    return "\n".join(lines)


def run_text_pipeline(text, data_source=None, debug=False, exclude_titles=None):
    text = str(text or "").strip()
    if not text:
        return {"handled": False, "response": None, "stages": {}}

    parsed = analyze_cooking_request(text, data_source)
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
    exclude_titles = exclude_titles or []

    if mode == "list_datasets":
        datasets = [dataset.get("name", "") for dataset in get_known_datasets() if dataset.get("name")]
        response = f"Подключенные датасеты: {join_items(datasets)}"
        return {
            "handled": True,
            "response": response,
            "stages": {"input": text, "nlp": parsed, "rules": {}, "decision": {"mode": mode}},
        }

    if mode == "dataset_detail" and entities.get("datasets"):
        response = _describe_dataset(entities["datasets"][0])
        return {
            "handled": True,
            "response": response,
            "stages": {"input": text, "nlp": parsed, "rules": {}, "decision": {"mode": mode}},
        }

    if mode == "list_recipes" and min_cal is None and max_cal is None and not include_ingredients:
        response = _format_dataset_preview_response()
        return {
            "handled": True,
            "response": response,
            "stages": {"input": text, "nlp": parsed, "rules": {}, "decision": {"mode": mode}},
        }

    if _is_similarity_request(text, parsed):
        if recipenlg_ready():
            dataset_ranked = rank_recipenlg_candidates(
                text,
                include_ingredients=include_ingredients,
                exclude_ingredients=exclude_ingredients,
                exclude_titles=exclude_titles,
                meal_type=meal_type,
                limit=limit,
            )
            if dataset_ranked:
                response = _format_dataset_similarity_response(text, dataset_ranked, debug=debug)
                return {
                    "handled": True,
                    "response": response,
                    "stages": {
                        "input": text,
                        "nlp": parsed,
                        "rules": {
                            "candidate_count_before": "RecipeNLG",
                            "candidate_count_after": len(dataset_ranked),
                        },
                        "decision": {
                            "mode": "similarity_search",
                            "seed_recipe": None,
                            "strategy": "recipenlg_cosine_fuzzy",
                            "ranked": dataset_ranked,
                        },
                    },
                }
        return {
            "handled": True,
            "response": "Не нашел похожих рецептов в RecipeNLG. Попробуйте уточнить блюдо или ингредиенты.",
            "stages": {"input": text, "nlp": parsed, "rules": {}, "decision": {"mode": mode}},
        }

    wants_recommendation = (
        bool(include_ingredients)
        or bool(exclude_ingredients)
        or bool(exclude_allergens)
        or bool(meal_type)
        or min_cal is not None
        or max_cal is not None
        or "рецепт" in _normalize(text)
        or "приготов" in _normalize(text)
        or "посоветуй" in _normalize(text)
        or len(text.split()) <= 4
    )

    if wants_recommendation and recipenlg_ready():
        dataset_ranked = rank_recipenlg_candidates(
            text,
            include_ingredients=include_ingredients,
            exclude_ingredients=exclude_ingredients,
            exclude_titles=exclude_titles,
            meal_type=meal_type,
            limit=limit,
        )
        if dataset_ranked:
            wants_list = mode in {"list_recipes", "list_ingredients"} or "рецепты" in _normalize(text)
            response = (
                _format_dataset_similarity_response(text, dataset_ranked, debug=debug)
                if wants_list
                else _format_dataset_recipe_response(dataset_ranked[0], debug=debug)
            )
            return {
                "handled": True,
                "response": response,
                "stages": {
                    "input": text,
                    "nlp": parsed,
                    "rules": {
                        "candidate_count_before": "RecipeNLG",
                        "candidate_count_after": len(dataset_ranked),
                        "include_ingredients": include_ingredients,
                        "exclude_ingredients": exclude_ingredients,
                        "exclude_allergens": exclude_allergens,
                        "meal_type": meal_type,
                        "min_calories": min_cal,
                        "max_calories": max_cal,
                    },
                    "decision": {
                        "mode": mode,
                        "strategy": "recipenlg_cosine_fuzzy_rules",
                        "ranked": dataset_ranked,
                    },
                },
            }
        return {
            "handled": True,
            "response": _build_no_results_message(
                meal_type,
                include_ingredients,
                exclude_ingredients,
                exclude_allergens,
                min_cal,
                max_cal,
            ),
            "stages": {
                "input": text,
                "nlp": parsed,
                "rules": {
                    "include_ingredients": include_ingredients,
                    "exclude_ingredients": exclude_ingredients,
                    "exclude_allergens": exclude_allergens,
                    "meal_type": meal_type,
                    "min_calories": min_cal,
                    "max_calories": max_cal,
                },
                "decision": {
                    "mode": mode,
                    "strategy": "recipenlg_only",
                    "ranked": [],
                },
            },
        }

    return {"handled": False, "response": None, "stages": {"input": text, "nlp": parsed}}


def run_image_pipeline(image_bytes, data_source=None):
    result = analyze_food_photo(image_bytes, data_source)
    label = result.get("predicted_label")
    confidence = float(result.get("confidence", 0.0) or 0.0)
    rules_stage = {
        "low_confidence": confidence < 0.35,
        "ocr_used": bool(result.get("ocr_text")),
        "ingredient_hints_used": bool(result.get("ingredient_hints")),
        "dataset_recipe_used": "RecipeNLG" in str(result.get("recipe_text", "")),
    }
    return {
        "handled": True,
        "response": result.get("recipe_text", "Рецепт не найден."),
        "vision_result": result,
        "stages": {
            "input": {"type": "image"},
            "cv": {
                "predicted_label": label,
                "confidence": confidence,
                "top_candidates": result.get("top_candidates", []),
                "ocr_text": result.get("ocr_text", ""),
            },
            "rules": rules_stage,
            "decision": {
                "recipe_name": result.get("recipe_name"),
                "dataset_status": result.get("dataset_status", {}),
            },
        },
    }
