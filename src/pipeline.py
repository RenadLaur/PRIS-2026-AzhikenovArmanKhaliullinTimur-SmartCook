try:
    from .nlp import analyze_cooking_request, get_known_datasets
    from .recommender import (
        filter_recipe_candidates,
        find_recipe_by_name,
        format_ranked_recipe_list,
        format_recipe_answer,
        graph_lists,
        join_items,
        rank_recipe_candidates,
        rank_recipenlg_candidates,
        rank_similar_recipes,
        recipenlg_ready,
    )
    from .vision import analyze_food_photo
except ImportError:
    from nlp import analyze_cooking_request, get_known_datasets
    from recommender import (
        filter_recipe_candidates,
        find_recipe_by_name,
        format_ranked_recipe_list,
        format_recipe_answer,
        graph_lists,
        join_items,
        rank_recipe_candidates,
        rank_recipenlg_candidates,
        rank_similar_recipes,
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
    return f"Не нашел рецепт ({suffix}). Попробуйте смягчить ограничения."


def _is_similarity_request(query_text, parsed):
    mode = parsed.get("query_mode")
    query = _normalize(query_text)
    if mode == "similarity_search":
        return True
    return any(marker in query for marker in ["похож", "похожие", "аналог", "что похоже", "similar"])


def _format_similarity_response(seed_recipe, ranked):
    if not ranked:
        return f"Для блюда '{seed_recipe}' не нашел похожих вариантов."

    lines = [f"Похожие блюда на '{seed_recipe}':"]
    for idx, item in enumerate(ranked, start=1):
        shared = ", ".join(item.get("shared_ingredients", [])) or "без явных общих ингредиентов"
        lines.append(
            f"{idx}. {item['recipe']} ({item['calories']} ккал, cosine: {item['cosine_similarity']}, "
            f"fuzzy: {item['fuzzy_score']}, общие ингредиенты: {shared})"
        )
    return "\n".join(lines)


def _format_similarity_by_query_response(query_text, ranked):
    if not ranked:
        return f"По описанию '{query_text}' не нашел похожих вариантов."

    lines = [f"Похожие блюда по описанию '{query_text}':"]
    for idx, item in enumerate(ranked, start=1):
        lines.append(
            f"{idx}. {item['recipe']} ({item['calories']} ккал, cosine: {item['cosine_similarity']}, "
            f"fuzzy: {item['fuzzy_score']}) - {item['match_reason']}"
        )
    return "\n".join(lines)


def _format_dataset_similarity_response(query_text, ranked):
    if not ranked:
        return f"По описанию '{query_text}' не нашел похожих рецептов в RecipeNLG."

    lines = [f"Похожие рецепты из RecipeNLG по запросу '{query_text}':"]
    for idx, item in enumerate(ranked, start=1):
        lines.append(
            f"{idx}. {item['title_ru']} (score: {item['total_score']}, cosine: {item['cosine_similarity']}, "
            f"fuzzy: {item['fuzzy_score']}) - {item['match_reason']}"
        )
    return "\n".join(lines)


def _format_dataset_recipe_response(item):
    ingredients_text = ", ".join(item.get("ingredients_ru", [])) if item.get("ingredients_ru") else "нет данных"
    steps_text = " ".join(item.get("directions_ru", [])) if item.get("directions_ru") else "нет шагов"
    return (
        f"**Рецепт: {item['title_ru']}**\n\n"
        f"**Ингредиенты:** {ingredients_text}\n\n"
        f"**Шаги:** {steps_text}\n\n"
        f"Почему выбран: {item['match_reason']}. "
        f"Hybrid score: {item['total_score']} "
        f"(cosine: {item['cosine_similarity']}, fuzzy: {item['fuzzy_score']}, rules: {item['rule_score']})."
    )


def _format_hybrid_list_response(ranked, limit):
    if not ranked:
        return "Нет подходящих рецептов."

    lines = ["Подобрал варианты гибридным пайплайном (NLP -> правила -> cosine/fuzzy -> решение):"]
    for idx, item in enumerate(ranked[:limit], start=1):
        lines.append(
            f"{idx}. {item['recipe']} ({item['calories']} ккал, score: {item['total_score']}) "
            f"- {item['match_reason']}"
        )
    return "\n".join(lines)


def _significant_tokens(text):
    query = _normalize(text)
    return [token for token in query.split() if len(token) >= 3]


def _prefer_graph_for_recommendation(text, include_ingredients, exclude_ingredients, exclude_allergens, meal_type, min_cal, max_cal):
    query = _normalize(text)
    tokens = _significant_tokens(text)

    if exclude_allergens or min_cal is not None or max_cal is not None:
        return True

    if "датасет" in query or "recipenlg" in query:
        return False

    if len(tokens) <= 5 and (include_ingredients or meal_type):
        return True

    short_markers = ["подбери", "посоветуй", "ужин", "обед", "завтрак", "рецепты с", "что приготовить"]
    if any(marker in query for marker in short_markers) and len(query) <= 40:
        return True

    return False


def _try_fast_similarity_response(text, graph):
    query = _normalize(text)
    if not any(marker in query for marker in ["похож", "похожие", "аналог", "what similar", "similar"]):
        return None

    seed_recipe = find_recipe_by_name(graph, text)
    if not seed_recipe:
        return None

    ranked = rank_similar_recipes(graph, seed_recipe, limit=8)
    return {
        "handled": True,
        "response": _format_similarity_response(seed_recipe, ranked),
        "stages": {
            "input": text,
            "nlp": {"engine": "fast-path", "query_mode": "similarity_search"},
            "rules": {"candidate_count_before": "graph", "candidate_count_after": len(ranked)},
            "decision": {
                "mode": "similarity_search",
                "seed_recipe": seed_recipe,
                "strategy": "fast_graph_similarity",
                "ranked": ranked,
            },
        },
    }


def run_text_pipeline(text, graph):
    text = str(text or "").strip()
    if not text or graph is None or not hasattr(graph, "nodes"):
        return {"handled": False, "response": None, "stages": {}}

    fast_similarity = _try_fast_similarity_response(text, graph)
    if fast_similarity is not None:
        return fast_similarity

    parsed = analyze_cooking_request(text, graph)
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

    recipes, ingredients, allergens = graph_lists(graph)

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

    if mode == "list_allergens":
        response = f"Доступные аллергены: {join_items(allergens)}"
        return {
            "handled": True,
            "response": response,
            "stages": {"input": text, "nlp": parsed, "rules": {}, "decision": {"mode": mode}},
        }

    if mode == "list_ingredients":
        response = f"Доступные ингредиенты: {join_items(ingredients)}"
        return {
            "handled": True,
            "response": response,
            "stages": {"input": text, "nlp": parsed, "rules": {}, "decision": {"mode": mode}},
        }

    if mode == "list_recipes" and min_cal is None and max_cal is None and not include_ingredients:
        response = f"Доступные рецепты: {join_items(recipes)}"
        return {
            "handled": True,
            "response": response,
            "stages": {"input": text, "nlp": parsed, "rules": {}, "decision": {"mode": mode}},
        }

    if _is_similarity_request(text, parsed):
        seed_recipe = None
        if entities.get("recipes"):
            seed_recipe = entities["recipes"][0]
        if seed_recipe is None:
            seed_recipe = find_recipe_by_name(graph, text)

        if recipenlg_ready():
            dataset_ranked = rank_recipenlg_candidates(
                text,
                include_ingredients=include_ingredients,
                exclude_ingredients=exclude_ingredients,
                meal_type=meal_type,
                limit=limit,
            )
            if dataset_ranked:
                response = _format_dataset_similarity_response(text, dataset_ranked)
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

        if seed_recipe:
            ranked = rank_similar_recipes(graph, seed_recipe, limit=limit)
            response = _format_similarity_response(seed_recipe, ranked)
            return {
                "handled": True,
                "response": response,
                "stages": {
                    "input": text,
                    "nlp": parsed,
                    "rules": {"candidate_count_before": len(recipes), "candidate_count_after": len(ranked)},
                    "decision": {"mode": "similarity_search", "seed_recipe": seed_recipe, "ranked": ranked},
                },
            }

        ranked = rank_recipe_candidates(
            graph,
            recipes,
            text,
            include_ingredients=include_ingredients,
            meal_type=meal_type,
            target_calories=target_cal,
            limit=limit,
        )
        response = _format_similarity_by_query_response(text, ranked)
        return {
            "handled": True,
            "response": response,
            "stages": {
                "input": text,
                "nlp": parsed,
                "rules": {"candidate_count_before": len(recipes), "candidate_count_after": len(ranked)},
                "decision": {
                    "mode": "similarity_search",
                    "seed_recipe": None,
                    "strategy": "query_to_recipe_cosine_fuzzy",
                    "ranked": ranked,
                },
            },
        }

    filtered = filter_recipe_candidates(
        graph,
        include_ingredients=include_ingredients,
        exclude_ingredients=exclude_ingredients,
        exclude_allergens=exclude_allergens,
        meal_type=meal_type,
        min_calories=min_cal,
        max_calories=max_cal,
    )

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
    )

    if (
        wants_recommendation
        and recipenlg_ready()
        and not _prefer_graph_for_recommendation(
            text,
            include_ingredients,
            exclude_ingredients,
            exclude_allergens,
            meal_type,
            min_cal,
            max_cal,
        )
    ):
        dataset_ranked = rank_recipenlg_candidates(
            text,
            include_ingredients=include_ingredients,
            exclude_ingredients=exclude_ingredients,
            meal_type=meal_type,
            limit=limit,
        )
        if dataset_ranked:
            wants_list = mode in {"list_recipes", "list_ingredients"} or "рецепты" in _normalize(text)
            response = (
                _format_dataset_similarity_response(text, dataset_ranked)
                if wants_list
                else _format_dataset_recipe_response(dataset_ranked[0])
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
                        "meal_type": meal_type,
                    },
                    "decision": {
                        "mode": mode,
                        "strategy": "recipenlg_cosine_fuzzy_rules",
                        "ranked": dataset_ranked,
                    },
                },
            }

    if wants_recommendation:
        ranked = rank_recipe_candidates(
            graph,
            filtered,
            text,
            include_ingredients=include_ingredients,
            meal_type=meal_type,
            target_calories=target_cal,
            limit=limit,
        )

        if not ranked:
            response = _build_no_results_message(
                meal_type,
                include_ingredients,
                exclude_ingredients,
                exclude_allergens,
                min_cal,
                max_cal,
            )
        else:
            wants_list = mode in {"list_recipes", "list_ingredients"} or "рецепты" in _normalize(text)
            if wants_list:
                response = _format_hybrid_list_response(ranked, limit)
            else:
                best = ranked[0]
                response = (
                    f"{format_recipe_answer(graph, best['recipe'])}\n"
                    f"Почему выбран: {best['match_reason']}. "
                    f"Hybrid score: {best['total_score']} "
                    f"(cosine: {best['cosine_similarity']}, fuzzy: {best['fuzzy_score']}, "
                    f"rules: {best['rule_score']})."
                )

        return {
            "handled": True,
            "response": response,
            "stages": {
                "input": text,
                "nlp": parsed,
                "rules": {
                    "candidate_count_before": len(recipes),
                    "candidate_count_after": len(filtered),
                    "include_ingredients": include_ingredients,
                    "exclude_ingredients": exclude_ingredients,
                    "exclude_allergens": exclude_allergens,
                    "meal_type": meal_type,
                    "min_calories": min_cal,
                    "max_calories": max_cal,
                },
                "decision": {
                    "mode": mode,
                    "strategy": "hybrid_cosine_fuzzy_rules",
                    "ranked": ranked if ranked else [],
                    "preview": format_ranked_recipe_list(ranked, limit=limit) if ranked else "",
                },
            },
        }

    return {"handled": False, "response": None, "stages": {"input": text, "nlp": parsed}}


def run_image_pipeline(image_bytes, graph):
    result = analyze_food_photo(image_bytes, graph)
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
