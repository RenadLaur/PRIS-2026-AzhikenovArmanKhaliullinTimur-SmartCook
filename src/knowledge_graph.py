import networkx as nx

try:
    from .models import Allergen, Ingredient, Recipe
except ImportError:  # Support direct execution from src/
    from models import Allergen, Ingredient, Recipe


def _build_sample_data():
    recipes = [
        Recipe(name="Курица с рисом", ingredients=["курица", "рис", "брокколи"], calories=550),
        Recipe(name="Паста с сыром", ingredients=["паста", "сыр", "сливки"], calories=720),
        Recipe(
            name="Овощной салат",
            ingredients=["помидор", "огурец", "зелень", "оливковое масло"],
            calories=240,
        ),
        Recipe(name="Ореховый батончик", ingredients=["арахис", "мед", "овсянка"], calories=380),
        Recipe(
            name="Лосось с киноа",
            ingredients=["лосось", "киноа", "лимон", "оливковое масло"],
            calories=510,
        ),
        Recipe(name="Омлет с сыром", ingredients=["яйцо", "сыр", "молоко", "зелень"], calories=430),
        Recipe(
            name="Йогурт с ягодами",
            ingredients=["йогурт", "клубника", "мед", "орех грецкий"],
            calories=290,
        ),
        Recipe(
            name="Тофу боул",
            ingredients=["тофу", "рис", "морковь", "соевый соус"],
            calories=470,
        ),
        Recipe(
            name="Чечевичный суп",
            ingredients=["чечевица", "лук", "морковь", "сельдерей"],
            calories=340,
        ),
        Recipe(name="Панкейки", ingredients=["мука", "молоко", "яйцо", "сахар"], calories=610),
        Recipe(name="Хумус", ingredients=["нут", "кунжут", "лимон", "оливковое масло"], calories=320),
        Recipe(name="Сэндвич с тунцом", ingredients=["тунец", "хлеб", "яйцо", "майонез"], calories=520),
        Recipe(
            name="Бешбармак",
            ingredients=["конина", "лапша", "лук", "бульон", "картофель"],
            calories=680,
        ),
        Recipe(
            name="Плов по-казахски",
            ingredients=["баранина", "рис", "морковь", "лук", "чеснок"],
            calories=640,
        ),
        Recipe(
            name="Манты",
            ingredients=["баранина", "мука", "лук", "вода"],
            calories=590,
        ),
        Recipe(
            name="Лагман",
            ingredients=["говядина", "лапша", "болгарский перец", "лук", "морковь"],
            calories=620,
        ),
        Recipe(
            name="Гречка с грибами",
            ingredients=["гречка", "шампиньоны", "лук", "оливковое масло"],
            calories=410,
        ),
        Recipe(
            name="Шакшука",
            ingredients=["яйцо", "помидор", "болгарский перец", "лук"],
            calories=360,
        ),
        Recipe(
            name="Сырники",
            ingredients=["творог", "яйцо", "мука", "сахар"],
            calories=540,
        ),
        Recipe(
            name="Окрошка",
            ingredients=["кефир", "огурец", "яйцо", "зелень", "картофель"],
            calories=330,
        ),
        Recipe(
            name="Креветки с овощами",
            ingredients=["креветки", "болгарский перец", "морковь", "соевый соус"],
            calories=390,
        ),
        Recipe(
            name="Рамен",
            ingredients=["лапша", "курица", "яйцо", "соевый соус"],
            calories=560,
        ),
        Recipe(
            name="Говядина с картофелем",
            ingredients=["говядина", "картофель", "лук", "морковь"],
            calories=530,
        ),
        Recipe(
            name="Цезарь с курицей",
            ingredients=["курица", "сыр", "сухарики", "салат романо"],
            calories=480,
        ),
    ]

    allergens = [
        Allergen(name="Молоко", sources=["сыр", "сливки", "молоко", "йогурт"], severity=0.8),
        Allergen(name="Орехи", sources=["арахис", "орех грецкий"], severity=0.9),
        Allergen(name="Глютен", sources=["паста", "овсянка", "мука", "хлеб", "лапша"], severity=0.6),
        Allergen(name="Яйца", sources=["яйцо", "майонез"], severity=0.7),
        Allergen(name="Рыба", sources=["лосось", "тунец"], severity=0.8),
        Allergen(name="Морепродукты", sources=["креветки"], severity=0.8),
        Allergen(name="Соя", sources=["тофу", "соевый соус"], severity=0.7),
        Allergen(name="Кунжут", sources=["кунжут"], severity=0.7),
    ]

    ingredients = [
        Ingredient(name="курица", allergens=[]),
        Ingredient(name="рис", allergens=[]),
        Ingredient(name="брокколи", allergens=[]),
        Ingredient(name="паста", allergens=["Глютен"]),
        Ingredient(name="сыр", allergens=["Молоко"]),
        Ingredient(name="сливки", allergens=["Молоко"]),
        Ingredient(name="помидор", allergens=[]),
        Ingredient(name="огурец", allergens=[]),
        Ingredient(name="зелень", allergens=[]),
        Ingredient(name="оливковое масло", allergens=[]),
        Ingredient(name="арахис", allergens=["Орехи"]),
        Ingredient(name="мед", allergens=[]),
        Ingredient(name="овсянка", allergens=["Глютен"]),
        Ingredient(name="лосось", allergens=["Рыба"]),
        Ingredient(name="киноа", allergens=[]),
        Ingredient(name="лимон", allergens=[]),
        Ingredient(name="яйцо", allergens=["Яйца"]),
        Ingredient(name="молоко", allergens=["Молоко"]),
        Ingredient(name="йогурт", allergens=["Молоко"]),
        Ingredient(name="клубника", allergens=[]),
        Ingredient(name="орех грецкий", allergens=["Орехи"]),
        Ingredient(name="тофу", allergens=["Соя"]),
        Ingredient(name="морковь", allergens=[]),
        Ingredient(name="соевый соус", allergens=["Соя"]),
        Ingredient(name="чечевица", allergens=[]),
        Ingredient(name="лук", allergens=[]),
        Ingredient(name="сельдерей", allergens=[]),
        Ingredient(name="мука", allergens=["Глютен"]),
        Ingredient(name="сахар", allergens=[]),
        Ingredient(name="нут", allergens=[]),
        Ingredient(name="кунжут", allergens=["Кунжут"]),
        Ingredient(name="тунец", allergens=["Рыба"]),
        Ingredient(name="хлеб", allergens=["Глютен"]),
        Ingredient(name="майонез", allergens=["Яйца"]),
        Ingredient(name="конина", allergens=[]),
        Ingredient(name="лапша", allergens=["Глютен"]),
        Ingredient(name="бульон", allergens=[]),
        Ingredient(name="картофель", allergens=[]),
        Ingredient(name="баранина", allergens=[]),
        Ingredient(name="чеснок", allergens=[]),
        Ingredient(name="вода", allergens=[]),
        Ingredient(name="говядина", allergens=[]),
        Ingredient(name="болгарский перец", allergens=[]),
        Ingredient(name="гречка", allergens=[]),
        Ingredient(name="шампиньоны", allergens=[]),
        Ingredient(name="творог", allergens=["Молоко"]),
        Ingredient(name="кефир", allergens=["Молоко"]),
        Ingredient(name="креветки", allergens=["Морепродукты"]),
        Ingredient(name="сухарики", allergens=["Глютен"]),
        Ingredient(name="салат романо", allergens=[]),
    ]

    return recipes, ingredients, allergens


def create_graph():
    G = nx.Graph()

    recipes, ingredients, allergens = _build_sample_data()

    for recipe in recipes:
        G.add_node(recipe.name, type="recipe", data=recipe)
    for ingredient in ingredients:
        G.add_node(ingredient.name, type="ingredient", data=ingredient)
    for allergen in allergens:
        G.add_node(allergen.name, type="allergen", data=allergen)

    for recipe in recipes:
        for ingredient in recipe.ingredients:
            G.add_edge(recipe.name, ingredient, relation="содержит")

    for ingredient in ingredients:
        for allergen in ingredient.allergens:
            G.add_edge(ingredient.name, allergen, relation="является")

    return G


def load_graph():
    """
    Точка входа для UI: возвращает готовый граф знаний.
    """
    return create_graph()


def find_related_entities(graph, start_node):
    if start_node not in graph:
        return []

    results = []
    for neighbor in graph.neighbors(start_node):
        relation = graph.get_edge_data(start_node, neighbor).get("relation", "связано")
        neighbor_type = graph.nodes[neighbor].get("type", "unknown")
        results.append((neighbor, relation, neighbor_type))

    return results


def find_related_recipes_for_allergen(graph, allergen_name):
    if allergen_name not in graph:
        return []

    if graph.nodes[allergen_name].get("type") != "allergen":
        return []

    recipes = set()
    for ingredient in graph.neighbors(allergen_name):
        if graph.nodes[ingredient].get("type") != "ingredient":
            continue
        for recipe in graph.neighbors(ingredient):
            if graph.nodes[recipe].get("type") == "recipe":
                recipes.add(recipe)

    return sorted(recipes)


def exclude_recipes_by_allergen(graph, allergen_name):
    if allergen_name not in graph:
        return [], []

    if graph.nodes[allergen_name].get("type") != "allergen":
        return [], []

    all_recipes = sorted(
        node for node in graph.nodes if graph.nodes[node].get("type") == "recipe"
    )
    excluded_recipes = set(find_related_recipes_for_allergen(graph, allergen_name))
    safe_recipes = [recipe for recipe in all_recipes if recipe not in excluded_recipes]

    return safe_recipes, sorted(excluded_recipes)
