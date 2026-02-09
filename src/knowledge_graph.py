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
    ]

    allergens = [
        Allergen(name="Молоко", sources=["сыр", "сливки"], severity=0.8),
        Allergen(name="Орехи", sources=["арахис"], severity=0.9),
        Allergen(name="Глютен", sources=["паста", "овсянка"], severity=0.6),
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
