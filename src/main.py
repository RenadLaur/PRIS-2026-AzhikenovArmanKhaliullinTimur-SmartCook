import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

try:
    from .knowledge_graph import (
        create_graph,
        exclude_recipes_by_allergen,
        find_related_entities,
        find_related_recipes_for_allergen,
    )
except ImportError:
    from knowledge_graph import (
        create_graph,
        exclude_recipes_by_allergen,
        find_related_entities,
        find_related_recipes_for_allergen,
    )

st.title("SmartCook Knowledge Graph Explorer 🕸")
st.write("Исследуйте связи между рецептами, ингредиентами и аллергенами.")

G = create_graph()

node_types = sorted({G.nodes[node].get("type", "unknown") for node in G.nodes})

with st.sidebar:
    st.header("Фильтры")
    selected_type = st.selectbox("Тип узла:", ["Все"] + node_types)

if selected_type == "Все":
    available_nodes = sorted(G.nodes())
else:
    available_nodes = sorted(
        [node for node in G.nodes() if G.nodes[node].get("type") == selected_type]
    )

selected_node = st.selectbox("Выберите объект для поиска связей:", available_nodes)

if st.button("Найти связи"):
    results = find_related_entities(G, selected_node)

    if results:
        st.success(f"Объект '{selected_node}' связан с {len(results)} узлами:")
        for neighbor, relation, neighbor_type in results:
            st.write(f"- {neighbor} — {relation} (тип: {neighbor_type})")
    else:
        st.warning("Связи не найдены.")

    if G.nodes[selected_node].get("type") == "allergen":
        linked_recipes = find_related_recipes_for_allergen(G, selected_node)
        if linked_recipes:
            st.info(
                "Рецепты, связанные с аллергеном через ингредиенты: "
                + ", ".join(linked_recipes)
            )
        else:
            st.info("Нет рецептов, связанных с этим аллергеном через ингредиенты.")

st.write("### Безопасный фильтр по аллергену")
allergen_nodes = sorted(
    [node for node in G.nodes() if G.nodes[node].get("type") == "allergen"]
)
selected_allergen = st.selectbox(
    "Исключить все рецепты, связанные с аллергеном:",
    allergen_nodes,
)

if st.button("Применить исключение"):
    safe_recipes, excluded_recipes = exclude_recipes_by_allergen(G, selected_allergen)

    if excluded_recipes:
        st.warning("Исключены рецепты: " + ", ".join(excluded_recipes))
    else:
        st.info("Связанные рецепты для исключения не найдены.")

    if safe_recipes:
        st.success("Остались безопасные рецепты: " + ", ".join(safe_recipes))
    else:
        st.error("После фильтрации безопасных рецептов не осталось.")

st.write("### Визуализация структуры")
fig, ax = plt.subplots(figsize=(9, 7))

pos = nx.spring_layout(G, seed=42)

color_map = {
    "recipe": "#FFD166",
    "ingredient": "#118AB2",
    "allergen": "#EF476F",
    "unknown": "#CCCCCC",
}
node_colors = [color_map.get(G.nodes[node].get("type", "unknown")) for node in G.nodes()]

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=node_colors,
    edge_color="#999999",
    node_size=2000,
    font_size=9,
    ax=ax,
)

edge_labels = nx.get_edge_attributes(G, "relation")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

legend_handles = [
    mpatches.Patch(color=color_map["recipe"], label="Рецепт"),
    mpatches.Patch(color=color_map["ingredient"], label="Ингредиент"),
    mpatches.Patch(color=color_map["allergen"], label="Аллерген"),
]
ax.legend(handles=legend_handles, loc="upper left")

st.pyplot(fig)
