import streamlit as st
from mock_data import test_entity as default_data
from logic import check_rules

st.title("SmartCook Rule-Based Debugger 🛠")
st.write("### Настройка входных данных")

with st.sidebar:
    st.header("Параметры")
    calories = st.number_input(
        "Калорийность блюда (ккал):",
        min_value=0,
        value=int(default_data["calories"]),
        step=10,
    )
    has_allergy_info = st.checkbox(
        "У пользователя указаны аллергии",
        value=default_data["has_allergy_info"],
    )
    meal_type = st.text_input("Тип блюда:", value=default_data["meal_type"])
    ingredients_text = st.text_input(
        "Ингредиенты (через запятую):",
        value=", ".join(default_data["ingredients"]),
    )

if st.button("Запустить проверку"):
    ingredients = [item.strip() for item in ingredients_text.split(",") if item.strip()]
    current_test_data = {
        "meal_type": meal_type,
        "calories": calories,
        "ingredients": ingredients,
        "has_allergy_info": has_allergy_info,
    }

    result = check_rules(current_test_data)

    if "✅" in result:
        st.success(result)
    elif "⛔️" in result:
        st.error(result)
    else:
        st.warning(result)
