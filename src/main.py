import hashlib

import pandas as pd
import streamlit as st

try:
    from .app_service import (
        compact_text,
        get_api_catalog,
        get_dashboard_data,
        get_runtime_status,
        handle_chat_message,
        handle_image_message,
        get_sample_queries,
    )
except ImportError:
    from app_service import (
        compact_text,
        get_api_catalog,
        get_dashboard_data,
        get_runtime_status,
        handle_chat_message,
        handle_image_message,
        get_sample_queries,
    )


st.set_page_config(page_title="SmartCook Chat", page_icon="🤖")
st.title("AI Assistant")
st.write("Спросите термин из базы знаний (рецепт, ингредиент или аллерген).")


@st.cache_resource
def get_ui_status():
    return get_runtime_status()


@st.cache_data
def get_dashboard_view():
    return get_dashboard_data()


status = get_ui_status()
dashboard = get_dashboard_view()
api_catalog = get_api_catalog()
sample_queries = get_sample_queries()
graph_status = status["graph"]
spacy_status = status["nlp"]["status"]
nlp_runtime = status["nlp"]["runtime"]
vision_status = status["vision"]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Привет! Я бот SmartCook. Введите название рецепта, ингредиента или аллергена.",
        }
    ]

if "query_history" not in st.session_state:
    st.session_state.query_history = []

with st.sidebar:
    st.header("Управление")
    if st.button("Очистить чат", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Чат очищен. Введите название рецепта, ингредиента или аллергена.",
            }
        ]
        st.session_state.query_history = []
        st.rerun()

    st.header("История запросов")
    if st.session_state.query_history:
        for idx, query in enumerate(st.session_state.query_history, start=1):
            st.write(f"{idx}. {query}")
    else:
        st.caption("Пока нет запросов")

    st.header("NLP Статус")
    if not spacy_status["spacy_installed"]:
        st.error("spaCy не найден в текущем окружении.")
    elif not spacy_status["model_found"]:
        st.warning("spaCy установлен, но модель (`ru_core_news_sm`) не найдена.")
    elif not nlp_runtime.get("ok"):
        st.warning("spaCy найден, но прогрев модели не удался.")
    else:
        st.success("spaCy и модель доступны.")

    st.caption(f"Python: {spacy_status['python_executable']}")
    st.caption("Рекомендуемый запуск: `.venv/bin/streamlit run src/main.py`")

    st.header("CV/OCR Статус")
    if vision_status["food11_ready"]:
        st.success("Food-11 найден.")
    else:
        st.warning("Food-11 не найден. Будет использован fallback подбор рецепта.")
    if vision_status["easyocr_installed"]:
        st.success("EasyOCR установлен.")
    else:
        st.info("EasyOCR не установлен. OCR-анализ отключен.")
    if vision_status.get("cnn_model_ready"):
        st.success("CNN-модель Food-11 готова.")
    else:
        st.warning(
            "CNN-модель Food-11 не найдена. Запустите: "
            "`.venv/bin/python scripts/train_food11_cnn.py --epochs 3`"
        )
    if vision_status.get("recipenlg_ready"):
        st.success("RecipeNLG подключен.")
    else:
        st.warning("RecipeNLG не найден. Рецепты будут браться из fallback-логики.")
    st.caption(f"Food-11 path: {vision_status['food11_path'] or 'не задан'}")
    st.caption(f"RecipeNLG path: {vision_status.get('recipenlg_path') or 'не задан'}")

if not graph_status["ok"]:
    st.error(f"Ошибка загрузки базы знаний: {graph_status['error']}")

def submit_chat_query(raw_text):
    clean_input = str(raw_text or "").strip()
    if not clean_input:
        return

    st.session_state.query_history.append(clean_input)
    st.session_state.messages.append({"role": "user", "content": compact_text(clean_input)})
    result = handle_chat_message(clean_input)
    st.session_state.messages.append(
        {"role": "assistant", "content": compact_text(result["response"])}
    )


tab_chat, tab_vision, tab_dashboard, tab_api = st.tabs(
    ["Чат", "Фото блюда", "Аналитика", "API/Backend"]
)

with tab_chat:
    st.subheader("Быстрые сценарии")
    sample_columns = st.columns(3)
    for idx, query in enumerate(sample_queries):
        column = sample_columns[idx % len(sample_columns)]
        if column.button(query, key=f"sample_query_{idx}", use_container_width=True):
            submit_chat_query(query)

    with st.form("chat_form", clear_on_submit=True):
        chat_text = st.text_input("Введите запрос")
        chat_submitted = st.form_submit_button("Отправить", type="primary")

    if chat_submitted:
        submit_chat_query(chat_text)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

with tab_vision:
    st.subheader("Фото блюда: распознавание и рецепт")
    with st.form("image_analysis_form"):
        uploaded_image = st.file_uploader(
            "Загрузите фото блюда (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"]
        )
        analyze_button = st.form_submit_button("Определить блюдо и предложить рецепт", type="primary")

    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()
        image_hash = hashlib.sha1(image_bytes).hexdigest()
        st.image(image_bytes, caption="Загруженное изображение", width="stretch")

        if analyze_button:
            with st.spinner("Анализирую фото..."):
                pipeline_result = handle_image_message(image_bytes)
            st.session_state[f"vision_result_{image_hash}"] = pipeline_result

        cached_result = st.session_state.get(f"vision_result_{image_hash}")
        if cached_result:
            vision_result = cached_result.get("vision_result", {})
            if vision_result.get("error"):
                st.error(vision_result["error"])
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Предсказанный класс", vision_result.get("predicted_label") or "не определено")
                with col2:
                    st.metric("Confidence", vision_result.get("confidence", 0.0))

                if vision_result.get("ocr_warning"):
                    st.warning(vision_result["ocr_warning"])
                if vision_result.get("classification_note"):
                    st.warning(vision_result["classification_note"])
                if vision_result.get("ocr_text"):
                    st.caption(f"OCR текст: {vision_result['ocr_text']}")
                if vision_result.get("ingredient_hints"):
                    hints = ", ".join(vision_result["ingredient_hints"])
                    st.caption(f"Найденные ингредиентные подсказки: {hints}")

                top_candidates = vision_result.get("top_candidates") or []
                if top_candidates:
                    candidates_df = pd.DataFrame(
                        [{"class": name, "score": score} for name, score in top_candidates]
                    )
                    st.dataframe(candidates_df, use_container_width=True, hide_index=True)
                    st.bar_chart(candidates_df.set_index("class"))

                st.markdown(cached_result.get("response", "Рецепт не найден."))

with tab_dashboard:
    st.subheader("Аналитика базы знаний")
    summary = dashboard["summary"]
    metric_cols = st.columns(3)
    metric_cols[0].metric("Рецепты", summary["recipes_count"])
    metric_cols[1].metric("Ингредиенты", summary["ingredients_count"])
    metric_cols[2].metric("Аллергены", summary["allergens_count"])

    recipes_df = pd.DataFrame(dashboard["recipes"])
    top_calories_df = pd.DataFrame(dashboard["top_calories"])
    allergen_df = pd.DataFrame(dashboard["allergen_stats"])
    meal_df = pd.DataFrame(dashboard["meal_stats"])

    st.markdown("**Таблица рецептов**")
    st.dataframe(
        recipes_df[["recipe", "calories", "ingredients_count", "allergens_count", "meal_tags"]],
        use_container_width=True,
        hide_index=True,
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Топ-10 по калориям**")
        if not top_calories_df.empty:
            st.bar_chart(top_calories_df.set_index("recipe")[["calories"]])
    with chart_col2:
        st.markdown("**Распределение по приемам пищи**")
        if not meal_df.empty:
            st.bar_chart(meal_df.set_index("meal_type")[["recipes_count"]])

    st.markdown("**Аллергены по рецептам**")
    if not allergen_df.empty:
        st.dataframe(allergen_df, use_container_width=True, hide_index=True)
        st.bar_chart(allergen_df.set_index("allergen")[["recipes_count"]])

with tab_api:
    st.subheader("Backend и API")
    st.markdown("**Статус backend**")
    st.json(status)

    st.markdown("**Доступные endpoints**")
    api_df = pd.DataFrame(api_catalog)
    st.dataframe(api_df, use_container_width=True, hide_index=True)

    st.markdown("**Пример запроса**")
    st.code(
        "curl -X POST http://127.0.0.1:8000/chat "
        "-H 'Content-Type: application/json' "
        "-d '{\"message\":\"похожие на плов\"}'",
        language="bash",
    )
