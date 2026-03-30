import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from .app_service import (
        compact_text,
        get_api_catalog,
        get_dashboard_data,
        get_dataset_inventory,
        get_runtime_status,
        handle_chat_message,
        handle_image_message,
        initial_chat_context,
        initial_chat_messages,
        get_sample_queries,
    )
    CHEF_ICON_PATH = Path(__file__).resolve().parent.parent / "assets" / "chef-hat.svg"
except ImportError:
    from app_service import (
        compact_text,
        get_api_catalog,
        get_dashboard_data,
        get_dataset_inventory,
        get_runtime_status,
        handle_chat_message,
        handle_image_message,
        initial_chat_context,
        initial_chat_messages,
        get_sample_queries,
    )
    CHEF_ICON_PATH = Path(__file__).resolve().parent.parent / "assets" / "chef-hat.svg"


st.set_page_config(page_title="SmartCook Chat", page_icon=str(CHEF_ICON_PATH))
st.logo(str(CHEF_ICON_PATH))
st.title("SmartCook")
st.write("Опишите блюдо или рецепт на русском, либо загрузите фото блюда.")
st.markdown(
    """
    <style>
    [data-testid="stStatusWidget"] {visibility: hidden !important;}
    .smartcook-run-indicator {
        position: fixed;
        top: 0.65rem;
        right: 15.2rem;
        z-index: 999999;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        color: #c86b00;
        font-weight: 700;
        font-size: 0.95rem;
        pointer-events: none;
    }
    .smartcook-run-indicator__icon {
        display: inline-block;
        font-size: 1.15rem;
        animation: smartcookPulse 1.4s ease-in-out infinite;
        transform-origin: center;
    }
    @keyframes smartcookPulse {
        0% { transform: rotate(0deg) scale(1); opacity: 0.9; }
        50% { transform: rotate(-10deg) scale(1.08); opacity: 1; }
        100% { transform: rotate(0deg) scale(1); opacity: 0.9; }
    }
    </style>
    <div class="smartcook-run-indicator">
        <span class="smartcook-run-indicator__icon">🍳</span>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_ui_status():
    return get_runtime_status()


@st.cache_data
def get_dashboard_view(cache_version="datasets-v2"):
    _ = cache_version
    return get_dashboard_data()


status = get_ui_status()
dashboard = get_dashboard_view()
api_catalog = get_api_catalog()
sample_queries = get_sample_queries()
datasets_status = status["datasets"]
spacy_status = status["nlp"]["status"]
nlp_runtime = status["nlp"]["runtime"]
translation_status = status["nlp"]["translation"]
vision_status = status["vision"]

if "messages" not in st.session_state:
    st.session_state.messages = initial_chat_messages()

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "chat_context" not in st.session_state:
    st.session_state.chat_context = initial_chat_context()


def reset_chat_state():
    st.session_state.messages = initial_chat_messages()
    st.session_state.query_history = []
    st.session_state.chat_context = initial_chat_context()


def submit_chat_query(raw_text):
    clean_input = str(raw_text or "").strip()
    if not clean_input:
        return

    st.session_state.query_history.append(clean_input)
    st.session_state.messages.append({"role": "user", "content": compact_text(clean_input)})
    result = handle_chat_message(clean_input, context=st.session_state.chat_context)
    st.session_state.messages.append(
        {"role": "assistant", "content": compact_text(result["response"])}
    )


def render_sidebar():
    st.header("Управление")
    if st.button("Очистить чат", width="stretch"):
        reset_chat_state()
        st.session_state.messages[0]["content"] = (
            "Чат очищен. Опишите блюдо или рецепт на русском, либо загрузите фото блюда."
        )
        st.rerun()

    st.header("История запросов")
    if st.session_state.query_history:
        for idx, query in enumerate(st.session_state.query_history, start=1):
            st.markdown(
                f"<a href='#chat-msg-{idx}' style='text-decoration:none'>{idx}. {query}</a>",
                unsafe_allow_html=True,
            )
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

    st.header("Перевод")
    if translation_status.get("ru_translator_ready") and translation_status.get("en_translator_ready"):
        st.success("RU<->EN переводчик доступен.")
    else:
        st.warning("Переводчик недоступен. Часть рецептов может остаться на английском.")
    if translation_status.get("ru_last_error") or translation_status.get("en_last_error"):
        st.caption(
            "Последние ошибки перевода: "
            f"RU={translation_status.get('ru_last_error') or 'нет'}, "
            f"EN={translation_status.get('en_last_error') or 'нет'}"
        )

    st.header("CV/OCR Статус")
    if vision_status["food11_ready"]:
        st.success("Food-11 найден.")
    else:
        st.warning("Food-11 не найден. Распознавание изображения будет ограничено.")
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
        st.warning("RecipeNLG не найден. Текстовые рецепты и рекомендации будут недоступны.")
    st.caption(f"Food-11 path: {vision_status['food11_path'] or 'не задан'}")
    st.caption(f"RecipeNLG path: {vision_status.get('recipenlg_path') or 'не задан'}")
    st.caption(
        f"RecipeNLG ready: {'да' if datasets_status.get('recipenlg_ready') else 'нет'}, "
        f"Food-11 ready: {'да' if datasets_status.get('food11_ready') else 'нет'}"
    )
    search_index = datasets_status.get("search_index", {})
    if search_index.get("ready"):
        st.success(f"Поисковый индекс RecipeNLG готов ({search_index.get('row_count', 0)} строк).")
    elif search_index.get("needs_rebuild"):
        st.warning("Поисковый индекс RecipeNLG отсутствует или устарел. Первый поиск может занять больше времени.")
    if search_index.get("last_error"):
        st.caption(f"Статус индекса: {search_index['last_error']}")


def render_chat_tab():
    st.subheader("Быстрые сценарии")
    sample_columns = st.columns(3)
    for idx, query in enumerate(sample_queries):
        column = sample_columns[idx % len(sample_columns)]
        if column.button(query, key=f"sample_query_{idx}", width="stretch"):
            submit_chat_query(query)

    user_message_idx = 0
    for message in st.session_state.messages:
        if message["role"] == "user":
            user_message_idx += 1
            st.markdown(f"<div id='chat-msg-{user_message_idx}'></div>", unsafe_allow_html=True)
        with st.chat_message(message["role"]):
            st.write(message["content"])

    chat_text = st.chat_input("Введите запрос")
    if chat_text:
        submit_chat_query(chat_text)
        st.rerun()


def render_vision_result(cached_result):
    vision_result = cached_result.get("vision_result", {})
    if vision_result.get("error"):
        st.error(vision_result["error"])
        return

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
        st.dataframe(candidates_df, width="stretch", hide_index=True)
        st.bar_chart(candidates_df.set_index("class"))

    st.markdown(cached_result.get("response", "Рецепт не найден."))


def render_vision_tab():
    st.subheader("Фото блюда: распознавание и рецепт")
    with st.form("image_analysis_form"):
        uploaded_image = st.file_uploader(
            "Загрузите фото блюда (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"]
        )
        analyze_button = st.form_submit_button(
            "Определить блюдо и предложить рецепт",
            type="primary",
        )

    if uploaded_image is None:
        return

    image_bytes = uploaded_image.getvalue()
    image_hash = hashlib.sha1(image_bytes).hexdigest()
    st.image(image_bytes, caption="Загруженное изображение", width="stretch")

    if analyze_button:
        with st.spinner("Анализирую фото..."):
            pipeline_result = handle_image_message(image_bytes)
        st.session_state[f"vision_result_{image_hash}"] = pipeline_result

    cached_result = st.session_state.get(f"vision_result_{image_hash}")
    if cached_result:
        render_vision_result(cached_result)


def render_dashboard_tab():
    st.subheader("Аналитика датасетов")
    summary = dashboard["summary"]
    dataset_inventory = dashboard.get("dataset_inventory") or get_dataset_inventory()
    external_dataset_data = dashboard.get("external_dataset_data") or {}

    st.info(
        "Этот экран показывает только реальные внешние датасеты проекта: RecipeNLG и Food-11."
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("RecipeNLG строк", summary["recipenlg_rows"])
    metric_cols[1].metric("Food-11 train", summary["food11_train_images"])
    metric_cols[2].metric("Food-11 test", summary["food11_test_images"])
    metric_cols[3].metric("Food-11 классы", summary["food11_classes"])

    dataset_df = pd.DataFrame(
        [
            {"dataset": "RecipeNLG", "records": dataset_inventory["recipenlg_rows"]},
            {"dataset": "Food-11 train images", "records": dataset_inventory["food11_train"]["images"]},
            {"dataset": "Food-11 test images", "records": dataset_inventory["food11_test"]["images"]},
        ]
    )
    recipenlg_preview_df = pd.DataFrame(external_dataset_data.get("recipenlg_preview", []))
    food11_class_df = pd.DataFrame(external_dataset_data.get("food11_class_stats", []))

    st.markdown("**Подключенные датасеты**")
    st.dataframe(dataset_df, width="stretch", hide_index=True)
    st.bar_chart(dataset_df.set_index("dataset"))

    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.markdown("**RecipeNLG: реальные примеры**")
        if not recipenlg_preview_df.empty:
            st.dataframe(
                recipenlg_preview_df[["title_ru", "ingredient_count"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("RecipeNLG preview недоступен.")
    with preview_col2:
        st.markdown("**Food-11: классы train**")
        if not food11_class_df.empty:
            st.dataframe(food11_class_df, width="stretch", hide_index=True)
            st.bar_chart(food11_class_df.set_index("class")[["train_images"]])
        else:
            st.caption("Food-11 stats недоступны.")


def render_api_tab():
    st.subheader("Backend и API")
    st.markdown("**Статус backend**")
    st.json(status)

    st.markdown("**Доступные endpoints**")
    api_df = pd.DataFrame(api_catalog)
    st.dataframe(api_df, width="stretch", hide_index=True)

    st.markdown("**Пример запроса**")
    st.code(
        "curl -X POST http://127.0.0.1:8000/chat "
        "-H 'Content-Type: application/json' "
        "-d '{\"message\":\"похожие на плов\"}'",
        language="bash",
    )


with st.sidebar:
    render_sidebar()

tab_chat, tab_vision, tab_dashboard, tab_api = st.tabs(
    ["Чат", "Фото блюда", "Аналитика", "API/Backend"]
)

with tab_chat:
    render_chat_tab()

with tab_vision:
    render_vision_tab()

with tab_dashboard:
    render_dashboard_tab()

with tab_api:
    render_api_tab()
