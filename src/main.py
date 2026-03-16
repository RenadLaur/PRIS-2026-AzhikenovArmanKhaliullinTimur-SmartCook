import hashlib

import streamlit as st

try:
    from .knowledge_graph import load_graph
    from .logic import process_text_message
    from .nlp import get_spacy_status, warmup_spacy_model
    from .pipeline import run_image_pipeline
    from .vision import get_vision_status
except ImportError:
    from knowledge_graph import load_graph
    from logic import process_text_message
    from nlp import get_spacy_status, warmup_spacy_model
    from pipeline import run_image_pipeline
    from vision import get_vision_status


st.set_page_config(page_title="SmartCook Chat", page_icon="🤖")
st.title("AI Assistant")
st.write("Спросите термин из базы знаний (рецепт, ингредиент или аллерген).")


@st.cache_resource
def get_data_source():
    return load_graph()


@st.cache_resource
def get_nlp_runtime():
    return warmup_spacy_model()


def _compact(text):
    return " ".join(str(text).split())


data_source = None
data_source_error = None
try:
    data_source = get_data_source()
except Exception as exc:  # pragma: no cover - UI safeguard
    data_source_error = str(exc)

spacy_status = get_spacy_status()
nlp_runtime = get_nlp_runtime()
vision_status = get_vision_status()

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

if data_source_error:
    st.error(f"Ошибка загрузки базы знаний: {data_source_error}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

st.divider()
st.subheader("Фото блюда: распознавание и рецепт")
uploaded_image = st.file_uploader(
    "Загрузите фото блюда (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"]
)
if uploaded_image is not None:
    image_bytes = uploaded_image.getvalue()
    image_hash = hashlib.sha1(image_bytes).hexdigest()
    st.image(image_bytes, caption="Загруженное изображение", width="stretch")

    analyze_button = st.button("Определить блюдо и предложить рецепт", type="primary")
    if analyze_button:
        if data_source is None:
            st.error("База знаний не загружена. Невозможно подобрать рецепт.")
        else:
            with st.spinner("Анализирую фото..."):
                pipeline_result = run_image_pipeline(image_bytes, data_source)
            st.session_state[f"vision_result_{image_hash}"] = pipeline_result

    cached_result = st.session_state.get(f"vision_result_{image_hash}")
    if cached_result:
        vision_result = cached_result.get("vision_result", {})
        if vision_result.get("error"):
            st.error(vision_result["error"])
        else:
            label = vision_result.get("predicted_label") or "не определено"
            confidence = vision_result.get("confidence", 0.0)
            st.markdown(f"**Предсказанный класс:** `{label}` (confidence: `{confidence}`)")
            if vision_result.get("top_candidates"):
                top_text = ", ".join(
                    f"{name}: {score}" for name, score in vision_result["top_candidates"]
                )
                st.caption(f"Top кандидаты: {top_text}")
            if vision_result.get("ocr_warning"):
                st.warning(vision_result["ocr_warning"])
            if vision_result.get("classification_note"):
                st.warning(vision_result["classification_note"])
            if vision_result.get("ocr_text"):
                st.caption(f"OCR текст: {vision_result['ocr_text']}")
            if vision_result.get("ingredient_hints"):
                hints = ", ".join(vision_result["ingredient_hints"])
                st.caption(f"Найденные ингредиентные подсказки: {hints}")
            st.markdown(cached_result.get("response", "Рецепт не найден."))

if user_input := st.chat_input("Введите ваш запрос..."):
    clean_input = user_input.strip()
    if clean_input:
        st.session_state.query_history.append(clean_input)

    st.session_state.messages.append({"role": "user", "content": _compact(user_input)})
    with st.chat_message("user"):
        st.write(_compact(user_input))

    if data_source is None:
        bot_response = "Ошибка: база знаний не загружена. Проверьте консоль сервера."
    else:
        try:
            bot_response = process_text_message(clean_input, data_source)
        except Exception as exc:  # pragma: no cover - UI safeguard
            bot_response = f"Ошибка обработки запроса: {exc}"

    compact_response = _compact(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": compact_response})
    with st.chat_message("assistant"):
        st.write(compact_response)
