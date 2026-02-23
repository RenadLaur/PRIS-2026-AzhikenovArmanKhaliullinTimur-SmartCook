import streamlit as st

try:
    from .knowledge_graph import load_graph
    from .logic import process_text_message
    from .nlp import get_spacy_status
except ImportError:
    from knowledge_graph import load_graph
    from logic import process_text_message
    from nlp import get_spacy_status


st.set_page_config(page_title="SmartCook Chat", page_icon="🤖")
st.title("AI Assistant")
st.write("Спросите термин из базы знаний (рецепт, ингредиент или аллерген).")


@st.cache_resource
def get_data_source():
    return load_graph()


def _compact(text):
    return " ".join(str(text).split())


data_source = None
data_source_error = None
try:
    data_source = get_data_source()
except Exception as exc:  # pragma: no cover - UI safeguard
    data_source_error = str(exc)

spacy_status = get_spacy_status()

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
    else:
        st.success("spaCy и модель доступны.")

    st.caption(f"Python: {spacy_status['python_executable']}")
    st.caption("Рекомендуемый запуск: `.venv/bin/streamlit run src/main.py`")

if data_source_error:
    st.error(f"Ошибка загрузки базы знаний: {data_source_error}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

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
