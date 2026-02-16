import streamlit as st

try:
    from .knowledge_graph import load_graph
    from .logic import process_text_message
except ImportError:
    from knowledge_graph import load_graph
    from logic import process_text_message


st.set_page_config(page_title="SmartCook Chat", page_icon="🤖")
st.title("AI Assistant")
st.write("Спросите термин из базы знаний (рецепт, ингредиент или аллерген).")


@st.cache_resource
def get_data_source():
    return load_graph()


data_source = get_data_source()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Привет! Я бот SmartCook.\n"
                "Введите название рецепта, ингредиента или аллергена."
            ),
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
                "content": (
                    "Чат очищен.\n"
                    "Введите название рецепта, ингредиента или аллергена."
                ),
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Введите ваш запрос..."):
    clean_input = user_input.strip()
    if clean_input:
        st.session_state.query_history.append(clean_input)

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    bot_response = process_text_message(clean_input, data_source)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
