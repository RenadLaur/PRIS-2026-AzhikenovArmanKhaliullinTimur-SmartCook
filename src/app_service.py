from functools import lru_cache

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


def compact_text(text):
    return " ".join(str(text or "").split())


@lru_cache(maxsize=1)
def get_data_source():
    return load_graph()


@lru_cache(maxsize=1)
def get_nlp_runtime():
    return warmup_spacy_model()


def get_runtime_status():
    graph_ok = True
    graph_error = ""
    try:
        get_data_source()
    except Exception as exc:  # pragma: no cover - runtime safeguard
        graph_ok = False
        graph_error = str(exc)

    return {
        "graph": {
            "ok": graph_ok,
            "error": graph_error,
        },
        "nlp": {
            "status": get_spacy_status(),
            "runtime": get_nlp_runtime(),
        },
        "vision": get_vision_status(),
    }


def handle_chat_message(text):
    clean_text = str(text or "").strip()
    if not clean_text:
        return {
            "ok": False,
            "query": "",
            "response": "Пустой запрос.",
        }

    try:
        graph = get_data_source()
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return {
            "ok": False,
            "query": clean_text,
            "response": f"Ошибка загрузки базы знаний: {exc}",
        }

    try:
        response = process_text_message(clean_text, graph)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return {
            "ok": False,
            "query": clean_text,
            "response": f"Ошибка обработки запроса: {exc}",
        }

    return {
        "ok": True,
        "query": clean_text,
        "response": compact_text(response),
    }


def handle_image_message(image_bytes):
    if not image_bytes:
        return {
            "ok": False,
            "response": "Пустое изображение.",
            "vision_result": {"error": "Пустое изображение."},
        }

    try:
        graph = get_data_source()
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return {
            "ok": False,
            "response": f"Ошибка загрузки базы знаний: {exc}",
            "vision_result": {"error": f"Ошибка загрузки базы знаний: {exc}"},
        }

    try:
        result = run_image_pipeline(image_bytes, graph)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return {
            "ok": False,
            "response": f"Ошибка анализа изображения: {exc}",
            "vision_result": {"error": f"Ошибка анализа изображения: {exc}"},
        }

    result["ok"] = not bool(result.get("vision_result", {}).get("error"))
    return result

