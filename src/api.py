from fastapi import FastAPI, File, UploadFile

try:
    from .app_service import handle_chat_message, handle_image_message, get_runtime_status
    from .api_schemas import ChatRequest, ChatResponse, HealthResponse, StatusResponse
except ImportError:
    from app_service import handle_chat_message, handle_image_message, get_runtime_status
    from api_schemas import ChatRequest, ChatResponse, HealthResponse, StatusResponse


def create_app():
    app = FastAPI(
        title="SmartCook API",
        version="1.0.0",
        description="REST API для SmartCook: чат, NLP/CV статусы и анализ изображений.",
    )

    @app.get("/health", response_model=HealthResponse)
    def healthcheck():
        return {"ok": True, "service": "smartcook-api"}

    @app.get("/status", response_model=StatusResponse)
    def status():
        return get_runtime_status()

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest):
        return handle_chat_message(payload.message)

    @app.post("/image/analyze")
    async def analyze_image(file: UploadFile = File(...)):
        image_bytes = await file.read()
        result = handle_image_message(image_bytes)
        return result

    return app


app = create_app()
