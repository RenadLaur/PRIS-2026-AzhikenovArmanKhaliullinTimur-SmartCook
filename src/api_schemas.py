from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Текст запроса пользователя")


class ChatResponse(BaseModel):
    ok: bool
    query: str
    response: str
    recipe_title: str = ""
    query_bucket: str = ""


class StatusResponse(BaseModel):
    datasets: dict[str, Any]
    nlp: dict[str, Any]
    vision: dict[str, Any]


class HealthResponse(BaseModel):
    ok: bool
    service: str
