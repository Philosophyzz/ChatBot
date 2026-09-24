"""Bounded, transient pet actions and observations. No history side effects."""
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Literal
from core.companion import CompanionDirector

router = APIRouter(prefix="/api/pet")


class Parameter(BaseModel):
    min: float = Field(ge=-10000, le=10000, allow_inf_nan=False)
    max: float = Field(ge=-10000, le=10000, allow_inf_nan=False)
    default: float = Field(default=0, ge=-10000, le=10000, allow_inf_nan=False)


class ActionRequest(BaseModel):
    text: str = Field(default="", max_length=1000)
    persona_id: str = Field(default="", max_length=100)
    capabilities: dict[str, Parameter] = Field(default_factory=dict, max_length=80)


class Observation(BaseModel):
    enabled: bool = False
    scene: Literal["normal", "movie", "music", "quiet"] = "normal"
    busy: bool = False
    protected: bool = False
    pet_idle_s: float = Field(default=0, ge=0, le=1e9, allow_inf_nan=False)
    idle_s: float = Field(default=0, ge=0, le=1e9, allow_inf_nan=False)
    activity_ticks: int = Field(default=0, ge=0, le=10000)
    mouse_distance: float = Field(default=0, ge=0, le=1e7, allow_inf_nan=False)
    disk_kbps: float = Field(default=0, ge=0, le=1e9, allow_inf_nan=False)
    app: str = Field(default="", max_length=120)
    title: str = Field(default="", max_length=200)
    media_title: str = Field(default="", max_length=160)
    media_artist: str = Field(default="", max_length=100)
    media_playing: bool = False
    media_status: str = Field(default="unavailable", max_length=40)
    audio_rms: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    audio_peak: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)


class ObserveRequest(ActionRequest):
    observation: Observation
    image: str = Field(default="", max_length=1_200_000)


def director(request):
    from api.routes import _app
    app = _app(request)
    if not hasattr(app, "companion"):
        app.companion = CompanionDirector(app)
    return app.companion


@router.post("/action")
async def action(request: Request, payload: ActionRequest):
    return await director(request).action(payload.text, {k: v.model_dump() for k, v in payload.capabilities.items()}, payload.persona_id)


@router.post("/observe")
async def observe(request: Request, payload: ObserveRequest):
    return await director(request).observe(payload.observation.model_dump(), payload.image, payload.persona_id,
                                          {k: v.model_dump() for k, v in payload.capabilities.items()})


@router.get("/awareness")
async def awareness(request: Request):
    return director(request).last
