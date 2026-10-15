"""HTTP API: answer prompts with GPT-2 Small and record each request's top SAE features.

The model work sits behind an `Engine`, so tests can build the app with a fake one via
`create_app` and never download GPT-2 or the SAE.
"""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import torch
from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from backend import storage
from monitor.extract import FeatureHit, extract_features
from monitor.model import MODEL_NAME, load_model_and_sae

MAX_NEW_TOKENS = 40
SEED = 0


@dataclass(frozen=True)
class EngineResult:
    """A generated reply, the prompt's top features, and how long each step took."""

    reply: str
    hits: list[FeatureHit]
    gen_ms: float
    extract_ms: float


class Engine(Protocol):
    """Anything that can answer a prompt and extract its features."""

    model_name: str
    device: str

    def run(self, prompt: str) -> EngineResult: ...


class GPT2Engine:
    """GPT-2 Small with its block-8 SAE, serving one request at a time."""

    def __init__(self) -> None:
        self.model, _ = load_model_and_sae()
        self.model_name = MODEL_NAME
        self.device = str(self.model.cfg.device)
        # The seed is process-wide, so concurrent requests would disturb each other's sampling.
        self._lock = threading.Lock()

    def run(self, prompt: str) -> EngineResult:
        """Sample a seeded reply, then extract features from the prompt."""
        with self._lock:
            start = time.perf_counter()
            torch.manual_seed(SEED)
            prompt_length = self.model.to_tokens(prompt, prepend_bos=True).shape[1]
            tokens = self.model.generate(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                verbose=False,
                return_type="tokens",
            )
            reply = self.model.to_string(tokens[0, prompt_length:])
            gen_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            hits = extract_features(prompt)
            extract_ms = (time.perf_counter() - start) * 1000
        return EngineResult(reply, hits, gen_ms, extract_ms)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    scenario: str | None = None


class FeatureOut(BaseModel):
    feature_id: int
    activation: float


class ChatResponse(BaseModel):
    reply: str
    request_id: int
    top_features: list[FeatureOut]


def create_app(load_engine: Callable[[], Engine]) -> FastAPI:
    """Build the API; `load_engine` runs once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        storage.init_db()
        app.state.engine = await run_in_threadpool(load_engine)
        yield

    app = FastAPI(title="Feature Monitor", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        engine: Engine = request.app.state.engine
        return {"status": "ok", "model": engine.model_name, "device": engine.device}

    @app.post("/chat")
    async def chat(body: ChatRequest, request: Request) -> ChatResponse:
        engine: Engine = request.app.state.engine
        result = await run_in_threadpool(engine.run, body.prompt)
        request_id = await run_in_threadpool(
            storage.save_request,
            body.prompt,
            result.reply,
            engine.model_name,
            result.hits,
            result.gen_ms,
            result.extract_ms,
            body.scenario,
        )
        return ChatResponse(
            reply=result.reply,
            request_id=request_id,
            top_features=[
                FeatureOut(feature_id=h.feature_id, activation=h.activation) for h in result.hits
            ],
        )

    @app.get("/features/recent")
    def features_recent(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
        requests = storage.recent_requests(limit)
        hits = storage.hits_for_requests([r["id"] for r in requests])
        return [{**r, "features": hits[r["id"]]} for r in requests]

    @app.get("/features/stats")
    def features_stats(minutes: int = Query(60, ge=1, le=10080)) -> dict:
        until = datetime.now(UTC)
        since = until - timedelta(minutes=minutes)
        counts = storage.feature_counts(since, until)
        return {
            "since": storage.format_timestamp(since),
            "until": storage.format_timestamp(until),
            "request_count": storage.request_count(since, until),
            "counts": dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)),
        }

    return app


app = create_app(GPT2Engine)
