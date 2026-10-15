"""HTTP API: answer prompts with GPT-2 Small and record each request's top SAE features.

The model work sits behind an `Engine`, so tests can build the app with a fake one via
`create_app` and never download GPT-2 or the SAE. A background scheduler runs the drift job
(monitor/job.py) every check_interval_seconds and stores each check's scores and any alerts.
The /dashboard endpoints (backend/dashboard.py) read those back for the frontend.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import torch
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from backend import dashboard, storage
from monitor.config import load_config
from monitor.extract import FeatureHit, extract_features
from monitor.job import Detector, DriftJob, load_drift_job
from monitor.model import MODEL_NAME, load_model_and_sae

MAX_NEW_TOKENS = 40
SEED = 0
# Set to "0" to skip feature extraction, e.g. to measure its latency overhead.
EXTRACT_ENV = "FEATURE_MONITOR_EXTRACT"


def extraction_enabled() -> bool:
    """Return False only when the extraction flag is set to \"0\"."""
    return os.environ.get(EXTRACT_ENV, "1") != "0"


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
    extract: bool

    def run(self, prompt: str) -> EngineResult: ...


class GPT2Engine:
    """GPT-2 Small with its block-8 SAE, serving one request at a time."""

    def __init__(self) -> None:
        self.model, _ = load_model_and_sae()
        self.model_name = MODEL_NAME
        self.device = str(self.model.cfg.device)
        self.extract = extraction_enabled()
        # The seed is process-wide, so concurrent requests would disturb each other's sampling.
        self._lock = threading.Lock()

    def run(self, prompt: str) -> EngineResult:
        """Sample a seeded reply, then extract features from the prompt if enabled."""
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

            hits: list[FeatureHit] = []
            extract_ms = 0.0
            if self.extract:
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


def create_app(
    load_engine: Callable[[], Engine], load_job: Callable[[], DriftJob] | None = None
) -> FastAPI:
    """Build the API; `load_engine` and `load_job` run once at startup. No job, no scheduler."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        storage.init_db()
        app.state.engine = await run_in_threadpool(load_engine)
        app.state.scheduler = None
        app.state.job = None
        if load_job is not None:
            job = await run_in_threadpool(load_job)
            app.state.job = job
            scheduler = BackgroundScheduler()
            # One check at a time. A check that starts late (the server is busy with a request)
            # still runs instead of being skipped, and several missed checks run once.
            scheduler.add_job(
                job.run,
                "interval",
                seconds=load_config().check_interval_seconds,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=None,
            )
            scheduler.start()
            app.state.scheduler = scheduler
        yield
        if app.state.scheduler is not None:
            app.state.scheduler.shutdown(wait=False)

    app = FastAPI(title="Feature Monitor", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        engine: Engine = request.app.state.engine
        return {
            "status": "ok",
            "model": engine.model_name,
            "device": engine.device,
            "extract": engine.extract,
        }

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

    @app.get("/alerts")
    def alerts(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
        return storage.list_alerts(limit)

    @app.get("/dashboard/timeline")
    def dashboard_timeline(hours: int = Query(24, ge=1, le=168)) -> dict:
        return dashboard.timeline(hours)

    @app.get("/dashboard/drifting")
    def dashboard_drifting(request: Request, limit: int = Query(10, ge=1, le=100)) -> dict:
        job, detector = sae_detector(request)
        return dashboard.drifting(job, detector, limit)

    @app.get("/dashboard/feature/{feature_id}")
    def dashboard_feature(
        request: Request,
        feature_id: int = Path(ge=0),
        hours: int = Query(24, ge=1, le=168),
    ) -> dict:
        _, detector = sae_detector(request)
        return dashboard.feature(feature_id, detector.describe(feature_id, [], []), hours)

    return app


def sae_detector(request: Request) -> tuple[DriftJob, Detector]:
    """The running drift job and its SAE detector; 503 when the app was built without them."""
    job: DriftJob | None = request.app.state.job
    detector = next((d for d in job.detectors if d.name == "sae"), None) if job else None
    if job is None or detector is None:
        raise HTTPException(503, "the SAE drift detector is not loaded")
    return job, detector


app = create_app(GPT2Engine, load_drift_job)
