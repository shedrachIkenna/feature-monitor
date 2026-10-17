"""HTTP API: answer prompts with GPT-2 Small and record each request's top SAE features.

The model work sits behind an `Engine`, so tests can build the app with a fake one via
`create_app` and never download GPT-2 or the SAE. A background scheduler runs the drift job
(monitor/job.py) every check_interval_seconds and stores each check's scores and any alerts.
The /dashboard endpoints (backend/dashboard.py) read those back for the frontend, which is served
as static files at / after every API route, so the API paths always win. The playground page sends
`record: false`, so its prompts are answered but never stored as monitored traffic.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path as FilePath
from typing import Protocol

import torch
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
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
# The model answers one prompt at a time; at most this many /chat requests may be running or
# waiting at once, and the rest get 429 straight away instead of piling up.
MAX_CHAT_QUEUE_ENV = "FEATURE_MONITOR_MAX_CHAT_QUEUE"
DEFAULT_MAX_CHAT_QUEUE = 4
# When set, recorded prompts (record: true) must carry this secret in the X-Record-Token header,
# so strangers cannot push traffic into the drift window. Unrecorded prompts never need it.
RECORD_TOKEN_ENV = "FEATURE_MONITOR_RECORD_TOKEN"
RECORD_TOKEN_HEADER = "X-Record-Token"
# Set to "1" (the Docker image does) to refuse recorded prompts when no token is set.
REQUIRE_RECORD_TOKEN_ENV = "FEATURE_MONITOR_REQUIRE_RECORD_TOKEN"
# The playground labels one prompt's top 20 features; uncached labels each cost a Neuronpedia call.
MAX_LABEL_IDS = 20
LABEL_IDS_PATTERN = r"^\d+(,\d+)*$"
FRONTEND_DIR = FilePath(__file__).resolve().parent.parent / "frontend"


def extraction_enabled() -> bool:
    """Return False only when the extraction flag is set to \"0\"."""
    return os.environ.get(EXTRACT_ENV, "1") != "0"


def max_chat_queue() -> int:
    """Return the /chat queue cap from its environment variable; it must be at least 1."""
    value = int(os.environ.get(MAX_CHAT_QUEUE_ENV, DEFAULT_MAX_CHAT_QUEUE))
    if value < 1:
        raise ValueError(f"{MAX_CHAT_QUEUE_ENV} must be at least 1, got {value}")
    return value


def check_record_token(expected: str | None, required: bool, given: str | None) -> None:
    """Raise 403 unless a recorded prompt may be stored.

    With no expected token, recording is open unless `required`; otherwise `given` must match.
    """
    if not expected:
        if required:
            raise HTTPException(403, "Recording is disabled on this server.")
        return
    if given is None or not secrets.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(403, f"Recorded prompts need a valid {RECORD_TOKEN_HEADER} header.")


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
    # False for playground prompts: answered and encoded, but never stored, so they cannot enter
    # the drift job's window, the scenario results, or the dashboard's example prompts.
    record: bool = True


class FeatureOut(BaseModel):
    feature_id: int
    activation: float


class ChatResponse(BaseModel):
    reply: str
    request_id: int | None
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
    chat_limit = max_chat_queue()
    record_token = os.environ.get(RECORD_TOKEN_ENV)
    require_record_token = os.environ.get(REQUIRE_RECORD_TOKEN_ENV) == "1"
    # Only touched on the event loop, between awaits, so it needs no lock.
    chat_in_flight = 0

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
        nonlocal chat_in_flight
        if body.record:
            check_record_token(
                record_token, require_record_token, request.headers.get(RECORD_TOKEN_HEADER)
            )
        if chat_in_flight >= chat_limit:
            raise HTTPException(
                429,
                "The model is busy with other prompts; try again in a moment.",
                headers={"Retry-After": "1"},
            )
        engine: Engine = request.app.state.engine
        chat_in_flight += 1
        try:
            result = await run_in_threadpool(engine.run, body.prompt)
            request_id = None
            if body.record:
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
        finally:
            chat_in_flight -= 1
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
    def dashboard_timeline(
        hours: int = Query(24, ge=1, le=168),
        bucket_minutes: int = Query(dashboard.DEFAULT_BUCKET_MINUTES),
    ) -> dict:
        return dashboard.timeline(hours, check_bucket(bucket_minutes))

    @app.get("/dashboard/drifting")
    def dashboard_drifting(request: Request, limit: int = Query(10, ge=1, le=100)) -> dict:
        job, detector = sae_detector(request)
        return dashboard.drifting(job, detector, limit)

    @app.get("/dashboard/feature/{feature_id}")
    def dashboard_feature(
        request: Request,
        feature_id: int = Path(ge=0),
        hours: int = Query(24, ge=1, le=168),
        bucket_minutes: int = Query(dashboard.DEFAULT_BUCKET_MINUTES),
    ) -> dict:
        _, detector = sae_detector(request)
        label = detector.describe(feature_id, [], [])
        return dashboard.feature(feature_id, label, hours, check_bucket(bucket_minutes))

    @app.get("/dashboard/labels")
    def dashboard_labels(request: Request, ids: str = Query(pattern=LABEL_IDS_PATTERN)) -> dict:
        _, detector = sae_detector(request)
        feature_ids = list(dict.fromkeys(int(i) for i in ids.split(",")))
        if len(feature_ids) > MAX_LABEL_IDS:
            raise HTTPException(422, f"at most {MAX_LABEL_IDS} feature ids")
        return {str(f): detector.describe(f, [], []) for f in feature_ids}

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


def check_bucket(bucket_minutes: int) -> int:
    """Return bucket_minutes if the dashboard supports it; 422 otherwise."""
    if bucket_minutes not in dashboard.BUCKET_MINUTES:
        raise HTTPException(422, f"bucket_minutes must be one of {dashboard.BUCKET_MINUTES}")
    return bucket_minutes


def sae_detector(request: Request) -> tuple[DriftJob, Detector]:
    """The running drift job and its SAE detector; 503 when the app was built without them."""
    job: DriftJob | None = request.app.state.job
    detector = next((d for d in job.detectors if d.name == "sae"), None) if job else None
    if job is None or detector is None:
        raise HTTPException(503, "the SAE drift detector is not loaded")
    return job, detector


app = create_app(GPT2Engine, load_drift_job)
