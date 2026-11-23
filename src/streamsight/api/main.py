"""StreamSight event API and dashboard.

Run locally:  streamsight-api            (SQLite file ./streamsight.db)
Production:   DATABASE_URL=postgresql://user:pass@host/db streamsight-api
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, make_asgi_app
from sqlalchemy.orm import Session

from streamsight import __version__
from streamsight.api import service
from streamsight.api.db import Database
from streamsight.api.schemas import CameraOut, EventOut, IdentityOut, IngestResult, SummaryOut, TimelineEntry
from streamsight.events import EventBatch

STATIC = Path(__file__).parent / "static"
INGESTED = Counter("streamsight_api_events_ingested_total", "Events written to the database")


def create_app(database_url: str | None = None, api_key: str | None = None) -> FastAPI:
    db = Database(database_url)
    db.create_all()
    api_key = api_key if api_key is not None else os.getenv("STREAMSIGHT_API_KEY") or None

    app = FastAPI(title="StreamSight API", version=__version__)
    app.state.db = db

    def get_db() -> Session:
        yield from db.session()

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if api_key and not (x_api_key and secrets.compare_digest(x_api_key, api_key)):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post(
        "/api/v1/events/batch",
        response_model=IngestResult,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_key)],
        tags=["ingest"],
    )
    def ingest_events(batch: EventBatch, s: Session = Depends(get_db)) -> IngestResult:
        n = service.ingest(s, batch.events)
        INGESTED.inc(n)
        return IngestResult(accepted=n)

    @app.get("/api/v1/cameras", response_model=list[CameraOut], tags=["query"])
    def cameras(s: Session = Depends(get_db)) -> list[dict]:
        return service.list_cameras(s)

    @app.get("/api/v1/events", response_model=list[EventOut], tags=["query"])
    def events(
        camera_id: str | None = None,
        event_type: str | None = None,
        global_id: int | None = None,
        since: datetime | None = None,
        limit: int = Query(100, ge=1, le=1000),
        s: Session = Depends(get_db),
    ):
        return service.list_events(s, camera_id, event_type, global_id, since, limit)

    @app.get("/api/v1/identities", response_model=list[IdentityOut], tags=["query"])
    def identities(
        min_cameras: int = Query(1, ge=1),
        limit: int = Query(100, ge=1, le=1000),
        s: Session = Depends(get_db),
    ):
        return service.list_identities(s, min_cameras, limit)

    @app.get("/api/v1/identities/{global_id}/timeline", response_model=list[TimelineEntry], tags=["query"])
    def timeline(global_id: int, s: Session = Depends(get_db)):
        entries = service.identity_timeline(s, global_id)
        if not entries:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown identity")
        return entries

    @app.get("/api/v1/stats/summary", response_model=SummaryOut, tags=["query"])
    def stats(s: Session = Depends(get_db)) -> dict:
        return service.summary(s)

    app.mount("/metrics", make_asgi_app())
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


def run() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="StreamSight API server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    run()
