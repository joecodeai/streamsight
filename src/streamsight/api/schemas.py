from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict


def _as_utc(dt: datetime) -> datetime:
    # SQLite returns naive datetimes; everything is stored in UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: str
    ts: UtcDatetime
    event_type: str
    local_track_id: int
    global_id: int | None
    label: str
    confidence: float
    bbox: list[float]
    zone: str | None
    attributes: dict


class CameraOut(BaseModel):
    id: str
    name: str
    first_seen: UtcDatetime
    last_seen: UtcDatetime
    online: bool
    events_last_5m: int
    active_tracks: int


class IdentityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    global_id: int
    first_seen: UtcDatetime
    last_seen: UtcDatetime
    cameras: list[str]
    sightings: int


class TimelineEntry(BaseModel):
    camera_id: str
    first_seen: UtcDatetime
    last_seen: UtcDatetime
    local_track_id: int


class SummaryOut(BaseModel):
    cameras: int
    cameras_online: int
    events_last_5m: int
    identities_total: int
    identities_multi_camera: int
    active_tracks: int
    zone_occupancy: dict[str, int]
    falls_last_hour: int


class IngestResult(BaseModel):
    accepted: int
