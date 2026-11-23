"""Write and query logic, kept out of the route handlers so it can be tested directly."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from streamsight.api.models import Camera, Event, Identity
from streamsight.events import TrackEvent

ONLINE_WINDOW = timedelta(seconds=30)


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def ingest(db: Session, events: list[TrackEvent]) -> int:
    if not events:
        return 0
    now = datetime.now(timezone.utc)

    cam_ids = {e.camera_id for e in events}
    existing = {c.id: c for c in db.scalars(select(Camera).where(Camera.id.in_(cam_ids)))}
    for cid in cam_ids:
        if cid in existing:
            existing[cid].last_seen = now
        else:
            db.add(Camera(id=cid, name=cid, first_seen=now, last_seen=now))
    db.flush()

    db.add_all(
        Event(
            camera_id=e.camera_id,
            ts=_utc(e.ts),
            event_type=e.event_type,
            local_track_id=e.local_track_id,
            global_id=e.global_id,
            label=e.label,
            confidence=e.confidence,
            bbox=e.bbox,
            keypoints=e.keypoints,
            zone=e.zone,
            attributes=e.attributes,
        )
        for e in events
    )

    by_gid: dict[int, list[TrackEvent]] = {}
    for e in events:
        if e.global_id is not None:
            by_gid.setdefault(e.global_id, []).append(e)
    if by_gid:
        known = {i.global_id: i for i in db.scalars(select(Identity).where(Identity.global_id.in_(by_gid)))}
        for gid, evs in by_gid.items():
            ts = [_utc(e.ts) for e in evs]
            cams = sorted({e.camera_id for e in evs})
            ident = known.get(gid)
            if ident is None:
                db.add(
                    Identity(
                        global_id=gid, first_seen=min(ts), last_seen=max(ts), cameras=cams, sightings=len(evs)
                    )
                )
            else:
                ident.last_seen = max(_utc(ident.last_seen), max(ts))
                ident.cameras = sorted(set(ident.cameras) | set(cams))
                ident.sightings += len(evs)
    db.commit()
    return len(events)


def _active_track_counts(db: Session, since: datetime) -> dict[str, int]:
    """Tracks with a start/update in the window and no end, per camera."""
    started = (
        select(Event.camera_id, Event.local_track_id)
        .where(Event.ts >= since, Event.event_type.in_(["track_start", "track_update", "reid_match"]))
        .distinct()
    )
    ended = select(Event.camera_id, Event.local_track_id).where(
        Event.ts >= since, Event.event_type == "track_end"
    )
    ended_set = {(c, t) for c, t in db.execute(ended)}
    counts: dict[str, int] = {}
    for cam, tid in db.execute(started):
        if (cam, tid) not in ended_set:
            counts[cam] = counts.get(cam, 0) + 1
    return counts


def list_cameras(db: Session) -> list[dict]:
    now = datetime.now(timezone.utc)
    recent = dict(
        db.execute(
            select(Event.camera_id, func.count())
            .where(Event.ts >= now - timedelta(minutes=5))
            .group_by(Event.camera_id)
        ).all()
    )
    active = _active_track_counts(db, now - timedelta(seconds=10))
    out = []
    for c in db.scalars(select(Camera).order_by(Camera.id)):
        out.append(
            {
                "id": c.id,
                "name": c.name,
                "first_seen": c.first_seen,
                "last_seen": c.last_seen,
                "online": now - _utc(c.last_seen) <= ONLINE_WINDOW,
                "events_last_5m": recent.get(c.id, 0),
                "active_tracks": active.get(c.id, 0),
            }
        )
    return out


def list_events(
    db: Session,
    camera_id: str | None = None,
    event_type: str | None = None,
    global_id: int | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[Event]:
    q = select(Event)
    conds = []
    if camera_id:
        conds.append(Event.camera_id == camera_id)
    if event_type:
        conds.append(Event.event_type == event_type)
    if global_id is not None:
        conds.append(Event.global_id == global_id)
    if since:
        conds.append(Event.ts >= _utc(since))
    if conds:
        q = q.where(and_(*conds))
    return list(db.scalars(q.order_by(Event.ts.desc(), Event.id.desc()).limit(limit)))


def list_identities(db: Session, min_cameras: int = 1, limit: int = 100) -> list[Identity]:
    rows = db.scalars(select(Identity).order_by(Identity.last_seen.desc()).limit(limit * 5))
    return [i for i in rows if len(i.cameras) >= min_cameras][:limit]


def identity_timeline(db: Session, global_id: int) -> list[dict]:
    rows = db.execute(
        select(Event.camera_id, Event.local_track_id, func.min(Event.ts), func.max(Event.ts))
        .where(Event.global_id == global_id)
        .group_by(Event.camera_id, Event.local_track_id)
        .order_by(func.min(Event.ts))
    ).all()
    return [
        {"camera_id": cam, "local_track_id": tid, "first_seen": first, "last_seen": last}
        for cam, tid, first, last in rows
    ]


def summary(db: Session) -> dict:
    now = datetime.now(timezone.utc)
    cams = list_cameras(db)
    identities = list(db.scalars(select(Identity)))

    occupancy: dict[str, int] = {}
    zone_events = db.execute(
        select(Event.camera_id, Event.zone, Event.local_track_id, Event.event_type)
        .where(Event.zone.is_not(None), Event.ts >= now - timedelta(minutes=10))
        .order_by(Event.ts, Event.id)
    ).all()
    inside: dict[tuple[str, str], set[int]] = {}
    for cam, zone, tid, kind in zone_events:
        key = (cam, zone)
        s = inside.setdefault(key, set())
        if kind == "zone_enter":
            s.add(tid)
        else:
            s.discard(tid)
    for (cam, zone), s in inside.items():
        occupancy[f"{cam}/{zone}"] = len(s)

    falls = db.scalar(
        select(func.count()).where(Event.event_type == "fall_suspected", Event.ts >= now - timedelta(hours=1))
    )
    return {
        "cameras": len(cams),
        "cameras_online": sum(c["online"] for c in cams),
        "events_last_5m": sum(c["events_last_5m"] for c in cams),
        "identities_total": len(identities),
        "identities_multi_camera": sum(1 for i in identities if len(i.cameras) > 1),
        "active_tracks": sum(c["active_tracks"] for c in cams),
        "zone_occupancy": occupancy,
        "falls_last_hour": falls or 0,
    }
