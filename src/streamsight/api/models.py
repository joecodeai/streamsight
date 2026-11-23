from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, BigInteger, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from streamsight.api.db import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    local_track_id: Mapped[int] = mapped_column(Integer)
    global_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    label: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[list] = mapped_column(JSON)
    keypoints: Mapped[list | None] = mapped_column(JSON, nullable=True)
    zone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (Index("ix_events_camera_ts", "camera_id", "ts"),)


class Identity(Base):
    """A person known across cameras. Maintained on ingest so queries do not scan events."""

    __tablename__ = "identities"

    global_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    cameras: Mapped[list] = mapped_column(JSON, default=list)
    sightings: Mapped[int] = mapped_column(Integer, default=0)
