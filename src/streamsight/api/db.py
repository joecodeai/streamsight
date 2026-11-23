"""Database setup. PostgreSQL in production; SQLite works for local runs and tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def make_engine(url: str | None = None) -> Engine:
    url = url or os.getenv("DATABASE_URL", "sqlite:///./streamsight.db")
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(pool_size=10, max_overflow=20)
    return create_engine(url, **kwargs)


class Database:
    def __init__(self, url: str | None = None) -> None:
        self.engine = make_engine(url)
        self.SessionLocal = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        from streamsight.api import models  # noqa: F401  (register tables)

        Base.metadata.create_all(self.engine)

    def session(self) -> Iterator[Session]:
        with self.SessionLocal() as s:
            yield s
