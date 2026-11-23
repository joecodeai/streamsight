"""Identity galleries: where appearance embeddings of known people are stored and searched.

``InMemoryGallery`` is used for tests and single-process runs. ``MilvusGallery``
persists embeddings in Milvus so identities survive restarts and can be shared
by several pipeline processes (one per GPU, for example).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class GalleryHit:
    global_id: int
    score: float  # cosine similarity
    camera_id: str


class Gallery(ABC):
    @abstractmethod
    def add(self, global_id: int, camera_id: str, embedding: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self, embedding: np.ndarray, top_k: int = 5, exclude_ids: set[int] | None = None
    ) -> list[GalleryHit]:
        """Best match per identity, highest similarity first."""

    @abstractmethod
    def next_global_id(self) -> int: ...


class InMemoryGallery(Gallery):
    """Keeps the most recent ``samples_per_id`` embeddings for each identity."""

    def __init__(self, samples_per_id: int = 20) -> None:
        self.samples_per_id = samples_per_id
        self._vectors: dict[int, list[tuple[str, np.ndarray]]] = {}
        self._counter = 0

    def add(self, global_id: int, camera_id: str, embedding: np.ndarray) -> None:
        samples = self._vectors.setdefault(global_id, [])
        samples.append((camera_id, embedding.astype(np.float32)))
        if len(samples) > self.samples_per_id:
            del samples[0]

    def search(
        self, embedding: np.ndarray, top_k: int = 5, exclude_ids: set[int] | None = None
    ) -> list[GalleryHit]:
        exclude_ids = exclude_ids or set()
        hits = []
        for gid, samples in self._vectors.items():
            if gid in exclude_ids or not samples:
                continue
            mat = np.stack([v for _, v in samples])
            sims = mat @ embedding
            best = int(sims.argmax())
            hits.append(GalleryHit(gid, float(sims[best]), samples[best][0]))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def next_global_id(self) -> int:
        self._counter += 1
        return self._counter

    def __len__(self) -> int:
        return len(self._vectors)


class MilvusGallery(Gallery):
    """Milvus-backed gallery. ``uri`` may be a server (http://host:19530) or a local Milvus Lite file."""

    def __init__(
        self, uri: str, collection: str = "streamsight_reid", dim: int = 512, token: str = ""
    ) -> None:
        from pymilvus import DataType, MilvusClient

        self.client = MilvusClient(uri=uri, token=token)
        self.collection = collection
        if not self.client.has_collection(collection):
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field("pk", DataType.INT64, is_primary=True)
            schema.add_field("global_id", DataType.INT64)
            schema.add_field("camera_id", DataType.VARCHAR, max_length=64)
            schema.add_field("ts", DataType.INT64)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
            index = self.client.prepare_index_params()
            index.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
            self.client.create_collection(collection, schema=schema, index_params=index)
        self.client.load_collection(collection)
        # Millisecond timestamp seed: unique across restarts without a central id service.
        self._counter = int(time.time() * 1000) * 1000

    def add(self, global_id: int, camera_id: str, embedding: np.ndarray) -> None:
        row = {
            "global_id": int(global_id),
            "camera_id": camera_id,
            "ts": int(time.time() * 1000),
            "embedding": embedding.astype(np.float32).tolist(),
        }
        self.client.insert(self.collection, [row])

    def search(
        self, embedding: np.ndarray, top_k: int = 5, exclude_ids: set[int] | None = None
    ) -> list[GalleryHit]:
        flt = ""
        if exclude_ids:
            flt = f"global_id not in [{', '.join(str(i) for i in exclude_ids)}]"
        res = self.client.search(
            self.collection,
            data=[embedding.astype(np.float32).tolist()],
            limit=top_k * 4,
            filter=flt,
            output_fields=["global_id", "camera_id"],
            search_params={"metric_type": "COSINE"},
        )[0]
        best: dict[int, GalleryHit] = {}
        for hit in res:
            ent = hit["entity"]
            gid = int(ent["global_id"])
            score = float(hit["distance"])
            if gid not in best or score > best[gid].score:
                best[gid] = GalleryHit(gid, score, ent["camera_id"])
        return sorted(best.values(), key=lambda h: h.score, reverse=True)[:top_k]

    def next_global_id(self) -> int:
        self._counter += 1
        return self._counter


def create_gallery(cfg, dim: int = 512) -> Gallery:
    if cfg.gallery == "milvus":
        return MilvusGallery(cfg.milvus_uri, cfg.milvus_collection, dim=dim)
    return InMemoryGallery()
