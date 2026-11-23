import numpy as np
import pytest

from streamsight.reid.gallery import InMemoryGallery, MilvusGallery
from streamsight.reid.matcher import CrossCameraMatcher
from streamsight.tracking.byte_tracker import Track
from streamsight.tracking.kalman import xyxy_to_xyah

from .conftest import model_path


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def make_track(tid, emb, age=10):
    t = Track(
        track_id=tid,
        mean=np.r_[xyxy_to_xyah(np.array([0, 0, 10, 20])), np.zeros(4)],
        cov=np.eye(8),
        score=0.9,
        class_id=0,
        start_frame=0,
        last_frame=age - 1,
    )
    t.embedding = unit(emb)
    return t


def test_same_person_on_two_cameras_shares_identity():
    m = CrossCameraMatcher(InMemoryGallery(), match_threshold=0.8, min_track_frames=5)
    a = m.assign("cam1", make_track(1, [1, 0.1, 0]))
    b = m.assign("cam2", make_track(7, [1, 0.12, 0]))
    assert a.matched is False
    assert b.matched is True and b.global_id == a.global_id and b.from_camera == "cam1"


def test_different_people_get_different_identities():
    m = CrossCameraMatcher(InMemoryGallery(), match_threshold=0.8, min_track_frames=5)
    a = m.assign("cam1", make_track(1, [1, 0, 0]))
    b = m.assign("cam2", make_track(2, [0, 1, 0]))
    assert a.global_id != b.global_id and not b.matched


def test_identity_visible_in_same_camera_is_not_reused():
    m = CrossCameraMatcher(InMemoryGallery(), match_threshold=0.8, min_track_frames=5)
    first = m.assign("cam1", make_track(1, [1, 0, 0]))
    twin = m.assign("cam1", make_track(2, [1, 0, 0]))  # lookalike in the same view at the same time
    assert twin.global_id != first.global_id

    m.release("cam1", make_track(1, [1, 0, 0]))  # first person leaves
    back = m.assign("cam1", make_track(3, [1, 0.01, 0]))
    assert back.matched


def test_young_tracks_wait_for_a_stable_embedding():
    m = CrossCameraMatcher(InMemoryGallery(), min_track_frames=5)
    assert m.assign("cam1", make_track(1, [1, 0, 0], age=2)) is None


def test_decision_is_made_once_per_track():
    m = CrossCameraMatcher(InMemoryGallery())
    t = make_track(1, [1, 0, 0])
    assert m.assign("cam1", t) is not None
    assert m.assign("cam1", t) is None
    assert t.global_id is not None


def test_milvus_lite_gallery(tmp_path):
    pytest.importorskip("pymilvus")
    try:
        g = MilvusGallery(str(tmp_path / "reid.db"), dim=4)
    except Exception as exc:  # milvus-lite not available on this platform
        pytest.skip(f"milvus lite unavailable: {exc}")
    gid = g.next_global_id()
    g.add(gid, "cam1", unit([1, 0, 0, 0]))
    g.add(gid + 1, "cam2", unit([0, 1, 0, 0]))
    hits = g.search(unit([0.9, 0.1, 0, 0]), top_k=2)
    assert hits[0].global_id == gid and hits[0].score > 0.9
    assert g.search(unit([1, 0, 0, 0]), exclude_ids={gid})[0].global_id == gid + 1


def test_embedder_output_is_normalised():
    from streamsight.reid.embedder import ReIdEmbedder

    emb = ReIdEmbedder(model_path("osnet_x0_25_msmt17.onnx"), device="cpu")
    frame = np.random.default_rng(0).integers(0, 255, (288, 360, 3), dtype=np.uint8)
    out = emb(frame, np.array([[10, 10, 60, 150], [100, 20, 160, 200], [0, 0, 0, 0]], np.float32))
    assert out.shape == (3, emb.dim)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-4)
