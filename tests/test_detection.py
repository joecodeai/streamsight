import numpy as np
import pytest

from streamsight.detection.postprocess import batched_nms, box_iou, decode_yolo, nms
from streamsight.detection.preprocess import letterbox, letterbox_batch, scale_boxes, scale_keypoints


def test_letterbox_keeps_aspect_ratio_and_maps_back():
    img = np.zeros((480, 640, 3), np.uint8)
    canvas, info = letterbox(img, 640)
    assert canvas.shape == (640, 640, 3)
    assert info.scale == pytest.approx(1.0)
    assert info.pad_y == 80 and info.pad_x == 0

    # a box drawn in network space maps back to the original image
    net_box = np.array([[100, 180, 200, 280]], np.float32)
    orig = scale_boxes(net_box, info)
    np.testing.assert_allclose(orig, [[100, 100, 200, 200]])


def test_letterbox_batch_layout():
    imgs = [np.full((360, 288, 3), 255, np.uint8), np.zeros((100, 400, 3), np.uint8)]
    batch, infos = letterbox_batch(imgs, 320)
    assert batch.shape == (2, 3, 320, 320)
    assert batch.dtype == np.float32
    assert batch.min() >= 0.0 and batch.max() <= 1.0
    assert len(infos) == 2


def test_scale_keypoints_clips_to_image():
    _, info = letterbox(np.zeros((100, 200, 3), np.uint8), 200)
    kpts = np.array([[[10, 50, 0.9], [500, 500, 0.1]]], np.float32)
    out = scale_keypoints(kpts, info)
    assert out[0, 1, 0] == 200 and out[0, 1, 1] == 100
    assert out[0, 0, 2] == pytest.approx(0.9)


def test_iou_and_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    iou = box_iou(boxes[:1], boxes)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 2] == 0
    keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [0, 2]


def test_batched_nms_does_not_suppress_across_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    keep = batched_nms(boxes, np.array([0.9, 0.8], np.float32), np.array([0, 1]), 0.5)
    assert sorted(keep.tolist()) == [0, 1]


def _fake_output(num_classes: int, anchors: list[tuple], kpt=0) -> np.ndarray:
    """Build a (1, C, N) YOLO output from (cx, cy, w, h, class, score) tuples."""
    c = 4 + num_classes + kpt * 3
    out = np.zeros((1, c, len(anchors)), np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(anchors):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + cls, i] = score
        if kpt:
            out[0, 4 + num_classes :, i] = np.tile([cx, cy, 0.9], kpt)
    return out


def test_decode_filters_confidence_class_and_duplicates():
    raw = _fake_output(
        80,
        [
            (100, 100, 20, 40, 0, 0.9),  # person
            (101, 101, 20, 40, 0, 0.8),  # duplicate person -> NMS
            (300, 300, 50, 50, 2, 0.95),  # car -> filtered by class
            (500, 500, 20, 20, 0, 0.1),  # low confidence
        ],
    )
    ((boxes, scores, cls, kpts),) = decode_yolo(raw, conf_threshold=0.3, iou_threshold=0.5, classes=[0])
    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0], [90, 80, 110, 120])
    assert cls.tolist() == [0]
    assert kpts is None


def test_decode_pose_head():
    raw = _fake_output(1, [(100, 100, 20, 40, 0, 0.9)], kpt=17)
    ((boxes, _, _, kpts),) = decode_yolo(raw, conf_threshold=0.3, num_keypoints=17)
    assert kpts.shape == (1, 17, 3)
    assert kpts[0, 0].tolist() == pytest.approx([100, 100, 0.9])


def test_decode_rejects_bad_shape():
    with pytest.raises(ValueError):
        decode_yolo(np.zeros((84, 10), np.float32))
