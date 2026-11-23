"""Download and export every model StreamSight uses into ./models.

* YOLO11 detector and YOLO11 pose model -> ONNX with a dynamic batch axis (Ultralytics)
* OSNet x0.25 person re-identification model (MSMT17), batch axis made dynamic

Usage:
    python scripts/export_models.py               # yolo11n + yolo11n-pose + osnet
    python scripts/export_models.py --size s      # yolo11s variants instead
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

MODELS = Path("models")
OSNET_URL = "https://huggingface.co/kornia/osnet/resolve/main/osnet_x0_25_msmt17.onnx"


def export_yolo(weights: str, imgsz: int, opset: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(weights)
    exported = Path(model.export(format="onnx", dynamic=True, simplify=True, imgsz=imgsz, opset=opset))
    target = MODELS / exported.name
    shutil.move(str(exported), target)
    print(f"exported {target}")
    return target


def make_batch_dynamic(path: Path) -> None:
    """Rewrite a fixed batch dimension (e.g. 16) as a symbolic 'batch' axis."""
    import onnx

    model = onnx.load(str(path))
    for tensor in list(model.graph.input) + list(model.graph.output):
        dim = tensor.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch"
    # Reshape nodes with a hard-coded batch size would break; OSNet flattens with -1 so this is safe.
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def fetch_osnet() -> Path:
    target = MODELS / "osnet_x0_25_msmt17.onnx"
    if not target.exists():
        print(f"downloading {OSNET_URL}")
        urllib.request.urlretrieve(OSNET_URL, target)
    try:
        make_batch_dynamic(target)
        print(f"{target}: batch axis is dynamic")
    except Exception as exc:  # keep the static model; the embedder pads batches instead
        print(f"{target}: kept static batch ({exc})")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--size", default="n", choices=list("nsmlx"), help="YOLO11 model size")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--skip-reid", action="store_true")
    args = parser.parse_args()

    MODELS.mkdir(exist_ok=True)
    export_yolo(f"yolo11{args.size}.pt", args.imgsz, args.opset)
    if not args.skip_pose:
        export_yolo(f"yolo11{args.size}-pose.pt", args.imgsz, args.opset)
    if not args.skip_reid:
        fetch_osnet()
    for pt in Path(".").glob("yolo11*.pt"):
        shutil.move(str(pt), MODELS / pt.name)


if __name__ == "__main__":
    main()
