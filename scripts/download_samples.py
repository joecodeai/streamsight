"""Download sample multi-camera footage into data/videos and convert it to H.264 MP4.

Default: the EPFL "campus" sequence, three synchronised, overlapping views of the same
people (good for testing cross-camera re-identification). The originals are Indeo 5 AVI,
which NVDEC cannot decode, so they are re-encoded to H.264, the codec real IP cameras
stream. The dataset is published by CVLab, EPFL for research use; see
https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/ for its terms.

Usage:
    python scripts/download_samples.py
    python scripts/download_samples.py --scene terrace1   # if available
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path

BASE = "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www"
OUT = Path("data/videos")


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)


def to_h264(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to convert the videos")
    print(f"encoding {dst}")
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(src),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-g", "50", "-movflags", "+faststart", "-an", str(dst),
        ],
        check=True,
    )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", default="campus4")
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--keep-avi", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for v in range(args.views):
        avi = OUT / f"{args.scene}-c{v}.avi"
        fetch(f"{BASE}/{avi.name}", avi)
        to_h264(avi, avi.with_suffix(".mp4"))
        if not args.keep_avi:
            avi.unlink()
    print("ready:", ", ".join(str(p) for p in sorted(OUT.glob(f"{args.scene}-c*.mp4"))))


if __name__ == "__main__":
    main()
