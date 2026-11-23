from __future__ import annotations

import argparse
import json
import logging
import signal

from streamsight.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the StreamSight multi-camera pipeline.")
    parser.add_argument("-c", "--config", default="configs/pipeline.yaml")
    parser.add_argument("--duration", type=float, help="stop after N seconds")
    parser.add_argument("--max-frames", type=int, help="stop after N frames in total")
    parser.add_argument("--save-dir", help="write annotated video per camera here")
    parser.add_argument("--no-api", action="store_true", help="do not send events to the API")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.save_dir:
        cfg.output.save_dir = args.save_dir
    if args.no_api:
        cfg.events.api_url = None

    from streamsight.pipeline.runner import Pipeline

    pipeline = Pipeline(cfg)
    signal.signal(signal.SIGINT, lambda *_: pipeline.stop())
    signal.signal(signal.SIGTERM, lambda *_: pipeline.stop())
    summary = pipeline.run(duration_s=args.duration, max_frames=args.max_frames)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
