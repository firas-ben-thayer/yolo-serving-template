import argparse
import asyncio
import json
import os
import sys
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="client.cli", description="YOLO serving client CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pred = sub.add_parser("predict", help="Send image for prediction")
    pred.add_argument("image", help="Path to image file")
    pred.add_argument("--url", help="Server base URL (overrides YOLO_SERVER_URL env)", default=os.getenv("YOLO_SERVER_URL", "http://127.0.0.1:8000"))
    pred.add_argument("--api-key", help="API key to send as Bearer token", default=None)
    pred.add_argument("--inproc", help="Run prediction in-process (requires importable app)", action="store_true")
    pred.add_argument("--adapter", help="Adapter to use for inproc predict (stub|yolovx)", default=None)
    pred.add_argument("--weights", help="Path to model weights for inproc predict", default=None)
    pred.add_argument("--async", dest="use_async", action="store_true", help="Use async HTTP client for prediction")

    return p


def _print_json(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def cmd_predict(args: argparse.Namespace) -> int:
    image = args.image
    if args.inproc:
        # in-process prediction
        try:
            from client.inproc import predict_inproc
        except Exception as e:
            print(f"inproc predict unavailable: {e}", file=sys.stderr)
            return 2

        try:
            out = predict_inproc(image, adapter=args.adapter, weights=args.weights)
            _print_json(out)
            return 0
        except Exception as e:
            print(f"inproc prediction failed: {e}", file=sys.stderr)
            return 3

    # HTTP path
    try:
        from client.http import YoloHTTPClient
    except Exception as e:
        print(f"http client unavailable: {e}", file=sys.stderr)
        return 4

    client = YoloHTTPClient(args.url, api_key=args.api_key)

    try:
        if args.use_async:
            out = asyncio.run(client.predict_async(image))
        else:
            out = client.predict(image)
        _print_json(out)
        return 0
    except Exception as e:
        print(f"prediction failed: {e}", file=sys.stderr)
        return 5


def main(argv: Optional[list] = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv)
    if args.cmd == "predict":
        return cmd_predict(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
