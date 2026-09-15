"""Task 1 adapter for a chat-completions-compatible HTTP endpoint (stdlib only)."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark.tasks.task1 import build_messages, parse_prediction  # noqa: E402


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Endpoint must be exact; do not forward credentials to a redirect target.
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("ALPS_API_URL"),
                        help="Full endpoint URL, including /v1/chat/completions if required.")
    parser.add_argument("--model", default=os.environ.get("ALPS_MODEL"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--temperature", type=float,
                        help="Omitted by default; set only if your provider supports it.")
    parser.add_argument("--max-tokens", type=int,
                        help="Optional provider-supported max_tokens parameter.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the request body without making a network call (standalone only).")
    args = parser.parse_args()
    if not args.model:
        parser.error("Set ALPS_MODEL or --model.")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite.")
    if args.temperature is not None and not math.isfinite(args.temperature):
        parser.error("--temperature must be finite.")
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max-tokens must be positive.")
    try:
        row = json.load(sys.stdin)
        if not isinstance(row, dict):
            raise ValueError("stdin must contain one model_input object.")
        body = {"model": args.model, "messages": build_messages(row)}
        if args.temperature is not None:
            body["temperature"] = args.temperature
        if args.max_tokens is not None:
            body["max_tokens"] = args.max_tokens
        if args.dry_run:
            json.dump(body, sys.stdout, ensure_ascii=True)
            return
        endpoint = urllib.parse.urlsplit(args.url or "")
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            raise ValueError("Set ALPS_API_URL or --url to the full HTTP(S) completion endpoint.")
        if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise ValueError("Use ALPS_API_KEY for credentials, not URL credentials or query parameters.")
        key = os.environ.get("ALPS_API_KEY", "")
        local = endpoint.hostname in {"localhost", "127.0.0.1", "::1"}
        if not local and endpoint.scheme != "https":
            raise ValueError("Remote endpoints require HTTPS; HTTP is supported for loopback servers.")
        if not local and not key:
            raise ValueError("Set ALPS_API_KEY for the remote endpoint.")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            args.url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=headers,
        )
        opener = urllib.request.build_opener(NoRedirect())
        try:
            with opener.open(request, timeout=args.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise ValueError(f"Model API returned HTTP {exc.code}; check endpoint, access, and quota.") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ValueError("Model API connection failed or timed out.") from None
        except (json.JSONDecodeError, UnicodeError):
            raise ValueError("Model API returned an invalid JSON response.") from None
        try:
            choice = result["choices"][0]
            if choice.get("finish_reason") not in ("stop", None):
                raise ValueError("Model did not finish normally (possibly truncated, refused, or tool output).")
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ValueError("Expected choices[0].message.content in the API response.") from None
        prediction = parse_prediction(content, row["benchmark_id"])
        json.dump(prediction, sys.stdout, ensure_ascii=True)
        sys.stdout.write("\n")
    except (ValueError, UnicodeError) as exc:
        print(f"Task 1 adapter: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
