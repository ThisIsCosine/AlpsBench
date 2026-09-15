import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from src.benchmark import data
from src.benchmark.reports import read_jsonl, write_jsonl
from src.benchmark.runner import run_public_evaluation
from src.benchmark.tasks.task1 import build_messages, extraction_prompt, parse_prediction


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapter_example" / "task1_api_adapter.py"
MEMORY = {
    "type": "direct", "label": "Preferences/Food", "label_suggestion": None,
    "value": "喜欢素食", "confidence": 0.95, "evidence_text": "我一直喜欢素食。",
}
ROW = {
    "benchmark_id": "fixture-session", "task": "task1",
    "input": {"dialogue": [{"role": "user", "text": MEMORY["evidence_text"]}],
              "metadata": {"secret": "DO_NOT_SEND_METADATA"}},
    "gold": {"secret": "DO_NOT_SEND_GOLD"},
}


@pytest.fixture
def api(monkeypatch):
    state = {"requests": [], "status": 200, "finish_reason": "stop",
             "content": json.dumps({"memory_items": [MEMORY]}, ensure_ascii=False)}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, body, self.headers.get("Authorization")))
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {"choices": [{"finish_reason": state["finish_reason"],
                                     "message": {"content": state["content"]}}]}
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ALPS_API_URL", f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
    monkeypatch.setenv("ALPS_MODEL", "fixture-model")
    monkeypatch.setenv("ALPS_API_KEY", "fixture-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    try:
        yield state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def invoke(row=ROW, *args):
    return subprocess.run([sys.executable, str(ADAPTER), *args],
                          input=json.dumps(row), capture_output=True, text=True, timeout=10)


def test_original_prompt_fingerprint():
    # SHA-256 of DEFAULT_EXTRACT_PROMPT in the maintainer's supplied extraction script.
    assert hashlib.sha256(extraction_prompt().encode()).hexdigest() == (
        "a9cecb7416c11569c1efc96cd98f44e4eb3ffd259a080bbecd8df2d3d94ec090"
    )


def test_messages_only_contain_prompt_and_dialogue():
    messages = build_messages(ROW)
    assert messages[0] == {"role": "system", "content": extraction_prompt()}
    assert json.loads(messages[1]["content"]) == {"dialogue": ROW["input"]["dialogue"]}
    fallback = {"task": "task1", "benchmark_id": "s", "input": {
        "sessions": [{"turns": [{"role": "user", "text": "hello", "extra": "private"}]}],
    }}
    assert json.loads(build_messages(fallback)[1]["content"]) == {
        "dialogue": [{"role": "user", "text": "hello"}],
    }


def test_http_adapter_preserves_predictions(api):
    result = invoke()
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"benchmark_id": ROW["benchmark_id"], "memory_items": [MEMORY]}
    endpoint, body, auth = api["requests"][0]
    assert endpoint == "/v1/chat/completions"
    assert body == {"model": "fixture-model", "messages": build_messages(ROW)}
    assert auth == "Bearer fixture-key"
    assert result.stderr == ""


@pytest.mark.parametrize("content", ["not json", '{"memory_items":{}}',
                                    '{"memory_items":[{}]}', '{"memory_items":[],"extra":1}'])
def test_bad_model_output_is_not_silently_scored(api, content):
    api["content"] = content
    result = invoke()
    assert result.returncode != 0
    assert result.stdout == ""


@pytest.mark.parametrize("status,finish", [(401, "stop"), (429, "stop"), (200, "length")])
def test_api_failure_and_truncation_stop_the_adapter(api, status, finish):
    api.update(status=status, finish_reason=finish)
    result = invoke()
    assert result.returncode != 0
    assert result.stdout == ""
    assert "fixture-key" not in result.stderr


def test_empty_memories_and_markdown_wrapper():
    assert parse_prediction('```json\n{"memory_items": []}\n```', "s") == {
        "benchmark_id": "s", "memory_items": [],
    }


def test_dry_run_makes_no_api_call(api):
    result = invoke(ROW, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["messages"] == build_messages(ROW)
    assert api["requests"] == []


def test_adapter_through_public_evaluation_on_dev_and_test(api, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "BENCHMARK_DATA_ROOT", tmp_path / "dataset")
    for split in ("dev", "test"):
        track = data.BENCHMARK_DATA_ROOT / split / "task1"
        write_jsonl(track / "model_input.jsonl", [ROW])
        if split == "dev":
            write_jsonl(track / "reference_output.jsonl", [
                {"benchmark_id": ROW["benchmark_id"], "gold": {"memory_items": [MEMORY]}},
            ])
        output = tmp_path / "runs" / split
        summary = run_public_evaluation(task="task1", split=split,
            predict_argv=[sys.executable, str(ADAPTER)], output_dir=str(output))
        assert read_jsonl(output / "predictions.jsonl")[0]["memory_items"] == [MEMORY]
        if split == "dev":
            assert summary["main_score"] == 1.0
        else:
            assert summary["local_scoring_available"] is False
            assert not (output / "scores.jsonl").exists()
    for _, body, _ in api["requests"]:
        assert "DO_NOT_SEND" not in json.dumps(body)


def test_documented_cli_with_shipped_example(api, tmp_path):
    output = tmp_path / "cli"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "evaluate.py"),
        "--task", "task1", "--split", "examples", "--limit", "1",
        "--predict-program", sys.executable, "--predict-arg", str(ADAPTER),
        "--output-dir", str(output),
    ], cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["num_rows"] == 1
    assert summary["prediction_source"] == "predict_program"
    example = read_jsonl(ROOT / "benchmark_data/examples/task1/model_input.jsonl")[0]
    assert read_jsonl(output / "predictions.jsonl") == [
        {"benchmark_id": example["benchmark_id"], "memory_items": [MEMORY]},
    ]
    assert api["requests"][0][1]["messages"] == build_messages(example)
