from src.benchmark.runner import smoke_examples


def test_smoke_examples_runs() -> None:
    summary = smoke_examples()
    assert summary["status"] == "ok"
    assert "examples" in summary["splits"]
