from src.benchmark.metrics import metric_surface, supports_local_scoring


def test_local_scoring_policy() -> None:
    assert supports_local_scoring("examples") is True
    assert supports_local_scoring("dev") is True
    assert supports_local_scoring("validation") is True
    assert supports_local_scoring("test") is False


def test_metric_surface_is_exposed() -> None:
    surface = metric_surface()
    assert surface["status"] == "ready"
    assert "prediction_contract" in surface
    assert "task1" in surface["task_metrics"]
