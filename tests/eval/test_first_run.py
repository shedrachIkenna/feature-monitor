from eval.first_run import summarise
from monitor.job import Detector, DriftJob, sae_sets


def detector(name: str) -> Detector:
    return Detector(name, {1: 1}, 1, 0.1, sae_sets, lambda f, *_: "")


def alert(kind: str, first: int, last: int, score: float) -> dict:
    details = {"first_request_id": first, "last_request_id": last, "window_size": last - first + 1}
    return {"kind": kind, "score": score, "details": {**details, "top": [{"id": 9}]}}


def test_summarise_maps_alerts_to_request_numbers_and_shift_share() -> None:
    # Request ids 11..20 are scenario requests 1..10; the last four are shifted.
    rows = [{"request_id": 11 + i, "index": i, "is_shift": i >= 6} for i in range(10)]
    job = DriftJob([detector("sae"), detector("embedding")], window_size=4)
    job.checks.append({"last_request_id": 18, "scores": {"sae": 0.2, "embedding": 0.05}})
    alerts = [alert("sae", 17, 20, 0.3), alert("sae", 15, 18, 0.2)]

    summary = summarise(rows, job, alerts)

    assert summary["checks"] == [{"request_number": 8, "scores": {"sae": 0.2, "embedding": 0.05}}]
    sae = summary["detectors"]["sae"]
    assert sae["alerts"] == 2
    assert sae["alert_request_numbers"] == [8, 10]
    assert sae["first_alert_request_number"] == 8
    assert sae["first_alert_score"] == 0.2
    assert sae["first_alert_window_shift_share"] == 0.5  # requests 5-8: 7 and 8 are shifted
    assert sae["first_alert_top"] == [{"id": 9}]
    assert summary["detectors"]["embedding"]["alerts"] == 0
    assert summary["detectors"]["embedding"]["first_alert_request_number"] is None
