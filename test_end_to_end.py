"""Full pipeline on the tiny profile: data, models, graph, index, agents, evaluation and API."""
import json

from fastapi.testclient import TestClient


def test_pipeline_outputs_exist(tiny_run):
    s = tiny_run
    raw = s.path("raw")
    for name in ("historian/C201.parquet", "alarms.jsonl", "shift_logs.jsonl", "work_orders.jsonl", "lab_results.csv",
                 "ground_truth/episodes.json", "knowledge/manifest.json"):
        assert (raw / name).exists(), name
    reports = json.loads((s.path("reports") / "all_reports.json").read_text())
    assert reports, "no episodes investigated"
    for r in reports:
        assert r["verification"]["claims_supported"] >= 5
        for c in r["claims"]:
            assert c["evidence"], c
            assert all(e in r["evidence"] for e in c["evidence"])


def test_evaluation_is_sane(tiny_run):
    s = tiny_run
    det = json.loads((s.path("metrics") / "detection.json").read_text())
    assert det["event_level"]["scenario_recall"] >= 0.6
    rc = json.loads((s.path("metrics") / "root_cause.json").read_text())
    assert rc["signals_and_records"]["top1_accuracy"] >= 0.6
    rt = json.loads((s.path("metrics") / "retrieval.json").read_text())
    assert rt["modes"]["full"]["ndcg@10"] >= rt["modes"]["bm25"]["ndcg@10"]
    f = json.loads((s.path("metrics") / "faithfulness.json").read_text())
    assert f["injected_error_detection"]["fake_evidence_id"]["rate"] == 1.0


def test_api(tiny_run):
    from weir_whisper.api.app import create_app
    client = TestClient(create_app(tiny_run))
    h = client.get("/health").json()
    assert h["status"] == "ok"
    eps = client.get("/episodes").json()
    assert eps
    rep = client.post(f"/investigate/{eps[0]['episode_id']}").json()
    assert rep["root_cause"]["cause"]
    md = client.get(f"/reports/{eps[0]['episode_id']}.md")
    assert md.status_code == 200 and md.text.startswith("# Investigation")
    hits = client.post("/search", json={"query": "steam valve hunting", "column": "C201", "k": 5}).json()
    assert len(hits) == 5
    ans = client.post("/ask", json={"question": "Which column lost the most money?"}).json()
    assert ans["answer"]
    assert client.get("/episodes/DE9999").status_code == 404
