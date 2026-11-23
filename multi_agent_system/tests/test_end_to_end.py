from core.llm import DemoLLM
from core.orchestrator import Config
from core.tools import HTTPFailure, canned_http_get
from main import build_system


def test_demo_pipeline_end_to_end():
    rep = build_system(DemoLLM(), canned_http_get).run("multi-agent systems")
    assert rep.status == "success"
    assert "wikipedia.org" in rep.answer                      # provenance survives to the answer
    assert [h["step"] for h in rep.state.history] == ["s1", "s2", "s3"]


def test_demo_pipeline_when_api_is_down_degrades_gracefully():
    def down(url, timeout=8.0):
        raise HTTPFailure("HTTP 500", transient=False)
    rep = build_system(DemoLLM(), down, Config()).run("anything")
    # retriever reports the failure text instead of crashing; pipeline still completes
    assert rep.status in ("success", "partial")
    assert "Retrieval failed" in rep.state.results["s1"].output
