# tests/test_metrics.py
from fluxft.eval.evaluator import MetricComputer
from fluxft.config import GlobalConfig
def test_metric_stub(tmp_path, monkeypatch):
    # monkeypatch heavy calls
    monkeypatch.setattr(MetricComputer, "_generate", lambda self, p,n: [])
    monkeypatch.setattr(MetricComputer, "clip_score", lambda s,i,p: 0.5)
    monkeypatch.setattr(MetricComputer, "fid", lambda s,i,p: 1.0)
    cfg = GlobalConfig()
    m = MetricComputer(cfg, tmp_path)
    res = m.run([], cfg.data.data_dir)
    assert "FID" in res