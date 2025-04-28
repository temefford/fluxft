# tests/test_loader.py
from fluxft.config import GlobalConfig
from fluxft.data.loader import build_dataloaders, default_transforms
def test_loader_smoke(tmp_path):
    cfg = GlobalConfig()
    cfg.data.data_dir = tmp_path  # empty folder
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "dummy.jpg").touch()
    ds, _ = build_dataloaders(cfg.data, 1, 64)
    assert ds is not None
