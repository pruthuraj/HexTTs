"""Tests for backend routing logic in shared dataloader factory."""

from hextts.data import dataloaders as dl


def test_create_dataloaders_selects_raw_backend(monkeypatch):
    """When caching is disabled, the raw dataset backend should be used."""
    called = {"ok": False}

    def fake_factory(config, batch_size, num_workers):
        called["ok"] = True
        return "train", "val"

    monkeypatch.setattr(dl.vits_data, "create_dataloaders", fake_factory)
    train, val = dl.create_dataloaders({"batch_size": 2, "num_workers": 0, "use_cached_features": False})

    assert called["ok"] is True
    assert (train, val) == ("train", "val")


def test_create_dataloaders_selects_cached_backend(monkeypatch):
    """When caching is enabled, the cached feature backend should be used."""
    called = {"ok": False}

    def fake_factory(config, batch_size, num_workers):
        called["ok"] = True
        return "train_cached", "val_cached"

    monkeypatch.setattr(dl.vits_data_cached, "create_dataloaders", fake_factory)
    train, val = dl.create_dataloaders({"batch_size": 2, "num_workers": 0, "use_cached_features": True})

    assert called["ok"] is True
    assert (train, val) == ("train_cached", "val_cached")
