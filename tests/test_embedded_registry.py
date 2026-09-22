from __future__ import annotations

from lwb import embedded


class FakeBackend:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_embedded_registry_reuses_identical_configuration(monkeypatch):
    created = []

    def create(**configuration):
        backend = FakeBackend()
        created.append((backend, configuration))
        return backend

    monkeypatch.setattr(embedded, "_create_embedded_backend", create)
    registry = embedded.EmbeddedModelRegistry()
    first = registry.load(model_path="a.gguf", family="auto")
    second = registry.load(family="auto", model_path="a.gguf")
    assert first is second
    assert len(created) == 1
    assert registry.release(first) is True
    assert first.closed is True
