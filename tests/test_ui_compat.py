"""Legacy UI import paths stay live after the package move."""
from __future__ import annotations


def test_widget_path_forwards_live(monkeypatch):
    import spektr.ui.widget as implementation
    import spektr.widget as legacy

    sentinel = object()
    monkeypatch.setattr(implementation, "AudioVisualizer", sentinel)
    assert legacy.AudioVisualizer is sentinel


def test_picker_path_forwards_live(monkeypatch):
    import spektr.pickers as legacy
    import spektr.ui.pickers as implementation

    sentinel = object()
    monkeypatch.setattr(implementation, "Picker", sentinel)
    assert legacy.Picker is sentinel
