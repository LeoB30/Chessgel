"""Tests for multi-engine registry, custom engine addition, and aggressive engine evaluations."""

from __future__ import annotations
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import chess
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from chess_tracker.core.engine_registry import (
    EngineDefinition,
    get_all_available_engines,
    probe_uci_engine,
    save_custom_engine,
    remove_custom_engine,
)
from chess_tracker.ui.main_window import MainWindow
from chess_tracker.ui.workers import EngineWorker


def test_aggressive_engines_present():
    """Verifies that the three aggressive engines (Patricia, Berserk, OpenTal) are present and registered."""
    engines = get_all_available_engines()
    ids = [e.id for e in engines]
    print(f"Available engine IDs: {ids}")

    assert "stockfish" in ids, "Stockfish missing from registry"
    assert "patricia" in ids, "Patricia (aggressive) missing from registry"
    assert "berserk" in ids, "Berserk (aggressive) missing from registry"
    assert "opental" in ids, "OpenTal (aggressive) missing from registry"
    assert "minichess" in ids, "MiniChess missing from registry"
    assert "none" in ids, "None missing from registry"
    print("test_aggressive_engines_present PASSED!")


def test_engine_switching_and_analysis():
    """Verifies per-side engine switching and asynchronous multi-PV analysis."""
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    worker = EngineWorker()
    worker.start()

    received_results = []
    worker.sig_evaluation_ready.connect(lambda res: received_results.append(res))

    # Test White with Patricia (aggressive)
    worker.set_engine_for_side("white", "patricia")
    worker.request_analysis("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", depth=6)
    
    # Wait for result
    for _ in range(30):
        app.processEvents()
        if received_results:
            break
        time.sleep(0.1)

    assert len(received_results) > 0, "No evaluation received from Patricia"
    last_res = received_results[-1]
    print(f"Patricia result: engine={last_res.engine_name}, best={last_res.best_move_san}, top_moves={len(last_res.top_moves)}")
    assert "Patricia" in last_res.engine_name
    assert last_res.best_move is not None

    worker.stop()
    print("test_engine_switching_and_analysis PASSED!")


def test_custom_engine_registration():
    """Tests probing, saving, and removing a custom engine definition."""
    dummy_id = "test_custom_engine"
    test_def = EngineDefinition(
        id=dummy_id,
        name="Test Engine 1.0",
        binary_path=r"c:\Users\leoba\Desktop\AAA III\__Chessi__\chess_tracker\engines\patricia.exe",
        engine_type="uci",
        is_builtin=False,
    )

    # Save
    assert save_custom_engine(test_def), "Failed to save custom engine"
    all_engines = get_all_available_engines()
    assert any(e.id == dummy_id for e in all_engines), "Custom engine not found in available engines"

    # Remove
    assert remove_custom_engine(dummy_id), "Failed to remove custom engine"
    all_engines_after = get_all_available_engines()
    assert not any(e.id == dummy_id for e in all_engines_after), "Custom engine still present after removal"
    print("test_custom_engine_registration PASSED!")


if __name__ == "__main__":
    test_aggressive_engines_present()
    test_engine_switching_and_analysis()
    test_custom_engine_registration()
    print("All engine tests PASSED successfully!")
