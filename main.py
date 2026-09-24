from __future__ import annotations
import sys
from engine.uci import UCIEngine


def main() -> None:
    """Entry point for UCI chess engine."""
    engine: UCIEngine = UCIEngine()
    engine.run()


if __name__ == "__main__":
    main()
