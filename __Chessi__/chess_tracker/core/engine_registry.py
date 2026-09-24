"""Engine Registry managing built-in aggressive engines and user custom UCI engines."""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import logging
import os
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from chess_tracker.config import STOCKFISH_PATH

import sys

logger = logging.getLogger("ChessTracker.EngineRegistry")

# Directory where pre-packaged / downloaded engines reside
if getattr(sys, 'frozen', False):
    # When running as a PyInstaller bundle, use the directory containing the .exe
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENGINES_DIR = os.path.join(BASE_DIR, "engines")
CUSTOM_ENGINES_FILE = os.path.join(ENGINES_DIR, "custom_engines.json")


@dataclass
class EngineDefinition:
    """Metadata describing a chess engine."""
    id: str
    name: str
    binary_path: Optional[str] = None
    engine_type: str = "uci"  # "uci", "minichess", "none"
    is_builtin: bool = True
    description: str = ""
    extra_options: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "binary_path": self.binary_path,
            "engine_type": self.engine_type,
            "is_builtin": self.is_builtin,
            "description": self.description,
            "extra_options": self.extra_options,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EngineDefinition:
        return cls(
            id=data["id"],
            name=data["name"],
            binary_path=data.get("binary_path"),
            engine_type=data.get("engine_type", "uci"),
            is_builtin=data.get("is_builtin", False),
            description=data.get("description", ""),
            extra_options=data.get("extra_options", {}),
        )


def probe_uci_engine(binary_path: str, timeout: float = 3.0) -> Tuple[bool, str, str]:
    """Tests if a binary is a valid UCI engine and extracts its declared name.

    Returns:
        (is_valid, engine_name, error_message)
    """
    if not os.path.isfile(binary_path):
        return False, "", f"File does not exist: {binary_path}"

    try:
        working_dir = os.path.dirname(os.path.abspath(binary_path))
        proc = subprocess.Popen(
            [binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=working_dir,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        proc.stdin.write("uci\n")
        proc.stdin.flush()

        engine_name = ""
        is_uci = False
        start_time = time.time()

        while time.time() - start_time < timeout:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("id name "):
                engine_name = line[8:].strip()
            elif line.startswith("id ") and not line.startswith("id author") and not engine_name:
                engine_name = line[3:].strip()
            if line == "uciok":
                is_uci = True
                break

        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            proc.terminate()
            proc.wait(timeout=0.5)
        except Exception:
            pass

        if not is_uci:
            return False, "", "Executable did not respond with 'uciok' to UCI protocol handshake."

        if not engine_name:
            engine_name = os.path.splitext(os.path.basename(binary_path))[0].title()

        return True, engine_name, ""

    except Exception as e:
        return False, "", f"Failed to execute binary: {e}"


def get_builtin_engines() -> List[EngineDefinition]:
    """Returns the curated list of default engines, including the top aggressive engines."""
    engines: List[EngineDefinition] = []

    # 1. Stockfish (Objective Gold Standard)
    engines.append(
        EngineDefinition(
            id="stockfish",
            name="Stockfish 15.1",
            binary_path=STOCKFISH_PATH,
            engine_type="uci",
            is_builtin=True,
            description="The world's strongest objective chess engine.",
        )
    )

    # 2. Patricia 5.1 (Top Hyper-Aggressive Sacrificial Engine)
    patricia_path = os.path.join(ENGINES_DIR, "patricia.exe")
    engines.append(
        EngineDefinition(
            id="patricia",
            name="Patricia 5.1 (Hyper-Aggressive)",
            binary_path=patricia_path,
            engine_type="uci",
            is_builtin=True,
            description="Hyper-aggressive sacrificing neural engine, plays wild, attacking, tactical lines.",
        )
    )

    # 3. Berserk 14 (Tactical Attacker Powerhouse)
    berserk_path = os.path.join(ENGINES_DIR, "berserk.exe")
    engines.append(
        EngineDefinition(
            id="berserk",
            name="Berserk 14 (Tactical Attacker)",
            binary_path=berserk_path,
            engine_type="uci",
            is_builtin=True,
            description="Aggressive tactical engine renowned for relentless king-attacks and sharp complications.",
        )
    )

    # 4. OpenTal 1.1 (Mikhail Tal Sacrificial Style)
    opental_path = os.path.join(ENGINES_DIR, "opental.exe")
    engines.append(
        EngineDefinition(
            id="opental",
            name="OpenTal 1.1 (Mikhail Tal Style)",
            binary_path=opental_path,
            engine_type="uci",
            is_builtin=True,
            description="Engine engineered to mimic the legendary 8th World Champion Mikhail Tal's tactical sacrifices.",
            extra_options={"UseBook": "false"},
        )
    )

    # 5. Reckless (Highly Aggressive)
    reckless_path = os.path.join(ENGINES_DIR, "reckless.exe")
    engines.append(
        EngineDefinition(
            id="reckless",
            name="Reckless 1.0 (Highly Aggressive)",
            binary_path=reckless_path,
            engine_type="uci",
            is_builtin=True,
            description="Reckless is an ultra-aggressive engine rated >3000 that despises draws and loves to attack.",
        )
    )

    # 6. MiniChess (Custom AAA III Python Engine)
    engines.append(
        EngineDefinition(
            id="minichess",
            name="MiniChess (AAA III)",
            binary_path=None,
            engine_type="minichess",
            is_builtin=True,
            description="Custom built-in alpha-beta search with quiescence and opening book from AAA III.",
        )
    )

    # 7. None (Analysis Disabled)
    engines.append(
        EngineDefinition(
            id="none",
            name="None (Disabled)",
            binary_path=None,
            engine_type="none",
            is_builtin=True,
            description="Disable engine evaluation.",
        )
    )

    return engines


def load_custom_engines() -> List[EngineDefinition]:
    """Loads user-registered custom engines from custom_engines.json."""
    if not os.path.isfile(CUSTOM_ENGINES_FILE):
        return []

    try:
        with open(CUSTOM_ENGINES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        engines = []
        for item in data.get("engines", []):
            if os.path.isfile(item.get("binary_path", "")):
                engines.append(EngineDefinition.from_dict(item))
            else:
                logger.warning("Custom engine binary not found at %s, skipping.", item.get("binary_path"))
        return engines
    except Exception as e:
        logger.error("Failed to load custom engines file: %s", e)
        return []


def save_custom_engine(engine_def: EngineDefinition) -> bool:
    """Persists a new or updated custom engine definition."""
    os.makedirs(ENGINES_DIR, exist_ok=True)
    current = load_custom_engines()
    # Replace if ID exists, otherwise append
    new_list = [e for e in current if e.id != engine_def.id]
    new_list.append(engine_def)

    try:
        with open(CUSTOM_ENGINES_FILE, "w", encoding="utf-8") as f:
            json.dump({"engines": [e.to_dict() for e in new_list]}, f, indent=2)
        return True
    except Exception as e:
        logger.error("Failed to save custom engines: %s", e)
        return False


def remove_custom_engine(engine_id: str) -> bool:
    """Removes a custom engine from persistent storage."""
    if not os.path.isfile(CUSTOM_ENGINES_FILE):
        return False

    current = load_custom_engines()
    new_list = [e for e in current if e.id != engine_id]

    try:
        with open(CUSTOM_ENGINES_FILE, "w", encoding="utf-8") as f:
            json.dump({"engines": [e.to_dict() for e in new_list]}, f, indent=2)
        return True
    except Exception as e:
        logger.error("Failed to remove custom engine: %s", e)
        return False


def get_all_available_engines() -> List[EngineDefinition]:
    """Returns combined list of all built-in and active custom engines."""
    builtins = get_builtin_engines()
    customs = load_custom_engines()
    # Insert custom engines right before 'None (Disabled)'
    if builtins and builtins[-1].id == "none":
        return builtins[:-1] + customs + [builtins[-1]]
    return builtins + customs
