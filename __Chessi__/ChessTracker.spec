# -*- mode: python ; coding: utf-8 -*-

import os

base_dir = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(base_dir, 'main.py')],
    pathex=[base_dir],
    binaries=[],
    datas=[],
    hiddenimports=[
        'chess_tracker',
        'chess_tracker.config',
        'chess_tracker.main',
        'chess_tracker.core',
        'chess_tracker.core.engine',
        'chess_tracker.core.engine_registry',
        'chess_tracker.core.game_state',
        'chess_tracker.core.minichess_adapter',
        'chess_tracker.ui',
        'chess_tracker.ui.board_mirror',
        'chess_tracker.ui.eval_bar',
        'chess_tracker.ui.main_window',
        'chess_tracker.ui.screen_selector',
        'chess_tracker.ui.workers',
        'chess_tracker.vision',
        'chess_tracker.vision.capture',
        'chess_tracker.vision.detector',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ChessTracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
