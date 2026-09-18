# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Windows desktop build.

Run it through `build_windows.ps1`, or directly:

    .venv\\Scripts\\pyinstaller.exe --noconfirm az_interview_copilot.spec
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()

hidden_imports = [
    # keyring finds its backends by entry point, which PyInstaller cannot see.
    "keyring.backends.Windows",
    "keyring.backends.fail",
    "keyring.backends.null",
    *collect_submodules("openai"),
]

# Qt modules this app never touches; dropping them roughly halves the bundle.
excluded_qt = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtQuick3D",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtBluetooth",
    "PySide6.QtPositioning", "PySide6.QtSerialPort", "PySide6.QtSql", "PySide6.QtTest",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtQuick", "PySide6.QtQml",
]

excludes = [*excluded_qt, "tkinter", "matplotlib", "scipy", "pandas", "IPython", "pytest"]


a = Analysis(
    [str(SPEC_DIR / "run.py")],
    pathex=[str(SPEC_DIR / "src")],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AZInterviewCopilot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # a desktop app, not a console program
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AZInterviewCopilot",
)
