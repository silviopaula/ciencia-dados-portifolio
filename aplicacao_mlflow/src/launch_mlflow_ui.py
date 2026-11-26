"""Inicializador portátil da interface web oficial do MLflow UI (SQLite local)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = PROJECT_ROOT / "mlflow_store"
DB_PATH = STORE_DIR / "mlflow.db"
ART_DIR = STORE_DIR / "artifacts"


def launch_ui(port: int = 5000, open_browser: bool = True) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    backend_uri = f"sqlite:///{DB_PATH.as_posix()}"
    artifact_root = ART_DIR.resolve().as_uri()

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        backend_uri,
        "--default-artifact-root",
        artifact_root,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    print(f"Iniciando MLflow UI em http://127.0.0.1:{port}")
    print(f"  Backend Store : {backend_uri}")
    print(f"  Artifacts Root: {artifact_root}")
    proc = subprocess.Popen(cmd)
    if open_browser:
        time.sleep(2.5)
        webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nMLflow UI encerrado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abrir MLflow UI local")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    launch_ui(port=args.port, open_browser=not args.no_browser)
