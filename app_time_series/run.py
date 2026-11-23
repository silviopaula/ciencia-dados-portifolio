"""
TimeSeries Studio Pro — Launcher
Starts the FastAPI server and opens the application in the browser.
Usage:
    python run.py
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 68)
    print("  TimeSeries Studio Pro — Unified Time Series Platform")
    print("  Acesse no navegador: http://127.0.0.1:8055")
    print("=" * 68)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8055, reload=False)
