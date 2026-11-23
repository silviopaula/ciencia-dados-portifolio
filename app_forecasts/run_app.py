"""
Launcher script for App Forecasts (Python FastAPI Backend + Modern Web Frontend).
Usage:
    .\.venv\Scripts\python.exe run_app.py
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 68)
    print("  APP FORECASTS — Python Backend + Modern Frontend")
    print("  Acesse no navegador: http://127.0.0.1:8060")
    print("=" * 68)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8060, reload=False)
