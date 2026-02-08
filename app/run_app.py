import sys
import threading
import webbrowser
from pathlib import Path

from streamlit.web import bootstrap


def _get_app_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "streamlit_app.py"
    return Path(__file__).with_name("streamlit_app.py")


def _open_browser():
    webbrowser.open("http://localhost:8501", new=2)


def main():
    app_path = _get_app_path()
    args = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        "--server.port=8501",
        "--browser.serverAddress=localhost",
    ]
    threading.Timer(1.5, _open_browser).start()
    bootstrap.run(str(app_path), "", args, {})


if __name__ == "__main__":
    main()
