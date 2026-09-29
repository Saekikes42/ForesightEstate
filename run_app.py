"""Start Dash and Streamlit together; stop the helper when Streamlit exits."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
dash = subprocess.Popen([sys.executable, str(ROOT / "dashboard.py")], cwd=ROOT, creationflags=flags)
try:
    print("Streamlit: http://127.0.0.1:8501")
    print("Dash:      http://127.0.0.1:8050")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.address", "127.0.0.1", "--server.port", "8501"], cwd=ROOT, check=False)
finally:
    dash.terminate()
    try:
        dash.wait(timeout=5)
    except subprocess.TimeoutExpired:
        dash.kill()
