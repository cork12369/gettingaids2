"""
app.py — Zeabur entrypoint
Minimal Flask dashboard to trigger pipeline stages and view outputs.
"""

import os
import base64
import hmac
import threading
import subprocess
import json
from pathlib import Path
from datetime import datetime
from functools import wraps
from flask import (
    Flask, jsonify, render_template_string,
    request, session, redirect, url_for
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],          # no global limit — apply per route
    storage_uri="memory://",    # in-memory is fine for single worker
)

# ── Auth ──────────────────────────────────────────────────────────────────────
# Set ACCESS_KEY in Zeabur env vars — generate with:
#   python -c "import base64, os; print(base64.b32encode(os.urandom(10)).decode())"
# Example output: MFRA2YTBMJQXIZLB
#
# SESSION_SECRET should be a separate random string, also in env vars:
#   python -c "import secrets; print(secrets.token_hex(32))"

ACCESS_KEY     = os.environ.get("ACCESS_KEY", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "change-me-in-zeabur-env")

app.secret_key = SESSION_SECRET


def _key_valid(provided: str) -> bool:
    """Constant-time comparison to avoid timing attacks."""
    if not ACCESS_KEY:
        return False  # refuse all access if key not configured
    provided_bytes = provided.strip().upper().encode()
    expected_bytes = ACCESS_KEY.strip().upper().encode()
    return hmac.compare_digest(provided_bytes, expected_bytes)


def require_auth(f):
    """Decorator — redirects to /login if session not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated


LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Pipeline — Login</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: monospace;
      background: #0d1117;
      color: #c9d1d9;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
    }
    .card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 40px;
      width: 360px;
    }
    h2 { color: #58a6ff; margin-top: 0; letter-spacing: 1px; }
    p  { color: #8b949e; font-size: 13px; margin-bottom: 24px; }
    input[type=text] {
      width: 100%;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 4px;
      color: #c9d1d9;
      padding: 10px 12px;
      font-family: monospace;
      font-size: 15px;
      letter-spacing: 2px;
      margin-bottom: 16px;
    }
    input[type=text]:focus {
      outline: none;
      border-color: #58a6ff;
    }
    button {
      width: 100%;
      background: #238636;
      color: white;
      border: none;
      border-radius: 4px;
      padding: 10px;
      font-family: monospace;
      font-size: 14px;
      cursor: pointer;
    }
    button:hover { background: #2ea043; }
    .error {
      background: #3d1c1c;
      border: 1px solid #f85149;
      color: #f85149;
      border-radius: 4px;
      padding: 8px 12px;
      font-size: 13px;
      margin-bottom: 16px;
    }
  </style>
</head>
<body>
  <div class="card">
    <h2>🔩 PIPELINE</h2>
    <p>Enter your access key to continue.</p>
    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}
    <form method="POST">
      <input type="text" name="key" placeholder="XXXX-XXXX-XXXX"
             autocomplete="off" autocorrect="off" autocapitalize="characters"
             autofocus spellcheck="false">
      <input type="hidden" name="next" value="{{ next }}">
      <button type="submit">→ Enter</button>
    </form>
  </div>
</body>
</html>
"""


@app.errorhandler(429)
def rate_limited(e):
    return render_template_string(LOGIN_HTML,
        error="Too many attempts. Wait a minute and try again.",
        next=request.args.get("next", "/")
    ), 429


@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute; 30 per hour")
def login():
    error = None
    next_url = request.args.get("next", "/")

    if request.method == "POST":
        provided = request.form.get("key", "")
        next_url = request.form.get("next", "/")
        if _key_valid(provided):
            session["authenticated"] = True
            session.permanent = True
            return redirect(next_url)
        error = "Invalid key."

    return render_template_string(LOGIN_HTML, error=error, next=next_url)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ── State ─────────────────────────────────────────────────────────────────────
# Simple in-memory job tracker (good enough for a student project)
job_state = {
    "running":   False,
    "stage":     None,
    "log":       [],
    "started_at": None,
    "completed": [],
}

STAGES = [
    ("01_scrape_data",       "Scrape Text + Images"),
    ("03_sentiment_analysis","Sentiment Analysis"),
    ("04_image_processing",  "Image Processing"),
    ("05_cross_analysis",    "Cross Analysis"),
]

OUTPUT_DIR = Path("/data/output")
DATA_DIR   = Path("/data")

# ── Pipeline Runner ───────────────────────────────────────────────────────────

def run_stage(script_name):
    """Run a pipeline script as a subprocess, streaming stdout to job_state log."""
    script_path = Path(f"/app/{script_name}.py")
    job_state["stage"] = script_name
    job_state["log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {script_name}...")

    try:
        proc = subprocess.Popen(
            ["python", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/app",
        )
        for line in proc.stdout:
            line = line.rstrip()
            job_state["log"].append(f"  {line}")
            # Keep log from growing unbounded
            if len(job_state["log"]) > 500:
                job_state["log"] = job_state["log"][-400:]

        proc.wait()
        if proc.returncode == 0:
            job_state["completed"].append(script_name)
            job_state["log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ {script_name} complete")
            return True
        else:
            job_state["log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ {script_name} failed (exit {proc.returncode})")
            return False

    except Exception as e:
        job_state["log"].append(f"  ERROR: {e}")
        return False


def run_pipeline(stages_to_run):
    """Run selected stages sequentially in a background thread."""
    job_state["running"]    = True
    job_state["started_at"] = datetime.now().isoformat()
    job_state["log"]        = []

    for script_name, _ in STAGES:
        if script_name in stages_to_run:
            success = run_stage(script_name)
            if not success:
                job_state["log"].append("Pipeline stopped due to error.")
                break

    job_state["running"] = False
    job_state["stage"]   = None
    job_state["log"].append("=== Pipeline finished ===")


# ── Routes ────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Manhole Pipeline</title>
  <meta http-equiv="refresh" content="5">  <!-- auto-refresh every 5s -->
  <style>
    body { font-family: monospace; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #0d1117; color: #c9d1d9; }
    h1 { color: #58a6ff; }
    .status { padding: 8px 16px; border-radius: 4px; display: inline-block; margin-bottom: 20px; }
    .running  { background: #1f6feb; }
    .idle     { background: #21262d; }
    .stages   { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
    .stage    { padding: 6px 12px; border-radius: 4px; border: 1px solid #30363d; cursor: pointer; }
    .stage.done { border-color: #3fb950; color: #3fb950; }
    .stage.active { border-color: #58a6ff; color: #58a6ff; }
    form { margin-bottom: 20px; }
    select { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 6px; border-radius: 4px; }
    button { background: #238636; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-left: 8px; }
    button:disabled { background: #21262d; cursor: not-allowed; }
    pre { background: #161b22; border: 1px solid #30363d; padding: 16px; border-radius: 6px; overflow-x: auto; max-height: 400px; overflow-y: scroll; font-size: 12px; }
    .outputs { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .output-file { background: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 4px; }
    a { color: #58a6ff; }
  </style>
</head>
<body>
  <h1>🔩 Manhole Cover Pipeline</h1>
  <a href="/logout" style="float:right; color:#8b949e; font-size:12px; text-decoration:none;">logout</a>

  <div class="status {{ 'running' if job.running else 'idle' }}">
    {{ '⚙ Running: ' + job.stage if job.running else '● Idle' }}
  </div>

  <div class="stages">
    {% for script, label in stages %}
    <div class="stage {{ 'done' if script in job.completed else 'active' if job.stage == script else '' }}">
      {{ '✓ ' if script in job.completed else '' }}{{ label }}
    </div>
    {% endfor %}
  </div>

  <form method="POST" action="/run">
    <select name="stage" {{ 'disabled' if job.running }}>
      <option value="all">Run Full Pipeline</option>
      {% for script, label in stages %}
      <option value="{{ script }}">{{ label }} only</option>
      {% endfor %}
    </select>
    <button type="submit" {{ 'disabled' if job.running }}>▶ Run</button>
  </form>

  <h3>Log</h3>
  <pre>{{ log }}</pre>

  <h3>Outputs</h3>
  <div class="outputs">
    {% for f in output_files %}
    <div class="output-file">
      <a href="/output/{{ f }}">{{ f }}</a>
    </div>
    {% endfor %}
  </div>
</body>
</html>
"""


@app.route("/")
@require_auth
def dashboard():
    output_files = []
    if OUTPUT_DIR.exists():
        output_files = [f.name for f in OUTPUT_DIR.iterdir() if f.is_file()]

    return render_template_string(
        DASHBOARD_HTML,
        job=job_state,
        stages=STAGES,
        log="\n".join(job_state["log"][-100:]),
        output_files=sorted(output_files),
    )


@app.route("/run", methods=["POST"])
@require_auth
def trigger_run():
    if job_state["running"]:
        return "Already running", 400

    selected = request.form.get("stage", "all")
    if selected == "all":
        stages_to_run = [s for s, _ in STAGES]
    else:
        stages_to_run = [selected]

    thread = threading.Thread(target=run_pipeline, args=(stages_to_run,), daemon=True)
    thread.start()

    return jsonify({"status": "started", "stages": stages_to_run})


@app.route("/status")
@require_auth
def status():
    return jsonify({
        "running":   job_state["running"],
        "stage":     job_state["stage"],
        "completed": job_state["completed"],
        "log_tail":  job_state["log"][-20:],
    })


@app.route("/output/<filename>")
@require_auth
def serve_output(filename):
    """Serve output files (images/CSVs) for download."""
    from flask import send_from_directory
    return send_from_directory(str(OUTPUT_DIR), filename)


if __name__ == "__main__":
    # Zeabur injects PORT env var
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
