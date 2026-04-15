"""
app.py — Zeabur entrypoint
Flask dashboard to trigger pipeline stages, view outputs, and run human grading.
"""

import os, csv, uuid, hmac, shutil, threading, subprocess, time, json, io, re, base64, hashlib
from pathlib import Path
from functools import wraps
from flask import Flask, jsonify, render_template_string, request, session, redirect, url_for, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "change-me-in-zeabur-env")
limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")

ACCESS_KEY = os.environ.get("ACCESS_KEY", "")
ADMIN_RESET_PASSWORD = os.environ.get("ADMIN_RESET_PASSWORD", "")

# Allowed pipeline scripts (whitelist)
ALLOWED_SCRIPTS = frozenset([
    "01_scrape_data.py",
    "03_sentiment_analysis.py", 
    "04_image_processing.py",
    "05_cross_analysis.py",
    "06_confusion_matrix.py",
    "06_visualize_images.py",
    "07_evaluate_design.py",
    "08_generate_report.py"
])

# Thread-safe job state management
_job_lock = threading.Lock()
job_state = {"running": False, "script": None, "started_at": None, "log": [], "thread": None}
MAX_LOG_ENTRIES = 1000  # Limit log growth to prevent memory issues

# ── Sample Generation Guard ────────────────────────────────────────────────────
_sample_lock = threading.Lock()
_sample_generating = False  # True while background thread builds grade_sample.csv
_sample_generation_done = False  # True once sample has been built this lifecycle

def _key_valid(p):
    if not ACCESS_KEY: return False
    return hmac.compare_digest(p.strip().upper().encode(), ACCESS_KEY.strip().upper().encode())

def _admin_key_valid(p):
    if not ADMIN_RESET_PASSWORD: return False
    # Strip whitespace from both for consistency
    return hmac.compare_digest(p.strip().encode(), ADMIN_RESET_PASSWORD.strip().encode())

def _is_script_allowed(script):
    """Validate script against whitelist to prevent path traversal attacks."""
    if not script or ".." in script or "/" in script or "\\" in script:
        return False
    return script in ALLOWED_SCRIPTS

def require_auth(f):
    @wraps(f)
    def dec(*a, **kw):
        if not session.get("authenticated"):
            return redirect(url_for("login", next=request.path))
        return f(*a, **kw)
    return dec

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/data")
OUTPUT_DIR = DATA_DIR / "output"
IMAGES_DIR = DATA_DIR / "images"
GRADE_SAMPLE = DATA_DIR / "grade_sample.csv"
GRADER_ASSIGN = DATA_DIR / "grader_assignments.csv"
HUMAN_GRADES = DATA_DIR / "human_grades.csv"
CHUNK_SIZE = 10
SAMPLE_TARGET = 2000

# ── Grading Helpers ───────────────────────────────────────────────────────────

def _ensure_sample():
    """Non-blocking sample generation — spawns a background thread on first call.
    
    Thread-safe: only one thread does the expensive CSV read + sampling.
    Returns True immediately if sample already exists or generation was started.
    Returns False only if source data is missing.
    """
    # Fast path: already generated this lifecycle or on disk
    if _sample_generation_done or GRADE_SAMPLE.exists():
        return True

    with _sample_lock:
        # Double-check after acquiring lock (another thread may have started it)
        if _sample_generation_done or GRADE_SAMPLE.exists():
            return True

        # If a background thread is already building the sample, return True
        # and let the thread complete — the frontend timeout handles waiting
        if _sample_generating:
            return True

        src = DATA_DIR / "text_with_sentiment.csv"
        if not src.exists():
            return False

        # Mark as generating and hand off to background thread
        global _sample_generating
        _sample_generating = True

        def _build_sample():
            """Background worker: read full CSV, stratified sample, write grade_sample.csv."""
            try:
                import pandas as pd
                df = pd.read_csv(src)
                if "country" not in df.columns or len(df) == 0:
                    return
                countries = df["country"].unique().tolist()
                if "unknown" in countries:
                    countries.remove("unknown")
                if not countries:
                    return
                per_country = max(SAMPLE_TARGET // len(countries), 1)
                frames = []
                for c in countries:
                    sub = df[df["country"] == c].sample(
                        n=min(per_country, len(df[df["country"] == c])), random_state=42
                    )
                    frames.append(sub)
                sample = pd.concat(frames, ignore_index=True)
                if len(sample) > SAMPLE_TARGET:
                    sample = sample.sample(n=SAMPLE_TARGET, random_state=42)
                sample = sample.reset_index(drop=True)
                sample["snippet_id"] = sample.index
                sample.to_csv(GRADE_SAMPLE, index=False)
            except Exception:
                pass
            finally:
                global _sample_generating, _sample_generation_done
                _sample_generating = False
                _sample_generation_done = True

        t = threading.Thread(target=_build_sample, daemon=True)
        t.start()
        return True

def _total_chunks():
    if not GRADE_SAMPLE.exists():
        return 0
    import pandas as pd
    return (len(pd.read_csv(GRADE_SAMPLE)) + CHUNK_SIZE - 1) // CHUNK_SIZE

def _get_or_create_grader_id():
    gid = session.get("grader_id")
    if gid:
        return gid
    gid = f"grader_{uuid.uuid4().hex[:8]}"
    session["grader_id"] = gid
    return gid

def _read_assignments():
    if not GRADER_ASSIGN.exists():
        return []
    with open(GRADER_ASSIGN, newline="") as f:
        return list(csv.DictReader(f))

def _write_assignments(rows):
    with open(GRADER_ASSIGN, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["grader_id", "chunk_index", "start", "end"])
        w.writeheader()
        w.writerows(rows)

def _read_grades():
    if not HUMAN_GRADES.exists():
        return []
    with open(HUMAN_GRADES, newline="") as f:
        return list(csv.DictReader(f))

def _append_grade(snippet_id, grader_id, score):
    grades = _read_grades()
    file_exists = HUMAN_GRADES.exists()
    with open(HUMAN_GRADES, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snippet_id", "grader_id", "human_score", "timestamp"])
        if not file_exists:
            w.writeheader()
        w.writerow({"snippet_id": snippet_id, "grader_id": grader_id,
                     "human_score": score, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})

def _claim_chunk(grader_id):
    """Return the grader's existing incomplete chunk, or claim the next available."""
    assignments = _read_assignments()
    grades = _read_grades()
    graded_ids = {g["snippet_id"] for g in grades if g["grader_id"] == grader_id}
    total = _total_chunks()

    # Check existing assignments for this grader
    for a in assignments:
        if a["grader_id"] == grader_id:
            ci = int(a["chunk_index"])
            start = ci * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, SAMPLE_TARGET)
            # Count grades specifically in this chunk's snippet range
            chunk_snippet_ids = {str(i) for i in range(start, end)}
            graded_in_this_chunk = len(graded_ids & chunk_snippet_ids)
            if graded_in_this_chunk < (end - start):
                return ci
            # Chunk completed, continue to find a new unclaimed chunk

    # Find next unclaimed chunk
    claimed = {int(a["chunk_index"]) for a in assignments}
    for i in range(total):
        if i not in claimed:
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, SAMPLE_TARGET)
            assignments.append({"grader_id": grader_id, "chunk_index": i,
                                "start": start, "end": end})
            _write_assignments(assignments)
            return i
    return None  # All claimed

def _get_chunk_snippets(chunk_idx):
    import pandas as pd
    df = pd.read_csv(GRADE_SAMPLE)
    start = chunk_idx * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, len(df))
    return df.iloc[start:end]

# ── HTML Templates ────────────────────────────────────────────────────────────

LOGIN_HTML = """<!DOCTYPE html><html><head><title>Pipeline Login</title>
<style>body{font-family:monospace;background:#0d1117;color:#c9d1d9;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:40px;width:360px}
h2{color:#58a6ff;margin-top:0}input{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:4px;color:#c9d1d9;padding:10px;margin-bottom:16px}
button{width:100%;background:#238636;color:white;border:none;border-radius:4px;padding:10px;cursor:pointer}</style>
</head><body><div class="card"><h2>Pipeline Login</h2>
<form method="post" action="/login"><input type="text" name="key" placeholder="ACCESS KEY" autofocus required><button type="submit">Continue</button></form>
</div></body></html>"""

DASHBOARD_HTML = """<!DOCTYPE html><html><head><title>Manhole Cover Pipeline</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:linear-gradient(160deg,#0d1117 0%,#161b22 50%,#0d1117 100%);color:#e6edf3;min-height:100vh}
.container{max-width:1600px;margin:0 auto;padding:20px}

/* ── Header ── */
header{background:linear-gradient(135deg,#161b22 0%,#1c2333 100%);border:1px solid #30363d;border-radius:12px;padding:16px 24px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center;box-shadow:0 4px 24px rgba(0,0,0,0.3)}
header h1{color:#58a6ff;font-size:20px;font-weight:600;letter-spacing:-0.3px}
.header-actions{display:flex;gap:8px;align-items:center}
.auth-btn{background:transparent;border:1px solid #f85149;color:#f85149;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:500;transition:all .2s}
.auth-btn:hover{background:#f8514920}
.admin-toggle-btn{background:transparent;border:1px solid #d29922;color:#d29922;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:500;transition:all .2s}
.admin-toggle-btn:hover{background:#d2992220}
.nav-link{background:#8b5cf620;border:1px solid #8b5cf6;color:#c4b5fd;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:500;text-decoration:none;transition:all .2s}
.nav-link:hover{background:#8b5cf640;color:white}

/* ── Status Strip ── */
.status-strip{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:20px}
.status-card{background:linear-gradient(135deg,#161b22,#1c2333);border:1px solid #30363d;border-radius:10px;padding:16px;text-align:center;transition:all .3s;position:relative;overflow:hidden}
.status-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:#30363d;transition:background .3s}
.status-card.completed::before{background:linear-gradient(90deg,#238636,#2ea043)}
.status-card h3{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;font-weight:600}
.status-card .count{font-size:26px;font-weight:700;color:#58a6ff}
.status-card.completed .count{color:#3fb950}

/* ── Filter Bar ── */
.filter-bar{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px 16px;margin-bottom:24px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.filter-bar .label{color:#8b949e;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-right:8px}
.filter-btn{background:#21262d;color:#8b949e;border:1px solid #30363d;border-radius:6px;padding:7px 18px;cursor:pointer;font-size:13px;font-weight:500;transition:all .2s}
.filter-btn:hover{background:#30363d;color:#e6edf3}
.filter-btn.active{background:linear-gradient(135deg,#1f6feb,#388bfd);color:white;border-color:#1f6feb;box-shadow:0 2px 12px rgba(31,111,235,0.3)}

/* ── Section Headers ── */
.section-header{display:flex;align-items:center;gap:12px;margin:28px 0 16px;padding-bottom:10px;border-bottom:1px solid #21262d}
.section-icon{width:36px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0}
.section-icon.sentiment{background:linear-gradient(135deg,#1f6feb40,#388bfd20);border:1px solid #1f6feb60}
.section-icon.images{background:linear-gradient(135deg,#23863640,#2ea04320);border:1px solid #23863660}
.section-icon.cross{background:linear-gradient(135deg,#d2992240,#e3b34120);border:1px solid #d2992260}
.section-icon.confusion{background:linear-gradient(135deg,#8b5cf640,#7c3aed20);border:1px solid #8b5cf660}
.section-title{font-size:16px;font-weight:600;color:#e6edf3}
.section-subtitle{font-size:12px;color:#8b949e;margin-top:2px}

/* ── Graph Grid ── */
.graph-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:16px;margin-bottom:8px}
@media(max-width:560px){.graph-grid{grid-template-columns:1fr}}
.graph-card{background:linear-gradient(135deg,#161b22,#1c2333);border:1px solid #30363d;border-radius:10px;overflow:hidden;transition:all .3s}
.graph-card:hover{border-color:#484f58;box-shadow:0 4px 20px rgba(0,0,0,0.3);transform:translateY(-2px)}
.graph-card-header{padding:12px 16px;border-bottom:1px solid #21262d;display:flex;align-items:center;gap:8px}
.graph-card-header .dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.graph-card-header .dot.blue{background:#58a6ff}
.graph-card-header .dot.green{background:#3fb950}
.graph-card-header .dot.yellow{background:#d29922}
.graph-card-header .dot.purple{background:#8b5cf6}
.graph-card-header span{font-size:13px;font-weight:500;color:#c9d1d9}
.graph-card-body{padding:12px;text-align:center;background:#0d1117;min-height:120px;display:flex;align-items:center;justify-content:center}
.graph-card-body img{max-width:100%;height:auto;border-radius:4px;display:block}
.graph-card-body .no-data{color:#484f58;font-style:italic;font-size:13px}

/* ── Buttons ── */
.btn-grid{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin:20px 0}
.btn{background:#238636;color:white;border:none;border-radius:8px;padding:10px 20px;cursor:pointer;font-size:13px;font-weight:500;transition:all .2s}
.btn:hover{background:#2ea043;transform:translateY(-1px);box-shadow:0 2px 8px rgba(35,134,54,0.3)}
.btn-run-all{background:linear-gradient(135deg,#1f6feb,#388bfd)}
.btn-run-all:hover{background:linear-gradient(135deg,#388bfd,#58a6ff);box-shadow:0 2px 8px rgba(31,111,235,0.3)}
.btn-admin{background:#f85149}
.btn-admin:hover{background:#da3633;box-shadow:0 2px 8px rgba(248,81,73,0.3)}
.btn-soft-reset{background:#d29922}
.btn-soft-reset:hover{background:#e3b341}
.btn-stage6{background:linear-gradient(135deg,#8b5cf6,#7c3aed)}
.btn-stage6:hover{background:linear-gradient(135deg,#7c3aed,#6d28d9);box-shadow:0 2px 8px rgba(139,92,246,0.3)}

/* ── Output Log ── */
.output-section{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:16px;margin-top:20px}
.output-section h3{color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px}
.output-section pre{color:#c9d1d9;font-family:'Cascadia Code',Consolas,monospace;font-size:12px;overflow-x:auto;white-space:pre-wrap;max-height:300px;line-height:1.5}

/* ── Admin Section ── */
.admin-section{background:linear-gradient(135deg,#161b22,#1c2333);border:1px solid #f8514940;border-radius:10px;padding:20px;margin-top:20px;display:none;box-shadow:0 0 20px rgba(248,81,73,0.1)}
.admin-section h3{color:#f85149;font-size:14px;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid #21262d}
.admin-section .warning{color:#d29922;font-size:12px;margin-bottom:15px}
.admin-grid{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:15px}
.admin-grid input{flex:1;min-width:200px;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#c9d1d9;padding:10px;font-size:13px}
.pipeline-status{background:#21262d;border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:12px}
.pipeline-status .running{color:#3fb950;font-weight:600}
.pipeline-status .idle{color:#8b949e}
.pipeline-status .script{color:#58a6ff}

/* ── Pipeline Progress ── */
.pipeline-progress{background:linear-gradient(135deg,#161b22,#1c2333);border:1px solid #1f6feb40;border-radius:10px;padding:16px 20px;margin:16px 0;box-shadow:0 0 20px rgba(31,111,235,0.1)}
.progress-info{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.progress-label{color:#58a6ff;font-size:14px;font-weight:600}
.progress-time{color:#8b949e;font-size:13px;font-variant-numeric:tabular-nums}
.progress-track{width:100%;height:10px;background:#21262d;border-radius:6px;overflow:hidden}
.progress-animated{height:100%;width:100%;background:linear-gradient(90deg,#1f6feb,#58a6ff,#1f6feb);background-size:200% 100%;border-radius:6px;animation:progressShimmer 1.5s ease infinite}
@keyframes progressShimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}

/* ── Divider ── */
.divider{height:1px;background:linear-gradient(90deg,transparent,#30363d,transparent);margin:24px 0}
</style>
</head><body><div class="container">

<!-- Header -->
<header><h1>🔵 Manhole Cover Pipeline</h1><div class="header-actions">
<a href="/evaluate" class="nav-link">🎨 Evaluate</a>
<a href="/grade" class="nav-link">📝 Grade</a>
<a href="/grade/team" class="nav-link">📊 Team</a>
<a href="/history" class="nav-link">📜 History</a>
<a href="/report/download/csv" class="nav-link" id="report-csv" title="Download CSV report">📄 Report CSV</a>
<a href="/report/download/pdf" class="nav-link" id="report-pdf" title="Download PDF report" target="_blank">📄 Report PDF</a>
<button class="admin-toggle-btn" onclick="toggleAdminPanel()">⚙️ Admin</button>
<form action="/logout" method="get" style="display:inline"><button class="auth-btn">Logout</button></form>
</div></header>

<!-- Status Strip -->
<div class="status-strip">
<div class="status-card" id="status-scrape"><h3>Scrape Data</h3><div class="count">0</div></div>
<div class="status-card" id="status-sentiment"><h3>Sentiment</h3><div class="count">0</div></div>
<div class="status-card" id="status-images"><h3>Images</h3><div class="count">0</div></div>
<div class="status-card" id="status-cross"><h3>Cross Analysis</h3><div class="count">0</div></div>
<div class="status-card" id="status-confusion"><h3>Confusion</h3><div class="count">—</div></div>
</div>

<!-- Filter Bar -->
<div class="filter-bar">
<span class="label">Filter</span>
<button class="filter-btn active" onclick="filterSection('all',this)">📊 All</button>
<button class="filter-btn" onclick="filterSection('sentiment',this)">💬 Sentiment</button>
<button class="filter-btn" onclick="filterSection('images',this)">🖼️ Images</button>
<button class="filter-btn" onclick="filterSection('cross',this)">🔀 Cross Analysis</button>
<button class="filter-btn" onclick="filterSection('confusion',this)">🎯 Confusion Matrix</button>
</div>

<!-- Sentiment Section -->
<div class="dashboard-section" data-section="sentiment">
<div class="section-header">
<div class="section-icon sentiment">💬</div>
<div><div class="section-title">Sentiment Analysis</div><div class="section-subtitle">5 graphs from Stage 2</div></div>
</div>
<div class="graph-grid">
<div class="graph-card"><div class="graph-card-header"><span class="dot blue"></span><span>Weighted Sentiment by Country</span></div><div class="graph-card-body" id="g-sentiment-0"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot blue"></span><span>Sentiment Composition</span></div><div class="graph-card-body" id="g-sentiment-1"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot blue"></span><span>Text Volume by Country</span></div><div class="graph-card-body" id="g-sentiment-2"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot blue"></span><span>Keyword Heatmap</span></div><div class="graph-card-body" id="g-sentiment-3"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot blue"></span><span>Confidence Distribution</span></div><div class="graph-card-body" id="g-sentiment-4"><div class="no-data">Run pipeline to generate</div></div></div>
</div></div>

<!-- Images Section -->
<div class="dashboard-section" data-section="images">
<div class="section-header">
<div class="section-icon images">🖼️</div>
<div><div class="section-title">Image Analysis</div><div class="section-subtitle">4 graphs from Stage 3</div></div>
</div>
<div class="graph-grid">
<div class="graph-card"><div class="graph-card-header"><span class="dot green"></span><span>Image Volume by Country</span></div><div class="graph-card-body" id="g-images-0"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot green"></span><span>Country Image Distribution</span></div><div class="graph-card-body" id="g-images-1"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot green"></span><span>Source Distribution</span></div><div class="graph-card-body" id="g-images-2"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot green"></span><span>Resolution Distribution</span></div><div class="graph-card-body" id="g-images-3"><div class="no-data">Run pipeline to generate</div></div></div>
</div></div>

<!-- Cross Analysis Section -->
<div class="dashboard-section" data-section="cross">
<div class="section-header">
<div class="section-icon cross">🔀</div>
<div><div class="section-title">Cross Analysis</div><div class="section-subtitle">7 graphs from Stage 4</div></div>
</div>
<div class="graph-grid">
<div class="graph-card"><div class="graph-card-header"><span class="dot yellow"></span><span>Text vs Image by Country</span></div><div class="graph-card-body" id="g-cross-0"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot yellow"></span><span>Sentiment vs Image Volume</span></div><div class="graph-card-body" id="g-cross-1"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot yellow"></span><span>Combined Country Summary</span></div><div class="graph-card-body" id="g-cross-2"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot yellow"></span><span>Balance Ratio Chart</span></div><div class="graph-card-body" id="g-cross-3"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot yellow"></span><span>Coverage Summary</span></div><div class="graph-card-body" id="g-cross-4"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot yellow"></span><span>Sentiment Heatmap</span></div><div class="graph-card-body" id="g-cross-5"><div class="no-data">Run pipeline to generate</div></div></div>
<div class="graph-card"><div class="graph-card-header"><span class="dot yellow"></span><span>Human vs AI Agreement</span></div><div class="graph-card-body" id="g-cross-6"><div class="no-data">Grade snippets to generate</div></div></div>
</div></div>

<!-- Confusion Matrix Section -->
<div class="dashboard-section" data-section="confusion">
<div class="section-header">
<div class="section-icon confusion">🎯</div>
<div><div class="section-title">Confusion Matrix — Human vs AI</div><div class="section-subtitle">Human-AI sentiment validation from Stage 6</div></div>
</div>
<div class="graph-grid">
<div class="graph-card" style="grid-column:1/-1"><div class="graph-card-header"><span class="dot purple"></span><span>Confusion Matrix & Metrics</span></div><div class="graph-card-body" id="g-confusion-0"><div class="no-data">Run Stage 6 to generate confusion matrix</div></div></div>
</div></div>

<div class="divider"></div>

<!-- Pipeline Controls -->
<div class="btn-grid"><button class="btn btn-run-all" onclick="runAll()">▶ Run Full Pipeline</button>
<button class="btn" onclick="runStage('01_scrape_data.py')">Scrape</button>
<button class="btn" onclick="runStage('03_sentiment_analysis.py')">Sentiment</button>
<button class="btn" onclick="runStage('04_image_processing.py')">Images</button>
<button class="btn" onclick="runStage('05_cross_analysis.py')">Cross Analysis</button>
<button class="btn" onclick="runStage('06_visualize_images.py')">Stage 5: Image Viz</button>
<button class="btn btn-stage6" onclick="runStage('06_confusion_matrix.py')">Stage 6: Confusion Matrix</button></div>

<!-- Pipeline Progress Bar -->
<div class="pipeline-progress" id="pipeline-progress" style="display:none">
<div class="progress-info"><span id="progress-label" class="progress-label">Running...</span><span id="progress-time" class="progress-time">0s</span></div>
<div class="progress-track"><div class="progress-animated" id="progress-fill"></div></div>
</div>

<!-- Admin Section -->
<div class="admin-section"><h3>⚙️ Admin Controls</h3>
<div class="warning">⚠️ Requires ADMIN_RESET_PASSWORD. Use with caution.</div>
<div class="pipeline-status" id="pipeline-status">Pipeline Status: <span class="idle">Checking...</span></div>
<div class="admin-grid"><input type="password" id="admin-password" placeholder="Admin Password">
<button class="btn btn-soft-reset" onclick="adminReset('soft')">🔓 Soft Reset</button>
<button class="btn btn-admin" onclick="adminReset('analysis')">🗑️ Reset Analysis</button>
<button class="btn btn-admin" onclick="adminReset('full')">⚠️ Full Reset</button>
<button class="btn btn-admin" onclick="adminResetAndRerun()">🔄 Reset & Rerun</button>
<button class="btn" onclick="showLogs()" style="background:#388bfd">📋 View Logs</button></div>
<div id="admin-message" style="margin-top:10px;font-size:12px;color:#8b949e;"></div></div>

<div class="output-section"><h3>Output Log</h3><pre id="output-log">Ready. Run a pipeline stage to see output.</pre></div></div>

<script>
/* ── Graph Definitions ── */
var GRAPH_MAP={
sentiment:['sentiment_by_country.png','sentiment_composition.png','text_volume_by_country.png','keyword_heatmap.png','confidence_distribution.png'],
// Note: Stage 3 (04_image_processing.py) only generates CSVs, not PNG graphs.
// Image visualization graphs are generated by Stage 5 (06_visualize_images.py) or use data from /counts
images:['image_volume_by_country.png','country_image_distribution.png','source_distribution.png','image_resolution_distribution.png'],
cross:['cross_analysis_visualizations/text_vs_image_by_country.png','cross_analysis_visualizations/sentiment_vs_image_volume.png','cross_analysis_visualizations/combined_country_summary.png','cross_analysis_visualizations/balance_ratio_chart.png','cross_analysis_visualizations/coverage_summary.png','cross_analysis_visualizations/sentiment_heatmap.png','cross_analysis_visualizations/human_ai_agreement.png'],
confusion:['confusion_matrix.png']
};

/* ── Load all graphs on page load ── */
function loadAllGraphs(){
var loaded=0,total=0;
Object.keys(GRAPH_MAP).forEach(function(section){
GRAPH_MAP[section].forEach(function(file,idx){
var el=document.getElementById('g-'+section+'-'+idx);
if(!el)return;
total++;
el.innerHTML='<div class="no-data" style="color:#484f58">Loading...</div>';
var img=document.createElement('img');
img.src='/output/'+file;
img.alt=file.replace(/_/g,' ').replace('.png','');
img.onload=function(){loaded++;el.innerHTML='';el.appendChild(img);};
img.onerror=function(){
loaded++;
el.innerHTML='<div class="no-data">'+(section==='images'?'Run Stage 5 (Image Viz) to generate':'Not yet generated')+'</div>';};
});
});
console.log('loadAllGraphs: '+total+' graphs expected');
}

/* ── Filter sections ── */
function filterSection(section,btn){
document.querySelectorAll('.filter-btn').forEach(function(b){b.classList.remove('active');});
btn.classList.add('active');
document.querySelectorAll('.dashboard-section').forEach(function(s){
if(section==='all'){s.style.display='';}
else{s.style.display=s.getAttribute('data-section')===section?'':'none';}
});
}

/* ── Status updates ── */
function updateStatus(s,st,c){var card=document.getElementById('status-'+s);if(card){card.querySelector('.count').textContent=c;card.className='status-card '+st;}}
function updatePipelineStatus(){fetch('/admin/status').then(r=>r.json()).then(d=>{var el=document.getElementById('pipeline-status');if(d.running)el.innerHTML='Pipeline: <span class="running">RUNNING</span> — <span class="script">'+d.script+'</span> ('+d.started_ago+'s)';else el.innerHTML='Pipeline: <span class="idle">IDLE</span>';}).catch(()=>{});}

/* ── Pipeline runners ── */
function runAll(){var stages=['01_scrape_data.py','03_sentiment_analysis.py','04_image_processing.py','05_cross_analysis.py','06_visualize_images.py'];var i=0;var out=document.getElementById('output-log');function next(){if(i>=stages.length){out.textContent+='\\n✓ Full pipeline complete!\\n';loadAllGraphs();return;}var s=stages[i];out.textContent+='['+(i+1)+'/'+stages.length+'] Starting '+s+'...\\n';fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({script:s})}).then(r=>r.json()).then(d=>{out.textContent+=d.message+'\\n';if(d.status==='error'){out.textContent+='Aborting pipeline.\\n';return;}i++;pollThenNext();}).catch(e=>{out.textContent+='Error: '+e+'\\n';});}function pollThenNext(){out.textContent+='Waiting for '+stages[i-1]+' to finish...\\n';var tries=0;function p(){tries++;fetch('/admin/status').then(r=>r.json()).then(d=>{if(!d.running){out.textContent+='✓ '+stages[i-1]+' done.\\n';setTimeout(next,1000);}else if(tries>120){out.textContent+='Timeout waiting for '+stages[i-1]+'\\n';}else{setTimeout(p,5000);}}).catch(()=>setTimeout(p,5000));}p();}next();}
function runStage(script){var out=document.getElementById('output-log');out.textContent+='Starting '+script+'...\\n';fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({script:script})}).then(r=>r.json()).then(d=>{out.textContent+=d.message+'\\n';}).catch(e=>{out.textContent+='Error: '+e+'\\n';});}

/* ── Admin functions ── */
function adminReset(type){var pw=document.getElementById('admin-password').value;if(!pw){document.getElementById('admin-message').innerHTML='<span style="color:#f85149">Enter admin password</span>';return;}
var msg=document.getElementById('admin-message');msg.innerHTML='Processing...';fetch('/admin/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw,reset_type:type})}).then(r=>r.json()).then(d=>{msg.innerHTML=d.success?'<span style="color:#3fb950">✓ '+d.message+'</span>':'<span style="color:#f85149">✗ '+d.message+'</span>';document.getElementById('admin-password').value='';if(d.success)loadAllGraphs();}).catch(e=>{msg.innerHTML='<span style="color:#f85149">Error</span>';});}
function adminResetAndRerun(){var pw=document.getElementById('admin-password').value;if(!pw){document.getElementById('admin-message').innerHTML='<span style="color:#f85149">Enter admin password</span>';return;}
var msg=document.getElementById('admin-message');msg.innerHTML='Resetting...';fetch('/admin/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw,reset_type:'full'})}).then(r=>r.json()).then(d=>{if(d.success){msg.innerHTML='<span style="color:#3fb950">✓ '+d.message+' Starting pipeline sequentially...</span>';document.getElementById('admin-password').value='';loadAllGraphs();setTimeout(runAll,1500);}else msg.innerHTML='<span style="color:#f85149">✗ '+d.message+'</span>';}).catch(()=>{});}
function showLogs(){var out=document.getElementById('output-log');out.textContent='Fetching logs...\\n';fetch('/admin/logs').then(r=>r.json()).then(d=>{out.textContent='=== Pipeline Logs ===\\n';if(d.logs.length===0)out.textContent+='(no logs yet)\\n';else d.logs.forEach(function(l){out.textContent+=l+'\\n';});out.textContent+='\\nStatus: '+(d.running?'RUNNING ('+d.script+')':'IDLE')+'\\n';}).catch(e=>{out.textContent+='Error fetching logs: '+e+'\\n';});}
function toggleAdminPanel(){var s=document.querySelector('.admin-section');s.style.display=s.style.display==='none'?'block':'none';}

/* ── Auto-refresh ── */
setInterval(()=>{fetch('/counts').then(r=>r.json()).then(d=>{if(d.scrape)updateStatus('scrape','completed',d.scrape);if(d.sentiment)updateStatus('sentiment','completed',d.sentiment);if(d.images)updateStatus('images','completed',d.images);if(d.cross)updateStatus('cross','completed',d.cross);if(d.confusion)updateStatus('confusion','completed','✓');});updatePipelineStatus();updateProgressBar();},3000);
setTimeout(updatePipelineStatus,100);
loadAllGraphs();

/* ── Progress Bar ── */
function updateProgressBar(){
fetch('/admin/status').then(r=>r.json()).then(d=>{
var bar=document.getElementById('pipeline-progress');
var label=document.getElementById('progress-label');
var timer=document.getElementById('progress-time');
if(d.running){
bar.style.display='block';
var friendly=d.script?d.script.replace('.py','').replace(/_/g,' '):'Running...';
label.textContent=friendly;
timer.textContent=d.started_ago+'s';
}else{
bar.style.display='none';
}
}).catch(()=>{});
}
</script></body></html>"""

GRADE_HTML = """<!DOCTYPE html><html><head><title>Grade Snippets</title>
<style>*{box-sizing:border-box}body{font-family:monospace;background:#0d1117;color:#c9d1d9;min-height:100vh;margin:0;display:flex;flex-direction:column;align-items:center;padding:20px}
h1{color:#58a6ff;font-size:22px}.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:24px;width:100%;max-width:700px;margin-bottom:16px}
.country-tag{display:inline-block;background:#1f6feb;color:white;padding:3px 10px;border-radius:4px;font-size:12px;margin-bottom:8px}
.snippet-id{float:right;color:#8b949e;font-size:12px}.text-body{font-size:14px;line-height:1.6;margin:12px 0;white-space:pre-wrap;max-height:200px;overflow-y:auto}
.ai-label{padding:8px;background:#21262d;border-radius:4px;font-size:12px;color:#8b949e;margin-top:8px}
.ai-label .pos{color:#2ecc71}.ai-label .neg{color:#e74c3c}.ai-label .neu{color:#f39c12}
.btn-row{display:flex;gap:10px;justify-content:center;margin-top:16px}
.gbtn{border:none;border-radius:6px;padding:14px 28px;cursor:pointer;font-size:16px;font-weight:bold;min-width:140px;transition:transform .1s}
.gbtn:hover{transform:scale(1.05)}.gbtn-neg{background:#e74c3c;color:white}.gbtn-neu{background:#f39c12;color:white}.gbtn-pos{background:#2ecc71;color:white}
.flash{position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;opacity:0;transition:opacity .2s}
.progress-bar{width:100%;max-width:700px;height:8px;background:#21262d;border-radius:4px;margin-bottom:16px;overflow:hidden}
.progress-fill{height:100%;background:#238636;border-radius:4px;transition:width .3s}
.info{color:#8b949e;font-size:12px;text-align:center;margin:8px 0}
a{color:#58a6ff;text-decoration:none}</style></head><body>
<h1>📝 Grade Snippets</h1>
<div class="progress-bar"><div class="progress-fill" id="prog" style="width:0%"></div></div>
<div class="info" id="meta">Loading...</div>
<div class="card" id="snippet-card" style="display:none">
<div><span class="country-tag" id="country"></span><span class="snippet-id" id="sid"></span></div>
<div class="text-body" id="body"></div>
<div class="ai-label" id="ailabel"></div>
</div>
<div class="btn-row" id="btns" style="display:none">
<button class="gbtn gbtn-neg" onclick="grade(0)">Negative (0)</button>
<button class="gbtn gbtn-neu" onclick="grade(1)">Neutral (1)</button>
<button class="gbtn gbtn-pos" onclick="grade(2)">Positive (2)</button>
</div>
<div class="card" id="done" style="display:none;text-align:center"><h2 style="color:#238636">✓ Chunk Complete!</h2>
<p>Return to <a href="/grade">claim next chunk</a> or view <a href="/grade/team">team progress</a>.</p></div>
<div class="card" id="no-data" style="display:none;text-align:center"><h2 style="color:#f39c12">No Data</h2>
<p>Run Stage 2 (Sentiment Analysis) first.</p></div>
<div class="card" id="all-done" style="display:none;text-align:center"><h2 style="color:#58a6ff">All Chunks Claimed!</h2>
<p>Check <a href="/grade/team">team progress</a>.</p></div>
<div class="flash" id="flash"></div>
<p class="info"><a href="/">← Dashboard</a> &nbsp;|&nbsp; <a href="/grade/team">Team Progress →</a></p>
<script>
var snippets=[],idx=0,chunkIdx=-1;
var FETCH_TIMEOUT=20000;
document.addEventListener('keydown',function(e){if(e.key==='0')grade(0);else if(e.key==='1')grade(1);else if(e.key==='2')grade(2);});
function fetchT(url,opts){return new Promise(function(res,rej){var c=new AbortController();var t=setTimeout(function(){c.abort();},FETCH_TIMEOUT);fetch(url,Object.assign({},opts||{},{signal:c.signal})).then(function(r){clearTimeout(t);res(r);}).catch(function(e){clearTimeout(t);rej(e);});});}
function hideAll(){['snippet-card','btns','done','no-data','all-done'].forEach(function(id){document.getElementById(id).style.display='none';});}
function showError(msg){hideAll();document.getElementById('meta').innerHTML='<span style="color:#f85149">'+msg+'</span> <a href="/grade" style="color:#58a6ff;margin-left:8px">Retry</a>';}
function load(){document.getElementById('meta').textContent='Loading snippets...';fetchT('/grade/api/start').then(function(r){if(!r.ok)throw new Error('Server returned '+r.status);return r.json();}).then(function(d){
if(d.error){hideAll();document.getElementById('no-data').style.display='block';document.getElementById('meta').textContent=d.error;return;}
if(d.all_claimed){hideAll();document.getElementById('all-done').style.display='block';document.getElementById('meta').textContent='';return;}
chunkIdx=d.chunk_index;snippets=d.snippets;idx=d.graded_in_chunk;
if(!snippets||snippets.length===0){showError('No snippets received.');return;}
show();}).catch(function(e){if(e.name==='AbortError')showError('Request timed out. Server may be busy generating sample data.');else showError('Failed to load: '+e.message);});}
function show(){if(idx>=snippets.length){document.getElementById('snippet-card').style.display='none';
document.getElementById('btns').style.display='none';document.getElementById('done').style.display='block';return;}
var s=snippets[idx];document.getElementById('country').textContent=s.country||'?';
document.getElementById('sid').textContent='#'+s.snippet_id;
document.getElementById('body').textContent=(s.text||'').substring(0,600);
var lbl=s.label||'',conf=s.confidence?Math.round(s.confidence*100)+'%':'';
var cls=lbl.includes('POS')?'pos':lbl.includes('NEG')?'neg':'neu';
document.getElementById('ailabel').innerHTML='AI says: <span class="'+cls+'">'+lbl+'</span> confidence: '+conf;
document.getElementById('snippet-card').style.display='block';
document.getElementById('btns').style.display='flex';
document.getElementById('meta').textContent='Chunk '+(chunkIdx+1)+' — Snippet '+(idx+1)+'/'+snippets.length;
document.getElementById('prog').style.width=((idx/snippets.length)*100)+'%';
document.getElementById('done').style.display='none';}
function grade(s){if(idx>=snippets.length)return;var sid=snippets[idx].snippet_id;
fetch('/grade/api/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({snippet_id:sid,score:s})}).then(r=>r.json()).then(d=>{if(d.success){
var colors=['rgba(231,76,60,0.3)','rgba(243,156,18,0.3)','rgba(46,204,113,0.3)'];
var fl=document.getElementById('flash');fl.style.background=colors[s];fl.style.opacity='1';
setTimeout(()=>{fl.style.opacity='0';},200);idx++;show();
if(d.chunk_complete){fetch('/refresh-cross-analysis').catch(()=>{});}});}
load();
</script></body></html>"""

TEAM_HTML = """<!DOCTYPE html><html><head><title>Team Progress</title>
<style>*{box-sizing:border-box}body{font-family:monospace;background:#0d1117;color:#c9d1d9;min-height:100vh;margin:0;display:flex;flex-direction:column;align-items:center;padding:20px}
h1{color:#58a6ff;font-size:22px}.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:24px;width:100%;max-width:800px;margin-bottom:16px}
.stat{display:inline-block;margin:0 20px;text-align:center}.stat .num{font-size:36px;font-weight:bold;color:#58a6ff}.stat .lbl{color:#8b949e;font-size:12px}
table{width:100%;border-collapse:collapse;margin-top:12px}th{color:#8b949e;text-align:left;border-bottom:1px solid #30363d;padding:8px}
td{padding:8px;border-bottom:1px solid #21262d}.pbar{width:100%;height:10px;background:#21262d;border-radius:4px;overflow:hidden}
.pfill{height:100%;background:#238636;border-radius:4px}a{color:#58a6ff;text-decoration:none}
.info{color:#8b949e;font-size:12px}</style></head><body>
<h1>📊 Team Grading Progress</h1>
<div class="card"><div style="text-align:center">
<div class="stat"><div class="num" id="graded">—</div><div class="lbl">Graded</div></div>
<div class="stat"><div class="num" id="total">—</div><div class="lbl">Total</div></div>
<div class="stat"><div class="num" id="chunks">—</div><div class="lbl">Chunks Claimed</div></div>
<div class="stat"><div class="num" id="chunkstotal">—</div><div class="lbl">Total Chunks</div></div>
</div></div>
<div class="card"><h3 style="color:#58a6ff">Per-Grader Progress</h3>
<table><tr><th>Grader</th><th>Chunk</th><th>Progress</th><th>Done</th></tr>
<tbody id="tbody"></tbody></table></div>
<p class="info"><a href="/grade">← Grade</a> &nbsp;|&nbsp; <a href="/">Dashboard →</a> &nbsp;|&nbsp; Auto-refreshes every 15s</p>
<script>
function load(){fetch('/grade/api/progress').then(r=>r.json()).then(d=>{
document.getElementById('graded').textContent=d.total_graded;
document.getElementById('total').textContent=d.total_snippets;
document.getElementById('chunks').textContent=d.chunks_claimed;
document.getElementById('chunkstotal').textContent=d.total_chunks;
var tb=document.getElementById('tbody');tb.innerHTML='';
d.graders.forEach(function(g){var pct=Math.round(g.graded/g.total*100);
tb.innerHTML+='<tr><td>'+g.id+'</td><td>'+g.chunk+'</td><td><div class="pbar"><div class="pfill" style="width:'+pct+'%"></div></div></td><td>'+g.graded+'/'+g.total+'</td></tr>';});});}
load();setInterval(load,15000);
</script></body></html>"""

HISTORY_HTML = """<!DOCTYPE html><html><head><title>Scrape History</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:linear-gradient(160deg,#0d1117,#161b22,#0d1117);color:#e6edf3;min-height:100vh}
.container{max-width:1200px;margin:0 auto;padding:20px}
header{background:linear-gradient(135deg,#161b22,#1c2333);border:1px solid #30363d;border-radius:12px;padding:16px 24px;margin-bottom:24px;display:flex;justify-content:space-between;align-items:center}
header h1{color:#58a6ff;font-size:20px}
header a{color:#8b5cf6;text-decoration:none;font-size:13px;border:1px solid #8b5cf6;padding:6px 14px;border-radius:6px}

/* Summary cards */
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:28px}
.summary-card{background:linear-gradient(135deg,#161b22,#1c2333);border:1px solid #30363d;border-radius:10px;padding:20px;text-align:center}
.summary-card h3{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.summary-card .num{font-size:32px;font-weight:700;color:#58a6ff}
.summary-card .sub{color:#8b949e;font-size:12px;margin-top:4px}

/* Tables */
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;margin-bottom:20px}
.card h2{color:#58a6ff;font-size:16px;margin-bottom:14px}
table{width:100%;border-collapse:collapse}
th{color:#8b949e;text-align:left;border-bottom:1px solid #30363d;padding:10px 12px;font-size:12px;text-transform:uppercase;letter-spacing:.5px}
td{padding:10px 12px;border-bottom:1px solid #21262d;font-size:13px}
tr:hover td{background:#161b2280}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.tag-ddg{background:#1f6feb30;color:#58a6ff}
.tag-wiki{background:#23863630;color:#3fb950}
.tag-other{background:#8b5cf630;color:#c4b5fd}
.count-bar{height:6px;background:#21262d;border-radius:3px;overflow:hidden;min-width:60px;display:inline-block;vertical-align:middle}
.count-bar .fill{height:100%;background:linear-gradient(90deg,#1f6feb,#58a6ff);border-radius:3px}
.no-data{text-align:center;padding:40px;color:#484f58;font-style:italic}
.timestamp{color:#8b949e;font-size:12px}
</style></head><body><div class="container">

<header><h1>📜 Data Collection History</h1>
<div><a href="/">← Dashboard</a></div></header>

<!-- Summary -->
<div class="summary-grid" id="summary">
<div class="summary-card"><h3>Total Text Records</h3><div class="num" id="total-text">—</div><div class="sub" id="text-sources"></div></div>
<div class="summary-card"><h3>Total Images</h3><div class="num" id="total-images">—</div><div class="sub" id="img-sources"></div></div>
<div class="summary-card"><h3>Countries</h3><div class="num" id="total-countries">—</div><div class="sub">with data</div></div>
<div class="summary-card"><h3>Last Scraped</h3><div class="num timestamp" id="last-scrape" style="font-size:18px">—</div></div>
</div>

<!-- Text by Country -->
<div class="card"><h2>📝 Text Records by Country</h2>
<div id="text-table"><div class="no-data">No text data yet. Run Stage 1 (Scrape) first.</div></div></div>

<!-- Images by Country -->
<div class="card"><h2>🖼️ Images by Country</h2>
<div id="image-table"><div class="no-data">No image data yet. Run Stage 1 (Scrape) first.</div></div></div>

</div>

<script>
function load(){fetch('/history/api').then(r=>r.json()).then(d=>{
document.getElementById('total-text').textContent=d.total_text||0;
document.getElementById('total-images').textContent=d.total_images||0;
document.getElementById('total-countries').textContent=d.countries?d.countries.length:0;
document.getElementById('last-scrape').textContent=d.last_scrape||'Never';

/* Source breakdown labels */
var ts=d.text_sources||{};
var tsParts=[];
if(ts.ddg_snippet)tsParts.push('DuckDuckGo: '+ts.ddg_snippet);
if(ts.full_page)tsParts.push('Full Page: '+ts.full_page);
document.getElementById('text-sources').textContent=tsParts.join(' · ')||'—';

var is=d.image_sources||{};
var isParts=[];
if(is.ddg_image)isParts.push('DuckDuckGo: '+is.ddg_image);
if(is.wikimedia)isParts.push('Wikimedia: '+is.wikimedia);
if(is.other)isParts.push('Other: '+is.other);
document.getElementById('img-sources').textContent=isParts.join(' · ')||'—';

/* Text table */
var tt=document.getElementById('text-table');
if(d.text_by_country&&d.text_by_country.length){
var maxT=Math.max.apply(null,d.text_by_country.map(function(c){return c.count;}));
var h='<table><tr><th>Country</th><th>Records</th><th style="width:40%">Volume</th><th>Sources</th></tr>';
d.text_by_country.forEach(function(r){
var pct=Math.round(r.count/maxT*100);
var srcTags=[];
if(r.ddg_snippet)srcTags.push('<span class="tag tag-ddg">DDG ('+r.ddg_snippet+')</span>');
if(r.full_page)srcTags.push('<span class="tag tag-wiki">Page ('+r.full_page+')</span>');
if(r.other)srcTags.push('<span class="tag tag-other">Other ('+r.other+')</span>');
h+='<tr><td><strong>'+r.country+'</strong></td><td>'+r.count+'</td>';
h+='<td><div class="count-bar" style="width:100%"><div class="fill" style="width:'+pct+'%"></div></div></td>';
h+='<td>'+srcTags.join(' ')+'</td></tr>';
});
h+='</table>';tt.innerHTML=h;
}

/* Image table */
var it=document.getElementById('image-table');
if(d.image_by_country&&d.image_by_country.length){
var maxI=Math.max.apply(null,d.image_by_country.map(function(c){return c.count;}));
var h='<table><tr><th>Country</th><th>Images</th><th style="width:40%">Volume</th><th>Sources</th></tr>';
d.image_by_country.forEach(function(r){
var pct=Math.round(r.count/maxI*100);
var srcTags=[];
if(r.ddg_image)srcTags.push('<span class="tag tag-ddg">DDG ('+r.ddg_image+')</span>');
if(r.wikimedia)srcTags.push('<span class="tag tag-wiki">Wiki ('+r.wikimedia+')</span>');
if(r.other)srcTags.push('<span class="tag tag-other">Other ('+r.other+')</span>');
h+='<tr><td><strong>'+r.country+'</strong></td><td>'+r.count+'</td>';
h+='<td><div class="count-bar" style="width:100%"><div class="fill" style="width:'+pct+'%"></div></div></td>';
h+='<td>'+srcTags.join(' ')+'</td></tr>';
});
h+='</table>';it.innerHTML=h;
}
}).catch(()=>{});}
load();
</script></body></html>"""

# ── Auth Routes ───────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if _key_valid(request.form.get("key", "")):
            session["authenticated"] = True
            return redirect(request.args.get("next") or "/")
        return render_template_string(LOGIN_HTML)
    return render_template_string(LOGIN_HTML)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/")
@require_auth
def dashboard():
    return render_template_string(DASHBOARD_HTML)

# ── Grading Routes ────────────────────────────────────────────────────────────

@app.route("/grade")
@require_auth
def grade_page():
    return render_template_string(GRADE_HTML)

@app.route("/grade/team")
@require_auth
def team_page():
    return render_template_string(TEAM_HTML)

@app.route("/grade/api/start")
@require_auth
def grade_start():
    if not _ensure_sample():
        return jsonify({"error": "No sentiment data. Run Stage 2 first."})
    gid = _get_or_create_grader_id()
    ci = _claim_chunk(gid)
    if ci is None:
        return jsonify({"all_claimed": True})
    snippets_df = _get_chunk_snippets(ci)
    grades = _read_grades()
    graded_ids = {g["snippet_id"] for g in grades if g["grader_id"] == gid}
    rows = snippets_df.to_dict("records")
    graded_in_chunk = sum(1 for r in rows if str(r.get("snippet_id", "")) in graded_ids)
    return jsonify({
        "chunk_index": ci,
        "graded_in_chunk": graded_in_chunk,
        "snippets": rows
    })

@app.route("/grade/api/submit", methods=["POST"])
@require_auth
def grade_submit():
    d = request.get_json() or {}
    sid = d.get("snippet_id")
    score = d.get("score")
    if sid is None or score is None:
        return jsonify({"success": False, "error": "Missing data"}), 400
    gid = _get_or_create_grader_id()
    _append_grade(sid, gid, int(score))

    # Detect chunk completion
    chunk_complete = False
    try:
        assignments = _read_assignments()
        for a in assignments:
            if a["grader_id"] == gid:
                ci = int(a["chunk_index"])
                start = ci * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, SAMPLE_TARGET)
                chunk_snippet_ids = set(str(i) for i in range(start, end))
                grades = _read_grades()
                graded_in_chunk = {g["snippet_id"] for g in grades
                                   if g["grader_id"] == gid and g["snippet_id"] in chunk_snippet_ids}
                if len(graded_in_chunk) >= (end - start):
                    chunk_complete = True
                break
    except Exception:
        pass

    return jsonify({"success": True, "chunk_complete": chunk_complete})

@app.route("/grade/api/progress")
@require_auth
def grade_progress():
    if not GRADE_SAMPLE.exists():
        return jsonify({"total_snippets": 0, "total_graded": 0, "total_chunks": 0, "chunks_claimed": 0, "graders": []})
    import pandas as pd
    total_snippets = len(pd.read_csv(GRADE_SAMPLE))
    total_chunks = _total_chunks()
    assignments = _read_assignments()
    grades = _read_grades()
    graders = []
    seen = set()
    for a in assignments:
        gid = a["grader_id"]
        ci = int(a["chunk_index"])
        chunk_size = min(CHUNK_SIZE, total_snippets - ci * CHUNK_SIZE)
        done = sum(1 for g in grades if g["grader_id"] == gid and int(g["snippet_id"]) // CHUNK_SIZE == ci)
        if gid not in seen:
            graders.append({"id": gid, "chunk": ci, "graded": done, "total": chunk_size})
            seen.add(gid)
        else:
            for gr in graders:
                if gr["id"] == gid:
                    gr["graded"] += done
                    gr["total"] += chunk_size
    return jsonify({
        "total_snippets": total_snippets,
        "total_graded": len(grades),
        "total_chunks": total_chunks,
        "chunks_claimed": len(set(a["chunk_index"] for a in assignments)),
        "graders": graders
    })

# ── History Routes ────────────────────────────────────────────────────────────

@app.route("/history")
@require_auth
def history_page():
    return render_template_string(HISTORY_HTML)

@app.route("/history/api")
@require_auth
def history_api():
    result = {
        "total_text": 0, "total_images": 0,
        "countries": [], "last_scrape": "Never",
        "text_sources": {}, "image_sources": {},
        "text_by_country": [], "image_by_country": [],
    }
    try:
        import pandas as pd
        # Text data
        text_path = DATA_DIR / "text_raw.csv"
        if text_path.exists():
            df = pd.read_csv(text_path)
            result["total_text"] = len(df)
            if "source" in df.columns:
                src = df["source"].value_counts().to_dict()
                result["text_sources"] = {k: int(v) for k, v in src.items()}
            if "country" in df.columns:
                grp = df.groupby("country")
                for country, group in grp:
                    row = {"country": country, "count": len(group)}
                    if "source" in group.columns:
                        for s, c in group["source"].value_counts().items():
                            row[s] = int(c)
                    result["text_by_country"].append(row)
            result["countries"] = sorted(df["country"].unique().tolist()) if "country" in df.columns else []
            import datetime
            mtime = text_path.stat().st_mtime
            result["last_scrape"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        # Image data
        img_path = DATA_DIR / "image_metadata.csv"
        if img_path.exists():
            df = pd.read_csv(img_path)
            result["total_images"] = len(df)
            if "source" in df.columns:
                src = df["source"].value_counts().to_dict()
                result["image_sources"] = {k: int(v) for k, v in src.items()}
            if "country" in df.columns:
                grp = df.groupby("country")
                for country, group in grp:
                    row = {"country": country, "count": len(group)}
                    if "source" in group.columns:
                        for s, c in group["source"].value_counts().items():
                            row[s] = int(c)
                    result["image_by_country"].append(row)
                # Merge countries
                img_countries = set(result["countries"])
                img_countries.update(df["country"].unique().tolist())
                result["countries"] = sorted(img_countries)

            # Use most recent mtime
            import datetime
            mtime = img_path.stat().st_mtime
            if mtime > (text_path.stat().st_mtime if text_path.exists() else 0):
                result["last_scrape"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        result["error"] = str(e)
    return jsonify(result)

# ── Pipeline Routes ───────────────────────────────────────────────────────────

@app.route("/counts")
@require_auth
def get_counts():
    counts = {"scrape": 0, "sentiment": 0, "images": 0, "cross": 0, "confusion": 0}
    try:
        import pandas as pd
        for key, path in [("scrape", DATA_DIR/"text_raw.csv"), ("sentiment", DATA_DIR/"text_with_sentiment.csv")]:
            if path.exists(): counts[key] = len(pd.read_csv(path))
        img_meta = DATA_DIR / "image_metadata.csv"
        if img_meta.exists(): counts["images"] = len(pd.read_csv(img_meta))
        elif IMAGES_DIR.exists(): counts["images"] = len(list(IMAGES_DIR.rglob("*.*")))
        if (OUTPUT_DIR/"cross_analysis.csv").exists(): counts["cross"] = len(pd.read_csv(OUTPUT_DIR/"cross_analysis.csv"))
        if (OUTPUT_DIR/"classification_report.csv").exists(): counts["confusion"] = 1
    except: pass
    return jsonify(counts)

@app.route("/refresh-cross-analysis")
@require_auth
def refresh_cross_analysis():
    """Trigger cross-analysis refresh (used after chunk completion)."""
    with _job_lock:
        if job_state["running"]:
            return jsonify({"started": False, "reason": "Pipeline busy"})
        job_state["running"] = True
        job_state["script"] = "05_cross_analysis.py (auto-refresh)"
        job_state["started_at"] = time.time()

    def go():
        try:
            r = subprocess.run(["python", "05_cross_analysis.py"],
                               capture_output=True, text=True, timeout=300)
            log_entry = f"Auto cross-analysis refresh: exit {r.returncode}"
        except Exception as e:
            log_entry = f"Auto cross-analysis error: {e}"
        finally:
            with _job_lock:
                job_state["running"] = False
                job_state["script"] = None
                job_state["started_at"] = None
                if len(job_state["log"]) >= MAX_LOG_ENTRIES:
                    job_state["log"] = job_state["log"][-MAX_LOG_ENTRIES:]
                job_state["log"].append(log_entry)

    t = threading.Thread(target=go)
    t.start()
    return jsonify({"started": True})

@app.route("/admin/status")
@require_auth
def admin_status():
    with _job_lock:
        sa = 0
        if job_state["running"] and job_state["started_at"]:
            sa = int(time.time() - job_state["started_at"])
        return jsonify({"running": job_state["running"], "script": job_state["script"], "started_ago": sa})

@app.route("/admin/reset", methods=["POST"])
@limiter.limit("5 per minute")  # Rate limit admin reset operations
@require_auth
def admin_reset():
    d = request.get_json() or {}
    if not _admin_key_valid(d.get("password", "")):
        return jsonify({"success": False, "message": "Invalid admin password"}), 403
    rt = d.get("reset_type", "soft")
    # Validate reset_type against whitelist to prevent path traversal
    if rt not in ("soft", "analysis", "full"):
        return jsonify({"success": False, "message": "Invalid reset type"}), 400
    deleted = []; errors = []; dc = 0

    # Thread-safe state clearing
    with _job_lock:
        job_state["running"] = False; job_state["script"] = None; job_state["started_at"] = None
        job_state["log"] = []

    def _delete(p):
        nonlocal dc
        if p.is_file():
            try: p.unlink(); deleted.append(str(p)); dc += 1
            except Exception as e: errors.append(f"{p}: {e}")
        elif p.is_dir():
            try:
                shutil.rmtree(p); deleted.append(str(p)+"/"); dc += 1
            except Exception as e: errors.append(f"{p}: {e}")

    if rt == "soft":
        # Soft reset: clear logs + grading session data, keep scraped/raw data
        _delete(GRADER_ASSIGN)
        _delete(HUMAN_GRADES)
        _delete(GRADE_SAMPLE)
        # Clear output dir contents but keep the folder
        if OUTPUT_DIR.exists():
            for f in OUTPUT_DIR.iterdir():
                _delete(f)
        msg = f"Soft reset: cleared logs & grading state. Deleted {dc} items."
        if errors: msg += f" ({len(errors)} errors)"
        return jsonify({"success": True, "message": msg, "deleted_count": dc,
                        "deleted_files": deleted[:30], "errors": errors[:5]})

    # Grading files to include in analysis/full reset
    grading_files = [GRADE_SAMPLE, GRADER_ASSIGN, HUMAN_GRADES]

    targets = {"analysis": [DATA_DIR/"text_with_sentiment.csv", DATA_DIR/"image_metadata.csv", OUTPUT_DIR] + grading_files,
               "full": [DATA_DIR/"text_raw.csv", DATA_DIR/"text_with_sentiment.csv", DATA_DIR/"image_metadata.csv",
                        OUTPUT_DIR, IMAGES_DIR] + grading_files}
    for p in targets.get(rt, []):
        _delete(p)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if rt == "full": IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    msg = f"{rt.title()} reset. Deleted {dc} items."
    if errors: msg += f" ({len(errors)} errors)"
    return jsonify({"success": True, "message": msg, "deleted_count": dc,
                    "deleted_files": deleted[:30], "errors": errors[:5]})

@app.route("/admin/logs")
@require_auth
def admin_logs():
    return jsonify({"logs": job_state.get("log", []), "running": job_state["running"], "script": job_state["script"]})

@app.route("/run", methods=["POST"])
@require_auth
def run_stage():
    d = request.get_json() or {}
    script = d.get("script", "01_scrape_data.py")
    
    # Validate script against whitelist (security fix for path traversal)
    if not _is_script_allowed(script):
        return jsonify({"status": "error", "message": "Invalid or disallowed script"})
    
    # Thread-safe check for running state
    with _job_lock:
        if job_state["running"]:
            return jsonify({"status": "error", "message": "Pipeline already running"})
        job_state["running"] = True
        job_state["script"] = script
        job_state["started_at"] = time.time()
    
    def go():
        with _job_lock:
            running_state = job_state["running"]
            script_name = job_state["script"]
        
        try:
            r = subprocess.run(["python", script_name], capture_output=True, text=True, timeout=600)
            # Log exit code
            log_entry = f"Done {script_name}: exit {r.returncode}"
            # Capture stderr output for debugging (show last 500 chars if non-empty)
            stderr_output = r.stderr.strip() if r.stderr else ""
            if r.returncode != 0 and stderr_output:
                # Show error context from stderr
                error_lines = stderr_output.split('\n')[-10:]  # Last 10 lines of error
                log_entry += f"\n  STDERR ({len(stderr_output)} chars):\n    " + '\n    '.join(error_lines)
            elif r.stdout and len(r.stdout) > 2000:
                log_entry += f"\n  STDOUT ({len(r.stdout)} chars - truncated)"
            elif r.returncode == 0 and stderr_output:
                # Log warnings even on success
                log_entry += f"\n  Warnings: {stderr_output[:200]}..."
        except Exception as e:
            log_entry = f"Error {script_name}: {e}"
        finally:
            with _job_lock:
                job_state["running"] = False
                job_state["script"] = None
                job_state["started_at"] = None
                # Truncate logs to prevent memory leak (max 1000 entries)
                if len(job_state["log"]) >= MAX_LOG_ENTRIES:
                    job_state["log"] = job_state["log"][-MAX_LOG_ENTRIES:]
                job_state["log"].append(log_entry)
    
    t = threading.Thread(target=go)
    with _job_lock:
        job_state["thread"] = t
    t.start()
    return jsonify({"status": "started", "message": f"Started {script}", "script": script})

# ── Report Routes ─────────────────────────────────────────────────────────────

@app.route("/report/generate")
@require_auth
def report_generate():
    """Run 08_generate_report.py and return status."""
    with _job_lock:
        if job_state["running"]:
            return jsonify({"status": "error", "message": "Pipeline busy"})
        job_state["running"] = True
        job_state["script"] = "08_generate_report.py"
        job_state["started_at"] = time.time()

    def go():
        try:
            r = subprocess.run(["python", "08_generate_report.py"],
                               capture_output=True, text=True, timeout=120)
            with _job_lock:
                job_state["log"].append(f"Report generation: exit {r.returncode}")
        except Exception as e:
            with _job_lock:
                job_state["log"].append(f"Report error: {e}")
        finally:
            with _job_lock:
                job_state["running"] = False
                job_state["script"] = None
                job_state["started_at"] = None

    t = threading.Thread(target=go)
    t.start()
    return jsonify({"status": "started", "message": "Generating report..."})


@app.route("/report/download/csv")
@require_auth
def report_download_csv():
    """Serve the CSV report, generating on-demand if needed."""
    csv_path = OUTPUT_DIR / "pipeline_report.csv"
    if not csv_path.exists():
        try:
            subprocess.run(["python", "08_generate_report.py", "--format", "csv"],
                           capture_output=True, text=True, timeout=120)
        except Exception:
            pass
    if csv_path.exists():
        return send_file(str(csv_path), mimetype="text/csv",
                         as_attachment=True, download_name="pipeline_report.csv")
    return jsonify({"error": "Report not available. Run pipeline first."}), 404


@app.route("/report/download/pdf")
@require_auth
def report_download_pdf():
    """Serve the PDF report, generating on-demand if needed."""
    pdf_path = OUTPUT_DIR / "pipeline_report.pdf"
    if not pdf_path.exists():
        try:
            subprocess.run(["python", "08_generate_report.py", "--format", "pdf"],
                           capture_output=True, text=True, timeout=120)
        except Exception:
            pass
    if pdf_path.exists():
        return send_file(str(pdf_path), mimetype="application/pdf",
                         as_attachment=True, download_name="pipeline_report.pdf")
    return jsonify({"error": "Report not available. Run pipeline first."}), 404


@app.route("/output/<path:path>")
@require_auth
def serve_output(path):
    fp = OUTPUT_DIR / path
    if not fp.exists():
        return "", 404
    # Cache-busting: ensure fresh data on reload / different users
    resp = send_file(str(fp))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# ── Design Evaluation Route ───────────────────────────────────────────────────

WEIGHTS_JSON = OUTPUT_DIR / "design_weights.json"

# Ordinal maps (must match 05_cross_analysis.py exactly)
_EVAL_ORNAMENTATION_MAP = {
    "plain": 0.0, "minimal": 0.25, "moderate": 0.5,
    "ornate": 0.75, "highly_ornate": 1.0,
}
_EVAL_AESTHETIC_MAP = {"low": 0.0, "medium": 0.5, "high": 1.0}
_EVAL_ATTRIBUTES = ["ornamentation_level", "cultural_elements",
                    "aesthetic_appeal", "motif_diversity"]

EVALUATE_HTML = r"""<!DOCTYPE html><html><head><title>Design Evaluation</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:linear-gradient(160deg,#0d1117,#161b22,#0d1117);color:#e6edf3;min-height:100vh}
.container{max-width:1100px;margin:0 auto;padding:20px}
header{background:linear-gradient(135deg,#161b22,#1c2333);border:1px solid #30363d;border-radius:12px;padding:16px 24px;margin-bottom:24px;display:flex;justify-content:space-between;align-items:center}
header h1{color:#58a6ff;font-size:20px}
header a{color:#8b5cf6;text-decoration:none;font-size:13px;border:1px solid #8b5cf6;padding:6px 14px;border-radius:6px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:24px;margin-bottom:20px}
.card h2{color:#58a6ff;font-size:16px;margin-bottom:16px}

/* Drop zone */
.drop-zone{border:2px dashed #30363d;border-radius:10px;padding:40px;text-align:center;cursor:pointer;transition:all .3s;margin-bottom:16px}
.drop-zone:hover,.drop-zone.drag-over{border-color:#58a6ff;background:#58a6ff10}
.drop-zone p{color:#8b949e;font-size:14px}
.drop-zone .hint{font-size:12px;color:#484f58;margin-top:8px}

/* Preview thumbnails */
.preview-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.preview-item{position:relative;width:80px;height:80px;border-radius:8px;overflow:hidden;border:1px solid #30363d}
.preview-item img{width:100%;height:100%;object-fit:cover}
.preview-item .remove{position:absolute;top:2px;right:2px;background:#f85149;border:none;color:#fff;border-radius:50%;width:20px;height:20px;cursor:pointer;font-size:12px;display:flex;align-items:center;justify-content:center}

/* Country selects */
.country-row{display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
.country-row label{color:#8b949e;font-size:13px;font-weight:600}
.country-row select{background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#c9d1d9;padding:8px 14px;font-size:13px;min-width:140px}

/* Evaluate button */
.btn-eval{background:linear-gradient(135deg,#238636,#2ea043);color:#fff;border:none;border-radius:8px;padding:12px 32px;font-size:15px;font-weight:600;cursor:pointer;transition:all .2s;width:100%}
.btn-eval:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(35,134,54,.4)}
.btn-eval:disabled{opacity:.5;cursor:not-allowed;transform:none;box-shadow:none}

/* Loading */
.loading{text-align:center;padding:40px;color:#8b949e;display:none}
.loading .spinner{display:inline-block;width:24px;height:24px;border:3px solid #30363d;border-top-color:#58a6ff;border-radius:50%;animation:spin .8s linear infinite;margin-bottom:8px}
@keyframes spin{to{transform:rotate(360deg)}}

/* Results */
.results{display:none}
.result-card{background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:20px;margin-bottom:16px}
.result-header{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.result-thumb{width:60px;height:60px;border-radius:6px;object-fit:cover;border:1px solid #30363d}
.result-name{font-weight:600;font-size:15px;color:#e6edf3}
.result-prediction{font-size:12px;padding:3px 10px;border-radius:4px;font-weight:600}
.pred-positive{background:#23863640;color:#3fb950}
.pred-neutral{background:#d2992240;color:#d29922}
.pred-negative{background:#f8514940;color:#f85149}

/* Score bar */
.score-row{display:flex;align-items:center;gap:12px;margin:8px 0}
.score-label{width:140px;font-size:12px;color:#8b949e;text-align:right}
.score-bar-bg{flex:1;height:16px;background:#21262d;border-radius:4px;overflow:hidden;position:relative}
.score-bar-fill{height:100%;border-radius:4px;transition:width .5s}
.score-val{font-size:12px;color:#c9d1d9;min-width:45px}

/* Benchmark */
.benchmark{margin-top:12px;padding:12px;background:#161b22;border-radius:6px;font-size:13px}
.benchmark .gap-negative{color:#f85149}
.benchmark .gap-positive{color:#3fb950}
.benchmark .gap-neutral{color:#d29922}

/* Recommendation */
.recommendation{margin-top:10px;padding:10px 14px;background:#1f6feb15;border-left:3px solid #58a6ff;border-radius:0 6px 6px 0;font-size:13px;color:#c9d1d9}
.weakness{margin-top:6px;padding:10px 14px;background:#f8514915;border-left:3px solid #f85149;border-radius:0 6px 6px 0;font-size:13px;color:#c9d1d9}

/* Download buttons */
.download-row{display:flex;gap:10px;margin-top:20px;justify-content:center}
.dl-btn{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:8px 20px;cursor:pointer;font-size:13px;text-decoration:none;transition:all .2s}
.dl-btn:hover{background:#30363d;color:#fff}

/* No-weights warning */
.no-weights{background:#d2992220;border:1px solid #d29922;border-radius:8px;padding:16px;text-align:center;color:#d29922;margin-bottom:20px}
.no-weights a{color:#58a6ff}
</style></head><body><div class="container">

<header>
<h1>🎨 Design Evaluation Tool</h1>
<a href="/">← Dashboard</a>
</header>

<div id="weights-warning" class="no-weights" style="display:none">
<strong>⚠ No correlation model found.</strong> Run the <a href="/">full pipeline</a> (Stages 1–4) first to generate design_weights.json.
</div>

<div class="card">
<h2>Upload Candidate Designs</h2>
<div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
<p>📂 Drop images here or click to browse</p>
<p class="hint">Max 5 images · PNG, JPG, WebP</p>
</div>
<input type="file" id="file-input" accept="image/*" multiple style="display:none">
<div class="preview-row" id="preview-row"></div>

<div class="country-row" id="country-row">
<label>Compare against:</label>
<span id="benchmark-selects">Loading...</span>
</div>

<button class="btn-eval" id="btn-eval" onclick="evaluateDesigns()" disabled>Evaluate Designs</button>
</div>

<div class="loading" id="loading">
<div class="spinner"></div>
<p>Analyzing designs with VLM...</p>
</div>

<div class="results" id="results">
<h2 style="color:#58a6ff;margin-bottom:16px">Evaluation Results</h2>
<div id="results-container"></div>
<div class="download-row">
<a class="dl-btn" href="/evaluate/api/download/csv" id="dl-csv">📊 Download CSV</a>
<a class="dl-btn" href="/evaluate/api/download/pdf" id="dl-pdf" target="_blank">📄 Download PDF</a>
</div>
</div>

</div>

<script>
var files=[];
var countries=[];

/* Check weights */
fetch('/evaluate/api/status').then(r=>r.json()).then(d=>{
if(!d.has_weights) document.getElementById('weights-warning').style.display='block';
document.getElementById('btn-eval').disabled=!d.has_weights;
countries=d.countries||[];
renderBenchmarkSelects();
});

function renderBenchmarkSelects(){
var c=document.getElementById('benchmark-selects');
if(!countries.length){c.innerHTML='<span style="color:#8b949e">No countries in model yet</span>';return;}
var html='<select id="bench1">'+countries.map(c=>'<option value="'+c+'">'+c+'</option>').join('')+'</select>';
if(countries.length>1) html+=' <select id="bench2"><option value="">(none)</option>'+countries.map(c=>'<option value="'+c+'">'+c+'</option>').join('')+'</select>';
c.innerHTML=html;
}

/* File handling */
document.getElementById('file-input').addEventListener('change',function(e){addFiles(e.target.files);});
var dz=document.getElementById('drop-zone');
dz.addEventListener('dragover',function(e){e.preventDefault();dz.classList.add('drag-over');});
dz.addEventListener('dragleave',function(){dz.classList.remove('drag-over');});
dz.addEventListener('drop',function(e){e.preventDefault();dz.classList.remove('drag-over');addFiles(e.dataTransfer.files);});

function addFiles(fileList){
for(var i=0;i<fileList.length&&files.length<5;i++){
if(!fileList[i].type.startsWith('image/'))continue;
files.push(fileList[i]);
}
renderPreviews();
document.getElementById('btn-eval').disabled=files.length===0;
}

function removeFile(idx){files.splice(idx,1);renderPreviews();document.getElementById('btn-eval').disabled=files.length===0;}

function renderPreviews(){
var row=document.getElementById('preview-row');row.innerHTML='';
files.forEach(function(f,i){
var div=document.createElement('div');div.className='preview-item';
var img=document.createElement('img');img.src=URL.createObjectURL(f);
var btn=document.createElement('button');btn.className='remove';btn.textContent='×';btn.onclick=function(){removeFile(i);};
div.appendChild(img);div.appendChild(btn);row.appendChild(div);
});
}

/* Evaluate */
function evaluateDesigns(){
if(!files.length)return;
var fd=new FormData();
files.forEach(function(f){fd.append('images',f);});
var b1=document.getElementById('bench1');
var b2=document.getElementById('bench2');
if(b1)fd.append('benchmark1',b1.value);
if(b2&&b2.value)fd.append('benchmark2',b2.value);

document.getElementById('loading').style.display='block';
document.getElementById('results').style.display='none';

fetch('/evaluate/api/analyze',{method:'POST',body:fd})
.then(r=>r.json())
.then(d=>{
document.getElementById('loading').style.display='none';
if(d.error){alert(d.error);return;}
renderResults(d.results,d.benchmarks);
document.getElementById('results').style.display='block';
})
.catch(e=>{document.getElementById('loading').style.display='none';alert('Error: '+e);});
}

function renderResults(results,benchmarks){
var c=document.getElementById('results-container');c.innerHTML='';
results.sort(function(a,b){return b.score-a.score;});
results.forEach(function(r,i){
var predClass=r.predicted_label==='Positive'?'pred-positive':r.predicted_label==='Negative'?'pred-negative':'pred-neutral';
var barColor=r.score>=0.6?'#3fb950':r.score>=0.35?'#d29922':'#f85149';

var html='<div class="result-card">';
html+='<div class="result-header">';
html+='<img class="result-thumb" src="data:image/png;base64,'+r.thumbnail+'">';
html+='<div><div class="result-name">'+(r.name||'Design '+(i+1))+'</div>';
html+='<span class="result-prediction '+predClass+'">'+r.predicted_label+'</span>';
html+=' <span style="color:#8b949e;font-size:13px">Score: '+r.score.toFixed(2)+'</span></div></div>';

/* Overall bar */
html+='<div class="score-row"><span class="score-label">Overall Score</span>';
html+='<div class="score-bar-bg"><div class="score-bar-fill" style="width:'+Math.round(r.score*100)+'%;background:'+barColor+'"></div></div>';
html+='<span class="score-val">'+r.score.toFixed(2)+'</span></div>';

/* Per-attribute bars */
var attrs=['ornamentation_level','cultural_elements','aesthetic_appeal','motif_diversity'];
var labels=['Oramentation','Cultural Elements','Aesthetic Appeal','Motif Diversity'];
attrs.forEach(function(attr,idx){
var val=r.attribute_scores[attr]||0;
var aColor=val>=0.6?'#3fb950':val>=0.35?'#d29922':'#f85149';
html+='<div class="score-row"><span class="score-label">'+labels[idx]+'</span>';
html+='<div class="score-bar-bg"><div class="score-bar-fill" style="width:'+Math.round(val*100)+'%;background:'+aColor+'"></div></div>';
html+='<span class="score-val">'+val.toFixed(2)+'</span></div>';
});

/* Benchmark comparison */
if(benchmarks&&r.benchmark_comparison){
var bc=r.benchmark_comparison;
html+='<div class="benchmark">';
bc.forEach(function(b){
var gap=b.gap;
var gapClass=gap<0?'gap-negative':gap>0?'gap-positive':'gap-neutral';
var gapStr=(gap>=0?'+':'')+gap.toFixed(2);
html+='<div style="margin-bottom:4px"><strong>'+b.country+'</strong> avg: '+b.benchmark_score.toFixed(2)+' &nbsp;|&nbsp; <span class="'+gapClass+'">Gap: '+gapStr+'</span></div>';
});
html+='</div>';
}

/* Weakness + Recommendation */
if(r.weakness) html+='<div class="weakness"><strong>⚠ Weakness:</strong> '+r.weakness+'</div>';
if(r.recommendation) html+='<div class="recommendation"><strong>💡 Recommend:</strong> '+r.recommendation+'</div>';

html+='</div>';
c.innerHTML+=html;
});
}
</script></body></html>"""


def _get_weights():
    """Load design_weights.json, return dict or None."""
    if not WEIGHTS_JSON.exists():
        return None
    try:
        return json.loads(WEIGHTS_JSON.read_text())
    except Exception:
        return None


def _vlm_analyze_image(image_bytes, filename):
    """Send image to VLM via OpenRouter and extract design attributes.

    Returns dict with keys: ornamentation_level, cultural_elements,
    aesthetic_appeal, motifs, raw_response.
    """
    import openai

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return {"error": "OPENROUTER_API_KEY not set"}

    b64 = base64.b64encode(image_bytes).decode()
    ext = Path(filename).suffix.lower().replace(".", "")
    mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(ext, ext)
    data_url = f"data:image/{mime};base64,{b64}"

    prompt = (
        "Analyze this manhole cover / drain cover design. Return ONLY valid JSON "
        "with exactly these fields:\n"
        '{"ornamentation_level": "plain|minimal|moderate|ornate|highly_ornate", '
        '"cultural_elements": true|false, '
        '"aesthetic_appeal": "low|medium|high", '
        '"motifs": "pipe|wave|floral|geometric|tree|bird|fish|text|star|anchor|none", '
        '"is_manhole_cover": true|false, '
        '"design_description": "one sentence summary"}'
    )

    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        resp = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}
            ],
            max_tokens=300,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        parsed["raw_response"] = resp.choices[0].message.content
        return parsed
    except Exception as e:
        return {"error": str(e)}


def _encode_attributes(vlm_result):
    """Encode VLM categorical outputs to numeric scores (0-1)."""
    scores = {}

    # Ornamentation
    orn = str(vlm_result.get("ornamentation_level", "minimal")).lower().strip()
    scores["ornamentation_level"] = _EVAL_ORNAMENTATION_MAP.get(orn, 0.25)

    # Cultural elements
    ce = vlm_result.get("cultural_elements", False)
    scores["cultural_elements"] = 1.0 if str(ce).lower() in ("true", "yes", "1") else 0.0

    # Aesthetic appeal
    aa = str(vlm_result.get("aesthetic_appeal", "medium")).lower().strip()
    scores["aesthetic_appeal"] = _EVAL_AESTHETIC_MAP.get(aa, 0.5)

    # Motif diversity
    motifs_str = str(vlm_result.get("motifs", "none")).lower()
    motif_list = [m.strip() for m in motifs_str.split("|") if m.strip() and m.strip() != "none"]
    scores["motif_diversity"] = min(len(motif_list) / 5.0, 1.0)

    return scores


def _compute_weighted_score(attr_scores, weights):
    """predicted_sentiment = sum(score[attr] * weight)."""
    return sum(attr_scores.get(attr, 0) * weights.get(attr, 0)
               for attr in _EVAL_ATTRIBUTES)


def _generate_recommendation(attr_scores, weights):
    """Identify weakest area and generate actionable recommendation."""
    # Find the weakest weighted contribution
    contributions = {attr: attr_scores.get(attr, 0) * weights.get(attr, 0)
                     for attr in _EVAL_ATTRIBUTES}
    weakest = min(contributions, key=contributions.get)
    weak_val = attr_scores.get(weakest, 0)

    recs = {
        "ornamentation_level": (
            f"Low ornamentation (score: {weak_val:.2f}). Increase pattern complexity — "
            "add borders, geometric bands, or radial motifs to raise visual richness."
        ),
        "cultural_elements": (
            f"No cultural elements detected. Add local landmarks, regional symbols, "
            "or heritage motifs to strengthen cultural resonance and public sentiment."
        ),
        "aesthetic_appeal": (
            f"Aesthetic appeal below threshold (score: {weak_val:.2f}). Improve color "
            "contrast, symmetry, and visual balance — consider adding decorative borders "
            "or central emblems."
        ),
        "motif_diversity": (
            f"Low motif diversity (score: {weak_val:.2f}). Incorporate additional "
            "design elements — combine floral, geometric, and text motifs for richer "
            "visual storytelling."
        ),
    }

    weakness_map = {
        "ornamentation_level": "Ornamentation level is below benchmark",
        "cultural_elements": "No cultural elements detected",
        "aesthetic_appeal": "Aesthetic appeal below threshold",
        "motif_diversity": "Low motif diversity — design too plain",
    }

    return weakness_map.get(weakest, "Below benchmark"), recs.get(weakest, "Consider enhancing the design.")


@app.route("/evaluate")
@require_auth
def evaluate_page():
    return render_template_string(EVALUATE_HTML)


@app.route("/evaluate/api/status")
@require_auth
def evaluate_status():
    w = _get_weights()
    countries = []
    if w and "country_benchmarks" in w:
        countries = sorted(w["country_benchmarks"].keys())
    return jsonify({"has_weights": w is not None, "countries": countries})


@app.route("/evaluate/api/analyze", methods=["POST"])
@require_auth
def evaluate_analyze():
    weights_data = _get_weights()
    if not weights_data:
        return jsonify({"error": "No correlation model found. Run the full pipeline first."}), 400

    weights = weights_data.get("weights", {})
    benchmarks = weights_data.get("country_benchmarks", {})

    # Get uploaded images (max 5)
    uploaded = request.files.getlist("images")
    if not uploaded:
        return jsonify({"error": "No images uploaded"}), 400

    uploaded = [f for f in uploaded if f.filename][:5]

    bench1 = request.form.get("benchmark1", "")
    bench2 = request.form.get("benchmark2", "")

    results = []
    for f in uploaded:
        img_bytes = f.read()
        thumb_b64 = base64.b64encode(img_bytes).decode()

        # Run VLM
        vlm = _vlm_analyze_image(img_bytes, f.filename)
        if "error" in vlm:
            results.append({
                "name": f.filename,
                "error": vlm["error"],
                "score": 0, "thumbnail": thumb_b64,
                "attribute_scores": {}, "predicted_label": "Error",
                "benchmark_comparison": [], "weakness": vlm["error"],
                "recommendation": "Fix VLM error and retry.",
            })
            continue

        # Encode + score
        attr_scores = _encode_attributes(vlm)
        score = _compute_weighted_score(attr_scores, weights)

        # Normalize to 0-1 range (weights sum to 1, attribute scores are 0-1)
        score = max(0.0, min(1.0, score))

        # Predicted label
        label = "Positive" if score >= 0.55 else ("Negative" if score < 0.35 else "Neutral")

        # Benchmark comparison
        bench_comparison = []
        for bc in [bench1, bench2]:
            if bc and bc in benchmarks:
                bm = benchmarks[bc]
                bm_score = sum(bm.get(a, 0) * weights.get(a, 0) for a in _EVAL_ATTRIBUTES)
                bench_comparison.append({
                    "country": bc,
                    "benchmark_score": round(bm_score, 4),
                    "gap": round(score - bm_score, 4),
                })

        weakness, recommendation = _generate_recommendation(attr_scores, weights)

        results.append({
            "name": f.filename,
            "score": round(score, 4),
            "thumbnail": thumb_b64,
            "attribute_scores": {k: round(v, 4) for k, v in attr_scores.items()},
            "predicted_label": label,
            "vlm_raw": vlm,
            "benchmark_comparison": bench_comparison,
            "weakness": weakness,
            "recommendation": recommendation,
        })

    # Store in session for CSV/PDF download
    session["eval_results"] = results
    session["eval_benchmarks"] = {k: v for k, v in benchmarks.items()
                                  if k in [bench1, bench2]}

    return jsonify({"results": results, "benchmarks": benchmarks})


@app.route("/evaluate/api/download/csv")
@require_auth
def evaluate_download_csv():
    results = session.get("eval_results", [])
    if not results:
        return jsonify({"error": "No results to download"}), 404

    import pandas as pd
    rows = []
    for r in results:
        row = {
            "name": r.get("name", ""),
            "score": r.get("score", 0),
            "predicted_label": r.get("predicted_label", ""),
            "ornamentation_level": r.get("attribute_scores", {}).get("ornamentation_level", 0),
            "cultural_elements": r.get("attribute_scores", {}).get("cultural_elements", 0),
            "aesthetic_appeal": r.get("attribute_scores", {}).get("aesthetic_appeal", 0),
            "motif_diversity": r.get("attribute_scores", {}).get("motif_diversity", 0),
            "weakness": r.get("weakness", ""),
            "recommendation": r.get("recommendation", ""),
        }
        for bc in r.get("benchmark_comparison", []):
            row[f"benchmark_{bc['country']}_score"] = bc["benchmark_score"]
            row[f"benchmark_{bc['country']}_gap"] = bc["gap"]
        rows.append(row)

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name="design_evaluation.csv")


@app.route("/evaluate/api/download/pdf")
@require_auth
def evaluate_download_pdf():
    results = session.get("eval_results", [])
    if not results:
        return jsonify({"error": "No results to download"}), 404

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 4 * len(results)))
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        attrs = _EVAL_ATTRIBUTES
        labels = ["Ornamentation", "Cultural Elem.", "Aesthetic", "Motif Diversity"]
        vals = [r.get("attribute_scores", {}).get(a, 0) for a in attrs]
        colors = ["#3fb950" if v >= 0.6 else "#d29922" if v >= 0.35 else "#f85149" for v in vals]

        y = range(len(labels))
        ax.barh(y, vals, color=colors, edgecolor="#30363d", height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_title(f"{r.get('name', 'Design')} — Score: {r.get('score', 0):.2f} "
                     f"({r.get('predicted_label', '')})", fontsize=12, fontweight="bold",
                     color="#e6edf3")
        ax.set_facecolor("#0d1117")
        for i, v in enumerate(vals):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, color="#c9d1d9")

        # Benchmark lines
        for bc in r.get("benchmark_comparison", []):
            ax.axvline(bc["benchmark_score"], color="#58a6ff", linestyle="--",
                       linewidth=1, label=f'{bc["country"]} benchmark')

    fig.patch.set_facecolor("#161b22")
    for ax in axes:
        ax.tick_params(colors="#8b949e")
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        legend = ax.legend(fontsize=8, loc="lower right")
        if legend:
            for t in legend.get_texts():
                t.set_color("#c9d1d9")

    plt.tight_layout()
    fig.savefig(buf, format="pdf", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True, download_name="design_evaluation.pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=port, debug=False)
