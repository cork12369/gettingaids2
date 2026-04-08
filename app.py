"""
app.py — Zeabur entrypoint
Flask dashboard to trigger pipeline stages, view outputs, and run human grading.
"""

import os, csv, uuid, hmac, shutil, threading, subprocess, time, json
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
    "06_confusion_matrix.py"
])

# Thread-safe job state management
_job_lock = threading.Lock()
job_state = {"running": False, "script": None, "started_at": None, "log": [], "thread": None}
MAX_LOG_ENTRIES = 1000  # Limit log growth to prevent memory issues

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
    """Generate the fixed 2000-snippet stratified sample (once)."""
    if GRADE_SAMPLE.exists():
        return True
    src = DATA_DIR / "text_with_sentiment.csv"
    if not src.exists():
        return False
    import pandas as pd
    df = pd.read_csv(src)
    if "country" not in df.columns or len(df) == 0:
        return False
    countries = df["country"].unique().tolist()
    if "unknown" in countries:
        countries.remove("unknown")
    if not countries:
        return False
    # Stratified capped allocation
    per_country = max(SAMPLE_TARGET // len(countries), 1)
    frames = []
    for c in countries:
        sub = df[df["country"] == c].sample(n=min(per_country, len(df[df["country"] == c])), random_state=42)
        frames.append(sub)
    sample = pd.concat(frames, ignore_index=True)
    if len(sample) > SAMPLE_TARGET:
        sample = sample.sample(n=SAMPLE_TARGET, random_state=42)
    sample = sample.reset_index(drop=True)
    sample["snippet_id"] = sample.index
    sample.to_csv(GRADE_SAMPLE, index=False)
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

    # Check existing assignment
    for a in assignments:
        if a["grader_id"] == grader_id:
            # Already assigned — check if completed
            graded_in_chunk = len(graded_ids)
            if graded_in_chunk < CHUNK_SIZE:
                return int(a["chunk_index"])
            # else completed, will try to claim new below

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
<style>*{box-sizing:border-box}body{font-family:monospace;background:linear-gradient(135deg,#1a1a2b,#0d1117);color:#e6edf3;min-height:100vh;margin:0}
.container{max-width:1400px;margin:0 auto;padding:20px}header{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:15px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center}
header h1{color:#58a6ff;margin:0;font-size:20px}.header-actions{display:flex;gap:10px}
.auth-btn{background:transparent;border:1px solid #f85149;color:#f85149;padding:5px 10px;border-radius:4px;cursor:pointer}
.admin-toggle-btn{background:transparent;border:1px solid #d29922;color:#d29922;padding:5px 10px;border-radius:4px;cursor:pointer}
.toolbar{background:#21262d;border:1px solid #30363d;border-radius:8px;padding:10px;margin-bottom:20px;display:flex;flex-wrap:wrap;gap:10px}
.toolbar-btn{background:#238636;color:white;border:none;border-radius:4px;padding:8px 16px;cursor:pointer}
.toolbar-btn:hover{background:#2ea043}.toolbar-btn.active{background:#58a6ff}
.graph-panel{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin-bottom:20px;display:none}
.graph-panel.active{display:block}.graph-panel h3{color:#58a6ff;margin:0 0 15px;border-bottom:1px solid #30363d;padding-bottom:10px}
.graph-container{text-align:center}.graph-container img{max-width:100%;border-radius:4px;margin:10px 0}
.no-data{color:#8b949e;font-style:italic;padding:40px}
.status-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin-bottom:20px}
.status-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:15px;text-align:center}
.status-card h3{color:#8b949e;margin:0 0 8px;font-size:13px}.status-card .count{font-size:28px;font-weight:bold;color:#58a6ff}
.btn-grid{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-top:10px}
.btn{background:#238636;color:white;border:none;border-radius:6px;padding:12px 24px;cursor:pointer;font-size:14px}
.btn:hover{background:#2ea043}.btn-run-all{background:#1f6feb}.btn-run-all:hover{background:#388bfd}
.btn-admin{background:#f85149}.btn-admin:hover{background:#da3633}.btn-soft-reset{background:#d29922}.btn-soft-reset:hover{background:#e3b341}
.btn-grade{background:#8b5cf6}.btn-grade:hover{background:#7c3aed}
.output-section{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:15px;margin-top:20px}
.output-section h3{color:#8b949e;margin:0 0 10px}.output-section pre{color:#c9d1d9;font-size:12px;overflow-x:auto;white-space:pre-wrap;max-height:300px}
.admin-section{background:#161b22;border:2px solid #f85149;border-radius:8px;padding:20px;margin-top:20px;display:none}
.admin-section h3{color:#f85149;margin:0 0 15px;border-bottom:1px solid #30363d;padding-bottom:10px}
.admin-section .warning{color:#d29922;font-size:12px;margin-bottom:15px}
.admin-grid{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:15px}
.admin-grid input{flex:1;min-width:200px;background:#0d1117;border:1px solid #30363d;border-radius:4px;color:#c9d1d9;padding:10px}
.pipeline-status{background:#21262d;border-radius:4px;padding:10px;margin-bottom:10px;font-size:12px}
.pipeline-status .running{color:#238636}.pipeline-status .idle{color:#8b949e}.pipeline-status .script{color:#58a6ff}</style>
</head><body><div class="container">
<header><h1>Manhole Cover Pipeline</h1><div class="header-actions">
<a href="/grade" class="btn btn-grade" style="text-decoration:none;padding:5px 10px;font-size:12px">📝 Grade</a>
<a href="/grade/team" class="btn btn-grade" style="text-decoration:none;padding:5px 10px;font-size:12px">📊 Team</a>
<button class="admin-toggle-btn" onclick="toggleAdminPanel()">⚙️ Admin</button>
<form action="/logout" method="get" style="display:inline"><button class="auth-btn">Logout</button></form></div></header>

<div class="toolbar"><button class="toolbar-btn active" onclick="showGraph('sentiment')">Sentiment</button>
<button class="toolbar-btn" onclick="showGraph('images')">Images</button>
<button class="toolbar-btn" onclick="showGraph('cross')">Cross Analysis</button>
<button class="toolbar-btn" onclick="showGraph('confusion')">Confusion Matrix</button></div>

<div class="graph-panel active" id="panel-sentiment"><h3>Sentiment Analysis</h3><div class="graph-container" id="graphs-sentiment"><div class="no-data">Run pipeline to generate graphs</div></div></div>
<div class="graph-panel" id="panel-images"><h3>Image Analysis</h3><div class="graph-container" id="graphs-images"><div class="no-data">Run pipeline to generate graphs</div></div></div>
<div class="graph-panel" id="panel-cross"><h3>Cross Analysis</h3><div class="graph-container" id="graphs-cross"><div class="no-data">Run pipeline to generate graphs</div></div></div>
<div class="graph-panel" id="panel-confusion"><h3>Confusion Matrix — Human vs AI</h3><div class="graph-container" id="graphs-confusion"><div class="no-data">Run Stage 6 to generate confusion matrix</div></div></div>

<div class="status-grid">
<div class="status-card" id="status-scrape"><h3>Scrape Data</h3><div class="count">0</div></div>
<div class="status-card" id="status-sentiment"><h3>Sentiment</h3><div class="count">0</div></div>
<div class="status-card" id="status-images"><h3>Images</h3><div class="count">0</div></div>
<div class="status-card" id="status-cross"><h3>Cross Analysis</h3><div class="count">0</div></div>
<div class="status-card" id="status-confusion"><h3>Confusion Matrix</h3><div class="count">—</div></div></div>

<div class="btn-grid"><button class="btn btn-run-all" onclick="runAll()">Run Full Pipeline</button>
<button class="btn" onclick="runStage('01_scrape_data.py')">Scrape</button>
<button class="btn" onclick="runStage('03_sentiment_analysis.py')">Sentiment</button>
<button class="btn" onclick="runStage('04_image_processing.py')">Images</button>
<button class="btn" onclick="runStage('05_cross_analysis.py')">Cross Analysis</button>
<button class="btn" onclick="runStage('06_confusion_matrix.py')" style="background:#8b5cf6">Stage 6: Confusion Matrix</button></div>

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
function showGraph(t){document.querySelectorAll('.graph-panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.toolbar-btn').forEach(b=>b.classList.remove('active'));document.getElementById('panel-'+t).classList.add('active');event.target.classList.add('active');loadGraphs(t);}
function loadGraphs(t){var P={sentiment:['sentiment_by_country.png','sentiment_composition.png','text_volume_by_country.png','keyword_heatmap.png','confidence_distribution.png'],images:['image_volume_by_country.png','country_image_distribution.png','source_distribution.png','image_resolution_distribution.png'],cross:['cross_analysis_visualizations/text_vs_image_by_country.png','cross_analysis_visualizations/sentiment_vs_image_volume.png','cross_analysis_visualizations/combined_country_summary.png','cross_analysis_visualizations/balance_ratio_chart.png','cross_analysis_visualizations/coverage_summary.png','cross_analysis_visualizations/sentiment_heatmap.png'],confusion:['confusion_matrix.png']};var c=document.getElementById('graphs-'+t);c.innerHTML='';(P[t]||[]).forEach(function(f){var img=document.createElement('img');img.src='/output/'+f;img.style.maxWidth='100%';img.style.margin='10px';img.onerror=function(){img.style.display='none'};c.appendChild(img)});}
function updateStatus(s,st,c){var card=document.getElementById('status-'+s);if(card){card.querySelector('.count').textContent=c;card.className='status-card '+st;}}
function updatePipelineStatus(){fetch('/admin/status').then(r=>r.json()).then(d=>{var el=document.getElementById('pipeline-status');if(d.running)el.innerHTML='Pipeline: <span class="running">RUNNING</span> — <span class="script">'+d.script+'</span> ('+d.started_ago+'s)';else el.innerHTML='Pipeline: <span class="idle">IDLE</span>';}).catch(()=>{});}
function runAll(){var stages=['01_scrape_data.py','03_sentiment_analysis.py','04_image_processing.py','05_cross_analysis.py'];var i=0;var out=document.getElementById('output-log');function next(){if(i>=stages.length){out.textContent+='\\n✓ Full pipeline complete!\\n';return;}var s=stages[i];out.textContent+='['+(i+1)+'/'+stages.length+'] Starting '+s+'...\\n';fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({script:s})}).then(r=>r.json()).then(d=>{out.textContent+=d.message+'\\n';if(d.status==='error'){out.textContent+='Aborting pipeline.\\n';return;}i++;pollThenNext();}).catch(e=>{out.textContent+='Error: '+e+'\\n';});}function pollThenNext(){out.textContent+='Waiting for '+stages[i-1]+' to finish...\\n';var tries=0;function p(){tries++;fetch('/admin/status').then(r=>r.json()).then(d=>{if(!d.running){out.textContent+='✓ '+stages[i-1]+' done.\\n';setTimeout(next,1000);}else if(tries>120){out.textContent+='Timeout waiting for '+stages[i-1]+'\\n';}else{setTimeout(p,5000);}}).catch(()=>setTimeout(p,5000));}p();}next();}
function runStage(script){var out=document.getElementById('output-log');out.textContent+='Starting '+script+'...\\n';fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({script:script})}).then(r=>r.json()).then(d=>{out.textContent+=d.message+'\\n';}).catch(e=>{out.textContent+='Error: '+e+'\\n';});}
function adminReset(type){var pw=document.getElementById('admin-password').value;if(!pw){document.getElementById('admin-message').innerHTML='<span style="color:#f85149">Enter admin password</span>';return;}
var msg=document.getElementById('admin-message');msg.innerHTML='Processing...';fetch('/admin/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw,reset_type:type})}).then(r=>r.json()).then(d=>{msg.innerHTML=d.success?'<span style="color:#238636">✓ '+d.message+'</span>':'<span style="color:#f85149">✗ '+d.message+'</span>';document.getElementById('admin-password').value='';}).catch(e=>{msg.innerHTML='<span style="color:#f85149">Error</span>';});}
function adminResetAndRerun(){var pw=document.getElementById('admin-password').value;if(!pw){document.getElementById('admin-message').innerHTML='<span style="color:#f85149">Enter admin password</span>';return;}
var msg=document.getElementById('admin-message');msg.innerHTML='Resetting...';fetch('/admin/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw,reset_type:'full'})}).then(r=>r.json()).then(d=>{if(d.success){msg.innerHTML='<span style="color:#238636">✓ '+d.message+' Starting pipeline sequentially...</span>';document.getElementById('admin-password').value='';setTimeout(runAll,1500);}else msg.innerHTML='<span style="color:#f85149">✗ '+d.message+'</span>';}).catch(()=>{});}
function showLogs(){var out=document.getElementById('output-log');out.textContent='Fetching logs...\\n';fetch('/admin/logs').then(r=>r.json()).then(d=>{out.textContent='=== Pipeline Logs ===\\n';if(d.logs.length===0)out.textContent+='(no logs yet)\\n';else d.logs.forEach(function(l){out.textContent+=l+'\\n';});out.textContent+='\\nStatus: '+(d.running?'RUNNING ('+d.script+')':'IDLE')+'\\n';}).catch(e=>{out.textContent+='Error fetching logs: '+e+'\\n';});}
setInterval(()=>{fetch('/counts').then(r=>r.json()).then(d=>{if(d.scrape)updateStatus('scrape','completed',d.scrape);if(d.sentiment)updateStatus('sentiment','completed',d.sentiment);if(d.images)updateStatus('images','completed',d.images);if(d.cross)updateStatus('cross','completed',d.cross);if(d.confusion)updateStatus('confusion','completed','✓');});updatePipelineStatus();},3000);
setTimeout(updatePipelineStatus,100);
function toggleAdminPanel(){var s=document.querySelector('.admin-section');s.style.display=s.style.display==='none'?'block':'none';}
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
document.addEventListener('keydown',function(e){if(e.key==='0')grade(0);else if(e.key==='1')grade(1);else if(e.key==='2')grade(2);});
function load(){fetch('/grade/api/start').then(r=>r.json()).then(d=>{
if(d.error){document.getElementById('no-data').style.display='block';return;}
if(d.all_claimed){document.getElementById('all-done').style.display='block';return;}
chunkIdx=d.chunk_index;snippets=d.snippets;idx=d.graded_in_chunk;
show();}).catch(()=>{document.getElementById('no-data').style.display='block';});}
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
setTimeout(()=>{fl.style.opacity='0';},200);idx++;show();}});}
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
    return jsonify({"success": True})

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
            log_entry = f"Done {script_name}: exit {r.returncode}"
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

@app.route("/output/<path:path>")
@require_auth
def serve_output(path):
    fp = OUTPUT_DIR / path
    if fp.exists(): return send_file(str(fp))
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=port, debug=False)