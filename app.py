"""
app.py — Zeabur entrypoint
Minimal Flask dashboard to trigger pipeline stages and view outputs.
Includes admin-protected reset controls for pipeline state and storage.
"""

import os
import hmac
import shutil
import threading
import subprocess
import time
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

def _key_valid(provided):
    if not ACCESS_KEY:
        return False
    return hmac.compare_digest(provided.strip().upper().encode(), ACCESS_KEY.strip().upper().encode())

def _admin_key_valid(provided):
    if not ADMIN_RESET_PASSWORD:
        return False
    return hmac.compare_digest(provided.strip().encode(), ADMIN_RESET_PASSWORD.strip().encode())

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated

LOGIN_HTML = """<!DOCTYPE html>
<html><head><title>Pipeline Login</title>
<style>body{font-family:monospace;background:#0d1117;color:#c9d1d9;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:40px;width:360px}
h2{color:#58a6ff;margin-top:0}input{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:4px;color:#c9d1d9;padding:10px;margin-bottom:16px}
button{width:100%;background:#238636;color:white;border:none;border-radius:4px;padding:10px;cursor:pointer}</style>
</head><body><div class="card"><h2>Pipeline Login</h2>
<form method="post" action="/login"><input type="text" name="key" placeholder="ACCESS KEY" autofocus required><button type="submit">Continue</button></form>
</div></body></html>"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><title>Manhole Cover Pipeline</title>
<style>*{box-sizing:border-box}body{font-family:monospace;background:linear-gradient(135deg,#1a1a2b,#0d1117);color:#e6edf3;min-height:100vh;margin:0}
.container{max-width:1400px;margin:0 auto;padding:20px}header{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:15px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center}
header h1{color:#58a6ff;margin:0;font-size:20px}.auth-btn{background:transparent;border:1px solid #f85149;color:#f85149;padding:5px 10px;border-radius:4px;cursor:pointer}
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
.status-card.running{border-color:#238636}.status-card.running .count{color:#238636}
.status-card.completed{border-color:#238636}.status-card.completed .count{color:#238636}
.status-card.failed{border-color:#f85149}.status-card.failed .count{color:#f85149}
.btn-grid{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-top:10px}
.btn{background:#238636;color:white;border:none;border-radius:6px;padding:12px 24px;cursor:pointer;font-size:14px}
.btn:hover{background:#2ea043}.btn-run-all{background:#1f6feb}.btn-run-all:hover{background:#388bfd}
.btn-admin{background:#f85149}.btn-admin:hover{background:#da3633}
.btn-soft-reset{background:#d29922}.btn-soft-reset:hover{background:#e3b341}
.output-section{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:15px;margin-top:20px}
.output-section h3{color:#8b949e;margin:0 0 10px}.output-section pre{color:#c9d1d9;font-size:12px;overflow-x:auto;white-space:pre-wrap;max-height:300px}
.admin-section{background:#161b22;border:2px solid #f85149;border-radius:8px;padding:20px;margin-top:20px}
.admin-section h3{color:#f85149;margin:0 0 15px;border-bottom:1px solid #30363d;padding-bottom:10px}
.admin-section .warning{color:#d29922;font-size:12px;margin-bottom:15px}
.admin-grid{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:15px}
.admin-grid input{flex:1;min-width:200px;background:#0d1117;border:1px solid #30363d;border-radius:4px;color:#c9d1d9;padding:10px}
.pipeline-status{background:#21262d;border-radius:4px;padding:10px;margin-bottom:10px;font-size:12px}
.pipeline-status .running{color:#238636}.pipeline-status .idle{color:#8b949e}
.pipeline-status .script{color:#58a6ff}</style>
</head><body><div class="container">
<header><h1>Manhole Cover Pipeline</h1><form action="/logout" method="get"><button class="auth-btn">Logout</button></form></header>
<div class="toolbar"><button class="toolbar-btn active" onclick="showGraph('sentiment')">Sentiment</button>
<button class="toolbar-btn" onclick="showGraph('images')">Images</button>
<button class="toolbar-btn" onclick="showGraph('cross')">Cross Analysis</button>
<button class="toolbar-btn" onclick="showGraph('coverage')">Coverage</button></div>
<div class="graph-panel active" id="panel-sentiment"><h3>Sentiment Analysis</h3><div class="graph-container" id="graphs-sentiment"><div class="no-data">Run pipeline to generate graphs</div></div></div>
<div class="graph-panel" id="panel-images"><h3>Image Analysis</h3><div class="graph-container" id="graphs-images"><div class="no-data">Run pipeline to generate graphs</div></div></div>
<div class="graph-panel" id="panel-cross"><h3>Cross Analysis</h3><div class="graph-container" id="graphs-cross"><div class="no-data">Run pipeline to generate graphs</div></div></div>
<div class="graph-panel" id="panel-coverage"><h3>Coverage Balance</h3><div class="graph-container" id="graphs-coverage"><div class="no-data">Run pipeline to generate graphs</div></div></div>
<div class="status-grid"><div class="status-card" id="status-scrape"><h3>Scrape Data</h3><div class="count">0</div></div>
<div class="status-card" id="status-sentiment"><h3>Sentiment Analysis</h3><div class="count">0</div></div>
<div class="status-card" id="status-images"><h3>Image Processing</h3><div class="count">0</div></div>
<div class="status-card" id="status-cross"><h3>Cross Analysis</h3><div class="count">0</div></div></div>
<div class="btn-grid"><button class="btn btn-run-all" onclick="runAll()">Run Full Pipeline</button>
<button class="btn" onclick="runStage('01_scrape_data.py')">Scrape Only</button>
<button class="btn" onclick="runStage('03_sentiment_analysis.py')">Sentiment Only</button>
<button class="btn" onclick="runStage('04_image_processing.py')">Images Only</button>
<button class="btn" onclick="runStage('05_cross_analysis.py')">Cross Analysis</button></div>

<div class="admin-section">
<h3>⚙️ Admin Controls</h3>
<div class="warning">⚠️ Admin actions require the ADMIN_RESET_PASSWORD. Use with caution — some actions delete data.</div>
<div class="pipeline-status" id="pipeline-status">Pipeline Status: <span class="idle">Checking...</span></div>
<div class="admin-grid">
<input type="password" id="admin-password" placeholder="Admin Password">
<button class="btn btn-soft-reset" onclick="adminReset('soft')">🔓 Soft Reset</button>
<button class="btn btn-admin" onclick="adminReset('analysis')">🗑️ Reset Analysis</button>
<button class="btn btn-admin" onclick="adminReset('full')">⚠️ Full Reset</button>
<button class="btn btn-admin" onclick="adminResetAndRerun()">🔄 Reset & Rerun</button>
</div>
<div id="admin-message" style="margin-top:10px;font-size:12px;color:#8b949e;"></div>
</div>

<div class="output-section"><h3>Output Log</h3><pre id="output-log">Ready. Run a pipeline stage to see output.</pre></div></div>
<script>
function showGraph(type){document.querySelectorAll('.graph-panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.toolbar-btn').forEach(b=>b.classList.remove('active'));document.getElementById('panel-'+type).classList.add('active');event.target.classList.add('active');loadGraphs(type);}
function loadGraphs(type){var paths={sentiment:['sentiment_by_country.png','sentiment_composition.png','text_volume_by_country.png','keyword_heatmap.png','confidence_distribution.png'],images:['image_volume_by_country.png','country_image_distribution.png','source_distribution.png','image_resolution_distribution.png'],cross:['cross_analysis_visualizations/text_vs_image_by_country.png','cross_analysis_visualizations/sentiment_vs_image_volume.png','cross_analysis_visualizations/combined_country_summary.png','cross_analysis_visualizations/balance_ratio_chart.png','cross_analysis_visualizations/coverage_summary.png','cross_analysis_visualizations/sentiment_heatmap.png'],coverage:['cross_analysis_visualizations/text_vs_image_by_country.png','cross_analysis_visualizations/balance_ratio_chart.png','cross_analysis_visualizations/coverage_summary.png','text_volume_by_country.png','image_volume_by_country.png']};var container=document.getElementById('graphs-'+type);container.innerHTML='';paths[type].forEach(function(f){var img=document.createElement('img');img.src='/output/'+f;img.style.maxWidth='100%';img.style.margin='10px';img.onerror=function(){img.style.display='none';};container.appendChild(img);});}
function updateStatus(stage,status,count){var card=document.getElementById('status-'+stage);if(card){card.querySelector('.count').textContent=count;card.className='status-card '+status;}}
function updatePipelineStatus(){fetch('/admin/status').then(r=>r.json()).then(data=>{var el=document.getElementById('pipeline-status');if(data.running){el.innerHTML='Pipeline Status: <span class="running">RUNNING</span> - <span class="script">'+data.script+'</span> (started '+data.started_ago+'s ago)';}else{el.innerHTML='Pipeline Status: <span class="idle">IDLE</span>';}document.getElementById('admin-password').disabled=data.running;}).catch(e=>{document.getElementById('pipeline-status').innerHTML='Pipeline Status: <span class="idle">Unknown</span>';});}
function runAll(){runStage('01_scrape_data.py');setTimeout(function(){runStage('03_sentiment_analysis.py');},2000);setTimeout(function(){runStage('04_image_processing.py');},4000);setTimeout(function(){runStage('05_cross_analysis.py');},6000);}
function runStage(script){var output=document.getElementById('output-log');var stage=script.replace('.py','').replace('01_','').replace('03_','').replace('04_','').replace('05_','');output.textContent+='Starting '+script+'...\\n';updateStatus(stage,'running','...');fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({script:script})}).then(r=>r.json()).then(data=>{output.textContent+=data.message+'\\n';if(data.status==='completed')updateStatus(stage,'completed',data.count||'✓');else if(data.status==='error')updateStatus(stage,'failed','✗');else if(data.status==='started')updateStatus(stage,'running','...');}).catch(e=>{output.textContent+='Error: '+e+'\\n';updateStatus(stage,'failed','✗');});}
function adminReset(type){var password=document.getElementById('admin-password').value;if(!password){document.getElementById('admin-message').innerHTML='<span style="color:#f85149;">Please enter admin password</span>';return;}
var msgEl=document.getElementById('admin-message');var output=document.getElementById('output-log');msgEl.innerHTML='Processing...';fetch('/admin/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:password,reset_type:type})}).then(r=>r.json()).then(data=>{if(data.success){msgEl.innerHTML='<span style="color:#238636;">✓ '+data.message+'</span>';if(data.deleted_files&&data.deleted_files.length>0){output.textContent+='Admin reset ('+type+'): deleted '+data.deleted_files.length+' items\\n';}}else{msgEl.innerHTML='<span style="color:#f85149;">✗ '+data.message+'</span>';}document.getElementById('admin-password').value='';updatePipelineStatus();}).catch(e=>{msgEl.innerHTML='<span style="color:#f85149;">Error: '+e+'</span>';});}
function adminResetAndRerun(){var password=document.getElementById('admin-password').value;if(!password){document.getElementById('admin-message').innerHTML='<span style="color:#f85149;">Please enter admin password</span>';return;}
var msgEl=document.getElementById('admin-message');var output=document.getElementById('output-log');msgEl.innerHTML='Resetting and starting pipeline...';fetch('/admin/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:password,reset_type:'full'})}).then(r=>r.json()).then(data=>{if(data.success){msgEl.innerHTML='<span style="color:#238636;">✓ Reset complete. Starting pipeline...</span>';output.textContent+='Admin full reset: deleted '+data.deleted_count+' items\\n';setTimeout(function(){runAll();},500);}else{msgEl.innerHTML='<span style="color:#f85149;">✗ '+data.message+'</span>';}}).catch(e=>{msgEl.innerHTML='<span style="color:#f85149;">Error: '+e+'</span>';});}
setInterval(function(){fetch('/counts').then(r=>r.json()).then(data=>{if(data.scrape)updateStatus('scrape','completed',data.scrape);if(data.sentiment)updateStatus('sentiment','completed',data.sentiment);if(data.images)updateStatus('images','completed',data.images);if(data.cross)updateStatus('cross','completed',data.cross);});updatePipelineStatus();},2000);
setTimeout(updatePipelineStatus,100);
</script></body></html>"""

# Enhanced pipeline state tracking
job_state = {
    "running": False,
    "script": None,
    "started_at": None,
    "log": [],
    "thread": None
}

DATA_DIR = Path("/data")
OUTPUT_DIR = DATA_DIR / "output"
IMAGES_DIR = DATA_DIR / "images"


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


@app.route("/counts")
@require_auth
def get_counts():
    counts = {"scrape": 0, "sentiment": 0, "images": 0, "cross": 0}
    text_path = DATA_DIR / "text_raw.csv"
    sentiment_path = DATA_DIR / "text_with_sentiment.csv"
    image_meta = DATA_DIR / "image_metadata.csv"
    cross_path = OUTPUT_DIR / "cross_analysis.csv"
    
    try:
        if text_path.exists():
            import pandas as pd
            counts["scrape"] = len(pd.read_csv(text_path))
    except:
        pass
    
    try:
        if sentiment_path.exists():
            import pandas as pd
            counts["sentiment"] = len(pd.read_csv(sentiment_path))
    except:
        pass
    
    try:
        if image_meta.exists():
            import pandas as pd
            counts["images"] = len(pd.read_csv(image_meta))
        elif IMAGES_DIR.exists():
            counts["images"] = len(list(IMAGES_DIR.rglob("*.jpg")) + list(IMAGES_DIR.rglob("*.png")))
    except:
        pass
    
    try:
        if cross_path.exists():
            import pandas as pd
            counts["cross"] = len(pd.read_csv(cross_path))
    except:
        pass
    
    return jsonify(counts)


@app.route("/admin/status")
@require_auth
def admin_status():
    """Return current pipeline status for the UI."""
    started_ago = 0
    if job_state["running"] and job_state["started_at"]:
        started_ago = int(time.time() - job_state["started_at"])
    
    return jsonify({
        "running": job_state["running"],
        "script": job_state["script"],
        "started_at": job_state["started_at"],
        "started_ago": started_ago
    })


@app.route("/admin/reset", methods=["POST"])
@require_auth
def admin_reset():
    """Admin-protected reset endpoint. Requires ADMIN_RESET_PASSWORD."""
    data = request.get_json() or {}
    password = data.get("password", "")
    reset_type = data.get("reset_type", "soft")
    
    # Validate admin password
    if not _admin_key_valid(password):
        return jsonify({"success": False, "message": "Invalid admin password"}), 403
    
    deleted_files = []
    deleted_count = 0
    
    # Always clear the running state
    was_running = job_state["running"]
    job_state["running"] = False
    job_state["script"] = None
    job_state["started_at"] = None
    job_state["log"].append(f"Admin reset ({reset_type}) at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if reset_type == "soft":
        # Soft reset: just unlock the state, keep all data
        return jsonify({
            "success": True,
            "message": f"Soft reset complete. Unlocked pipeline (was: {'running' if was_running else 'idle'}).",
            "deleted_count": 0,
            "deleted_files": []
        })
    
    elif reset_type == "analysis":
        # Reset analysis outputs only, keep raw scrape data
        analysis_files = [
            DATA_DIR / "text_with_sentiment.csv",
            DATA_DIR / "image_metadata.csv",
            OUTPUT_DIR / "cross_analysis.csv",
        ]
        analysis_dirs = [
            OUTPUT_DIR,
        ]
        
        for f in analysis_files:
            if f.exists():
                try:
                    f.unlink()
                    deleted_files.append(str(f))
                    deleted_count += 1
                except Exception as e:
                    job_state["log"].append(f"Failed to delete {f}: {e}")
        
        for d in analysis_dirs:
            if d.exists():
                try:
                    for item in d.rglob("*"):
                        if item.is_file():
                            item.unlink()
                            deleted_files.append(str(item))
                            deleted_count += 1
                except Exception as e:
                    job_state["log"].append(f"Failed to clean {d}: {e}")
        
        return jsonify({
            "success": True,
            "message": f"Analysis reset complete. Deleted {deleted_count} analysis files.",
            "deleted_count": deleted_count,
            "deleted_files": deleted_files[:20]  # Limit response size
        })
    
    elif reset_type == "full":
        # Full reset: delete all pipeline data including scraped data and images
        full_files = [
            DATA_DIR / "text_raw.csv",
            DATA_DIR / "text_with_sentiment.csv",
            DATA_DIR / "image_metadata.csv",
        ]
        full_dirs = [
            OUTPUT_DIR,
            IMAGES_DIR,
        ]
        
        for f in full_files:
            if f.exists():
                try:
                    f.unlink()
                    deleted_files.append(str(f))
                    deleted_count += 1
                except Exception as e:
                    job_state["log"].append(f"Failed to delete {f}: {e}")
        
        for d in full_dirs:
            if d.exists():
                try:
                    shutil.rmtree(d)
                    deleted_files.append(str(d) + "/")
                    deleted_count += 1
                except Exception as e:
                    job_state["log"].append(f"Failed to remove {d}: {e}")
        
        # Recreate essential directories
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        
        return jsonify({
            "success": True,
            "message": f"Full reset complete. Deleted all pipeline data ({deleted_count} items).",
            "deleted_count": deleted_count,
            "deleted_files": deleted_files[:20]
        })
    
    else:
        return jsonify({"success": False, "message": f"Unknown reset type: {reset_type}"}), 400


@app.route("/run", methods=["POST"])
@require_auth
def run_stage():
    data = request.get_json() or {}
    script = data.get("script", "01_scrape_data.py")
    
    if job_state["running"]:
        return jsonify({"status": "error", "message": "Pipeline already running"})
    
    def run():
        job_state["running"] = True
        job_state["script"] = script
        job_state["started_at"] = time.time()
        try:
            result = subprocess.run(["python", script], capture_output=True, text=True, timeout=600)
            job_state["log"].append(f"Completed {script}: exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            job_state["log"].append(f"Timeout {script}: exceeded 600s limit")
        except Exception as e:
            job_state["log"].append(f"Error {script}: {str(e)}")
        finally:
            job_state["running"] = False
            job_state["script"] = None
            job_state["started_at"] = None
    
    thread = threading.Thread(target=run)
    job_state["thread"] = thread
    thread.start()
    
    return jsonify({"status": "started", "message": f"Started {script}", "script": script})


@app.route("/output/<path:path>")
@require_auth
def serve_output(path):
    file_path = OUTPUT_DIR / path
    if file_path.exists():
        return send_file(str(file_path))
    return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=port, debug=False)