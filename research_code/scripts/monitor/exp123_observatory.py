#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import psutil
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from los_baseline.config import load_config, resolve_path
from los_baseline.resource_manager import system_snapshot


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path, limit: int = 1000) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try: rows.append(json.loads(line))
                except Exception: continue
        return rows[-limit:]
    except Exception:
        return []


def csv_rows(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    if not path.is_file(): return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))[:limit]
    except Exception:
        return []


def tail_text(path: Path, lines: int = 500) -> str:
    if not path.is_file(): return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def safe_run(runs_dir: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise HTTPException(400, "Invalid run ID")
    path = (runs_dir / run_id).resolve()
    if path.parent != runs_dir.resolve() or not path.is_dir():
        raise HTTPException(404, "Run not found")
    return path


def run_phase(run_dir: Path) -> str:
    if (run_dir / "TRACK3_undersampled").exists(): return "Track 3 · Undersampled"
    if (run_dir / "common/staged_optimized").exists(): return "Track 2 · Optimized"
    if (run_dir / "common/staged_baseline").exists(): return "Track 1 · Baseline"
    return "Preparing"


def run_snapshot(run_dir: Path) -> dict[str, Any]:
    progress = read_json(run_dir / "common/progress/run_progress.json", {}) or {}
    tree = read_json(run_dir / "common/progress/progress_tree.json", {}) or {}
    tracking = read_json(run_dir / "common/tracking/summary.json", {}) or {}
    scheduler = read_json(run_dir / "common/progress/scheduler_state.json", {}) or {}
    status_rows = list((scheduler.get("status") or {}).values()) if isinstance(scheduler, dict) else []
    counts: dict[str, int] = {}
    for row in status_rows:
        state = str(row.get("status", "unknown")); counts[state] = counts.get(state, 0) + 1
    files = [p for p in run_dir.rglob("*") if p.is_file()]
    trials = len(list(run_dir.rglob("optuna_trials.csv")))
    return {
        "run_id": run_dir.name,
        "phase": run_phase(run_dir),
        "updated": datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat(),
        "overall_percent": progress.get("overall_percent", 0),
        "completed_nodes": progress.get("completed_count", 0),
        "failed_nodes": progress.get("failed_count", 0),
        "scheduler_counts": counts,
        "tracking_summary": tracking,
        "file_count": len(files),
        "optuna_study_count": trials,
        "progress_tree": tree,
    }


def build_app(config_path: Path) -> FastAPI:
    config = load_config(config_path)
    runs_dir = resolve_path(config, config["paths"]["runs_dir"])
    dashboard_cfg = dict(config.get("dashboard", {}))
    token = str(dashboard_cfg.get("token", os.environ.get("EXP123_DASHBOARD_TOKEN", "")))
    app = FastAPI(title="EXP123 Observatory", docs_url=None, redoc_url=None)

    def authorize(request: Request) -> None:
        if not token: return
        supplied = request.query_params.get("token") or request.headers.get("X-EXP123-Token")
        if supplied != token: raise HTTPException(401, "Invalid dashboard token")

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        authorize(request); return HTML

    @app.get("/api/runs")
    def runs(request: Request):
        authorize(request)
        runs_dir.mkdir(parents=True, exist_ok=True)
        rows = [run_snapshot(p) for p in runs_dir.iterdir() if p.is_dir()]
        rows.sort(key=lambda x: x["updated"], reverse=True)
        return {"runs": rows}

    @app.get("/api/system")
    def system(request: Request):
        authorize(request); return system_snapshot().to_dict()

    @app.get("/api/runs/{run_id}")
    def overview(run_id: str, request: Request):
        authorize(request); run_dir = safe_run(runs_dir, run_id)
        snapshot = run_snapshot(run_dir)
        snapshot["history"] = read_jsonl(run_dir / "common/tracking/history.jsonl", 2000)
        snapshot["resource_history"] = read_jsonl(run_dir / "common/progress/resource_metrics.jsonl", 500)
        snapshot["completed_operations"] = read_jsonl(run_dir / "common/progress/completed_operations.jsonl", 500)
        snapshot["config"] = read_json(run_dir / "common/tracking/run_config.json", {})
        snapshot["metrics"] = {
            "exp1": csv_rows(run_dir / "EXP1_flat_3class/model_summary_validation.csv"),
            "stage1": csv_rows(run_dir / "EXP2_two_stage/stage1_mortality/model_summary_validation.csv"),
            "stage2": csv_rows(run_dir / "EXP2_two_stage/stage2_long_stay/model_summary_validation.csv"),
            "track3": csv_rows(run_dir / "TRACK3_undersampled/track3_summary.csv"),
        }
        snapshot["trials"] = []
        for path in sorted(run_dir.rglob("optuna_trials.csv")):
            snapshot["trials"].append({"path": str(path.relative_to(run_dir)), "rows": csv_rows(path, 200)})
        snapshot["artifacts"] = read_jsonl(run_dir / "common/tracking/artifacts.jsonl", 1000)
        logs = []
        for path in sorted(run_dir.rglob("*.log")):
            logs.append({"name": str(path.relative_to(run_dir)), "size": path.stat().st_size})
        snapshot["logs"] = logs
        return snapshot

    @app.get("/api/runs/{run_id}/logs")
    def log(run_id: str, request: Request, path: str = Query(...), lines: int = 500):
        authorize(request); run_dir = safe_run(runs_dir, run_id)
        target = (run_dir / path).resolve()
        if run_dir not in target.parents or not target.is_file(): raise HTTPException(404, "Log not found")
        return {"path": path, "tail": tail_text(target, max(10, min(lines, 5000)))}

    @app.get("/api/runs/{run_id}/stream")
    async def stream(run_id: str, request: Request):
        authorize(request); run_dir = safe_run(runs_dir, run_id)
        async def events():
            while True:
                if await request.is_disconnected(): break
                payload = {"snapshot": run_snapshot(run_dir), "system": system_snapshot().to_dict()}
                yield "event: update\ndata: " + json.dumps(payload, ensure_ascii=False, default=str) + "\n\n"
                await asyncio.sleep(float(dashboard_cfg.get("refresh_seconds", 2)))
        return StreamingResponse(events(), media_type="text/event-stream")

    return app


HTML = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EXP123 Observatory</title><style>
:root{--bg:#f5f7fb;--surface:#fff;--line:#dfe5ef;--ink:#1a2438;--muted:#68758a;--blue:#4169e1;--cyan:#00a6a6;--green:#198754;--red:#c43a4b;--amber:#c48100;--nav:#111a2f;--shadow:0 10px 30px rgba(23,42,74,.08)}*{box-sizing:border-box}body{margin:0;background:var(--bg);font:14px/1.45 Inter,-apple-system,BlinkMacSystemFont,"Noto Sans KR",sans-serif;color:var(--ink)}button,select,input{font:inherit}.shell{display:grid;grid-template-columns:240px minmax(0,1fr);min-height:100vh}.nav{background:var(--nav);color:#d9e2f6;padding:22px 16px;position:sticky;top:0;height:100vh}.brand{font-size:19px;font-weight:800;color:white}.brand small{display:block;font-size:11px;color:#8fa1c5;font-weight:500;margin-top:4px}.nav a{display:block;color:#b9c5df;padding:9px 10px;border-radius:9px;margin:3px 0;text-decoration:none}.nav a:hover{background:#1b2948;color:white}.main{padding:25px;max-width:1700px;width:100%;margin:auto}.top{display:flex;justify-content:space-between;gap:12px;align-items:center}.top h1{margin:0;font-size:26px}.controls{display:flex;gap:8px;flex-wrap:wrap}select,button,input{background:white;border:1px solid var(--line);border-radius:9px;padding:8px 10px;color:var(--ink)}button{cursor:pointer}.badge{padding:5px 9px;border-radius:999px;background:#e9eefb;font-size:12px}.live{background:#e9f8f2;color:var(--green)}section{margin-top:18px}.grid{display:grid;gap:13px}.cards{grid-template-columns:repeat(6,minmax(0,1fr))}.two{grid-template-columns:repeat(2,minmax(0,1fr))}.three{grid-template-columns:repeat(3,minmax(0,1fr))}.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:15px;box-shadow:var(--shadow)}.kpi label{display:block;color:var(--muted);font-size:12px}.kpi strong{font-size:24px;display:block;margin-top:5px}.section-title{display:flex;justify-content:space-between;align-items:center}.section-title h2{font-size:17px;margin:0}.bar{height:9px;background:#edf1f7;border-radius:99px;overflow:hidden}.bar span{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));border-radius:99px}.tree-row{display:grid;grid-template-columns:minmax(220px,1fr) 100px 85px;gap:10px;padding:8px 0;border-bottom:1px solid #edf0f5;align-items:center}.tree-row small{color:var(--muted)}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:8px;border-bottom:1px solid #edf0f5;white-space:nowrap}.scroll{overflow:auto;max-height:460px}.tabs{display:flex;gap:4px;margin-bottom:10px}.tab{background:#edf1f8;border:0}.tab.active{background:var(--blue);color:white}.panel{display:none}.panel.active{display:block}canvas{width:100%;height:210px;background:white}.mono{font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#0d1425;color:#d7e2fa;padding:12px;border-radius:10px;white-space:pre-wrap;max-height:500px;overflow:auto}.artifact{padding:8px;border-bottom:1px solid #edf0f5}.muted{color:var(--muted)}@media(max-width:1200px){.cards{grid-template-columns:repeat(3,1fr)}.three{grid-template-columns:1fr 1fr}}@media(max-width:800px){.shell{display:block}.nav{height:auto;position:relative}.two,.three,.cards{grid-template-columns:1fr}.main{padding:12px}}
</style></head><body><div class="shell"><aside class="nav"><div class="brand">EXP123 Observatory<small>Local experiment workspace</small></div><nav><a href="#overview">Overview</a><a href="#progress">Progress</a><a href="#metrics">Metrics</a><a href="#trials">Optimization</a><a href="#system">System</a><a href="#artifacts">Artifacts</a><a href="#logs">Logs</a><a href="#config">Config</a></nav></aside><main class="main"><div class="top"><div><h1>Experiment Workspace</h1><div class="muted" id="subtitle">Run을 선택하세요.</div></div><div class="controls"><select id="run"></select><button id="refresh">새로고침</button><span id="connection" class="badge">연결 중</span></div></div>
<section id="overview" class="grid cards"><div class="card kpi"><label>Track</label><strong id="phase">-</strong></div><div class="card kpi"><label>Overall progress</label><strong id="percent">0%</strong></div><div class="card kpi"><label>Completed nodes</label><strong id="completed">0</strong></div><div class="card kpi"><label>Failed nodes</label><strong id="failed">0</strong></div><div class="card kpi"><label>Optuna studies</label><strong id="studies">0</strong></div><div class="card kpi"><label>Files</label><strong id="files">0</strong></div></section>
<section id="progress" class="grid two"><div class="card"><div class="section-title"><h2>Hierarchical progress</h2><span id="progressText" class="muted"></span></div><div class="bar"><span id="overallBar" style="width:0%"></span></div><div id="tree"></div></div><div class="card"><div class="section-title"><h2>Metric history</h2><select id="metricKey"></select></div><canvas id="metricCanvas" width="700" height="250"></canvas></div></section>
<section id="metrics" class="card"><div class="section-title"><h2>Model comparison</h2></div><div class="tabs"><button class="tab active" data-panel="exp1">EXP1</button><button class="tab" data-panel="stage1">Stage 1</button><button class="tab" data-panel="stage2">Stage 2</button><button class="tab" data-panel="track3">Track 3</button></div><div id="metricPanels"></div></section>
<section id="trials" class="card"><div class="section-title"><h2>Optimization trials</h2><select id="studySelect"></select></div><div id="trialTable" class="scroll"></div></section>
<section id="system" class="grid three"><div class="card"><h2>CPU & RAM</h2><div id="cpuRam"></div></div><div class="card"><h2>GPU</h2><div id="gpu"></div></div><div class="card"><h2>Recent operations</h2><div id="ops" class="scroll"></div></div></section>
<section id="artifacts" class="grid two"><div class="card"><h2>Artifacts & lineage</h2><div id="artifactList" class="scroll"></div></div><div id="logs" class="card"><h2>Logs</h2><div class="controls"><select id="logSelect"></select><button id="loadLog">열기</button></div><pre id="logText" class="mono">로그를 선택하세요.</pre></div></section>
<section id="config" class="card"><h2>Run configuration</h2><pre id="configText" class="mono"></pre></section></main></div><script>
const S={run:null,data:null,source:null,token:new URLSearchParams(location.search).get('token')||''};const $=id=>document.getElementById(id);const api=p=>p+(p.includes('?')?'&':'?')+(S.token?'token='+encodeURIComponent(S.token):'');const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));async function get(p){let r=await fetch(api(p));if(!r.ok)throw Error(await r.text());return r.json()}function bars(label,v,max=100){return `<div>${esc(label)} <b>${Number(v||0).toFixed(1)}%</b></div><div class="bar"><span style="width:${Math.min(100,Number(v||0)/max*100)}%"></span></div>`}
function renderRun(d){S.data=d;$('phase').textContent=d.phase;$('percent').textContent=Number(d.overall_percent||0).toFixed(1)+'%';$('completed').textContent=d.completed_nodes||0;$('failed').textContent=d.failed_nodes||0;$('studies').textContent=d.optuna_study_count||0;$('files').textContent=d.file_count||0;$('subtitle').textContent=d.run_id+' · '+d.phase;$('overallBar').style.width=(d.overall_percent||0)+'%';$('progressText').textContent=Number(d.overall_percent||0).toFixed(1)+'%';renderTree(d.progress_tree?.nodes||[]);renderHistory(d.history||[]);renderMetrics(d.metrics||{});renderTrials(d.trials||[]);renderOps(d.completed_operations||[]);renderArtifacts(d.artifacts||[]);renderLogs(d.logs||[]);$('configText').textContent=JSON.stringify(d.config||{},null,2)}
function renderTree(nodes){let rows=[...nodes].sort((a,b)=>(a.parent_id?1:0)-(b.parent_id?1:0)||String(a.operation).localeCompare(String(b.operation))).slice(-30);$('tree').innerHTML=rows.map(n=>`<div class="tree-row"><div><b>${esc(n.operation)}</b><br><small>${esc(n.worker)} · ${esc(n.status)}</small></div><div class="bar"><span style="width:${n.percent??(n.status==='completed'?100:0)}%"></span></div><div>${n.percent==null?'—':Number(n.percent).toFixed(1)+'%'}</div></div>`).join('')||'<p class="muted">진행 노드가 아직 없습니다.</p>'}
function renderHistory(rows){let keys=[...new Set(rows.flatMap(r=>Object.keys(r)).filter(k=>k!=='step'&&k!=='worker'&&typeof rows.find(x=>typeof x[k]==='number')?.[k]==='number'))];$('metricKey').innerHTML=keys.map(k=>`<option>${esc(k)}</option>`).join('');function draw(){let k=$('metricKey').value,vals=rows.filter(r=>typeof r[k]==='number').slice(-120);let c=$('metricCanvas'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.strokeStyle='#dfe5ef';for(let i=0;i<5;i++){let y=20+i*45;x.beginPath();x.moveTo(40,y);x.lineTo(c.width-10,y);x.stroke()}if(!vals.length)return;let ys=vals.map(r=>r[k]),min=Math.min(...ys),max=Math.max(...ys);if(max===min)max=min+1;x.strokeStyle='#4169e1';x.lineWidth=2;x.beginPath();vals.forEach((r,i)=>{let px=40+i*(c.width-55)/Math.max(1,vals.length-1),py=220-(r[k]-min)/(max-min)*190;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.fillStyle='#68758a';x.fillText(max.toFixed(4),2,25);x.fillText(min.toFixed(4),2,220)}$('metricKey').onchange=draw;draw()}
function table(rows){if(!rows?.length)return '<p class="muted">데이터가 없습니다.</p>';let keys=Object.keys(rows[0]).slice(0,12);return `<div class="scroll"><table><thead><tr>${keys.map(k=>`<th>${esc(k)}</th>`).join('')}</tr></thead><tbody>${rows.slice(0,100).map(r=>`<tr>${keys.map(k=>`<td>${esc(r[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
function renderMetrics(m){$('metricPanels').innerHTML=Object.entries(m).map(([k,v],i)=>`<div id="panel-${k}" class="panel ${i===0?'active':''}">${table(v)}</div>`).join('');document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');$('panel-'+b.dataset.panel)?.classList.add('active')})}
function renderTrials(studies){$('studySelect').innerHTML=studies.map((s,i)=>`<option value="${i}">${esc(s.path)}</option>`).join('');let show=()=>{$('trialTable').innerHTML=studies.length?table(studies[Number($('studySelect').value)||0].rows):'<p class="muted">Optuna 결과가 없습니다.</p>'};$('studySelect').onchange=show;show()}
function renderOps(rows){$('ops').innerHTML=rows.slice(-30).reverse().map(r=>`<div class="artifact"><b>${esc(r.operation)}</b><br><small>${esc(r.status)} · ${esc(r.timestamp_utc)}</small></div>`).join('')||'<p class="muted">기록 없음</p>'}function renderArtifacts(rows){$('artifactList').innerHTML=rows.map(r=>`<div class="artifact"><b>${esc(r.name)}</b> <span class="badge">${esc(r.type)}</span><br><small>${esc(r.path)}</small></div>`).join('')||'<p class="muted">Artifact 기록 없음</p>'}function renderLogs(rows){$('logSelect').innerHTML=rows.map(r=>`<option value="${esc(r.name)}">${esc(r.name)}</option>`).join('');$('loadLog').onclick=async()=>{let d=await get(`/api/runs/${encodeURIComponent(S.run)}/logs?path=${encodeURIComponent($('logSelect').value)}&lines=700`);$('logText').textContent=d.tail}}
function renderSystem(s){$('cpuRam').innerHTML=bars('CPU',s.cpu_percent)+bars('RAM',100*Number(s.used_ram_gb||0)/Math.max(1,Number(s.total_ram_gb||1)));$('gpu').innerHTML=(s.gpu||[]).map(g=>bars(`GPU${g.index} util`,g.utilization_percent)+bars(`GPU${g.index} memory`,100*g.memory_used_gb/Math.max(1,g.memory_total_gb))).join('')||'<p class="muted">GPU 정보 없음</p>'}
async function loadRuns(){let d=await get('/api/runs');$('run').innerHTML=d.runs.map(r=>`<option value="${esc(r.run_id)}">${esc(r.run_id)} · ${esc(r.phase)}</option>`).join('');if(!S.run&&d.runs.length)S.run=d.runs[0].run_id;$('run').value=S.run||'';await loadRun()}async function loadRun(){if(!S.run)return;let [d,s]=await Promise.all([get('/api/runs/'+encodeURIComponent(S.run)),get('/api/system')]);renderRun(d);renderSystem(s);connect()}function connect(){S.source?.close();S.source=new EventSource(api('/api/runs/'+encodeURIComponent(S.run)+'/stream'));S.source.addEventListener('update',e=>{let d=JSON.parse(e.data);renderSystem(d.system);$('connection').textContent='실시간';$('connection').className='badge live';let snap=d.snapshot;$('percent').textContent=Number(snap.overall_percent||0).toFixed(1)+'%';$('overallBar').style.width=(snap.overall_percent||0)+'%'});S.source.onerror=()=>{$('connection').textContent='재연결 중'}}$('run').onchange=e=>{S.run=e.target.value;loadRun()};$('refresh').onclick=loadRuns;loadRuns().catch(e=>{$('configText').textContent=e.stack||e});
</script></body></html>'''


def parse_args():
    p=argparse.ArgumentParser(description="Launch the W&B-inspired local EXP123 Observatory.")
    p.add_argument("--config", default="configs/experiment_cuda_v7.json")
    p.add_argument("--host", default=None); p.add_argument("--port", type=int, default=None)
    return p.parse_args()


def main():
    import uvicorn
    a=parse_args(); cfg_path=Path(a.config).expanduser().resolve(); cfg=load_config(cfg_path); d=dict(cfg.get("dashboard",{}))
    uvicorn.run(build_app(cfg_path), host=a.host or d.get("host","127.0.0.1"), port=a.port or int(d.get("port",8080)), log_level="info")

if __name__=="__main__": main()
