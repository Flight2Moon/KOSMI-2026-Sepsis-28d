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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def tail_text(path: Path, max_lines: int = 120) -> str:
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as exception:
        return f"Unable to read log: {exception}"


def csv_rows(path: Path, limit: int = 100) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))[:limit]
    except Exception:
        return []




def safe_run_directory(runs_dir: Path, run_id: str) -> Path:
    """Resolve one run directory without permitting path traversal."""
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise HTTPException(status_code=400, detail="Invalid Run ID")
    run_dir = (runs_dir / run_id).resolve()
    root = runs_dir.resolve()
    if run_dir.parent != root or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    return run_dir


def tail_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, limit):]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows

def latest_protocol_result(run_dir: Path) -> tuple[str | None, dict[str, Any]]:
    progress_dir = run_dir / "common" / "progress"
    candidates: list[tuple[int, str, Path]] = []
    for mode in ("discovery", "confirmatory"):
        path = progress_dir / f"resource_{mode}_protocol_result.json"
        if path.is_file():
            candidates.append((path.stat().st_mtime_ns, mode, path))
    if not candidates:
        return None, {}
    _, mode, path = max(candidates)
    return mode, read_json(path, {}) or {}


def detect_phase(run_dir: Path) -> str:
    if (run_dir / "final_run_freeze" / "EXP123_RUN_COMPLETE.lock.json").is_file():
        return "completed"
    mode, result = latest_protocol_result(run_dir)
    if result and int(result.get("failed", 0)) > 0:
        return f"{mode or 'protocol'}_failed"
    if (run_dir / "frozen_design" / "CONFIRMATORY_READY.lock.json").is_file():
        return "confirmatory_ready"
    if (run_dir / "EXP2_two_stage" / "super_learner" / "stage2_long_stay" / "artifact.json").is_file():
        return "discovery_completed"
    if (run_dir / "common" / "splits" / "split_hash.txt").is_file():
        return "discovery_running"
    if (run_dir / "common" / "preflight" / "PREFLIGHT_PASSED.lock.json").is_file():
        return "preflight_passed"
    return "created"


def run_summary(run_dir: Path) -> dict[str, Any]:
    scheduler = read_json(run_dir / "common" / "progress" / "scheduler_state.json", {}) or {}
    state_summary = scheduler.get("summary", {})
    modified = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()
    final_result = read_json(run_dir / "final_run_freeze" / "final_run_manifest.json", {}) or {}
    return {
        "run_id": run_dir.name,
        "phase": detect_phase(run_dir),
        "modified_at_utc": modified,
        "jobs": state_summary,
        "running_jobs": list((scheduler.get("running") or {}).keys()),
        "has_scheduler": bool(scheduler),
        "finalized": bool(final_result),
    }


def collect_metrics(run_dir: Path) -> dict[str, Any]:
    tables = {
        "exp1": run_dir / "EXP1_flat_3class" / "model_summary_validation.csv",
        "stage1": run_dir / "EXP2_two_stage" / "stage1_mortality" / "model_summary_validation.csv",
        "stage2": run_dir / "EXP2_two_stage" / "stage2_long_stay" / "model_summary_validation.csv",
        "final_comparison": run_dir / "final_comparison" / "manuscript_tables" / "combined_main_comparison_table.csv",
        "statistical_tests": run_dir / "final_comparison" / "statistical_tests_combined.csv",
    }
    return {name: csv_rows(path, 200) for name, path in tables.items()}


def create_app(config_path: Path) -> FastAPI:
    config = load_config(config_path)
    runs_dir = resolve_path(config, config["paths"]["runs_dir"])
    token = os.environ.get("EXP123_DASHBOARD_TOKEN", "").strip()
    app = FastAPI(title="EXP123 Experiment Dashboard", docs_url=None, redoc_url=None)

    def authorize(request: Request, query_token: str | None = None) -> None:
        if not token:
            return
        supplied = query_token or request.headers.get("X-EXP123-Token") or request.query_params.get("token")
        if supplied != token:
            raise HTTPException(status_code=401, detail="A valid dashboard token is required.")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, token_query: str | None = Query(default=None, alias="token")) -> str:
        authorize(request, token_query)
        return DASHBOARD_HTML

    @app.get("/api/runs")
    async def list_runs(request: Request, token_query: str | None = Query(default=None, alias="token")) -> dict[str, Any]:
        authorize(request, token_query)
        runs_dir.mkdir(parents=True, exist_ok=True)
        runs = [run_summary(path) for path in runs_dir.iterdir() if path.is_dir()]
        runs.sort(key=lambda item: item["modified_at_utc"], reverse=True)
        return {"timestamp_utc": utc_now(), "runs_dir": str(runs_dir), "runs": runs}

    @app.get("/api/system")
    async def system(request: Request, token_query: str | None = Query(default=None, alias="token")) -> dict[str, Any]:
        authorize(request, token_query)
        snapshot = system_snapshot().to_dict()
        snapshot["load_average"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
        snapshot["process_count"] = len(psutil.pids())
        return snapshot

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str, request: Request, token_query: str | None = Query(default=None, alias="token")) -> dict[str, Any]:
        authorize(request, token_query)
        run_dir = safe_run_directory(runs_dir, run_id)
        scheduler = read_json(run_dir / "common" / "progress" / "scheduler_state.json", {}) or {}
        job_states = []
        state_dir = run_dir / "common" / "progress" / "scheduler_jobs"
        if state_dir.is_dir():
            for path in sorted(state_dir.glob("*.json")):
                payload = read_json(path, {})
                if payload:
                    job_states.append(payload)
        completed = []
        event_path = run_dir / "common" / "progress" / "completed_operations.jsonl"
        if event_path.is_file():
            for line in event_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]:
                try:
                    completed.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        logs = []
        log_dir = run_dir / "common" / "progress" / "step_logs"
        if log_dir.is_dir():
            for path in sorted(log_dir.glob("*.log"), key=lambda item: item.stat().st_mtime, reverse=True)[:30]:
                logs.append({"name": path.name, "modified_at": path.stat().st_mtime, "size_bytes": path.stat().st_size})
        history_limit = int(config.get("dashboard", {}).get("history_points", 300))
        resource_history = tail_jsonl(
            run_dir / "common" / "progress" / "resource_metrics.jsonl",
            max(10, min(history_limit, 5000)),
        )
        protocol_mode, protocol_result = latest_protocol_result(run_dir)
        diagnosis = {}
        if protocol_mode:
            diagnosis = read_json(
                run_dir / "common" / "progress" / f"resource_{protocol_mode}_failure_diagnosis.json",
                {},
            ) or {}
        return {
            "summary": run_summary(run_dir),
            "scheduler": scheduler,
            "jobs": job_states,
            "completed_operations": completed,
            "metrics": collect_metrics(run_dir),
            "resource_history": resource_history,
            "logs": logs,
            "protocol_mode": protocol_mode,
            "protocol_result": protocol_result,
            "failure_diagnosis": diagnosis,
        }

    @app.get("/api/runs/{run_id}/diagnosis")
    async def diagnosis(run_id: str, request: Request, token_query: str | None = Query(default=None, alias="token")) -> dict[str, Any]:
        authorize(request, token_query)
        run_dir = safe_run_directory(runs_dir, run_id)
        mode, result = latest_protocol_result(run_dir)
        if not mode:
            raise HTTPException(status_code=404, detail="Protocol result not found")
        diagnosis_payload = read_json(
            run_dir / "common" / "progress" / f"resource_{mode}_failure_diagnosis.json",
            {},
        ) or {}
        return {
            "run_id": run_id,
            "mode": mode,
            "result": result,
            "diagnosis": diagnosis_payload,
        }

    @app.get("/api/runs/{run_id}/logs/{log_name}")
    async def log_tail(run_id: str, log_name: str, request: Request, lines: int = 200, token_query: str | None = Query(default=None, alias="token")) -> dict[str, Any]:
        authorize(request, token_query)
        run_dir = safe_run_directory(runs_dir, run_id)
        safe_name = Path(log_name).name
        path = run_dir / "common" / "progress" / "step_logs" / safe_name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Log not found")
        maximum = int(config.get("dashboard", {}).get("max_log_lines", 2000))
        return {"run_id": run_id, "log_name": safe_name, "tail": tail_text(path, max(20, min(lines, maximum)))}

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request, limit: int = 300, token_query: str | None = Query(default=None, alias="token")) -> dict[str, Any]:
        authorize(request, token_query)
        run_dir = safe_run_directory(runs_dir, run_id)
        path = run_dir / "common" / "progress" / "progress_events.jsonl"
        return {"events": tail_jsonl(path, max(1, min(limit, 2000)))}

    @app.get("/api/runs/{run_id}/stream")
    async def stream(run_id: str, request: Request, token_query: str | None = Query(default=None, alias="token")) -> StreamingResponse:
        authorize(request, token_query)
        run_dir = safe_run_directory(runs_dir, run_id)

        async def generator():
            last_payload = ""
            while True:
                if await request.is_disconnected():
                    break
                payload = json.dumps({
                    "timestamp_utc": utc_now(),
                    "summary": run_summary(run_dir),
                    "scheduler": read_json(run_dir / "common" / "progress" / "scheduler_state.json", {}) or {},
                    "system": system_snapshot().to_dict(),
                }, ensure_ascii=False, default=str)
                if payload != last_payload:
                    yield f"event: update\ndata: {payload}\n\n"
                    last_payload = payload
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(float(config.get("dashboard", {}).get("refresh_seconds", 2.0)))

        return StreamingResponse(generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


DASHBOARD_HTML = r'''
<div class="app-shell">
  <header>
    <div><h1>EXP123 Experiment Dashboard</h1><p>실시간 작업 상태, 자원 사용량, Validation 및 confirmatory 결과를 한 화면에서 확인합니다.</p></div>
    <span id="connection" class="badge">연결 중</span>
  </header>
  <section class="toolbar">
    <label>Run<select id="runSelect"></select></label>
    <button id="refreshButton" type="button">새로고침</button>
    <span id="updatedAt"></span>
  </section>
  <section id="summaryCards" class="cards"></section>
  <section class="grid two">
    <article><h2>시스템 자원</h2><div id="systemPanel"></div></article>
    <article><h2>실행 중 작업</h2><div id="runningPanel"></div></article>
  </section>
  <section><h2>최근 자원 사용 추이</h2><div id="resourceHistory" class="history-grid"></div></section>
  <section><h2>Scheduler Job 상태</h2><div class="table-wrap"><table><thead><tr><th>Job</th><th>상태</th><th>시도</th><th>CPU</th><th>RAM</th><th>장치</th><th>GPU</th><th>최근 정보</th></tr></thead><tbody id="jobRows"></tbody></table></div></section>
  <section class="grid two">
    <article><h2>Validation 결과</h2><div id="metricsPanel"></div></article>
    <article><h2>최근 완료 작업</h2><div id="completedPanel"></div></article>
  </section>
  <section><h2>로그</h2><div class="log-layout"><div id="logList"></div><pre id="logViewer">로그를 선택하세요.</pre></div></section>
</div>
<style>
:root{color-scheme:light dark;--bg:#0b1020;--panel:#151c31;--line:#2b3657;--text:#edf2ff;--muted:#a8b3cf;--accent:#75a7ff;--good:#55d69e;--bad:#ff7e8a;--wait:#f2c66d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif}.app-shell{max-width:1500px;margin:auto;padding:24px}header,.toolbar{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}h1{margin:0 0 6px;font-size:28px}h2{font-size:18px;margin:0 0 14px}p,#updatedAt{color:var(--muted)}section,article{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin-top:16px}.toolbar{background:transparent;border:0;padding:0}.toolbar label{display:flex;align-items:center;gap:8px}select,button{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:9px;padding:9px 12px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;background:transparent;border:0;padding:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}.card .label{color:var(--muted);font-size:13px}.card .value{font-size:25px;font-weight:700;margin-top:8px}.grid.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;background:transparent;border:0;padding:0}.grid.two article{margin:0}.badge,.status{display:inline-block;padding:5px 9px;border-radius:999px;background:#253154;font-size:12px}.completed{color:var(--good)}.failed,.resource_exceeded,.blocked{color:var(--bad)}.running{color:var(--accent)}.waiting_for_resources,.queued{color:var(--wait)}.bar{height:9px;border-radius:8px;background:#26304c;overflow:hidden;margin:7px 0 12px}.bar>span{display:block;height:100%;background:var(--accent)}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px;vertical-align:top}.table-wrap{overflow:auto}.log-layout{display:grid;grid-template-columns:280px 1fr;gap:12px}.log-button{width:100%;text-align:left;margin-bottom:6px}pre{margin:0;min-height:350px;max-height:650px;overflow:auto;background:#080c17;border:1px solid var(--line);border-radius:10px;padding:14px;white-space:pre-wrap}.metric-table{width:100%;margin-bottom:16px}.history-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}.history-row{border:1px solid var(--line);border-radius:10px;padding:10px}.history-row small{color:var(--muted)}.empty{color:var(--muted)}@media(max-width:900px){.grid.two,.log-layout{grid-template-columns:1fr}.app-shell{padding:12px}}
</style>
<script>
const state={runs:[],runId:null,source:null,token:new URLSearchParams(location.search).get('token')||''};
const q=(id)=>document.getElementById(id); const api=(path)=>path+(path.includes('?')?'&':'?')+(state.token?'token='+encodeURIComponent(state.token):'');
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function get(path){const r=await fetch(api(path));if(!r.ok)throw new Error(await r.text());return r.json();}
function pct(value,total){return total?Math.min(100,Math.max(0,value/total*100)):0}
function renderSystem(s){const ramUsed=s.used_ram_gb?.toFixed(1),ramTotal=s.total_ram_gb?.toFixed(1);let html=`<div>CPU ${esc(s.cpu_percent)}%</div><div class="bar"><span style="width:${pct(s.cpu_percent,100)}%"></span></div><div>RAM ${ramUsed}/${ramTotal} GB</div><div class="bar"><span style="width:${pct(s.used_ram_gb,s.total_ram_gb)}%"></span></div>`;(s.gpu||[]).forEach(g=>{html+=`<div>GPU ${g.index} ${esc(g.name)} · ${g.utilization_percent}% · ${g.memory_used_gb?.toFixed(1)}/${g.memory_total_gb?.toFixed(1)} GB</div><div class="bar"><span style="width:${pct(g.memory_used_gb,g.memory_total_gb)}%"></span></div>`});q('systemPanel').innerHTML=html;}
function renderSummary(d){const s=d.summary||{},jobs=s.jobs||{};const cards=[['Run',s.run_id],['Phase',s.phase],['완료',jobs.completed||0],['실행 중',jobs.running||0],['대기',Number(jobs.queued||0)+Number(jobs.waiting_for_resources||0)],['실패',Number(jobs.failed||0)+Number(jobs.resource_exceeded||0)+Number(jobs.blocked||0)]];q('summaryCards').innerHTML=cards.map(([l,v])=>`<div class="card"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('');}
function renderJobs(d){const jobs=[...(d.jobs||[])].sort((a,b)=>String(a.status).localeCompare(String(b.status)));q('jobRows').innerHTML=jobs.map(x=>{const r=x.job?.request||{};return `<tr><td>${esc(x.job?.name)}</td><td class="${esc(x.status)}">${esc(x.status)}</td><td>${esc(x.attempt)}</td><td>${esc(r.cpu_threads)}</td><td>${esc(r.ram_gb)} GB</td><td>${esc(r.accelerator)}</td><td>${esc(x.gpu_id??'')}</td><td>${esc(x.failure||x.waiting_reason||'')}</td></tr>`}).join('')||'<tr><td colspan="8" class="empty">Scheduler 상태가 아직 없습니다.</td></tr>';const running=d.scheduler?.running||{};q('runningPanel').innerHTML=Object.entries(running).map(([name,x])=>`<div><strong>${esc(name)}</strong><br><span class="running">PID ${esc(x.pid)} · GPU ${esc(x.gpu_id??'-')} · RSS ${Number(x.rss_gb||0).toFixed(2)} GB</span></div>`).join('<hr>')||'<span class="empty">실행 중인 작업이 없습니다.</span>';}
function renderMetrics(m){let html='';for(const [name,rows] of Object.entries(m||{})){if(!rows?.length)continue;const keys=Object.keys(rows[0]).slice(0,7);html+=`<h3>${esc(name)}</h3><div class="table-wrap"><table class="metric-table"><thead><tr>${keys.map(k=>`<th>${esc(k)}</th>`).join('')}</tr></thead><tbody>${rows.slice(0,10).map(r=>`<tr>${keys.map(k=>`<td>${esc(r[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}q('metricsPanel').innerHTML=html||'<span class="empty">아직 생성된 결과표가 없습니다.</span>';}

function renderHistory(rows){const recent=(rows||[]).slice(-12).reverse();q('resourceHistory').innerHTML=recent.map(x=>{const gpu=(x.gpu||[]).map(g=>`GPU${g.index} ${Number(g.utilization_percent||0).toFixed(0)}% · ${Number(g.memory_used_gb||0).toFixed(1)}/${Number(g.memory_total_gb||0).toFixed(1)}GB`).join('<br>');return `<div class="history-row"><strong>${new Date(x.timestamp_utc).toLocaleTimeString()}</strong><br>CPU ${Number(x.cpu_percent||0).toFixed(1)}% · RAM ${Number(x.used_ram_gb||0).toFixed(1)}/${Number(x.total_ram_gb||0).toFixed(1)}GB<br><small>${gpu||'GPU 정보 없음'}</small></div>`}).join('')||'<span class="empty">자원 이력이 아직 없습니다.</span>';}
function renderCompleted(rows){q('completedPanel').innerHTML=(rows||[]).slice(-15).reverse().map(x=>`<div><span class="status ${esc(x.status)}">${esc(x.status||x.event)}</span> <strong>${esc(x.operation||x.worker)}</strong><br><small>${esc(x.timestamp_utc)} · ${esc(x.message)}</small></div>`).join('<hr>')||'<span class="empty">완료 기록이 없습니다.</span>';}
function renderLogs(logs){q('logList').innerHTML=(logs||[]).map(x=>`<button type="button" class="log-button" data-log="${esc(x.name)}">${esc(x.name)}</button>`).join('')||'<span class="empty">로그가 없습니다.</span>';document.querySelectorAll('[data-log]').forEach(b=>b.addEventListener('click',async()=>{const d=await get(`/api/runs/${encodeURIComponent(state.runId)}/logs/${encodeURIComponent(b.dataset.log)}?lines=400`);q('logViewer').textContent=d.tail;}));}
async function loadRun(){if(!state.runId)return;const [d,s]=await Promise.all([get(`/api/runs/${encodeURIComponent(state.runId)}`),get('/api/system')]);renderSummary(d);renderSystem(s);renderJobs(d);renderMetrics(d.metrics);renderHistory(d.resource_history);renderCompleted(d.completed_operations);renderLogs(d.logs);q('updatedAt').textContent='업데이트 '+new Date().toLocaleTimeString();connectStream();}
function connectStream(){if(state.source)state.source.close();const url=api(`/api/runs/${encodeURIComponent(state.runId)}/stream`);state.source=new EventSource(url);state.source.addEventListener('update',e=>{const d=JSON.parse(e.data);renderSystem(d.system);const fake={summary:d.summary,scheduler:d.scheduler,jobs:Object.values(d.scheduler?.status||{}),metrics:{},completed_operations:[],logs:[]};q('connection').textContent='실시간 연결';q('connection').className='badge completed';loadLight();});state.source.onerror=()=>{q('connection').textContent='재연결 중';q('connection').className='badge waiting_for_resources';};}
let lightBusy=false;async function loadLight(){if(lightBusy)return;lightBusy=true;try{const d=await get(`/api/runs/${encodeURIComponent(state.runId)}`);renderSummary(d);renderJobs(d);renderMetrics(d.metrics);renderHistory(d.resource_history);renderCompleted(d.completed_operations);}finally{lightBusy=false;}}
async function loadRuns(){const d=await get('/api/runs');state.runs=d.runs||[];q('runSelect').innerHTML=state.runs.map(r=>`<option value="${esc(r.run_id)}">${esc(r.run_id)} · ${esc(r.phase)}</option>`).join('');if(!state.runId&&state.runs.length)state.runId=state.runs[0].run_id;q('runSelect').value=state.runId||'';await loadRun();}
q('runSelect').addEventListener('change',e=>{state.runId=e.target.value;loadRun()});q('refreshButton').addEventListener('click',loadRuns);loadRuns().catch(e=>{q('connection').textContent='오류';q('logViewer').textContent=e.stack||e;});
</script>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the EXP123 live and historical web dashboard.")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    dashboard = dict(config.get("dashboard", {}))
    host = args.host or str(dashboard.get("host", "127.0.0.1"))
    port = args.port or int(dashboard.get("port", 8080))
    if host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get("EXP123_DASHBOARD_TOKEN"):
        print("[WARNING] Dashboard is exposed beyond localhost without EXP123_DASHBOARD_TOKEN.")
    import uvicorn

    uvicorn.run(create_app(config_path), host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
