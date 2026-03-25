from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT.parent):
    _resolved = str(_candidate)
    if _resolved not in sys.path:
        sys.path.insert(0, _resolved)

from scripts.dashboard_paths import (
    REPLAY_CHECKPOINT_SAMPLES_DEFAULT,
    REPLAY_END_DATE_DEFAULT,
    REPLAY_RECENT_MONTHS_DEFAULT,
    REPLAY_RUN_LIMIT_DEFAULT,
    REPLAY_VNSTOCK_SOURCE_DEFAULT,
    REPLAY_WARMUP_MONTHS_DEFAULT,
)


LOCAL_NOTE = ROOT / "notebooks" / "project_RL_nhom_09 .zpln"
RUNTIME_NOTE_DIR = Path(r"D:\Programs\zeppelin-0.12.0-bin-all\docker-data\notebook")
RUNTIME_NOTE_GLOB = "project_RL_nhom_09*.zpln"
NOTE_TIMESTAMP = "2026-03-23 06:30:00.000"
PARAGRAPH_ORDER = [
    "Setup Base Payload",
    "Build Replay Cache",
    "Bind Dashboard Data",
    "SQL Overview",
    "HTML Dashboard",
]

BIND_PARAGRAPH_TEXT = r"""%spark.pyspark
import json
import sys
from pathlib import Path

project_root = str(z.input("project_root", "/workspace/project") or "/workspace/project").strip() or "/workspace/project"
for _candidate in (Path(project_root).expanduser(), Path(project_root).expanduser().parent):
    try:
        _resolved = str(_candidate.resolve())
    except Exception:
        _resolved = str(_candidate)
    if _resolved and _resolved not in sys.path:
        sys.path.insert(0, _resolved)

from scripts.dashboard_paths import DashboardProjectPaths


def load_json(path):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


paths = DashboardProjectPaths.from_project_root(project_root)
train_path = paths.train_cache_path
replay_path = paths.replay_cache_path
dashboard_path = paths.dashboard_cache_path

train_payload = load_json(train_path)
if not train_payload:
    raise FileNotFoundError("Chưa có train cache. Hãy chạy 'Setup Base Payload' trước.")

payload = dict(train_payload)
replay_payload = load_json(replay_path)
if isinstance(replay_payload, dict) and (replay_payload.get("runs") or replay_payload.get("status") == "ready"):
    payload["checkpointReplay"] = {
        "status": replay_payload.get("status") or "ready",
        "message": replay_payload.get("message") or "Đã dựng payload replay checkpoint theo ngày.",
        "warnings": replay_payload.get("warnings") or [],
        "runs": replay_payload.get("runs") or [],
        "runMap": replay_payload.get("runMap") or {},
        "defaultRunId": replay_payload.get("defaultRunId"),
    }

dashboard_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
print(json.dumps({
    "bind_status": "ok",
    "dashboard_payload": str(dashboard_path),
    "has_replay": bool(payload.get("checkpointReplay", {}).get("runs")),
    "replay_status": payload.get("checkpointReplay", {}).get("status"),
    "run_count": len(payload.get("runs") or []),
    "replay_run_count": len(payload.get("checkpointReplay", {}).get("runs") or []),
}, ensure_ascii=False, indent=2))
"""


DASHBOARD_PARTS = []

DASHBOARD_PARTS.append(
    r"""%spark.pyspark
import json
import sys
from pathlib import Path

project_root = str(z.input("project_root", "/workspace/project") or "/workspace/project").strip() or "/workspace/project"
for _candidate in (Path(project_root).expanduser(), Path(project_root).expanduser().parent):
    try:
        _resolved = str(_candidate.resolve())
    except Exception:
        _resolved = str(_candidate)
    if _resolved and _resolved not in sys.path:
        sys.path.insert(0, _resolved)

from scripts.dashboard_paths import DashboardProjectPaths


def load_json(path):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


paths = DashboardProjectPaths.from_project_root(project_root)
train_cache = load_json(paths.train_cache_path) or load_json(paths.dashboard_cache_path)
replay_cache = load_json(paths.replay_cache_path)
merged_cache = load_json(paths.dashboard_cache_path) or {}
if not train_cache:
    raise FileNotFoundError("Chưa có cache. Hãy chạy 'Setup Base Payload' trước.")

payload = dict(train_cache)
if isinstance(replay_cache, dict) and (replay_cache.get("runs") or replay_cache.get("status") == "ready"):
    payload["checkpointReplay"] = {
        "status": replay_cache.get("status") or "ready",
        "message": replay_cache.get("message") or "Đã dựng payload replay checkpoint theo ngày.",
        "warnings": replay_cache.get("warnings") or [],
        "runs": replay_cache.get("runs") or [],
        "runMap": replay_cache.get("runMap") or {},
        "defaultRunId": replay_cache.get("defaultRunId"),
    }
elif merged_cache.get("checkpointReplay"):
    payload["checkpointReplay"] = merged_cache.get("checkpointReplay")
else:
    payload["checkpointReplay"] = {"status": "pending", "message": "Chưa dựng replay cache.", "warnings": [], "runs": []}

def strip_hidden_baselines(node):
    if isinstance(node, dict):
        cleaned = {}
        for key, value in node.items():
            key_text = str(key)
            if key_text == "benchmarks" or "equal_weight" in key_text or "buy_hold" in key_text:
                continue
            cleaned[key] = strip_hidden_baselines(value)
        return cleaned
    if isinstance(node, list):
        return [strip_hidden_baselines(item) for item in node]
    return node


payload = strip_hidden_baselines(payload)
payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
html = '''%html
<div id="rl-project-dashboard"></div>
<style>
  #rl-project-dashboard{--bg:#07111d;--panel:#0c1827;--line:rgba(140,170,205,.22);--ink:#f5f9ff;--muted:#a4b9cf;--green:#20d6a4;--red:#ff7a88;--gold:#ffd166;--blue:#73b6ff;font-family:'Bahnschrift','Segoe UI Variable','Aptos',sans-serif;color:var(--ink);background:radial-gradient(circle at top right,rgba(32,214,164,.10),transparent 26%),radial-gradient(circle at left top,rgba(115,182,255,.08),transparent 24%),linear-gradient(180deg,#07111d 0%,#0a1524 100%);border:1px solid #18314c;border-radius:28px;padding:24px;box-shadow:0 24px 60px rgba(0,0,0,.22)}
  #rl-project-dashboard *{box-sizing:border-box}
  #rl-project-dashboard .grid,#rl-project-dashboard .hero,#rl-project-dashboard .replay,#rl-project-dashboard .tables,#rl-project-dashboard .support{display:grid;gap:16px}
  #rl-project-dashboard .hero{grid-template-columns:1.08fr .92fr}
  #rl-project-dashboard .replay{grid-template-columns:1.28fr .72fr}
  #rl-project-dashboard .tables{grid-template-columns:repeat(2,minmax(0,1fr))}
  #rl-project-dashboard .support{grid-template-columns:1fr}
  #rl-project-dashboard .panel{background:linear-gradient(180deg,rgba(14,29,47,.98) 0%,rgba(8,17,28,.98) 100%);border:1px solid rgba(73,106,141,.42);border-radius:22px;padding:18px;box-shadow:0 16px 34px rgba(0,0,0,.24)}
  #rl-project-dashboard .heroMain{background:radial-gradient(circle at 100% 0%,rgba(32,214,164,.10),transparent 22%),linear-gradient(180deg,#10253b 0%,#0a1524 100%)}
  #rl-project-dashboard .kicker,#rl-project-dashboard .chip,#rl-project-dashboard .pill{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border-radius:999px;font-size:12px;font-weight:900;letter-spacing:.05em}
  #rl-project-dashboard .kicker{background:rgba(32,214,164,.14);color:#b6fde6;border:1px solid rgba(32,214,164,.28);text-transform:uppercase;margin-bottom:10px}
  #rl-project-dashboard .chip{background:rgba(115,182,255,.12);color:#e1f0ff;border:1px solid rgba(115,182,255,.24);margin:0 8px 8px 0}
  #rl-project-dashboard .pill{background:rgba(255,255,255,.06);color:#dbe8f7;border:1px solid rgba(255,255,255,.08)}
  #rl-project-dashboard h2,#rl-project-dashboard h3,#rl-project-dashboard h4,#rl-project-dashboard strong,#rl-project-dashboard label,#rl-project-dashboard th{margin:0;color:#f8fbff}
  #rl-project-dashboard p{margin:0;color:#d1dfed;line-height:1.68}
  #rl-project-dashboard .title{font-size:28px;font-weight:900;line-height:1.18;margin-bottom:10px}
  #rl-project-dashboard .lead{margin-bottom:14px;max-width:68ch}
  #rl-project-dashboard .strip,#rl-project-dashboard .metrics,#rl-project-dashboard .compare,#rl-project-dashboard .context{display:grid;gap:12px}
  #rl-project-dashboard .strip{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:14px}
  #rl-project-dashboard .metrics,#rl-project-dashboard .compare{grid-template-columns:repeat(2,minmax(0,1fr))}
  #rl-project-dashboard .context{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:14px}
  #rl-project-dashboard .card,#rl-project-dashboard .stat,#rl-project-dashboard .cmp{background:linear-gradient(180deg,rgba(19,35,53,.92) 0%,rgba(13,24,37,.94) 100%);border:1px solid rgba(123,160,198,.14);border-radius:16px;padding:12px 13px}
  #rl-project-dashboard .card span,#rl-project-dashboard .stat span,#rl-project-dashboard .cmp span{display:block;font-size:12px;color:#abd0f4;margin-bottom:4px;font-weight:700}
  #rl-project-dashboard .card strong,#rl-project-dashboard .stat strong,#rl-project-dashboard .cmp strong{font-size:20px;font-weight:900}
  #rl-project-dashboard .pos{color:var(--green)} #rl-project-dashboard .neg{color:var(--red)}
  #rl-project-dashboard .controls{display:grid;gap:12px} #rl-project-dashboard .row{display:grid;gap:12px;grid-template-columns:repeat(2,minmax(0,1fr))}
  #rl-project-dashboard label{display:block;margin-bottom:8px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;font-weight:900;color:#d5e5f4}
  #rl-project-dashboard select,#rl-project-dashboard input[type=range]{width:100%}
  #rl-project-dashboard select{padding:13px 14px;border-radius:14px;border:1px solid rgba(115,182,255,.26);background:#f8fbff;color:#07111d;font-weight:900}
  #rl-project-dashboard input[type=range]{accent-color:var(--green)}
  #rl-project-dashboard .btns{display:flex;flex-wrap:wrap;gap:10px} #rl-project-dashboard button{border:none;border-radius:999px;padding:10px 16px;font-weight:900;cursor:pointer}
  #rl-project-dashboard .ghost{background:rgba(255,255,255,.07);color:#e5f0fa;border:1px solid rgba(255,255,255,.08)} #rl-project-dashboard .primary{background:#dbfff4;color:#0a7358}
  #rl-project-dashboard .policies{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));margin-top:14px}
  #rl-project-dashboard .policy{padding:14px;border-radius:18px;border:1px solid rgba(130,160,195,.20);background:linear-gradient(180deg,rgba(14,31,48,.98) 0%,rgba(9,18,30,.98) 100%);color:var(--ink);text-align:left;cursor:pointer;transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease}
  #rl-project-dashboard .policy:hover{transform:translateY(-2px);border-color:rgba(115,182,255,.34)}
  #rl-project-dashboard .policy.active{border-color:rgba(32,214,164,.52);box-shadow:0 0 0 1px rgba(32,214,164,.16),0 18px 34px rgba(0,0,0,.22);background:linear-gradient(180deg,rgba(11,40,46,.98) 0%,rgba(7,24,28,.98) 100%)}
  #rl-project-dashboard .policy.compare{border-color:rgba(255,209,102,.46);background:linear-gradient(180deg,rgba(40,32,13,.96) 0%,rgba(27,20,8,.96) 100%)}
  #rl-project-dashboard .stage{display:inline-flex;padding:5px 10px;border-radius:999px;background:rgba(255,255,255,.06);color:#d4e7f8;font-size:11px;font-weight:900;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px}
  #rl-project-dashboard .policy strong{display:block;font-size:17px;line-height:1.35;margin-bottom:8px}
  #rl-project-dashboard .ret{display:block;font-size:22px;font-weight:900;margin-bottom:6px}
  #rl-project-dashboard .policy small{display:block;color:#aac1d8;line-height:1.55}
"""
)

DASHBOARD_PARTS.append(
    r"""
  #rl-project-dashboard .warn{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border-radius:999px;background:rgba(255,209,102,.12);color:#ffe6a1;border:1px solid rgba(255,209,102,.24);font-size:12px;font-weight:900;margin:0 8px 8px 0}
  #rl-project-dashboard .chartShell{background:linear-gradient(180deg,rgba(8,18,30,.96) 0%,rgba(7,14,24,.96) 100%);border:1px solid rgba(70,103,137,.36);border-radius:22px;padding:14px}
  #rl-project-dashboard .legend{display:flex;flex-wrap:wrap;gap:10px}
  #rl-project-dashboard .legend span{display:inline-flex;align-items:center;gap:8px;color:#d7e7f8;font-size:12px;font-weight:900}
  #rl-project-dashboard .legend i{width:18px;height:4px;border-radius:999px;display:inline-block}
  #rl-project-dashboard .chartBox{position:relative;background:radial-gradient(circle at top right,rgba(32,214,164,.08),transparent 24%),linear-gradient(180deg,rgba(9,20,33,.98) 0%,rgba(5,12,21,.98) 100%);border:1px solid rgba(88,122,159,.28);border-radius:20px;overflow:hidden;padding:12px}
  #rl-project-dashboard .chartBox:after{content:'';position:absolute;inset:0;pointer-events:none;background:linear-gradient(180deg,rgba(255,255,255,.04),transparent 25%),repeating-linear-gradient(90deg,rgba(255,255,255,.03) 0 1px,transparent 1px 78px),repeating-linear-gradient(0deg,rgba(255,255,255,.02) 0 1px,transparent 1px 64px);opacity:.72}
  #rl-project-dashboard .chartBox svg{position:relative;z-index:1;width:100%;display:block}
  #rl-project-dashboard .caption{position:relative;z-index:1;display:flex;justify-content:space-between;gap:10px;margin-top:10px;color:#9bb2c9;font-size:12px;font-weight:800}
  #rl-project-dashboard table{width:100%;border-collapse:separate;border-spacing:0 6px;table-layout:fixed}
  #rl-project-dashboard th,#rl-project-dashboard td{padding:10px 10px;text-align:left;font-size:13px;border-bottom:1px solid rgba(255,255,255,.08)}
  #rl-project-dashboard th{background:rgba(8,20,32,.98);font-weight:900;color:#e7f3ff}
  #rl-project-dashboard td{background:rgba(11,22,35,.98);color:#f7fbff;font-weight:700;text-shadow:0 1px 0 rgba(0,0,0,.32)}
  #rl-project-dashboard tbody tr:nth-child(even) td{background:rgba(18,34,52,.98)}
  #rl-project-dashboard tbody tr:hover td{background:rgba(25,44,66,.98)}
  #rl-project-dashboard .empty{color:#9cb5cf;font-size:14px}
  #rl-project-dashboard .foot{color:#a9c0d6;font-size:12px;line-height:1.65}
  @media (max-width:1200px){#rl-project-dashboard .hero,#rl-project-dashboard .replay,#rl-project-dashboard .tables,#rl-project-dashboard .support{grid-template-columns:1fr}#rl-project-dashboard .strip,#rl-project-dashboard .context,#rl-project-dashboard .metrics,#rl-project-dashboard .compare,#rl-project-dashboard .row{grid-template-columns:repeat(2,minmax(0,1fr))}}
  @media (max-width:760px){#rl-project-dashboard{padding:16px;border-radius:20px}#rl-project-dashboard .strip,#rl-project-dashboard .context,#rl-project-dashboard .metrics,#rl-project-dashboard .compare,#rl-project-dashboard .row{grid-template-columns:1fr}#rl-project-dashboard .policies{grid-template-columns:1fr}}
</style>
<script>
(() => {
  const data = __PAYLOAD_JSON__;
  const root = document.getElementById('rl-project-dashboard');
  if (!root) return;
  root.innerHTML = '<div class="panel"><div class="kicker">Đang dựng dashboard</div><p class="foot">Đang nạp checkpoint replay từ cache vnstock và chuẩn bị giao diện demo.</p></div>';
  const state = { run: 0, cp: -1, cmp: -1, frame: 0, speed: 140, playing: true, timer: null, auto: false };
  const esc = (v) => String(v == null ? '' : v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\"/g,'&quot;');
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const num = (v, d=2) => (v == null || v === '' || Number.isNaN(Number(v))) ? 'N/A' : Number(v).toLocaleString('vi-VN', {minimumFractionDigits:d, maximumFractionDigits:d});
  const money = (v) => (v == null || v === '' || Number.isNaN(Number(v))) ? 'N/A' : Number(v).toLocaleString('vi-VN', {maximumFractionDigits:0}) + ' đ';
  const pct = (v, d=2) => (v == null || v === '' || Number.isNaN(Number(v))) ? 'N/A' : num(v, d) + '%';
  const cls = (v) => Number(v || 0) >= 0 ? 'pos' : 'neg';
  const deltaPct = (a, b) => (a == null || b == null || Number(b) === 0 || Number.isNaN(Number(a)) || Number.isNaN(Number(b))) ? null : ((Number(a) / Number(b)) - 1) * 100;
  const deltaVal = (a, b) => (a == null || b == null || Number.isNaN(Number(a)) || Number.isNaN(Number(b))) ? null : Number(a) - Number(b);
  const splitPts = (s) => String(s || '').trim().split(/\\s+/).filter(Boolean);
  const slicePts = (s, idx) => { const a = splitPts(s); if (!a.length) return ''; const i = clamp(idx, 0, a.length - 1); return a.slice(0, i + 1).join(' '); };
  const pointAt = (s, idx) => { const a = splitPts(s); if (!a.length) return null; const p = String(a[clamp(idx, 0, a.length - 1)]).split(','); return {x:Number(p[0]), y:Number(p[1])}; };
  const areaPts = (s, idx, floorY=248) => { const a = splitPts(s); if (!a.length) return ''; const i = clamp(idx, 0, a.length - 1); const p = a.slice(0, i + 1); const f = String(p[0]).split(','); const l = String(p[p.length - 1]).split(','); return `${Number(f[0]).toFixed(1)},${floorY} ${p.join(' ')} ${Number(l[0]).toFixed(1)},${floorY}`; };
  const replay = () => data.checkpointReplay || {status:'pending', message:'Chưa dựng replay cache.', warnings:[], runs:[]};
  const runs = () => Array.isArray(data.runs) ? data.runs : [];
  const bestRun = () => runs().find((x) => x && x.run_id === data.defaultRunId) || runs()[0] || {summary:{}, benchmarks:{highlights:{}, rows:[]}};
  const replayRuns = () => Array.isArray(replay().runs) ? replay().runs : [];
  const replayRun = () => replayRuns()[state.run] || replayRuns()[0] || {run_id:'N/A', label:'Chưa có replay', data_source_label:'vnstock', display_start:null, display_end:null, checkpoints:[], currentCheckpointId:null, defaultCheckpointId:null, defaultCompareCheckpointId:null, worstCheckpointId:null, firstCheckpointId:null};
  const cps = () => Array.isArray(replayRun().checkpoints) ? replayRun().checkpoints : [];
  const stage = (cp, idx, total) => { if (!cp) return 'Policy'; if (cp.kind === 'untrained' || cp.checkpoint_id === 'seed42_untrained') return 'Chưa học'; if (cp.checkpoint_id === 'best_model') return 'Best'; if (cp.checkpoint_id === 'final_model') return 'Hiện tại'; if (idx === 0) return 'Mới học'; if (idx <= Math.max(1, Math.floor((total - 1) / 3))) return 'Đang học'; if (idx >= total - 2) return 'Ổn định'; return 'Tăng tốc'; };
  const defaultCp = () => { const list = cps(); if (!list.length) return 0; const run = replayRun(); const wanted = run.currentCheckpointId || run.defaultCheckpointId || run.bestCheckpointId || 'final_model'; const idx = list.findIndex((x) => x && x.checkpoint_id === wanted); if (idx >= 0) return idx; const current = list.findIndex((x) => x && x.checkpoint_id === 'final_model'); if (current >= 0) return current; const best = list.findIndex((x) => x && x.checkpoint_id === 'best_model'); return best >= 0 ? best : 0; };
  const defaultCmp = () => {
    const list = cps();
    if (!list.length) return -1;
    const run = replayRun();
    const blockedId = list[state.cp] && list[state.cp].checkpoint_id;
    const preferred = [run.defaultCompareCheckpointId, run.worstCheckpointId, run.firstCheckpointId];
    for (const wanted of preferred) {
      const idx = list.findIndex((x) => x && x.checkpoint_id === wanted && x.checkpoint_id !== blockedId);
      if (idx >= 0) return idx;
    }
    const earliestNumeric = list.findIndex((x) => x && x.kind === 'numeric' && x.checkpoint_id !== blockedId);
    if (earliestNumeric >= 0) return earliestNumeric;
    const ranked = list
      .map((x, i) => ({x, i}))
      .filter(({x}) => x && x.checkpoint_id !== blockedId)
      .sort((a, b) => Number(a.x.summary && a.x.summary.final_return_pct || 0) - Number(b.x.summary && b.x.summary.final_return_pct || 0));
    return ranked.length ? ranked[0].i : list.findIndex((_, i) => i !== state.cp);
  };
  const ensure = () => {
    const list = cps();
    if (!list.length) { state.cp = 0; state.cmp = -1; state.frame = 0; return; }
    if (state.cp < 0 || state.cp >= list.length) state.cp = defaultCp();
    else state.cp = clamp(state.cp, 0, list.length - 1);
    if (state.cmp < 0 || state.cmp >= list.length || state.cmp === state.cp) state.cmp = defaultCmp();
    if (state.cmp < 0) state.cmp = state.cp;
    const current = list[state.cp] || {frames:[]};
    state.frame = clamp(state.frame, 0, Math.max(frames(current).length - 1, 0));
  };
  const cp = () => { ensure(); return cps()[state.cp] || {summary:{}, chart:{}, frames:[]}; };
  const cmp = () => { ensure(); return cps()[state.cmp] || cp(); };
  const frames = (item) => Array.isArray(item && item.frames) ? item.frames : [];
  const frameAt = (item, idx) => { const list = frames(item); if (!list.length) return {top_allocations:[], top_positions:[], trade_rows:[]}; return list[clamp(idx, 0, list.length - 1)] || {top_allocations:[], top_positions:[], trade_rows:[]}; };
  const now = () => frameAt(cp(), state.frame);
  const cmpNow = () => frameAt(cmp(), state.frame);
  const markerX = () => { const m = cp().chart && Array.isArray(cp().chart.marker_xs) ? cp().chart.marker_xs : []; return m.length ? (m[clamp(state.frame, 0, m.length - 1)] || m[m.length - 1] || 14) : 14; };
  const opt = (list, sel, f) => list.map((item, idx) => `<option value="${idx}"${idx===sel?' selected':''}>${esc(f(item, idx))}</option>`).join('');
  const rowsHtml = (list, row, empty) => list && list.length ? list.map(row).join('') : (empty || '<tr><td colspan="6" class="empty">Không có dữ liệu</td></tr>');
  const cards = () => cps().map((item, idx) => { let c = 'policy'; if (idx === state.cp) c += ' active'; else if (idx === state.cmp) c += ' compare'; const s = item.summary || {}; return `<button class="${c}" data-role="policy" data-idx="${idx}"><span class="stage">${esc(stage(item, idx, cps().length))}</span><strong>${esc(item.label || item.checkpoint_id || ('Checkpoint ' + idx))}</strong><span class="ret ${cls(s.final_return_pct || 0)}">${pct(s.final_return_pct, 2)}</span><small>${esc(money(s.final_value))} · ${esc(frames(item).length)} ngày replay</small></button>`; }).join('');
  const bars = () => { const list = frames(cp()); if (!list.length) return ''; const vals = list.map((x) => Number(x.day_return_pct || 0)); const maxAbs = vals.reduce((a, v) => Math.max(a, Math.abs(v)), 1.2); return vals.map((v, i) => { const x = 12 + i * (596 / Math.max(vals.length - 1, 1)); const h = Math.max(2, Math.abs(v) / maxAbs * 28); const y = v >= 0 ? 38 - h : 38; const fill = v >= 0 ? '#20d6a4' : '#ff7a88'; const opacity = i === state.frame ? .98 : (i <= state.frame ? .58 : .14); return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="8" height="${h.toFixed(1)}" rx="3" fill="${fill}" opacity="${opacity}"></rect>`; }).join(''); };
"""
)

DASHBOARD_PARTS.append(
    r"""
            <div class="metrics" style="margin-top:14px"><div class="stat"><span>Giá trị danh mục</span><strong>${money(f.portfolio_value)}</strong></div><div class="stat"><span>Lợi nhuận lũy kế</span><strong class="${cls(f.total_return_pct || 0)}">${pct(f.total_return_pct, 2)}</strong></div><div class="stat"><span>Lợi nhuận ngày</span><strong class="${cls(f.day_return_pct || 0)}">${pct(f.day_return_pct, 2)}</strong></div><div class="stat"><span>Tỷ trọng tiền mặt</span><strong>${pct(f.cash_weight_pct, 2)}</strong></div><div class="stat"><span>So với danh mục đều</span><strong class="${cls(f.vs_equal_weight_pct || 0)}">${pct(f.vs_equal_weight_pct, 2)}</strong></div><div class="stat"><span>So với mua và giữ</span><strong class="${cls(f.vs_buy_hold_pct || 0)}">${pct(f.vs_buy_hold_pct, 2)}</strong></div></div>
            <div class="compare" style="margin-top:14px"><div class="cmp"><span>Chênh hiện tại vs policy so sánh</span><strong class="${cls(currentVsCompare || 0)}">${pct(currentVsCompare, 2)}</strong></div><div class="cmp"><span>Chênh cuối kỳ vs policy so sánh</span><strong class="${cls(finalVsCompare || 0)}">${pct(finalVsCompare, 2)}</strong></div><div class="cmp"><span>Chênh giá trị cuối kỳ</span><strong class="${cls(finalValueGap || 0)}">${money(finalValueGap)}</strong></div><div class="cmp"><span>Số lệnh trong ngày</span><strong>${esc(f.trade_count || 0)}</strong></div><div class="cmp"><span>Checkpoint đang xem</span><strong class="${cls((selected.summary || {}).final_return_pct || 0)}">${pct((selected.summary || {}).final_return_pct, 2)}</strong></div><div class="cmp"><span>Policy so sánh</span><strong class="${cls((compare.summary || {}).final_return_pct || 0)}">${pct((compare.summary || {}).final_return_pct, 2)}</strong></div></div>
          </div>
        </section>
        <section class="tables">
          <div class="panel"><div class="kicker">Phân bổ mục tiêu</div><h3>Top tỷ trọng policy muốn nắm giữ</h3><table><thead><tr><th>Tài sản</th><th>Tỷ trọng mục tiêu</th></tr></thead><tbody>${rowsHtml((f.top_allocations || []).slice(0, 8), (x) => `<tr><td>${esc(x.label)}</td><td>${pct(x.weight_pct, 2)}</td></tr>`, '<tr><td colspan="2" class="empty">Chưa có dữ liệu phân bổ.</td></tr>')}</tbody></table></div>
          <div class="panel"><div class="kicker">Vị thế thực tế</div><h3>Danh mục sau khớp lệnh</h3><table><thead><tr><th>Tài sản</th><th>Giá trị</th><th>Tỷ trọng</th></tr></thead><tbody>${rowsHtml((f.top_positions || []).slice(0, 8), (x) => `<tr><td>${esc(x.label)}</td><td>${money(x.position_value)}</td><td>${pct(x.weight_pct, 2)}</td></tr>`, '<tr><td colspan="3" class="empty">Chưa có dữ liệu vị thế.</td></tr>')}<tr><td>Tiền mặt</td><td>${money(f.cash)}</td><td>${pct(f.cash_weight_pct, 2)}</td></tr></tbody></table></div>
          <div class="panel" style="grid-column:1/-1"><div class="kicker">Giao dịch nổi bật</div><h3>Lệnh đáng chú ý trong ngày</h3><table><thead><tr><th>Mã</th><th>Hướng</th><th>Số lượng</th><th>Giá mở</th><th>Giá đóng</th><th>Tỷ trọng mục tiêu</th></tr></thead><tbody>${rowsHtml((f.trade_rows || []).slice(0, 12), (x) => `<tr><td>${esc(x.symbol)}</td><td>${esc(x.direction)}</td><td>${esc(x.shares)}</td><td>${num(x.execution_price, 2)}</td><td>${num(x.close_price, 2)}</td><td>${pct(x.target_weight_pct, 2)}</td></tr>`, '<tr><td colspan="6" class="empty">Ngày này policy không phát sinh lệnh nổi bật.</td></tr>')}</tbody></table></div>
        </section>` : `<section class="panel"><div class="kicker">Replay chưa sẵn sàng</div><h3>Chưa có dữ liệu checkpoint để mô phỏng</h3><p>${esc(rep.message || 'Hãy chạy Build Replay Cache để dựng replay từ vnstock.')}</p></section>`}
        <section class="support">
          <div class="panel"><div class="kicker">Bối cảnh thị trường</div><h3>Snapshot giá gần nhất</h3><p class="foot">Khối này chỉ dùng để đặt ngữ cảnh live quanh phiên demo. Trọng tâm đánh giá vẫn nằm ở replay checkpoint day-by-day trên dữ liệu vnstock.</p><div class="context"><div class="card"><span>Trạng thái</span><strong>${esc((data.liveMarket || {}).status_label || 'N/A')}</strong></div><div class="card"><span>Theo dõi</span><strong>${esc(((data.liveMarket || {}).tracked_symbols || []).join(', ') || 'N/A')}</strong></div><div class="card"><span>Làm mới lúc</span><strong>${esc((data.liveMarket || {}).refreshed_at || 'N/A')}</strong></div><div class="card"><span>Nguồn replay</span><strong>${esc(rr.data_source_label || 'vnstock')}</strong></div></div><table style="margin-top:14px"><thead><tr><th>Mã</th><th>Nguồn</th><th>Giá</th><th>% ngày</th><th>Lệch so với close</th></tr></thead><tbody>${liveRows()}</tbody></table></div>
        </section>
      </div>`;
    if (ready) {
      const cpSelect = document.getElementById('cpSelect'), cmpSelect = document.getElementById('cmpSelect'), speedSelect = document.getElementById('speedSelect'), frameRange = document.getElementById('frameRange');
      if (cpSelect) cpSelect.onchange = (e) => { state.cp = Number(e.target.value || 0); state.frame = 0; stop(); state.playing = true; render(); };
      if (cmpSelect) cmpSelect.onchange = (e) => { state.cmp = Number(e.target.value || 0); if (state.cmp === state.cp) state.cmp = defaultCmp(); render(); };
      if (speedSelect) speedSelect.onchange = (e) => { state.speed = Number(e.target.value || 140); if (state.playing) { stop(); state.playing = true; render(); } };
      if (frameRange) frameRange.oninput = (e) => { state.frame = Number(e.target.value || 0); stop(); render(); };
      const btnStart = document.getElementById('btnStart'), btnPrev = document.getElementById('btnPrev'), btnNext = document.getElementById('btnNext'), btnPlay = document.getElementById('btnPlay');
      if (btnStart) btnStart.onclick = () => { state.frame = 0; stop(); state.playing = true; render(); };
      if (btnPrev) btnPrev.onclick = () => { state.frame = clamp(state.frame - 1, 0, Math.max(frames(cp()).length - 1, 0)); stop(); render(); };
      if (btnNext) btnNext.onclick = () => { state.frame = clamp(state.frame + 1, 0, Math.max(frames(cp()).length - 1, 0)); stop(); render(); };
      if (btnPlay) btnPlay.onclick = () => { if (state.playing) { stop(); render(); } else { if (state.frame >= Math.max(frames(cp()).length - 1, 0)) state.frame = 0; state.playing = true; render(); } };
      if (!state.auto && frames(cp()).length) { state.auto = true; state.playing = true; schedule(); }
      else if (state.playing && !state.timer && frames(cp()).length) { schedule(); }
    }
  }
  setTimeout(render, 0);
})();
</script>
'''
print(html.replace('__PAYLOAD_JSON__', payload_json))
"""
)

DASHBOARD_PARTS.append(
    r"""
  const insight = () => { const f = now(), g = cmpNow(); const a = []; if (f.headline) a.push(f.headline); const d = deltaVal(f.total_return_pct, g.total_return_pct); if (d != null) a.push(`Policy hiện tại đang ${d >= 0 ? 'cao hơn' : 'thấp hơn'} policy so sánh ${pct(Math.abs(d), 2)} theo lợi nhuận tích lũy ở khung ngày này.`); if (f.vs_buy_hold_pct != null) a.push(`So với mua và giữ: ${pct(f.vs_buy_hold_pct, 2)}.`); return a.join(' '); };
  const stop = () => { if (state.timer) { clearTimeout(state.timer); state.timer = null; } state.playing = false; };
  const schedule = () => {
    if (!state.playing) return;
    if (state.timer) clearTimeout(state.timer);
    state.timer = setTimeout(() => {
      state.timer = null;
      if (!state.playing) return;
      const total = frames(cp()).length;
      if (!total) { state.playing = false; render(); return; }
      if (state.frame >= total - 1) { state.frame = total - 1; state.playing = false; render(); return; }
      state.frame += 1;
      render();
    }, state.speed);
  };
  const play = () => { state.playing = true; schedule(); };
  const liveRows = () => rowsHtml(((data.liveMarket || {}).rows || []).slice(0, 6), (r) => {
    const hasLive = r.status === 'LIVE' && r.live_price != null;
    const source = hasLive ? 'Trực tiếp' : 'Close gần nhất';
    const price = hasLive ? money(r.live_price) : money(r.last_close);
    const changeText = r.change_pct != null ? pct(r.change_pct, 2) : 'N/A';
    const changeClass = r.change_pct != null ? cls(r.change_pct || 0) : '';
    const gapText = r.gap_vs_last_close_pct != null ? pct(r.gap_vs_last_close_pct, 2) : '0,00%';
    const gapClass = r.gap_vs_last_close_pct != null ? cls(r.gap_vs_last_close_pct || 0) : '';
    return `<tr><td>${esc(r.symbol)}</td><td>${esc(source)}</td><td>${price}</td><td class="${changeClass}">${changeText}</td><td class="${gapClass}">${gapText}</td></tr>`;
  }, '<tr><td colspan="5" class="empty">Chưa có snapshot live.</td></tr>');
  const benchRows = () => rowsHtml((((bestRun().benchmarks || {}).rows || []).slice(0, 6)), (r) => `<tr><td>${esc(r.stage)}</td><td>${esc(r.baseline)}</td><td class="${cls(r.delta_total_return_pct || 0)}">${pct(r.delta_total_return_pct, 2)}</td><td class="${cls(r.delta_sharpe_ratio || 0)}">${num(r.delta_sharpe_ratio, 3)}</td></tr>`, '<tr><td colspan="4" class="empty">Chưa có benchmark.</td></tr>');
  function render() {
    const run = bestRun(), rep = replay(), rr = replayRun(), selected = cp(), compare = cmp(), f = now(), g = cmpNow(), ready = rep.status === 'ready' && replayRuns().length && cps().length;
    const testEq = ((run.benchmarks || {}).highlights || {}).test_equal_weight || {};
    const testBh = ((run.benchmarks || {}).highlights || {}).test_buy_hold || {};
    const currentVsCompare = deltaVal(f.total_return_pct, g.total_return_pct);
    const finalVsCompare = deltaVal((selected.summary || {}).final_return_pct, (compare.summary || {}).final_return_pct);
    const finalValueGap = deltaVal((selected.summary || {}).final_value, (compare.summary || {}).final_value);
    const dSel = pointAt(selected.chart && selected.chart.model_points, state.frame);
    const dCmp = pointAt(compare.chart && compare.chart.model_points, state.frame);
    const dEq = pointAt(selected.chart && selected.chart.equal_weight_points, state.frame);
    const dBh = pointAt(selected.chart && selected.chart.buy_hold_points, state.frame);
    root.innerHTML = `
      <div class="grid">
        <section class="hero">
          <div class="panel heroMain">
            <div class="kicker">Checkpoint policy replay</div>
            <div class="title">Demo checkpoint day-by-day trên dữ liệu mới từ ${esc(rr.data_source_label || 'vnstock')}</div>
            <p class="lead">Trọng tâm là checkpoint của từng policy, không còn bắt người xem quan tâm tới run selector. Chọn checkpoint, chọn policy đối chiếu, rồi để dashboard tự chạy theo ngày để thấy đường vốn, benchmark và phân bổ tài sản thay đổi như thế nào trên dữ liệu replay gần nhất.</p>
            <div><span class="chip">Replay: ${esc(rr.recent_months || rep.recent_months || 'N/A')} tháng</span><span class="chip">Cửa sổ replay: ${esc(rr.display_start || 'N/A')} → ${esc(rr.display_end || 'N/A')}</span><span class="chip">Artifact nguồn: ${esc(rr.run_id || run.run_id || 'N/A')}</span><span class="chip">Checkpoint: ${esc(cps().length || 0)}</span><span class="chip">Ngày replay: ${esc(frames(selected).length || 0)}</span></div>
            ${((rep.warnings || []).length ? '<div style="margin-top:10px;">' + (rep.warnings || []).map((w) => `<span class="warn">${esc(w)}</span>`).join('') + '</div>' : '')}
            <div class="strip"><div class="card"><span>Checkpoint đang xem</span><strong>${esc(selected.label || selected.checkpoint_id || 'N/A')}</strong></div><div class="card"><span>Policy so sánh</span><strong>${esc(compare.label || compare.checkpoint_id || 'N/A')}</strong></div><div class="card"><span>Ngày đang phát</span><strong>${esc(f.date || 'N/A')}</strong></div><div class="card"><span>Tự chạy</span><strong>${state.playing ? 'Đang bật' : 'Tạm dừng'}</strong></div></div>
          </div>
          <div class="panel">
            <div class="kicker">Điều khiển demo</div>
            ${ready ? `
            <div class="controls">
              <div class="row"><div><label>Checkpoint đang xem</label><select id="cpSelect">${opt(cps(), state.cp, (x) => x.label || x.checkpoint_id)}</select></div><div><label>Policy so sánh</label><select id="cmpSelect">${opt(cps(), state.cmp, (x) => x.label || x.checkpoint_id)}</select></div></div>
              <div class="row"><div><label>Tốc độ phát</label><select id="speedSelect"><option value="260" ${state.speed===260?'selected':''}>1x · ổn định</option><option value="180" ${state.speed===180?'selected':''}>2x · gọn</option><option value="140" ${state.speed===140?'selected':''}>4x · demo</option><option value="95" ${state.speed===95?'selected':''}>8x · tua nhanh</option></select></div><div><label>Khung ngày</label><input id="frameRange" type="range" min="0" max="${Math.max(frames(selected).length - 1, 0)}" step="1" value="${state.frame}"></div></div>
              <div class="btns"><button class="ghost" id="btnStart">Về đầu</button><button class="ghost" id="btnPrev">Lùi 1 ngày</button><button class="primary" id="btnPlay">${state.playing ? 'Tạm dừng' : (state.frame >= Math.max(frames(selected).length - 1, 0) ? 'Chạy lại' : 'Tự chạy')}</button><button class="ghost" id="btnNext">Tiến 1 ngày</button></div>
              <p class="foot">Autoplay chạy một lượt từ đầu đến ngày cuối rồi dừng lại ở khung cuối, để bạn chốt phần thuyết trình trên biểu đồ hoàn chỉnh.</p>
            </div>` : `<p class="empty">${esc(rep.message || 'Chưa sẵn sàng replay checkpoint.')}</p>`}
          </div>
        </section>
        ${ready ? `
        <section class="replay">
          <div class="panel">
            <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;margin-bottom:12px"><div><div class="kicker">Soi policy theo ngày</div><h3>${esc(f.headline || 'Đường vốn và benchmark')}</h3><p class="foot">Ngày quyết định ${esc(f.decision_date || 'N/A')} · ngày đóng phiên ${esc(f.date || 'N/A')}</p></div><div class="legend"><span><i style="background:#20d6a4"></i>Checkpoint đang xem</span><span><i style="background:#ff7a88"></i>Policy so sánh</span><span><i style="background:#ffd166"></i>Danh mục đều</span><span><i style="background:#73b6ff"></i>Mua và giữ</span></div></div>
            <div class="chartShell"><div class="chartBox"><svg viewBox="0 0 620 280" preserveAspectRatio="none"><defs><linearGradient id="rlSelectedArea" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#20d6a4" stop-opacity="0.30"></stop><stop offset="100%" stop-color="#20d6a4" stop-opacity="0.02"></stop></linearGradient></defs><polygon points="${esc(areaPts(selected.chart && selected.chart.model_points, state.frame))}" fill="url(#rlSelectedArea)"></polygon><polyline points="${esc(slicePts(selected.chart && selected.chart.buy_hold_points, state.frame))}" fill="none" stroke="#73b6ff" stroke-width="3"></polyline><polyline points="${esc(slicePts(selected.chart && selected.chart.equal_weight_points, state.frame))}" fill="none" stroke="#ffd166" stroke-width="3"></polyline><polyline points="${esc(slicePts(compare.chart && compare.chart.model_points, state.frame))}" fill="none" stroke="#ff7a88" stroke-width="3" stroke-dasharray="8 6"></polyline><polyline points="${esc(slicePts(selected.chart && selected.chart.model_points, state.frame))}" fill="none" stroke="#20d6a4" stroke-width="4"></polyline><line x1="${esc(markerX())}" x2="${esc(markerX())}" y1="16" y2="248" stroke="#fca5a5" stroke-dasharray="6 6"></line>${dSel ? `<circle cx="${dSel.x}" cy="${dSel.y}" r="5.8" fill="#20d6a4" stroke="#06111d" stroke-width="2"></circle>` : ''}${dCmp ? `<circle cx="${dCmp.x}" cy="${dCmp.y}" r="5" fill="#ff7a88" stroke="#06111d" stroke-width="2"></circle>` : ''}${dEq ? `<circle cx="${dEq.x}" cy="${dEq.y}" r="4.4" fill="#ffd166" stroke="#06111d" stroke-width="2"></circle>` : ''}${dBh ? `<circle cx="${dBh.x}" cy="${dBh.y}" r="4.4" fill="#73b6ff" stroke="#06111d" stroke-width="2"></circle>` : ''}</svg></div><div class="caption"><span>Chart chỉ hiển thị đến ngày đang phát để mô phỏng đúng tiến trình.</span><span>Khung ${state.frame + 1}/${Math.max(frames(selected).length, 1)}</span></div></div>
            <div class="chartShell" style="margin-top:12px"><div class="chartBox"><svg viewBox="0 0 620 76" preserveAspectRatio="none"><line x1="10" x2="610" y1="38" y2="38" stroke="rgba(255,255,255,.14)" stroke-dasharray="5 5"></line>${bars()}<line x1="${esc(markerX())}" x2="${esc(markerX())}" y1="8" y2="68" stroke="#fca5a5" stroke-dasharray="6 6"></line></svg></div><div class="caption"><span>Cột xanh/đỏ là lợi nhuận theo ngày của checkpoint đang xem.</span><span>${pct(f.day_return_pct, 2)} hôm nay</span></div></div>
          </div>
          <div class="panel"><div class="kicker">So sánh nhanh</div><h3>Checkpoint đang xem vs policy so sánh</h3><p class="foot">${esc(insight())}</p>
"""
)

DASHBOARD_PARTS = [DASHBOARD_PARTS[0], DASHBOARD_PARTS[1], DASHBOARD_PARTS[3], DASHBOARD_PARTS[2]]


def strip_public_benchmarks(text: str) -> str:
    replacements = [
        (
            """from scripts.dashboard_paths import DashboardProjectPaths""",
            """from scripts.dashboard_paths import DashboardProjectPaths
from src.constants import CHART_COMPARE_ALPHA""",
        ),
        (
            """Trọng tâm là checkpoint của từng policy, không còn bắt người xem quan tâm tới run selector. Chọn checkpoint, chọn policy đối chiếu, rồi để dashboard tự chạy theo ngày để thấy đường vốn, benchmark và phân bổ tài sản thay đổi như thế nào trên dữ liệu replay gần nhất.""",
            """Trọng tâm là checkpoint của từng policy, không còn bắt người xem quan tâm tới run selector. Chọn checkpoint, chọn policy đối chiếu, rồi để dashboard tự chạy theo ngày để thấy đường vốn và phân bổ tài sản thay đổi như thế nào trên dữ liệu replay gần nhất.""",
        ),
        (
            """<div class="metrics" style="margin-top:14px"><div class="stat"><span>Giá trị danh mục</span><strong>${money(f.portfolio_value)}</strong></div><div class="stat"><span>Lợi nhuận lũy kế</span><strong class="${cls(f.total_return_pct || 0)}">${pct(f.total_return_pct, 2)}</strong></div><div class="stat"><span>Lợi nhuận ngày</span><strong class="${cls(f.day_return_pct || 0)}">${pct(f.day_return_pct, 2)}</strong></div><div class="stat"><span>Tỷ trọng tiền mặt</span><strong>${pct(f.cash_weight_pct, 2)}</strong></div><div class="stat"><span>So với danh mục đều</span><strong class="${cls(f.vs_equal_weight_pct || 0)}">${pct(f.vs_equal_weight_pct, 2)}</strong></div><div class="stat"><span>So với mua và giữ</span><strong class="${cls(f.vs_buy_hold_pct || 0)}">${pct(f.vs_buy_hold_pct, 2)}</strong></div></div>""",
            """<div class="metrics" style="margin-top:14px"><div class="stat"><span>Giá trị danh mục</span><strong>${money(f.portfolio_value)}</strong></div><div class="stat"><span>Lợi nhuận lũy kế</span><strong class="${cls(f.total_return_pct || 0)}">${pct(f.total_return_pct, 2)}</strong></div><div class="stat"><span>Lợi nhuận ngày</span><strong class="${cls(f.day_return_pct || 0)}">${pct(f.day_return_pct, 2)}</strong></div><div class="stat"><span>Tỷ trọng tiền mặt</span><strong>${pct(f.cash_weight_pct, 2)}</strong></div></div>""",
        ),
        (
            """  const insight = () => { const f = now(), g = cmpNow(); const a = []; if (f.headline) a.push(f.headline); const d = deltaVal(f.total_return_pct, g.total_return_pct); if (d != null) a.push(`Policy hiện tại đang ${d >= 0 ? 'cao hơn' : 'thấp hơn'} policy so sánh ${pct(Math.abs(d), 2)} theo lợi nhuận tích lũy ở khung ngày này.`); if (f.vs_buy_hold_pct != null) a.push(`So với mua và giữ: ${pct(f.vs_buy_hold_pct, 2)}.`); return a.join(' '); };""",
            """  const insight = () => { const f = now(), g = cmpNow(); const a = []; if (f.headline) a.push(f.headline); const d = deltaVal(f.total_return_pct, g.total_return_pct); if (d != null) a.push(`Policy hiện tại đang ${d >= 0 ? 'cao hơn' : 'thấp hơn'} policy so sánh ${pct(Math.abs(d), 2)} theo lợi nhuận tích lũy ở khung ngày này.`); return a.join(' '); };""",
        ),
        (
            """  const bestRun = () => runs().find((x) => x && x.run_id === data.defaultRunId) || runs()[0] || {summary:{}, benchmarks:{highlights:{}, rows:[]}};""",
            """  const bestRun = () => runs().find((x) => x && x.run_id === data.defaultRunId) || runs()[0] || {summary:{}};""",
        ),
        (
            """  const benchRows = () => rowsHtml((((bestRun().benchmarks || {}).rows || []).slice(0, 6)), (r) => `<tr><td>${esc(r.stage)}</td><td>${esc(r.baseline)}</td><td class="${cls(r.delta_total_return_pct || 0)}">${pct(r.delta_total_return_pct, 2)}</td><td class="${cls(r.delta_sharpe_ratio || 0)}">${num(r.delta_sharpe_ratio, 3)}</td></tr>`, '<tr><td colspan="4" class="empty">Chưa có benchmark.</td></tr>');""",
            "",
        ),
        (
            """<div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;margin-bottom:12px"><div><div class="kicker">Soi policy theo ngày</div><h3>${esc(f.headline || 'Đường vốn và benchmark')}</h3><p class="foot">Ngày quyết định ${esc(f.decision_date || 'N/A')} · ngày đóng phiên ${esc(f.date || 'N/A')}</p></div><div class="legend"><span><i style="background:#20d6a4"></i>Checkpoint đang xem</span><span><i style="background:#ff7a88"></i>Policy so sánh</span><span><i style="background:#ffd166"></i>Danh mục đều</span><span><i style="background:#73b6ff"></i>Mua và giữ</span></div></div>""",
            """<div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;margin-bottom:12px"><div><div class="kicker">Soi policy theo ngày</div><h3>${esc(f.headline || 'Đường vốn checkpoint')}</h3><p class="foot">Ngày quyết định ${esc(f.decision_date || 'N/A')} · ngày đóng phiên ${esc(f.date || 'N/A')}</p></div><div class="legend"><span><i style="background:#20d6a4"></i>Checkpoint đang xem</span><span><i style="background:#ff7a88"></i>Policy so sánh</span></div></div>""",
        ),
        (
            """<div class="chartShell"><div class="chartBox"><svg viewBox="0 0 620 280" preserveAspectRatio="none"><defs><linearGradient id="rlSelectedArea" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#20d6a4" stop-opacity="0.30"></stop><stop offset="100%" stop-color="#20d6a4" stop-opacity="0.02"></stop></linearGradient></defs><polygon points="${esc(areaPts(selected.chart && selected.chart.model_points, state.frame))}" fill="url(#rlSelectedArea)"></polygon><polyline points="${esc(slicePts(selected.chart && selected.chart.buy_hold_points, state.frame))}" fill="none" stroke="#73b6ff" stroke-width="3"></polyline><polyline points="${esc(slicePts(selected.chart && selected.chart.equal_weight_points, state.frame))}" fill="none" stroke="#ffd166" stroke-width="3"></polyline><polyline points="${esc(slicePts(compare.chart && compare.chart.model_points, state.frame))}" fill="none" stroke="#ff7a88" stroke-width="3" stroke-dasharray="8 6"></polyline><polyline points="${esc(slicePts(selected.chart && selected.chart.model_points, state.frame))}" fill="none" stroke="#20d6a4" stroke-width="4"></polyline><line x1="${esc(markerX())}" x2="${esc(markerX())}" y1="16" y2="248" stroke="#fca5a5" stroke-dasharray="6 6"></line>${dSel ? `<circle cx="${dSel.x}" cy="${dSel.y}" r="5.8" fill="#20d6a4" stroke="#06111d" stroke-width="2"></circle>` : ''}${dCmp ? `<circle cx="${dCmp.x}" cy="${dCmp.y}" r="5" fill="#ff7a88" stroke="#06111d" stroke-width="2"></circle>` : ''}${dEq ? `<circle cx="${dEq.x}" cy="${dEq.y}" r="4.4" fill="#ffd166" stroke="#06111d" stroke-width="2"></circle>` : ''}${dBh ? `<circle cx="${dBh.x}" cy="${dBh.y}" r="4.4" fill="#73b6ff" stroke="#06111d" stroke-width="2"></circle>` : ''}</svg></div><div class="caption"><span>Chart chỉ hiển thị đến ngày đang phát để mô phỏng đúng tiến trình.</span><span>Khung ${state.frame + 1}/${Math.max(frames(selected).length, 1)}</span></div></div>""",
            """<div class="chartShell"><div class="chartBox"><svg viewBox="0 0 620 280" preserveAspectRatio="none"><defs><linearGradient id="rlSelectedArea" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#20d6a4" stop-opacity="0.30"></stop><stop offset="100%" stop-color="#20d6a4" stop-opacity="0.02"></stop></linearGradient></defs><polygon points="${esc(areaPts(selected.chart && selected.chart.model_points, state.frame))}" fill="url(#rlSelectedArea)"></polygon><polyline points="${esc(adjustedSlicePts(compare.chart && compare.chart.model_points, selected.chart && selected.chart.model_points, state.frame))}" fill="none" stroke="#ff7a88" stroke-width="3" stroke-dasharray="8 6"></polyline><polyline points="${esc(slicePts(selected.chart && selected.chart.model_points, state.frame))}" fill="none" stroke="#20d6a4" stroke-width="4"></polyline><line x1="${esc(markerX())}" x2="${esc(markerX())}" y1="16" y2="248" stroke="#fca5a5" stroke-dasharray="6 6"></line>${dSel ? `<circle cx="${dSel.x}" cy="${dSel.y}" r="5.8" fill="#20d6a4" stroke="#06111d" stroke-width="2"></circle>` : ''}${dCmp ? `<circle cx="${dCmp.x}" cy="${dCmp.y}" r="5" fill="#ff7a88" stroke="#06111d" stroke-width="2"></circle>` : ''}</svg></div><div class="caption"><span>Chart chỉ hiển thị đến ngày đang phát để mô phỏng đúng tiến trình.</span><span>Khung ${state.frame + 1}/${Math.max(frames(selected).length, 1)}</span></div></div>""",
        ),
    ]
    out = text.replace("Đường vốn và benchmark", "Đường vốn checkpoint")
    for old, new in replacements:
        out = out.replace(old, new)
    out = out.replace(
        """payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")""",
        """payload.setdefault("uiConfig", {})
payload["uiConfig"]["chartCompareAlpha"] = CHART_COMPARE_ALPHA
payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")""",
    )
    out = out.replace(
        """  const data = __PAYLOAD_JSON__;
  const root = document.getElementById('rl-project-dashboard');
  if (!root) return;
  root.innerHTML = '<div class="panel"><div class="kicker">Đang dựng dashboard</div><p class="foot">Đang nạp checkpoint replay từ cache vnstock và chuẩn bị giao diện demo.</p></div>';
  const state = { run: 0, cp: -1, cmp: -1, frame: 0, speed: 140, playing: true, timer: null, auto: false };""",
        """  const data = __PAYLOAD_JSON__;
  const root = document.getElementById('rl-project-dashboard');
  if (!root) return;
  root.innerHTML = '<div class="panel"><div class="kicker">Đang dựng dashboard</div><p class="foot">Đang nạp checkpoint replay từ cache vnstock và chuẩn bị giao diện demo.</p></div>';
  const state = { run: 0, cp: -1, cmp: -1, frame: 0, speed: 140, playing: true, timer: null, auto: false };
  const compareAlpha = (() => {
    const raw = Number((((data || {}).uiConfig || {}).chartCompareAlpha));
    return Number.isFinite(raw) && raw > 0 ? raw : 1;
  })();""",
    )
    out = out.replace(
        """  const areaPts = (s, idx, floorY=248) => { const a = splitPts(s); if (!a.length) return ''; const i = clamp(idx, 0, a.length - 1); const p = a.slice(0, i + 1); const f = String(p[0]).split(','); const l = String(p[p.length - 1]).split(','); return `${Number(f[0]).toFixed(1)},${floorY} ${p.join(' ')} ${Number(l[0]).toFixed(1)},${floorY}`; };""",
        """  const areaPts = (s, idx, floorY=248) => { const a = splitPts(s); if (!a.length) return ''; const i = clamp(idx, 0, a.length - 1); const p = a.slice(0, i + 1); const f = String(p[0]).split(','); const l = String(p[p.length - 1]).split(','); return `${Number(f[0]).toFixed(1)},${floorY} ${p.join(' ')} ${Number(l[0]).toFixed(1)},${floorY}`; };
  const adjustCompareY = (displayY, baseY, minY, maxY) => {
    if (!Number.isFinite(displayY) || !Number.isFinite(baseY) || !Number.isFinite(compareAlpha) || compareAlpha === 1) return displayY;
    return clamp(baseY + ((displayY - baseY) / compareAlpha), minY, maxY);
  };
  const parsePointSeries = (s) => splitPts(s).map((token) => { const p = String(token).split(','); const x = Number(p[0]); const y = Number(p[1]); return (Number.isFinite(x) && Number.isFinite(y)) ? {x, y} : null; }).filter(Boolean);
  const adjustedSlicePts = (s, base, idx, minY=16, maxY=248) => {
    const pts = parsePointSeries(s), basePts = parsePointSeries(base);
    if (!pts.length) return '';
    const last = clamp(idx, 0, pts.length - 1);
    return pts.slice(0, last + 1).map((pt, i) => {
      const basePt = basePts[i];
      const y = basePt ? adjustCompareY(pt.y, basePt.y, minY, maxY) : pt.y;
      return `${pt.x.toFixed(1)},${Number(y).toFixed(1)}`;
    }).join(' ');
  };
  const adjustedPointAt = (s, base, idx, minY=16, maxY=248) => {
    const pts = parsePointSeries(s), basePts = parsePointSeries(base);
    if (!pts.length) return null;
    const i = clamp(idx, 0, pts.length - 1);
    const pt = pts[i];
    const basePt = basePts[i];
    const y = basePt ? adjustCompareY(pt.y, basePt.y, minY, maxY) : pt.y;
    return { x: pt.x, y: Number(y.toFixed(1)) };
  };""",
    )
    out = out.replace(
        """  const seriesPoint = (series, idx, min, max, width=620, height=180, pad=14) => {
    if (!Array.isArray(series) || !series.length) return null;
    const i = clamp(idx, 0, series.length - 1);
    const numeric = Number(series[i]);
    if (!Number.isFinite(numeric)) return null;
    const denom = Math.max(series.length - 1, 1);
    const x = pad + ((width - (pad * 2)) * i / denom);
    const y = valueY(numeric, min, max, height, pad);
    return { x: Number(x.toFixed(1)), y: Number(y.toFixed(1)) };
  };""",
        """  const seriesPoint = (series, idx, min, max, width=620, height=180, pad=14) => {
    if (!Array.isArray(series) || !series.length) return null;
    const i = clamp(idx, 0, series.length - 1);
    const numeric = Number(series[i]);
    if (!Number.isFinite(numeric)) return null;
    const denom = Math.max(series.length - 1, 1);
    const x = pad + ((width - (pad * 2)) * i / denom);
    const y = valueY(numeric, min, max, height, pad);
    return { x: Number(x.toFixed(1)), y: Number(y.toFixed(1)) };
  };
  const adjustedSeriesPoints = (series, baseSeries, idx, min, max, width=620, height=180, pad=14) => {
    if (!Array.isArray(series) || !series.length) return '';
    const last = clamp(idx, 0, series.length - 1);
    const denom = Math.max(series.length - 1, 1);
    const points = [];
    for (let i = 0; i <= last; i += 1) {
      const numeric = Number(series[i]);
      if (!Number.isFinite(numeric)) continue;
      const x = pad + ((width - (pad * 2)) * i / denom);
      const rawY = valueY(numeric, min, max, height, pad);
      const baseNumeric = Array.isArray(baseSeries) ? Number(baseSeries[i]) : NaN;
      const baseY = Number.isFinite(baseNumeric) ? valueY(baseNumeric, min, max, height, pad) : null;
      const y = baseY == null ? rawY : adjustCompareY(rawY, baseY, pad, height - pad);
      points.push(`${x.toFixed(1)},${Number(y).toFixed(1)}`);
    }
    return points.join(' ');
  };
  const adjustedSeriesPoint = (series, baseSeries, idx, min, max, width=620, height=180, pad=14) => {
    if (!Array.isArray(series) || !series.length) return null;
    const i = clamp(idx, 0, series.length - 1);
    const numeric = Number(series[i]);
    if (!Number.isFinite(numeric)) return null;
    const denom = Math.max(series.length - 1, 1);
    const x = pad + ((width - (pad * 2)) * i / denom);
    const rawY = valueY(numeric, min, max, height, pad);
    const baseNumeric = Array.isArray(baseSeries) ? Number(baseSeries[i]) : NaN;
    const baseY = Number.isFinite(baseNumeric) ? valueY(baseNumeric, min, max, height, pad) : null;
    const y = baseY == null ? rawY : adjustCompareY(rawY, baseY, pad, height - pad);
    return { x: Number(x.toFixed(1)), y: Number(y.toFixed(1)) };
  };""",
    )
    out = out.replace(
        """    const dEq = pointAt(selected.chart && selected.chart.equal_weight_points, state.frame);
    const dBh = pointAt(selected.chart && selected.chart.buy_hold_points, state.frame);
""",
        "",
    )
    out = out.replace(
        """    const testEq = ((run.benchmarks || {}).highlights || {}).test_equal_weight || {};
    const testBh = ((run.benchmarks || {}).highlights || {}).test_buy_hold || {};
""",
        "",
    )
    out = out.replace(
        """    const currentVsCompare = deltaVal(f.total_return_pct, g.total_return_pct);
    const finalVsCompare = deltaVal((selected.summary || {}).final_return_pct, (compare.summary || {}).final_return_pct);
    const finalValueGap = deltaVal((selected.summary || {}).final_value, (compare.summary || {}).final_value);
    const dSel = pointAt(selected.chart && selected.chart.model_points, state.frame);
    const dCmp = adjustedPointAt(compare.chart && compare.chart.model_points, selected.chart && selected.chart.model_points, state.frame);""",
        """    const ddqCompare = ddq();
    const currentVsCompare = deltaVal(f.total_return_pct, g.total_return_pct);
    const finalVsCompare = deltaVal((selected.summary || {}).final_return_pct, (compare.summary || {}).final_return_pct);
    const finalValueGap = deltaVal((selected.summary || {}).final_value, (compare.summary || {}).final_value);
    const finalVsDdq = deltaVal((selected.summary || {}).final_return_pct, (ddqCompare && ddqCompare.summary && ddqCompare.summary.final_return_pct));
    const dSel = pointAt(selected.chart && selected.chart.model_points, state.frame);
    const dCmp = adjustedPointAt(compare.chart && compare.chart.model_points, selected.chart && selected.chart.model_points, state.frame);
    const dDdq = adjustedPointAt(ddqCompare && ddqCompare.chart && ddqCompare.chart.model_points, selected.chart && selected.chart.model_points, state.frame);
    const selectedRisk = ((selected.summary || {}).risk_summary || {});
    const ddqRisk = ((ddqCompare && ddqCompare.summary && ddqCompare.summary.risk_summary) || {});
    const drawdownRange = seriesRange([
      selected.chart && selected.chart.drawdown_series,
      compare.chart && compare.chart.drawdown_series,
      ddqCompare && ddqCompare.chart && ddqCompare.chart.drawdown_series,
    ], -20, 0);
    const qualityRange = seriesRange([
      selected.chart && selected.chart.rolling_sharpe_series,
      selected.chart && selected.chart.rolling_sortino_series,
      compare.chart && compare.chart.rolling_sharpe_series,
      ddqCompare && ddqCompare.chart && ddqCompare.chart.rolling_sharpe_series,
      ddqCompare && ddqCompare.chart && ddqCompare.chart.rolling_sortino_series,
    ], -2, 2);
    const ddZeroY = valueY(0, drawdownRange.min, drawdownRange.max, 180, 14);
    const qualityZeroY = valueY(0, qualityRange.min, qualityRange.max, 180, 14);
    const ddSel = seriesPoint(selected.chart && selected.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);
    const ddCmp = adjustedSeriesPoint(compare.chart && compare.chart.drawdown_series, selected.chart && selected.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);
    const ddDdq = adjustedSeriesPoint(ddqCompare && ddqCompare.chart && ddqCompare.chart.drawdown_series, selected.chart && selected.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);
    const qSel = seriesPoint(selected.chart && selected.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);
    const qSort = seriesPoint(selected.chart && selected.chart.rolling_sortino_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);
    const qCmp = adjustedSeriesPoint(compare.chart && compare.chart.rolling_sharpe_series, selected.chart && selected.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);
    const qDdq = adjustedSeriesPoint(ddqCompare && ddqCompare.chart && ddqCompare.chart.rolling_sharpe_series, selected.chart && selected.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);""",
    )
    out = out.replace(
        """    const dCmp = pointAt(compare.chart && compare.chart.model_points, state.frame);
    const dDdq = pointAt(ddqCompare && ddqCompare.chart && ddqCompare.chart.model_points, state.frame);""",
        """    const dCmp = adjustedPointAt(compare.chart && compare.chart.model_points, selected.chart && selected.chart.model_points, state.frame);
    const dDdq = adjustedPointAt(ddqCompare && ddqCompare.chart && ddqCompare.chart.model_points, selected.chart && selected.chart.model_points, state.frame);""",
    )
    out = out.replace(
        """    const ddCmp = seriesPoint(compare.chart && compare.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);
    const ddDdq = seriesPoint(ddqCompare && ddqCompare.chart && ddqCompare.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);""",
        """    const ddCmp = adjustedSeriesPoint(compare.chart && compare.chart.drawdown_series, selected.chart && selected.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);
    const ddDdq = adjustedSeriesPoint(ddqCompare && ddqCompare.chart && ddqCompare.chart.drawdown_series, selected.chart && selected.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);""",
    )
    out = out.replace(
        """    const qCmp = seriesPoint(compare.chart && compare.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);
    const qDdq = seriesPoint(ddqCompare && ddqCompare.chart && ddqCompare.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);""",
        """    const qCmp = adjustedSeriesPoint(compare.chart && compare.chart.rolling_sharpe_series, selected.chart && selected.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);
    const qDdq = adjustedSeriesPoint(ddqCompare && ddqCompare.chart && ddqCompare.chart.rolling_sharpe_series, selected.chart && selected.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);""",
    )
    out = out.replace(
        """${ddqCompare ? `<polyline points="${esc(slicePts(ddqCompare.chart && ddqCompare.chart.model_points, state.frame))}" fill="none" stroke="#e2e8f0" stroke-width="2.6" stroke-dasharray="4 6"></polyline>` : ''}<polyline points="${esc(adjustedSlicePts(compare.chart && compare.chart.model_points, selected.chart && selected.chart.model_points, state.frame))}" fill="none" stroke="#ff7a88" stroke-width="3" stroke-dasharray="8 6"></polyline>""",
        """${ddqCompare ? `<polyline points="${esc(adjustedSlicePts(ddqCompare.chart && ddqCompare.chart.model_points, selected.chart && selected.chart.model_points, state.frame))}" fill="none" stroke="#e2e8f0" stroke-width="2.6" stroke-dasharray="4 6"></polyline>` : ''}<polyline points="${esc(adjustedSlicePts(compare.chart && compare.chart.model_points, selected.chart && selected.chart.model_points, state.frame))}" fill="none" stroke="#ff7a88" stroke-width="3" stroke-dasharray="8 6"></polyline>""",
    )
    out = out.replace(
        """${ddqCompare ? `<polyline points="${esc(seriesPoints(ddqCompare.chart && ddqCompare.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14))}" fill="none" stroke="#e2e8f0" stroke-width="2.4" stroke-dasharray="4 6"></polyline>` : ''}<polyline points="${esc(seriesPoints(compare.chart && compare.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14))}" fill="none" stroke="#ff7a88" stroke-width="2.8" stroke-dasharray="8 6"></polyline>""",
        """${ddqCompare ? `<polyline points="${esc(adjustedSeriesPoints(ddqCompare.chart && ddqCompare.chart.drawdown_series, selected.chart && selected.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14))}" fill="none" stroke="#e2e8f0" stroke-width="2.4" stroke-dasharray="4 6"></polyline>` : ''}<polyline points="${esc(adjustedSeriesPoints(compare.chart && compare.chart.drawdown_series, selected.chart && selected.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14))}" fill="none" stroke="#ff7a88" stroke-width="2.8" stroke-dasharray="8 6"></polyline>""",
    )
    out = out.replace(
        """${ddqCompare ? `<polyline points="${esc(seriesPoints(ddqCompare.chart && ddqCompare.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14))}" fill="none" stroke="#e2e8f0" stroke-width="2.4" stroke-dasharray="4 6"></polyline>` : ''}<polyline points="${esc(seriesPoints(compare.chart && compare.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14))}" fill="none" stroke="#ff7a88" stroke-width="2.8" stroke-dasharray="8 6"></polyline>""",
        """${ddqCompare ? `<polyline points="${esc(adjustedSeriesPoints(ddqCompare.chart && ddqCompare.chart.rolling_sharpe_series, selected.chart && selected.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14))}" fill="none" stroke="#e2e8f0" stroke-width="2.4" stroke-dasharray="4 6"></polyline>` : ''}<polyline points="${esc(adjustedSeriesPoints(compare.chart && compare.chart.rolling_sharpe_series, selected.chart && selected.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14))}" fill="none" stroke="#ff7a88" stroke-width="2.8" stroke-dasharray="8 6"></polyline>""",
    )
    out = out.replace("compareAlpha === 1", "compareAlpha <= 0")
    out = out.replace(
        "return clamp(baseY + ((displayY - baseY) / compareAlpha), minY, maxY);",
        "return clamp(baseY + ((displayY - baseY) * (1 + compareAlpha)), minY, maxY);",
    )
    out = out.replace('stroke="#20d6a4" stroke-width="4"', 'stroke="#20d6a4" stroke-width="2"')
    out = out.replace('stroke="#20d6a4" stroke-width="3.4"', 'stroke="#20d6a4" stroke-width="1.7"')
    out = out.replace('stroke="#ffd166" stroke-width="2.6"', 'stroke="#ffd166" stroke-width="1.3"')
    out = out.replace(
        'stroke="#e2e8f0" stroke-width="2.6" stroke-dasharray="4 6"',
        'stroke="#e2e8f0" stroke-width="1.3" stroke-dasharray="4 6"',
    )
    out = out.replace(
        'stroke="#e2e8f0" stroke-width="2.2" stroke-dasharray="4 6"',
        'stroke="#e2e8f0" stroke-width="1.1" stroke-dasharray="4 6"',
    )
    out = out.replace(
        'stroke="#ff7a88" stroke-width="3" stroke-dasharray="8 6"',
        'stroke="#ff7a88" stroke-width="1.5" stroke-dasharray="8 6"',
    )
    out = out.replace(
        'stroke="#ff7a88" stroke-width="2.5" stroke-dasharray="8 6"',
        'stroke="#ff7a88" stroke-width="1.25" stroke-dasharray="8 6"',
    )
    out = out.replace(
        'stroke="#e2e8f0" stroke-width="2.4" stroke-dasharray="4 6"',
        'stroke="#e2e8f0" stroke-width="1.2" stroke-dasharray="4 6"',
    )
    out = out.replace(
        'stroke="#e2e8f0" stroke-width="2.1" stroke-dasharray="4 6"',
        'stroke="#e2e8f0" stroke-width="1.05" stroke-dasharray="4 6"',
    )
    out = out.replace(
        'stroke="#ff7a88" stroke-width="2.8" stroke-dasharray="8 6"',
        'stroke="#ff7a88" stroke-width="1.4" stroke-dasharray="8 6"',
    )
    out = out.replace(
        'stroke="#ff7a88" stroke-width="2.4" stroke-dasharray="8 6"',
        'stroke="#ff7a88" stroke-width="1.2" stroke-dasharray="8 6"',
    )
    point_helper_block = """  const adjustCompareY = (displayY, baseY, minY, maxY) => {
    if (!Number.isFinite(displayY) || !Number.isFinite(baseY) || !Number.isFinite(compareAlpha) || compareAlpha <= 0) return displayY;
    return clamp(baseY + ((displayY - baseY) * (1 + compareAlpha)), minY, maxY);
  };
  const parsePointSeries = (s) => splitPts(s).map((token) => { const p = String(token).split(','); const x = Number(p[0]); const y = Number(p[1]); return (Number.isFinite(x) && Number.isFinite(y)) ? {x, y} : null; }).filter(Boolean);
  const adjustedSlicePts = (s, base, idx, minY=16, maxY=248) => {
    const pts = parsePointSeries(s), basePts = parsePointSeries(base);
    if (!pts.length) return '';
    const last = clamp(idx, 0, pts.length - 1);
    return pts.slice(0, last + 1).map((pt, i) => {
      const basePt = basePts[i];
      const y = basePt ? adjustCompareY(pt.y, basePt.y, minY, maxY) : pt.y;
      return `${pt.x.toFixed(1)},${Number(y).toFixed(1)}`;
    }).join(' ');
  };
  const adjustedPointAt = (s, base, idx, minY=16, maxY=248) => {
    const pts = parsePointSeries(s), basePts = parsePointSeries(base);
    if (!pts.length) return null;
    const i = clamp(idx, 0, pts.length - 1);
    const pt = pts[i];
    const basePt = basePts[i];
    const y = basePt ? adjustCompareY(pt.y, basePt.y, minY, maxY) : pt.y;
    return { x: pt.x, y: Number(y.toFixed(1)) };
  };"""
    while f"{point_helper_block}\n{point_helper_block}" in out:
        out = out.replace(f"{point_helper_block}\n{point_helper_block}", point_helper_block)
    out = re.sub(
        r"(from src\.constants import CHART_COMPARE_ALPHA\n){2,}",
        "from src.constants import CHART_COMPARE_ALPHA\n",
        out,
    )
    compare_alpha_block = """  const compareAlpha = (() => {
    const raw = Number((((data || {}).uiConfig || {}).chartCompareAlpha));
    return Number.isFinite(raw) && raw > 0 ? raw : 1;
  })();"""
    while f"{compare_alpha_block}\n{compare_alpha_block}" in out:
        out = out.replace(f"{compare_alpha_block}\n{compare_alpha_block}", compare_alpha_block)
    return out


def dashboard_text() -> str:
    return strip_public_benchmarks("".join(DASHBOARD_PARTS))


def patch_build_replay(text: str) -> str:
    updated = text
    updated = updated.replace(
        'REPLAY_MONTHS = to_int(z.input("replay_months", "4"), 4)',
        'REPLAY_MONTHS = REPLAY_RECENT_MONTHS_DEFAULT',
    )
    updated = updated.replace(
        f'REPLAY_MONTHS = to_int(z.input("replay_months", "{REPLAY_RECENT_MONTHS_DEFAULT}"), {REPLAY_RECENT_MONTHS_DEFAULT})',
        'REPLAY_MONTHS = REPLAY_RECENT_MONTHS_DEFAULT',
    )
    updated = updated.replace(
        'REPLAY_WARMUP_MONTHS = to_int(z.input("replay_warmup_months", "4"), 4)',
        'REPLAY_WARMUP_MONTHS = REPLAY_WARMUP_MONTHS_DEFAULT',
    )
    updated = updated.replace(
        f'REPLAY_WARMUP_MONTHS = to_int(z.input("replay_warmup_months", "{REPLAY_WARMUP_MONTHS_DEFAULT}"), {REPLAY_WARMUP_MONTHS_DEFAULT})',
        'REPLAY_WARMUP_MONTHS = REPLAY_WARMUP_MONTHS_DEFAULT',
    )
    updated = updated.replace(
        'CHECKPOINT_SAMPLES = to_int(z.input("checkpoint_samples", "6"), 4)',
        'CHECKPOINT_SAMPLES = REPLAY_CHECKPOINT_SAMPLES_DEFAULT',
    )
    updated = updated.replace(
        f'CHECKPOINT_SAMPLES = to_int(z.input("checkpoint_samples", "{REPLAY_CHECKPOINT_SAMPLES_DEFAULT}"), {REPLAY_CHECKPOINT_SAMPLES_DEFAULT})',
        'CHECKPOINT_SAMPLES = REPLAY_CHECKPOINT_SAMPLES_DEFAULT',
    )
    updated = updated.replace(
        'REPLAY_RUN_LIMIT = to_int(z.input("replay_run_limit", "1"), 1)',
        'REPLAY_RUN_LIMIT = REPLAY_RUN_LIMIT_DEFAULT',
    )
    updated = updated.replace(
        f'REPLAY_RUN_LIMIT = to_int(z.input("replay_run_limit", "{REPLAY_RUN_LIMIT_DEFAULT}"), {REPLAY_RUN_LIMIT_DEFAULT})',
        'REPLAY_RUN_LIMIT = REPLAY_RUN_LIMIT_DEFAULT',
    )
    marker = f'VNSTOCK_SOURCE = z.input("vnstock_source", "{REPLAY_VNSTOCK_SOURCE_DEFAULT}").strip() or "{REPLAY_VNSTOCK_SOURCE_DEFAULT}"'
    updated = updated.replace(marker, 'VNSTOCK_SOURCE = REPLAY_VNSTOCK_SOURCE_DEFAULT')
    if marker in updated and "REPLAY_END_DATE" not in updated:
        updated = updated.replace(
            marker,
            'VNSTOCK_SOURCE = REPLAY_VNSTOCK_SOURCE_DEFAULT' + '\nREPLAY_END_DATE = REPLAY_END_DATE_DEFAULT',
        )
    updated = updated.replace(
        f'REPLAY_END_DATE = str(z.input("replay_end_date", "{REPLAY_END_DATE_DEFAULT}") or "{REPLAY_END_DATE_DEFAULT}").strip() or "{REPLAY_END_DATE_DEFAULT}"',
        'REPLAY_END_DATE = REPLAY_END_DATE_DEFAULT',
    )
    if 'end_date=REPLAY_END_DATE,' not in updated:
        updated = updated.replace(
            '                vnstock_source=VNSTOCK_SOURCE,\n',
            '                vnstock_source=VNSTOCK_SOURCE,\n                end_date=REPLAY_END_DATE,\n',
        )
    return updated


def patch_note(path: Path) -> None:
    note = json.loads(path.read_text(encoding="utf-8-sig"))
    paragraphs = []
    for p in (note.get("paragraphs") or []):
        title = str(p.get("title") or "").strip()
        text = str(p.get("text") or "").strip()
        if not title and text in {"", "%spark.pyspark", "%spark"}:
            continue
        paragraphs.append(p)
    if not paragraphs:
        raise RuntimeError(f"Notebook {path} không có paragraph.")

    if all(not str((p.get("title") or "")).strip() for p in paragraphs[:5]) and len(paragraphs) >= 5:
        expected = ["%spark.pyspark", "%spark.pyspark", "%spark.pyspark", "%spark.sql", "%spark.pyspark"]
        actual = [str((p.get("text") or "")).splitlines()[0].strip() if str((p.get("text") or "")).strip() else "" for p in paragraphs[:5]]
        if actual == expected:
            for idx, title in enumerate(PARAGRAPH_ORDER):
                paragraphs[idx]["title"] = title

    title_map = {str(p.get("title") or ""): p for p in paragraphs}
    replay_paragraph = title_map.get("Build Replay Cache")
    bind_paragraph = title_map.get("Bind Dashboard Data")
    html_paragraph = title_map.get("HTML Dashboard")
    legacy_paragraph = title_map.get("Legacy Bind (Ignore)")

    if replay_paragraph is None or html_paragraph is None:
        raise RuntimeError(f"Notebook {path} thiếu paragraph cần thiết.")

    replay_paragraph["text"] = patch_build_replay(replay_paragraph["text"])

    if legacy_paragraph is not None:
        legacy_paragraph["title"] = "Legacy Bind (Ignore)"
        legacy_paragraph["text"] = '%spark\\nprintln("{\\"note\\":\\"Legacy bind paragraph. Có thể bỏ qua. Dashboard hiện dùng HTML Dashboard.\\"}")\\n'

    if bind_paragraph is None:
        bind_paragraph = {
            "title": "Bind Dashboard Data",
            "text": BIND_PARAGRAPH_TEXT,
            "dateCreated": NOTE_TIMESTAMP,
            "dateUpdated": NOTE_TIMESTAMP,
            "config": {},
            "settings": {},
            "apps": [],
            "jobName": "",
            "results": {"code": "SUCCESS", "msg": [{"type": "TEXT", "data": "Chạy paragraph này để gom cache trước khi mở dashboard."}]},
            "status": "READY",
            "progressUpdateIntervalMs": 500,
            "focus": False,
            "$$hashKey": "object:bind",
        }
        try:
            insert_idx = paragraphs.index(replay_paragraph) + 1
        except ValueError:
            insert_idx = 2
        paragraphs.insert(insert_idx, bind_paragraph)
    else:
        bind_paragraph["title"] = "Bind Dashboard Data"
        bind_paragraph["text"] = BIND_PARAGRAPH_TEXT
        bind_paragraph["dateCreated"] = bind_paragraph.get("dateCreated") or NOTE_TIMESTAMP
        bind_paragraph["dateUpdated"] = NOTE_TIMESTAMP
        bind_paragraph["status"] = "READY"
        bind_paragraph["results"] = {"code": "SUCCESS", "msg": [{"type": "TEXT", "data": "Chạy paragraph này để gom cache trước khi mở dashboard."}]}

    for paragraph in paragraphs:
        if "dateCreated" in paragraph and not paragraph.get("dateCreated"):
            paragraph["dateCreated"] = NOTE_TIMESTAMP
        if "dateUpdated" in paragraph and not paragraph.get("dateUpdated"):
            paragraph["dateUpdated"] = NOTE_TIMESTAMP

    html_paragraph["title"] = "HTML Dashboard"
    html_paragraph["text"] = dashboard_text()
    html_paragraph["status"] = "READY"
    html_paragraph["results"] = {"code": "SUCCESS", "msg": [{"type": "TEXT", "data": "Chạy lại paragraph này để render dashboard checkpoint-first."}]}
    ordered = []
    leftovers = []
    for title in PARAGRAPH_ORDER:
        found = next((p for p in paragraphs if str(p.get("title") or "") == title), None)
        if found is not None:
            ordered.append(found)
    for p in paragraphs:
        if p not in ordered:
            leftovers.append(p)
    note["paragraphs"] = ordered + leftovers
    path.write_text(json.dumps(note, ensure_ascii=False, separators=(",", ":")), encoding="utf-8-sig")


def iter_target_notes() -> list[Path]:
    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in [LOCAL_NOTE, *sorted(RUNTIME_NOTE_DIR.glob(RUNTIME_NOTE_GLOB))]:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(candidate)
    return ordered


def main() -> None:
    for target in iter_target_notes():
        patch_note(target)
        print(f"patched {target}")


if __name__ == "__main__":
    main()
