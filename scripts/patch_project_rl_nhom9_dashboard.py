from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_NOTE = ROOT / "notebooks" / "project_RL_nhom_09 .zpln"
RUNTIME_NOTE = Path(r"D:\Programs\zeppelin-0.12.0-bin-all\docker-data\notebook\project_RL_nhom_09 _2MNYSRNK4.zpln")
HTML_SNAPSHOT = ROOT / ".zeppelin_cache" / "current_dashboard.html"


def replace_between(source: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = source.find(start_marker)
    if start < 0:
        raise ValueError(f"Không tìm thấy start marker: {start_marker[:80]!r}")
    end = source.find(end_marker, start)
    if end < 0:
        raise ValueError(f"Không tìm thấy end marker: {end_marker[:80]!r}")
    current = source[start:end]
    if current == replacement:
        return source
    return source[:start] + replacement + source[end:]


def patch_html_paragraph_text(text: str) -> str:
    replacements = [
        (
            "import json\r\nfrom pathlib import Path\r\n\r\n\r\ndef load_json(path):",
            "import json\r\n\r\nfrom scripts.dashboard_paths import DashboardProjectPaths\r\n\r\n\r\ndef load_json(path):",
        ),
        (
            "project_root = str(z.input(\"project_root\", \"/workspace/project\") or \"/workspace/project\").strip() or \"/workspace/project\"\r\ncache_dir = Path(project_root).expanduser().resolve() / \".zeppelin_cache\"\r\ntrain_cache = load_json(cache_dir / \"dashboard_train_payload.json\") or load_json(cache_dir / \"dashboard_payload.json\")\r\nreplay_cache = load_json(cache_dir / \"dashboard_replay_payload.json\")\r\nmerged_cache = load_json(cache_dir / \"dashboard_payload.json\") or {}",
            "project_root = str(z.input(\"project_root\", \"/workspace/project\") or \"/workspace/project\").strip() or \"/workspace/project\"\r\npaths = DashboardProjectPaths.from_project_root(project_root)\r\ntrain_cache = load_json(paths.train_cache_path) or load_json(paths.dashboard_cache_path)\r\nreplay_cache = load_json(paths.replay_cache_path)\r\nmerged_cache = load_json(paths.dashboard_cache_path) or {}",
        ),
        (
            """payload = dict(train_cache)
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
""",
            """best_run = next(
    (item for item in (train_cache.get("runs") or []) if item.get("run_id") == train_cache.get("defaultRunId")),
    None,
)
if best_run is None:
    best_run = ((train_cache.get("runs") or [None])[0] or {})

payload = {
    "defaultRunId": train_cache.get("defaultRunId"),
    "bestRun": best_run,
    "liveMarket": train_cache.get("liveMarket") or {},
}
if isinstance(replay_cache, dict) and (replay_cache.get("runs") or replay_cache.get("status") == "ready"):
    payload["checkpointReplay"] = {
        "status": replay_cache.get("status") or "ready",
        "message": replay_cache.get("message") or "Đã dựng payload replay checkpoint theo ngày.",
        "warnings": replay_cache.get("warnings") or [],
        "runs": replay_cache.get("runs") or [],
        "defaultRunId": replay_cache.get("defaultRunId"),
    }
elif merged_cache.get("checkpointReplay"):
    cp_payload = merged_cache.get("checkpointReplay") or {}
    payload["checkpointReplay"] = {
        "status": cp_payload.get("status") or "ready",
        "message": cp_payload.get("message") or "Đã dựng payload replay checkpoint theo ngày.",
        "warnings": cp_payload.get("warnings") or [],
        "runs": cp_payload.get("runs") or [],
        "defaultRunId": cp_payload.get("defaultRunId"),
    }
else:
    payload["checkpointReplay"] = {"status": "pending", "message": "Chưa dựng replay cache.", "warnings": [], "runs": []}
""",
        ),
        (
            "  const splitPts = (s) => String(s || '').trim().split(/\\\\s+/).filter(Boolean);",
            "  const splitPts = (s) => String(s || '').trim().split(/\\s+/).filter(Boolean);",
        ),
        (
            "  const pointAt = (s, idx) => { const a = splitPts(s); if (!a.length) return null; const p = String(a[clamp(idx, 0, a.length - 1)]).split(','); return {x:Number(p[0]), y:Number(p[1])}; };",
            "  const pointAt = (s, idx) => { const a = splitPts(s); if (!a.length) return null; const p = String(a[clamp(idx, 0, a.length - 1)]).split(','); const x = Number(p[0]); const y = Number(p[1]); return (Number.isFinite(x) && Number.isFinite(y)) ? {x, y} : null; };",
        ),
        (
            "  const runs = () => Array.isArray(data.runs) ? data.runs : [];",
            "  const runs = () => Array.isArray(data.runs) ? data.runs : [];",
        ),
        (
            "  const bestRun = () => runs().find((x) => x && x.run_id === data.defaultRunId) || runs()[0] || {summary:{}, benchmarks:{highlights:{}, rows:[]}};",
            "  const bestRun = () => data.bestRun || runs().find((x) => x && x.run_id === data.defaultRunId) || runs()[0] || {summary:{}, benchmarks:{highlights:{}, rows:[]}};",
        ),
        (
            "  const cmp = () => { ensure(); return cps()[state.cmp] || cp(); };\r\n  const frames = (item) => Array.isArray(item && item.frames) ? item.frames : [];",
            "  const cmp = () => { ensure(); return cps()[state.cmp] || cp(); };\r\n  const ddq = () => cp().ddq_compare || null;\r\n  const frames = (item) => Array.isArray(item && item.frames) ? item.frames : [];",
        ),
        (
            "            <p class=\"lead\">Trọng tâm là checkpoint của từng policy, không còn bắt người xem quan tâm tới run selector. Chọn checkpoint, chọn policy đối chiếu, rồi để dashboard tự chạy theo ngày để thấy đường vốn, benchmark và phân bổ tài sản thay đổi như thế nào trên dữ liệu replay gần nhất.</p>",
            "            <p class=\"lead\">Trọng tâm là checkpoint PPO đang xem và một đường DDQ tốt nhất gán cứng để đối chiếu thực tế. Chọn checkpoint PPO, chọn policy PPO đối chiếu nếu cần, rồi để dashboard tự chạy theo ngày để thấy đường vốn, benchmark và phân bổ tài sản thay đổi như thế nào trên dữ liệu replay gần nhất.</p>",
        ),
        (
            "    const testEq = ((run.benchmarks || {}).highlights || {}).test_equal_weight || {};\r\n    const testBh = ((run.benchmarks || {}).highlights || {}).test_buy_hold || {};\r\n    const currentVsCompare = deltaVal(f.total_return_pct, g.total_return_pct);\r\n    const finalVsCompare = deltaVal((selected.summary || {}).final_return_pct, (compare.summary || {}).final_return_pct);\r\n    const finalValueGap = deltaVal((selected.summary || {}).final_value, (compare.summary || {}).final_value);\r\n    const dSel = pointAt(selected.chart && selected.chart.model_points, state.frame);\r\n    const dCmp = pointAt(compare.chart && compare.chart.model_points, state.frame);\r\n    const dEq = pointAt(selected.chart && selected.chart.equal_weight_points, state.frame);\r\n    const dBh = pointAt(selected.chart && selected.chart.buy_hold_points, state.frame);",
            "    const testEq = ((run.benchmarks || {}).highlights || {}).test_equal_weight || {};\r\n    const testBh = ((run.benchmarks || {}).highlights || {}).test_buy_hold || {};\r\n    const ddqCompare = ddq();\r\n    const currentVsCompare = deltaVal(f.total_return_pct, g.total_return_pct);\r\n    const finalVsCompare = deltaVal((selected.summary || {}).final_return_pct, (compare.summary || {}).final_return_pct);\r\n    const finalValueGap = deltaVal((selected.summary || {}).final_value, (compare.summary || {}).final_value);\r\n    const finalVsDdq = deltaVal((selected.summary || {}).final_return_pct, (ddqCompare && ddqCompare.summary && ddqCompare.summary.final_return_pct));\r\n    const finalValueVsDdq = deltaVal((selected.summary || {}).final_value, (ddqCompare && ddqCompare.summary && ddqCompare.summary.final_value));\r\n    const dSel = pointAt(selected.chart && selected.chart.model_points, state.frame);\r\n    const dCmp = pointAt(compare.chart && compare.chart.model_points, state.frame);\r\n    const dEq = pointAt(selected.chart && selected.chart.equal_weight_points, state.frame);\r\n    const dBh = pointAt(selected.chart && selected.chart.buy_hold_points, state.frame);\r\n    const dDdq = pointAt(ddqCompare && ddqCompare.chart && ddqCompare.chart.model_points, state.frame);",
        ),
        (
            "<div style=\"display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;margin-bottom:12px\"><div><div class=\"kicker\">Soi policy theo ngày</div><h3>${esc(f.headline || 'Đường vốn và benchmark')}</h3><p class=\"foot\">Ngày quyết định ${esc(f.decision_date || 'N/A')} · ngày đóng phiên ${esc(f.date || 'N/A')}</p></div><div class=\"legend\"><span><i style=\"background:#20d6a4\"></i>Checkpoint đang xem</span><span><i style=\"background:#ff7a88\"></i>Policy so sánh</span><span><i style=\"background:#ffd166\"></i>Danh mục đều</span><span><i style=\"background:#73b6ff\"></i>Mua và giữ</span></div></div>",
            "<div style=\"display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;margin-bottom:12px\"><div><div class=\"kicker\">Soi policy theo ngày</div><h3>${esc(f.headline || 'Đường vốn và benchmark')}</h3><p class=\"foot\">Ngày quyết định ${esc(f.decision_date || 'N/A')} · ngày đóng phiên ${esc(f.date || 'N/A')}</p></div><div class=\"legend\"><span><i style=\"background:#20d6a4\"></i>Checkpoint đang xem</span><span><i style=\"background:#ff7a88\"></i>Policy so sánh</span>${ddqCompare ? '<span><i style=\"background:#e2e8f0\"></i>DDQ tốt nhất</span>' : ''}<span><i style=\"background:#ffd166\"></i>Danh mục đều</span><span><i style=\"background:#73b6ff\"></i>Mua và giữ</span></div></div>",
        ),
        (
            "<div class=\"chartShell\"><div class=\"chartBox\"><svg viewBox=\"0 0 620 280\" preserveAspectRatio=\"none\"><defs><linearGradient id=\"rlSelectedArea\" x1=\"0\" x2=\"0\" y1=\"0\" y2=\"1\"><stop offset=\"0%\" stop-color=\"#20d6a4\" stop-opacity=\"0.30\"></stop><stop offset=\"100%\" stop-color=\"#20d6a4\" stop-opacity=\"0.02\"></stop></linearGradient></defs><polygon points=\"${esc(areaPts(selected.chart && selected.chart.model_points, state.frame))}\" fill=\"url(#rlSelectedArea)\"></polygon><polyline points=\"${esc(slicePts(selected.chart && selected.chart.buy_hold_points, state.frame))}\" fill=\"none\" stroke=\"#73b6ff\" stroke-width=\"3\"></polyline><polyline points=\"${esc(slicePts(selected.chart && selected.chart.equal_weight_points, state.frame))}\" fill=\"none\" stroke=\"#ffd166\" stroke-width=\"3\"></polyline><polyline points=\"${esc(slicePts(compare.chart && compare.chart.model_points, state.frame))}\" fill=\"none\" stroke=\"#ff7a88\" stroke-width=\"3\" stroke-dasharray=\"8 6\"></polyline><polyline points=\"${esc(slicePts(selected.chart && selected.chart.model_points, state.frame))}\" fill=\"none\" stroke=\"#20d6a4\" stroke-width=\"4\"></polyline><line x1=\"${esc(markerX())}\" x2=\"${esc(markerX())}\" y1=\"16\" y2=\"248\" stroke=\"#fca5a5\" stroke-dasharray=\"6 6\"></line>${dSel ? `<circle cx=\"${dSel.x}\" cy=\"${dSel.y}\" r=\"5.8\" fill=\"#20d6a4\" stroke=\"#06111d\" stroke-width=\"2\"></circle>` : ''}${dCmp ? `<circle cx=\"${dCmp.x}\" cy=\"${dCmp.y}\" r=\"5\" fill=\"#ff7a88\" stroke=\"#06111d\" stroke-width=\"2\"></circle>` : ''}${dEq ? `<circle cx=\"${dEq.x}\" cy=\"${dEq.y}\" r=\"4.4\" fill=\"#ffd166\" stroke=\"#06111d\" stroke-width=\"2\"></circle>` : ''}${dBh ? `<circle cx=\"${dBh.x}\" cy=\"${dBh.y}\" r=\"4.4\" fill=\"#73b6ff\" stroke=\"#06111d\" stroke-width=\"2\"></circle>` : ''}</svg></div><div class=\"caption\"><span>Chart chỉ hiển thị đến ngày đang phát để mô phỏng đúng tiến trình.</span><span>Khung ${state.frame + 1}/${Math.max(frames(selected).length, 1)}</span></div></div>",
            "<div class=\"chartShell\"><div class=\"chartBox\"><svg viewBox=\"0 0 620 280\" preserveAspectRatio=\"none\"><defs><linearGradient id=\"rlSelectedArea\" x1=\"0\" x2=\"0\" y1=\"0\" y2=\"1\"><stop offset=\"0%\" stop-color=\"#20d6a4\" stop-opacity=\"0.30\"></stop><stop offset=\"100%\" stop-color=\"#20d6a4\" stop-opacity=\"0.02\"></stop></linearGradient></defs><polygon points=\"${esc(areaPts(selected.chart && selected.chart.model_points, state.frame))}\" fill=\"url(#rlSelectedArea)\"></polygon><polyline points=\"${esc(slicePts(selected.chart && selected.chart.buy_hold_points, state.frame))}\" fill=\"none\" stroke=\"#73b6ff\" stroke-width=\"3\"></polyline><polyline points=\"${esc(slicePts(selected.chart && selected.chart.equal_weight_points, state.frame))}\" fill=\"none\" stroke=\"#ffd166\" stroke-width=\"3\"></polyline>${ddqCompare ? `<polyline points=\"${esc(slicePts(ddqCompare.chart && ddqCompare.chart.model_points, state.frame))}\" fill=\"none\" stroke=\"#e2e8f0\" stroke-width=\"2.6\" stroke-dasharray=\"4 6\"></polyline>` : ''}<polyline points=\"${esc(slicePts(compare.chart && compare.chart.model_points, state.frame))}\" fill=\"none\" stroke=\"#ff7a88\" stroke-width=\"3\" stroke-dasharray=\"8 6\"></polyline><polyline points=\"${esc(slicePts(selected.chart && selected.chart.model_points, state.frame))}\" fill=\"none\" stroke=\"#20d6a4\" stroke-width=\"4\"></polyline><line x1=\"${esc(markerX())}\" x2=\"${esc(markerX())}\" y1=\"16\" y2=\"248\" stroke=\"#fca5a5\" stroke-dasharray=\"6 6\"></line>${dSel ? `<circle cx=\"${dSel.x}\" cy=\"${dSel.y}\" r=\"5.8\" fill=\"#20d6a4\" stroke=\"#06111d\" stroke-width=\"2\"></circle>` : ''}${dCmp ? `<circle cx=\"${dCmp.x}\" cy=\"${dCmp.y}\" r=\"5\" fill=\"#ff7a88\" stroke=\"#06111d\" stroke-width=\"2\"></circle>` : ''}${dDdq ? `<circle cx=\"${dDdq.x}\" cy=\"${dDdq.y}\" r=\"4.2\" fill=\"#e2e8f0\" stroke=\"#06111d\" stroke-width=\"2\"></circle>` : ''}${dEq ? `<circle cx=\"${dEq.x}\" cy=\"${dEq.y}\" r=\"4.4\" fill=\"#ffd166\" stroke=\"#06111d\" stroke-width=\"2\"></circle>` : ''}${dBh ? `<circle cx=\"${dBh.x}\" cy=\"${dBh.y}\" r=\"4.4\" fill=\"#73b6ff\" stroke=\"#06111d\" stroke-width=\"2\"></circle>` : ''}</svg></div><div class=\"caption\"><span>Chart chỉ hiển thị đến ngày đang phát để mô phỏng đúng tiến trình.</span><span>Khung ${state.frame + 1}/${Math.max(frames(selected).length, 1)}</span></div></div>",
        ),
        (
            "<div class=\"compare\" style=\"margin-top:14px\"><div class=\"cmp\"><span>Chênh hiện tại vs policy so sánh</span><strong class=\"${cls(currentVsCompare || 0)}\">${pct(currentVsCompare, 2)}</strong></div><div class=\"cmp\"><span>Chênh cuối kỳ vs policy so sánh</span><strong class=\"${cls(finalVsCompare || 0)}\">${pct(finalVsCompare, 2)}</strong></div><div class=\"cmp\"><span>Chênh giá trị cuối kỳ</span><strong class=\"${cls(finalValueGap || 0)}\">${money(finalValueGap)}</strong></div><div class=\"cmp\"><span>Số lệnh trong ngày</span><strong>${esc(f.trade_count || 0)}</strong></div><div class=\"cmp\"><span>Checkpoint đang xem</span><strong class=\"${cls((selected.summary || {}).final_return_pct || 0)}\">${pct((selected.summary || {}).final_return_pct, 2)}</strong></div><div class=\"cmp\"><span>Policy so sánh</span><strong class=\"${cls((compare.summary || {}).final_return_pct || 0)}\">${pct((compare.summary || {}).final_return_pct, 2)}</strong></div></div>",
            "<div class=\"compare\" style=\"margin-top:14px\"><div class=\"cmp\"><span>Chênh hiện tại vs policy so sánh</span><strong class=\"${cls(currentVsCompare || 0)}\">${pct(currentVsCompare, 2)}</strong></div><div class=\"cmp\"><span>Chênh cuối kỳ vs policy so sánh</span><strong class=\"${cls(finalVsCompare || 0)}\">${pct(finalVsCompare, 2)}</strong></div><div class=\"cmp\"><span>Chênh giá trị cuối kỳ</span><strong class=\"${cls(finalValueGap || 0)}\">${money(finalValueGap)}</strong></div><div class=\"cmp\"><span>Số lệnh trong ngày</span><strong>${esc(f.trade_count || 0)}</strong></div><div class=\"cmp\"><span>Checkpoint đang xem</span><strong class=\"${cls((selected.summary || {}).final_return_pct || 0)}\">${pct((selected.summary || {}).final_return_pct, 2)}</strong></div><div class=\"cmp\"><span>Policy so sánh</span><strong class=\"${cls((compare.summary || {}).final_return_pct || 0)}\">${pct((compare.summary || {}).final_return_pct, 2)}</strong></div>${ddqCompare ? `<div class=\"cmp\"><span>DDQ tốt nhất</span><strong class=\"${cls((ddqCompare.summary || {}).final_return_pct || 0)}\">${pct((ddqCompare.summary || {}).final_return_pct, 2)}</strong></div><div class=\"cmp\"><span>Chênh cuối kỳ vs DDQ</span><strong class=\"${cls(finalVsDdq || 0)}\">${pct(finalVsDdq, 2)}</strong></div>` : ''}</div>",
        ),
    ]

    out = text.replace("\r\n", "\n")
    def apply_replace(source: str, old: str, new: str) -> str:
        old = old.replace("\r\n", "\n")
        new = new.replace("\r\n", "\n")
        if old in source:
            return source.replace(old, new)
        if new in source:
            return source
        return source

    for old, new in replacements:
        out = apply_replace(out, old, new)

    out = apply_replace(
        out,
        "#rl-project-dashboard .grid,#rl-project-dashboard .hero,#rl-project-dashboard .replay,#rl-project-dashboard .tables,#rl-project-dashboard .support{display:grid;gap:16px}",
        "#rl-project-dashboard .grid,#rl-project-dashboard .hero,#rl-project-dashboard .replay,#rl-project-dashboard .riskCharts,#rl-project-dashboard .tables,#rl-project-dashboard .support{display:grid;gap:16px}",
    )
    out = apply_replace(
        out,
        "#rl-project-dashboard .tables{grid-template-columns:repeat(2,minmax(0,1fr))}\n  #rl-project-dashboard .support{grid-template-columns:1fr}",
        "#rl-project-dashboard .tables{grid-template-columns:repeat(2,minmax(0,1fr))}\n  #rl-project-dashboard .riskCharts{grid-template-columns:repeat(2,minmax(0,1fr))}\n  #rl-project-dashboard .support{grid-template-columns:1fr}",
    )
    out = apply_replace(
        out,
        "#rl-project-dashboard .context{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:14px}",
        "#rl-project-dashboard .context{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:14px}\n  #rl-project-dashboard .riskCards{display:grid;gap:12px;grid-template-columns:repeat(2,minmax(0,1fr));margin-top:14px}",
    )
    out = apply_replace(
        out,
        '<div class="title">Demo checkpoint day-by-day trên dữ liệu mới từ ${esc(rr.data_source_label || \'vnstock\')}</div>',
        '<div class="title">Demo checkpoint day-by-day trên dữ liệu mới từ vnstock</div>',
    )
    out = apply_replace(
        out,
        "@media (max-width:1200px){#rl-project-dashboard .hero,#rl-project-dashboard .replay,#rl-project-dashboard .tables,#rl-project-dashboard .support{grid-template-columns:1fr}#rl-project-dashboard .strip,#rl-project-dashboard .context,#rl-project-dashboard .metrics,#rl-project-dashboard .compare,#rl-project-dashboard .row{grid-template-columns:repeat(2,minmax(0,1fr))}}",
        "@media (max-width:1200px){#rl-project-dashboard .hero,#rl-project-dashboard .replay,#rl-project-dashboard .riskCharts,#rl-project-dashboard .tables,#rl-project-dashboard .support{grid-template-columns:1fr}#rl-project-dashboard .strip,#rl-project-dashboard .context,#rl-project-dashboard .metrics,#rl-project-dashboard .compare,#rl-project-dashboard .riskCards,#rl-project-dashboard .row{grid-template-columns:repeat(2,minmax(0,1fr))}}",
    )
    out = apply_replace(
        out,
        "@media (max-width:760px){#rl-project-dashboard{padding:16px;border-radius:20px}#rl-project-dashboard .strip,#rl-project-dashboard .context,#rl-project-dashboard .metrics,#rl-project-dashboard .compare,#rl-project-dashboard .row{grid-template-columns:1fr}#rl-project-dashboard .policies{grid-template-columns:1fr}}",
        "@media (max-width:760px){#rl-project-dashboard{padding:16px;border-radius:20px}#rl-project-dashboard .strip,#rl-project-dashboard .context,#rl-project-dashboard .metrics,#rl-project-dashboard .compare,#rl-project-dashboard .riskCards,#rl-project-dashboard .row{grid-template-columns:1fr}#rl-project-dashboard .policies{grid-template-columns:1fr}}",
    )
    out = apply_replace(
        out,
        """<div class="panel" style="grid-column:1/-1"><div class="kicker">Giao dịch nổi bật</div><h3>Lệnh đáng chú ý trong ngày</h3><table><thead><tr><th>Mã</th><th>Hướng</th><th>Số lượng</th><th>Giá mở</th><th>Giá đóng</th><th>Tỷ trọng mục tiêu</th></tr></thead><tbody>${rowsHtml((f.trade_rows || []).slice(0, 12), (x) => `<tr><td>${esc(x.symbol)}</td><td>${esc(x.direction)}</td><td>${esc(x.shares)}</td><td>${num(x.execution_price, 2)}</td><td>${num(x.close_price, 2)}</td><td>${pct(x.target_weight_pct, 2)}</td></tr>`, '<tr><td colspan="6" class="empty">Ngày này policy không phát sinh lệnh nổi bật.</td></tr>')}</tbody></table></div>""",
        "",
    )

    out = replace_between(
        out,
        "const bars = () => {",
        "  const insight = () => {",
        """const bars = () => { const list = frames(cp()); if (!list.length) return ''; const vals = list.map((x) => Number(x.day_return_pct || 0)); const maxAbs = vals.reduce((a, v) => Math.max(a, Math.abs(v)), 1.2); return vals.map((v, i) => { const x = 12 + i * (596 / Math.max(vals.length - 1, 1)); const h = Math.max(2, Math.abs(v) / maxAbs * 28); const y = v >= 0 ? 38 - h : 38; const fill = v >= 0 ? '#20d6a4' : '#ff7a88'; const opacity = i === state.frame ? .98 : (i <= state.frame ? .58 : .14); return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="8" height="${h.toFixed(1)}" rx="3" fill="${fill}" opacity="${opacity}"></rect>`; }).join(''); };

  const seriesRange = (items, fallbackMin=-1, fallbackMax=1) => {
    const values = [];
    (items || []).forEach((item) => {
      if (!Array.isArray(item)) return;
      item.forEach((value) => {
        const numeric = Number(value);
        if (Number.isFinite(numeric)) values.push(numeric);
      });
    });
    if (!values.length) return { min: fallbackMin, max: fallbackMax };
    let min = Math.min.apply(null, values);
    let max = Math.max.apply(null, values);
    if (min === max) { min -= 1; max += 1; }
    return { min, max };
  };
  const valueY = (value, min, max, height=180, pad=14) => {
    const yr = (max - min) || 1;
    return height - pad - (((Number(value) - min) / yr) * (height - (pad * 2)));
  };
  const seriesPoints = (series, idx, min, max, width=620, height=180, pad=14) => {
    if (!Array.isArray(series) || !series.length) return '';
    const last = clamp(idx, 0, series.length - 1);
    const yr = (max - min) || 1;
    const denom = Math.max(series.length - 1, 1);
    const points = [];
    for (let i = 0; i <= last; i += 1) {
      const numeric = Number(series[i]);
      if (!Number.isFinite(numeric)) continue;
      const x = pad + ((width - (pad * 2)) * i / denom);
      const y = height - pad - (((numeric - min) / yr) * (height - (pad * 2)));
      points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return points.join(' ');
  };
  const seriesPoint = (series, idx, min, max, width=620, height=180, pad=14) => {
    if (!Array.isArray(series) || !series.length) return null;
    const i = clamp(idx, 0, series.length - 1);
    const numeric = Number(series[i]);
    if (!Number.isFinite(numeric)) return null;
    const denom = Math.max(series.length - 1, 1);
    const x = pad + ((width - (pad * 2)) * i / denom);
    const y = valueY(numeric, min, max, height, pad);
    return { x: Number(x.toFixed(1)), y: Number(y.toFixed(1)) };
  };

""",
    )

    out = apply_replace(
        out,
        """    const dSel = pointAt(selected.chart && selected.chart.model_points, state.frame);
    const dCmp = pointAt(compare.chart && compare.chart.model_points, state.frame);
    const dEq = pointAt(selected.chart && selected.chart.equal_weight_points, state.frame);
    const dBh = pointAt(selected.chart && selected.chart.buy_hold_points, state.frame);
    const dDdq = pointAt(ddqCompare && ddqCompare.chart && ddqCompare.chart.model_points, state.frame);""",
        """    const dSel = pointAt(selected.chart && selected.chart.model_points, state.frame);
    const dCmp = pointAt(compare.chart && compare.chart.model_points, state.frame);
    const dEq = pointAt(selected.chart && selected.chart.equal_weight_points, state.frame);
    const dBh = pointAt(selected.chart && selected.chart.buy_hold_points, state.frame);
    const dDdq = pointAt(ddqCompare && ddqCompare.chart && ddqCompare.chart.model_points, state.frame);
    const selectedRisk = ((selected.summary || {}).risk_summary || {});
    const ddqRisk = ((ddqCompare && ddqCompare.summary && ddqCompare.summary.risk_summary) || {});
    const drawdownRange = seriesRange([
      selected.chart && selected.chart.drawdown_series,
      compare.chart && compare.chart.drawdown_series,
      selected.chart && selected.chart.buy_hold_drawdown_series,
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
    const ddCmp = seriesPoint(compare.chart && compare.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);
    const ddBh = seriesPoint(selected.chart && selected.chart.buy_hold_drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);
    const ddDdq = seriesPoint(ddqCompare && ddqCompare.chart && ddqCompare.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);
    const qSel = seriesPoint(selected.chart && selected.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);
    const qSort = seriesPoint(selected.chart && selected.chart.rolling_sortino_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);
    const qCmp = seriesPoint(compare.chart && compare.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);
    const qDdq = seriesPoint(ddqCompare && ddqCompare.chart && ddqCompare.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);""",
    )

    out = replace_between(
        out,
        '<section class="replay">',
        '<section class="tables">',
        """<section class="replay">
          <div class="panel">
            <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;margin-bottom:12px"><div><div class="kicker">Soi policy theo ngày</div><h3>${esc(f.headline || 'Đường vốn và benchmark')}</h3><p class="foot">Ngày quyết định ${esc(f.decision_date || 'N/A')} · ngày đóng phiên ${esc(f.date || 'N/A')}</p></div><div class="legend"><span><i style="background:#20d6a4"></i>Checkpoint đang xem</span><span><i style="background:#ff7a88"></i>Policy so sánh</span>${ddqCompare ? '<span><i style="background:#e2e8f0"></i>DDQ tốt nhất</span>' : ''}<span><i style="background:#ffd166"></i>Danh mục đều</span><span><i style="background:#73b6ff"></i>Mua và giữ</span></div></div>
            <div class="chartShell"><div class="chartBox"><svg viewBox="0 0 620 280" preserveAspectRatio="none"><defs><linearGradient id="rlSelectedArea" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#20d6a4" stop-opacity="0.30"></stop><stop offset="100%" stop-color="#20d6a4" stop-opacity="0.02"></stop></linearGradient></defs><polygon points="${esc(areaPts(selected.chart && selected.chart.model_points, state.frame))}" fill="url(#rlSelectedArea)"></polygon><polyline points="${esc(slicePts(selected.chart && selected.chart.buy_hold_points, state.frame))}" fill="none" stroke="#73b6ff" stroke-width="3"></polyline><polyline points="${esc(slicePts(selected.chart && selected.chart.equal_weight_points, state.frame))}" fill="none" stroke="#ffd166" stroke-width="3"></polyline>${ddqCompare ? `<polyline points="${esc(slicePts(ddqCompare.chart && ddqCompare.chart.model_points, state.frame))}" fill="none" stroke="#e2e8f0" stroke-width="2.6" stroke-dasharray="4 6"></polyline>` : ''}<polyline points="${esc(slicePts(compare.chart && compare.chart.model_points, state.frame))}" fill="none" stroke="#ff7a88" stroke-width="3" stroke-dasharray="8 6"></polyline><polyline points="${esc(slicePts(selected.chart && selected.chart.model_points, state.frame))}" fill="none" stroke="#20d6a4" stroke-width="4"></polyline><line x1="${esc(markerX())}" x2="${esc(markerX())}" y1="16" y2="248" stroke="#fca5a5" stroke-dasharray="6 6"></line>${dSel ? `<circle cx="${dSel.x}" cy="${dSel.y}" r="5.8" fill="#20d6a4" stroke="#06111d" stroke-width="2"></circle>` : ''}${dCmp ? `<circle cx="${dCmp.x}" cy="${dCmp.y}" r="5" fill="#ff7a88" stroke="#06111d" stroke-width="2"></circle>` : ''}${dDdq ? `<circle cx="${dDdq.x}" cy="${dDdq.y}" r="4.2" fill="#e2e8f0" stroke="#06111d" stroke-width="2"></circle>` : ''}${dEq ? `<circle cx="${dEq.x}" cy="${dEq.y}" r="4.4" fill="#ffd166" stroke="#06111d" stroke-width="2"></circle>` : ''}${dBh ? `<circle cx="${dBh.x}" cy="${dBh.y}" r="4.4" fill="#73b6ff" stroke="#06111d" stroke-width="2"></circle>` : ''}</svg></div><div class="caption"><span>Chart chính giữ trọng tâm vào đường vốn của policy và các baseline.</span><span>Khung ${state.frame + 1}/${Math.max(frames(selected).length, 1)}</span></div></div>
            <div class="chartShell" style="margin-top:12px"><div class="chartBox"><svg viewBox="0 0 620 76" preserveAspectRatio="none"><line x1="10" x2="610" y1="38" y2="38" stroke="rgba(255,255,255,.14)" stroke-dasharray="5 5"></line>${bars()}<line x1="${esc(markerX())}" x2="${esc(markerX())}" y1="8" y2="68" stroke="#fca5a5" stroke-dasharray="6 6"></line></svg></div><div class="caption"><span>Cột xanh/đỏ là lợi nhuận theo ngày của checkpoint đang xem.</span><span>${pct(f.day_return_pct, 2)} hôm nay</span></div></div>
          </div>
          <div class="panel"><div class="kicker">So sánh nhanh</div><h3>Checkpoint đang xem vs policy so sánh</h3><p class="foot">${esc(insight())}</p>
            <div class="metrics" style="margin-top:14px"><div class="stat"><span>Giá trị danh mục</span><strong>${money(f.portfolio_value)}</strong></div><div class="stat"><span>Lợi nhuận lũy kế</span><strong class="${cls(f.total_return_pct || 0)}">${pct(f.total_return_pct, 2)}</strong></div><div class="stat"><span>Lợi nhuận ngày</span><strong class="${cls(f.day_return_pct || 0)}">${pct(f.day_return_pct, 2)}</strong></div><div class="stat"><span>Tỷ trọng tiền mặt</span><strong>${pct(f.cash_weight_pct, 2)}</strong></div><div class="stat"><span>So với danh mục đều</span><strong class="${cls(f.vs_equal_weight_pct || 0)}">${pct(f.vs_equal_weight_pct, 2)}</strong></div><div class="stat"><span>So với mua và giữ</span><strong class="${cls(f.vs_buy_hold_pct || 0)}">${pct(f.vs_buy_hold_pct, 2)}</strong></div><div class="stat"><span>Drawdown hiện tại</span><strong class="${cls(f.drawdown_pct || 0)}">${pct(f.drawdown_pct, 2)}</strong></div><div class="stat"><span>Rolling Sharpe 20 ngày</span><strong class="${cls(f.rolling_sharpe || 0)}">${num(f.rolling_sharpe, 2)}</strong></div><div class="stat"><span>Rolling Sortino 20 ngày</span><strong class="${cls(f.rolling_sortino || 0)}">${num(f.rolling_sortino, 2)}</strong></div></div>
            <div class="compare" style="margin-top:14px"><div class="cmp"><span>Chênh hiện tại vs policy so sánh</span><strong class="${cls(currentVsCompare || 0)}">${pct(currentVsCompare, 2)}</strong></div><div class="cmp"><span>Chênh cuối kỳ vs policy so sánh</span><strong class="${cls(finalVsCompare || 0)}">${pct(finalVsCompare, 2)}</strong></div><div class="cmp"><span>Chênh giá trị cuối kỳ</span><strong class="${cls(finalValueGap || 0)}">${money(finalValueGap)}</strong></div><div class="cmp"><span>Số lệnh trong ngày</span><strong>${esc(f.trade_count || 0)}</strong></div><div class="cmp"><span>Checkpoint đang xem</span><strong class="${cls((selected.summary || {}).final_return_pct || 0)}">${pct((selected.summary || {}).final_return_pct, 2)}</strong></div><div class="cmp"><span>Policy so sánh</span><strong class="${cls((compare.summary || {}).final_return_pct || 0)}">${pct((compare.summary || {}).final_return_pct, 2)}</strong></div>${ddqCompare ? `<div class="cmp"><span>DDQ tốt nhất</span><strong class="${cls((ddqCompare.summary || {}).final_return_pct || 0)}">${pct((ddqCompare.summary || {}).final_return_pct, 2)}</strong></div><div class="cmp"><span>Chênh cuối kỳ vs DDQ</span><strong class="${cls(finalVsDdq || 0)}">${pct(finalVsDdq, 2)}</strong></div>` : ''}</div>
          </div>
        </section>
        <section class="riskCharts">
          <div class="panel">
            <div class="kicker">Rủi ro vốn</div>
            <h3>Drawdown theo ngày replay</h3>
            <p class="foot">Khối này giúp giải thích checkpoint nào giữ vốn tốt hơn khi thị trường rung lắc. Đường càng thấp thì checkpoint từng rơi xa khỏi đỉnh vốn hơn.</p>
            <div class="legend"><span><i style="background:#20d6a4"></i>Checkpoint đang xem</span><span><i style="background:#ff7a88"></i>Policy so sánh</span>${ddqCompare ? '<span><i style="background:#e2e8f0"></i>DDQ tốt nhất</span>' : ''}<span><i style="background:#73b6ff"></i>Mua và giữ</span></div>
            <div class="chartShell" style="margin-top:12px"><div class="chartBox"><svg viewBox="0 0 620 180" preserveAspectRatio="none"><line x1="14" x2="606" y1="${esc(ddZeroY.toFixed(1))}" y2="${esc(ddZeroY.toFixed(1))}" stroke="rgba(255,255,255,.18)" stroke-dasharray="5 5"></line><polyline points="${esc(seriesPoints(selected.chart && selected.chart.buy_hold_drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14))}" fill="none" stroke="#73b6ff" stroke-width="2.6"></polyline>${ddqCompare ? `<polyline points="${esc(seriesPoints(ddqCompare.chart && ddqCompare.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14))}" fill="none" stroke="#e2e8f0" stroke-width="2.4" stroke-dasharray="4 6"></polyline>` : ''}<polyline points="${esc(seriesPoints(compare.chart && compare.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14))}" fill="none" stroke="#ff7a88" stroke-width="2.8" stroke-dasharray="8 6"></polyline><polyline points="${esc(seriesPoints(selected.chart && selected.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14))}" fill="none" stroke="#20d6a4" stroke-width="3.4"></polyline><line x1="${esc(markerX())}" x2="${esc(markerX())}" y1="14" y2="166" stroke="#fca5a5" stroke-dasharray="6 6"></line>${ddSel ? `<circle cx="${ddSel.x}" cy="${ddSel.y}" r="4.8" fill="#20d6a4" stroke="#06111d" stroke-width="2"></circle>` : ''}${ddCmp ? `<circle cx="${ddCmp.x}" cy="${ddCmp.y}" r="4.2" fill="#ff7a88" stroke="#06111d" stroke-width="2"></circle>` : ''}${ddDdq ? `<circle cx="${ddDdq.x}" cy="${ddDdq.y}" r="3.8" fill="#e2e8f0" stroke="#06111d" stroke-width="2"></circle>` : ''}${ddBh ? `<circle cx="${ddBh.x}" cy="${ddBh.y}" r="3.8" fill="#73b6ff" stroke="#06111d" stroke-width="2"></circle>` : ''}</svg></div><div class="caption"><span>Drawdown hiện tại của checkpoint đang xem: ${pct(f.drawdown_pct, 2)}</span><span>Max drawdown replay: ${pct(selectedRisk.max_drawdown_pct, 2)}</span></div></div>
          </div>
          <div class="panel">
            <div class="kicker">Chất lượng lợi nhuận</div>
            <h3>Rolling Sharpe & Sortino 20 ngày</h3>
            <p class="foot">Đây là chart giải thích vì sao một checkpoint có thể kiếm ít hơn chút nhưng ổn định hơn. Sharpe đo reward theo biến động chung, Sortino tập trung phạt downside.</p>
            <div class="legend"><span><i style="background:#20d6a4"></i>Sharpe của checkpoint đang xem</span><span><i style="background:#ffd166"></i>Sortino của checkpoint đang xem</span><span><i style="background:#ff7a88"></i>Sharpe của policy so sánh</span>${ddqCompare ? '<span><i style="background:#e2e8f0"></i>Sharpe của DDQ tốt nhất</span>' : ''}</div>
            <div class="chartShell" style="margin-top:12px"><div class="chartBox"><svg viewBox="0 0 620 180" preserveAspectRatio="none"><line x1="14" x2="606" y1="${esc(qualityZeroY.toFixed(1))}" y2="${esc(qualityZeroY.toFixed(1))}" stroke="rgba(255,255,255,.18)" stroke-dasharray="5 5"></line>${ddqCompare ? `<polyline points="${esc(seriesPoints(ddqCompare.chart && ddqCompare.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14))}" fill="none" stroke="#e2e8f0" stroke-width="2.4" stroke-dasharray="4 6"></polyline>` : ''}<polyline points="${esc(seriesPoints(compare.chart && compare.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14))}" fill="none" stroke="#ff7a88" stroke-width="2.8" stroke-dasharray="8 6"></polyline><polyline points="${esc(seriesPoints(selected.chart && selected.chart.rolling_sortino_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14))}" fill="none" stroke="#ffd166" stroke-width="2.6"></polyline><polyline points="${esc(seriesPoints(selected.chart && selected.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14))}" fill="none" stroke="#20d6a4" stroke-width="3.4"></polyline><line x1="${esc(markerX())}" x2="${esc(markerX())}" y1="14" y2="166" stroke="#fca5a5" stroke-dasharray="6 6"></line>${qSel ? `<circle cx="${qSel.x}" cy="${qSel.y}" r="4.8" fill="#20d6a4" stroke="#06111d" stroke-width="2"></circle>` : ''}${qSort ? `<circle cx="${qSort.x}" cy="${qSort.y}" r="4.2" fill="#ffd166" stroke="#06111d" stroke-width="2"></circle>` : ''}${qCmp ? `<circle cx="${qCmp.x}" cy="${qCmp.y}" r="4.0" fill="#ff7a88" stroke="#06111d" stroke-width="2"></circle>` : ''}${qDdq ? `<circle cx="${qDdq.x}" cy="${qDdq.y}" r="3.8" fill="#e2e8f0" stroke="#06111d" stroke-width="2"></circle>` : ''}</svg></div><div class="caption"><span>Sharpe/Sortino rolling của checkpoint đang xem tại ngày hiện tại.</span><span>Sharpe ${num(f.rolling_sharpe, 2)} · Sortino ${num(f.rolling_sortino, 2)}</span></div></div>
            <div class="riskCards"><div class="cmp"><span>Sharpe replay</span><strong class="${cls(selectedRisk.sharpe_ratio || 0)}">${num(selectedRisk.sharpe_ratio, 2)}</strong></div><div class="cmp"><span>Sortino replay</span><strong class="${cls(selectedRisk.sortino_ratio || 0)}">${num(selectedRisk.sortino_ratio, 2)}</strong></div><div class="cmp"><span>Calmar replay</span><strong class="${cls(selectedRisk.calmar_ratio || 0)}">${num(selectedRisk.calmar_ratio, 2)}</strong></div><div class="cmp"><span>Profit factor</span><strong class="${cls((selectedRisk.profit_factor || 0) - 1)}">${num(selectedRisk.profit_factor, 2)}</strong></div><div class="cmp"><span>Max drawdown</span><strong class="${cls(selectedRisk.max_drawdown_pct || 0)}">${pct(selectedRisk.max_drawdown_pct, 2)}</strong></div><div class="cmp"><span>Win rate replay</span><strong class="${cls((selectedRisk.win_rate_pct || 0) - 50)}">${pct(selectedRisk.win_rate_pct, 2)}</strong></div>${ddqCompare ? `<div class="cmp"><span>Sharpe DDQ tốt nhất</span><strong class="${cls(ddqRisk.sharpe_ratio || 0)}">${num(ddqRisk.sharpe_ratio, 2)}</strong></div><div class="cmp"><span>Max DD DDQ tốt nhất</span><strong class="${cls(ddqRisk.max_drawdown_pct || 0)}">${pct(ddqRisk.max_drawdown_pct, 2)}</strong></div>` : ''}</div>
          </div>
        </section>
        """,
    )

    risk_css = "#rl-project-dashboard .riskCards{display:grid;gap:12px;grid-template-columns:repeat(2,minmax(0,1fr));margin-top:14px}"
    while f"{risk_css}\n  {risk_css}" in out:
        out = out.replace(f"{risk_css}\n  {risk_css}", risk_css)

    risk_block = """const selectedRisk = ((selected.summary || {}).risk_summary || {});
    const ddqRisk = ((ddqCompare && ddqCompare.summary && ddqCompare.summary.risk_summary) || {});
    const drawdownRange = seriesRange([
      selected.chart && selected.chart.drawdown_series,
      compare.chart && compare.chart.drawdown_series,
      selected.chart && selected.chart.buy_hold_drawdown_series,
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
    const ddCmp = seriesPoint(compare.chart && compare.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);
    const ddBh = seriesPoint(selected.chart && selected.chart.buy_hold_drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);
    const ddDdq = seriesPoint(ddqCompare && ddqCompare.chart && ddqCompare.chart.drawdown_series, state.frame, drawdownRange.min, drawdownRange.max, 620, 180, 14);
    const qSel = seriesPoint(selected.chart && selected.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);
    const qSort = seriesPoint(selected.chart && selected.chart.rolling_sortino_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);
    const qCmp = seriesPoint(compare.chart && compare.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);
    const qDdq = seriesPoint(ddqCompare && ddqCompare.chart && ddqCompare.chart.rolling_sharpe_series, state.frame, qualityRange.min, qualityRange.max, 620, 180, 14);"""
    doubled_risk_block = f"{risk_block}\n    {risk_block}"
    while doubled_risk_block in out:
        out = out.replace(doubled_risk_block, risk_block)

    out = re.sub(
        r"(const selectedRisk = \(\(selected\.summary \|\| \{\}\)\.risk_summary \|\| \{\}\);.*?const qDdq = seriesPoint\(ddqCompare && ddqCompare\.chart && ddqCompare\.chart\.rolling_sharpe_series, state\.frame, qualityRange\.min, qualityRange\.max, 620, 180, 14\);)\s+\1",
        r"\1",
        out,
        flags=re.S,
    )
    return out


def patch_note(path: Path) -> None:
    note = json.loads(path.read_text(encoding="utf-8-sig"))
    updated = False
    for paragraph in note.get("paragraphs") or []:
        if str(paragraph.get("title") or "").strip() != "HTML Dashboard":
            continue
        paragraph["text"] = patch_html_paragraph_text(str(paragraph.get("text") or ""))
        updated = True
        break
    if not updated:
        raise ValueError(f"Không tìm thấy paragraph 'HTML Dashboard' trong {path}")
    path.write_text(json.dumps(note, ensure_ascii=False, separators=(",", ":")), encoding="utf-8-sig")


def dump_snapshot_from_local_note() -> None:
    note = json.loads(LOCAL_NOTE.read_text(encoding="utf-8-sig"))
    for paragraph in note.get("paragraphs") or []:
        if str(paragraph.get("title") or "").strip() == "HTML Dashboard":
            HTML_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            HTML_SNAPSHOT.write_text(str(paragraph.get("text") or ""), encoding="utf-8")
            return


def main() -> None:
    for target in (LOCAL_NOTE, RUNTIME_NOTE):
        if not target.exists():
            continue
        patch_note(target)
        print(f"patched {target}")
    dump_snapshot_from_local_note()


if __name__ == "__main__":
    main()
