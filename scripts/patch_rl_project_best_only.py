import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_NOTE = REPO_ROOT / "notebooks" / "RL_PROJECT.zpln"
RUNTIME_NOTE = Path(r"D:\Programs\zeppelin-0.12.0-bin-all\docker-data\notebook\RL_PROJECT_2MM3B3U1P.zpln")
FIX_TAG = "2026-03-23-fix12"


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Không tìm thấy đoạn cần thay: {old[:80]!r}")
    return text.replace(old, new, 1)


PY_HELPERS = """
def _metric_value(run):
    if not isinstance(run, dict):
        return float("-inf")
    summary = run.get("summary") or {}
    eval_payload = run.get("eval") or {}
    final_test = eval_payload.get("final_test") or {}
    for value in (summary.get("total_return"), final_test.get("total_return")):
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float("-inf")


def _keep_best_only(payload):
    payload = dict(payload or {})
    runs = [run for run in (payload.get("runs") or []) if isinstance(run, dict)]
    if not runs:
        payload["runs"] = []
        payload["runMap"] = {}
        payload["defaultRunId"] = None
        return payload
    best_run = max(
        runs,
        key=lambda item: (
            _metric_value(item),
            int(item.get("total_episodes") or 0),
            str(item.get("run_id") or ""),
        ),
    )
    payload["runs"] = [best_run]
    payload["runMap"] = {best_run.get("run_id"): best_run} if best_run.get("run_id") else {}
    payload["defaultRunId"] = best_run.get("run_id")
    return payload


def _agent_bucket(agent_name):
    agent_text = str(agent_name or "").upper()
    if "PPO" in agent_text:
        return "PPO"
    if "DDQ" in agent_text or "DQN" in agent_text:
        return "DQN"
    return None


def _summarize_payload_run(run):
    if not isinstance(run, dict):
        return None
    summary = run.get("summary") or {}
    eval_payload = run.get("eval") or {}
    final_test = eval_payload.get("final_test") or {}
    total_return = summary.get("total_return")
    if total_return in (None, ""):
        total_return = final_test.get("total_return")
    sharpe_ratio = summary.get("sharpe_ratio")
    if sharpe_ratio in (None, ""):
        sharpe_ratio = final_test.get("sharpe_ratio")
    win_rate = summary.get("win_rate")
    try:
        total_return_pct = round(float(total_return) * 100.0, 4)
    except (TypeError, ValueError):
        total_return_pct = None
    try:
        sharpe_ratio = round(float(sharpe_ratio), 4)
    except (TypeError, ValueError):
        sharpe_ratio = None
    try:
        win_rate_pct = round(float(win_rate) * 100.0, 4)
    except (TypeError, ValueError):
        win_rate_pct = None
    return {
        "algo": "PPO",
        "agent": run.get("agent") or "PPO_LSTM",
        "run_id": run.get("run_id"),
        "label": run.get("label") or run.get("run_id"),
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "win_rate_pct": win_rate_pct,
    }


def _summarize_summary_path(summary_path):
    if summary_path is None or not summary_path.exists():
        return None
    summary = _load_json(summary_path) or {}
    final_metrics = summary.get("final_metrics") or {}
    bucket = _agent_bucket(summary.get("agent"))
    if bucket is None:
        return None
    total_return = final_metrics.get("total_return")
    sharpe_ratio = final_metrics.get("sharpe_ratio")
    win_rate = final_metrics.get("win_rate")
    try:
        total_return_pct = round(float(total_return) * 100.0, 4)
    except (TypeError, ValueError):
        total_return_pct = None
    try:
        sharpe_ratio = round(float(sharpe_ratio), 4)
    except (TypeError, ValueError):
        sharpe_ratio = None
    try:
        win_rate_pct = round(float(win_rate) * 100.0, 4)
    except (TypeError, ValueError):
        win_rate_pct = None
    return {
        "algo": bucket,
        "agent": summary.get("agent"),
        "run_id": summary.get("run_id") or summary_path.parent.name,
        "label": summary.get("run_id") or summary_path.parent.name,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "win_rate_pct": win_rate_pct,
    }


def _build_algo_compare(project_root, payload):
    payload = dict(payload or {})
    selected_run = _summarize_payload_run(((payload.get("runs") or [None])[0]))
    results_root = Path(project_root).expanduser().resolve() / "results" / "runs"
    best_dqn = None
    if results_root.exists():
        for summary_path in results_root.glob("*/summary.json"):
            item = _summarize_summary_path(summary_path)
            if not item or item.get("algo") != "DQN":
                continue
            if (
                best_dqn is None
                or (item.get("total_return_pct") or float("-inf")) > (best_dqn.get("total_return_pct") or float("-inf"))
                or (
                    (item.get("total_return_pct") or float("-inf")) == (best_dqn.get("total_return_pct") or float("-inf"))
                    and (item.get("sharpe_ratio") or float("-inf")) > (best_dqn.get("sharpe_ratio") or float("-inf"))
                )
            ):
                best_dqn = item

    if not selected_run:
        return {
            "status": "empty",
            "message": "Không có PPO run để hiển thị.",
            "ppo": None,
            "dqn": best_dqn,
            "delta_total_return_pct": None,
            "delta_sharpe_ratio": None,
        }

    if not best_dqn:
        return {
            "status": "missing_dqn",
            "message": "Chưa tìm thấy artifact DQN/DDQ trong results/runs để so sánh. Dashboard hiện chỉ hiển thị best PPO run.",
            "ppo": selected_run,
            "dqn": None,
            "delta_total_return_pct": None,
            "delta_sharpe_ratio": None,
        }

    ppo_return = selected_run.get("total_return_pct")
    dqn_return = best_dqn.get("total_return_pct")
    ppo_sharpe = selected_run.get("sharpe_ratio")
    dqn_sharpe = best_dqn.get("sharpe_ratio")
    return {
        "status": "ready",
        "message": "Đã tìm thấy best PPO run và best DQN/DDQ run để so sánh.",
        "ppo": selected_run,
        "dqn": best_dqn,
        "delta_total_return_pct": round(float(ppo_return) - float(dqn_return), 4) if ppo_return is not None and dqn_return is not None else None,
        "delta_sharpe_ratio": round(float(ppo_sharpe) - float(dqn_sharpe), 4) if ppo_sharpe is not None and dqn_sharpe is not None else None,
    }


def _filter_replay_payload(replay_payload, best_run_id):
    replay_payload = dict(replay_payload or {})
    runs = [run for run in (replay_payload.get("runs") or []) if isinstance(run, dict)]
    if not runs:
        replay_payload["runs"] = []
        replay_payload["runMap"] = {}
        replay_payload["defaultRunId"] = None
        return replay_payload
    filtered = [run for run in runs if run.get("run_id") == best_run_id] or runs[:1]
    replay_payload["runs"] = filtered
    replay_payload["runMap"] = {run.get("run_id"): run for run in filtered if run.get("run_id")}
    replay_payload["defaultRunId"] = (filtered[0] or {}).get("run_id") if filtered else None
    return replay_payload
"""


JS_ALGO_HELPERS = """
    function algoCompare() { return data.algoCompare || { status:'missing_dqn', message:'Chưa có artifact DQN/DDQ để so sánh.', ppo:null, dqn:null }; }
    function algoCompareSection(algo) {
      if (!algo || algo.status !== 'ready') {
        return '<section class="rl-panel"><div class="rl-kicker">PPO vs DQN/DDQ</div><h3>So sánh best run theo thuật toán</h3><p>'+esc((algo && algo.message) || 'Chưa có artifact DQN/DDQ để so sánh.')+'</p><span class="rl-alert">'+esc((algo && algo.message) || 'Chưa có artifact DQN/DDQ trong results/runs.')+'</span></section>';
      }
      var ppo = algo.ppo || {};
      var dqn = algo.dqn || {};
      return '<section class="rl-panel"><div class="rl-kicker">PPO vs DQN/DDQ</div><h3>So sánh best run theo thuật toán</h3><p>Dashboard đang khóa vào best PPO run. Nếu trong repo có artifact DDQ/DQN, phần này sẽ đặt best DQN/DDQ cạnh best PPO để so nhanh hiệu quả và rủi ro.</p><div class="rl-compare-grid">' +
        '<div class="rl-compare-card"><span>Best PPO run</span><strong>'+esc(ppo.run_id || 'N/A')+'</strong></div>' +
        '<div class="rl-compare-card"><span>Best DQN/DDQ run</span><strong>'+esc(dqn.run_id || 'N/A')+'</strong></div>' +
        '<div class="rl-compare-card"><span>PPO - DQN lợi nhuận</span><strong class="'+pctClass(algo.delta_total_return_pct || 0)+'">'+num(algo.delta_total_return_pct, 2)+' điểm%</strong></div>' +
        '<div class="rl-compare-card"><span>PPO - DQN Sharpe</span><strong class="'+pctClass(algo.delta_sharpe_ratio || 0)+'">'+num(algo.delta_sharpe_ratio, 3)+'</strong></div>' +
        '</div><div class="rl-mini-grid" style="margin-top:12px;">' +
        '<div class="rl-mini"><span>PPO lợi nhuận cuối</span><strong>'+num(ppo.total_return_pct, 2)+'%</strong></div>' +
        '<div class="rl-mini"><span>DQN/DDQ lợi nhuận cuối</span><strong>'+num(dqn.total_return_pct, 2)+'%</strong></div>' +
        '<div class="rl-mini"><span>PPO Sharpe</span><strong>'+num(ppo.sharpe_ratio, 3)+'</strong></div>' +
        '<div class="rl-mini"><span>DQN/DDQ Sharpe</span><strong>'+num(dqn.sharpe_ratio, 3)+'</strong></div>' +
        '</div></section>';
    }
"""


def patch_html_dashboard(text: str) -> str:
    text = replace_once(text, 'def _load_json(path):\n    if path is None or not path.exists():\n        return None\n    return json.loads(path.read_text(encoding="utf-8"))\n', 'def _load_json(path):\n    if path is None or not path.exists():\n        return None\n    return json.loads(path.read_text(encoding="utf-8"))\n' + PY_HELPERS)
    text = replace_once(text, 'payload = dict(train_payload)\ndebug = dict((dashboard_payload or {}).get("debug") or payload.get("debug") or {})\n', 'payload = _keep_best_only(dict(train_payload))\npayload["algoCompare"] = _build_algo_compare(PROJECT_ROOT, payload)\ndebug = dict((dashboard_payload or {}).get("debug") or payload.get("debug") or {})\nbest_run_id = payload.get("defaultRunId")\n')
    text = replace_once(text, 'payload["checkpointReplay"] = checkpoint_replay\n', 'checkpoint_replay = _filter_replay_payload(checkpoint_replay, best_run_id)\npayload["checkpointReplay"] = checkpoint_replay\n')
    text = text.replace('debug["build_tag"] = "2026-03-23-fix11"', f'debug["build_tag"] = "{FIX_TAG}"')
    text = replace_once(text, 'debug["used_replay_cache"] = bool(replay_payload and (replay_payload.get("runs") or replay_payload.get("status") == "ready"))\n', 'debug["used_replay_cache"] = bool(replay_payload and (replay_payload.get("runs") or replay_payload.get("status") == "ready"))\ndebug["algo_compare_status"] = (payload.get("algoCompare") or {}).get("status")\ndebug["algo_compare_message"] = (payload.get("algoCompare") or {}).get("message")\n')
    text = replace_once(text, "    function htmlRows(list, rowFn, emptyHtml) {{ return list && list.length ? list.map(rowFn).join('') : (emptyHtml || '<tr><td colspan=\"6\">Không có dữ liệu</td></tr>'); }}\n", "    function htmlRows(list, rowFn, emptyHtml) {{ return list && list.length ? list.map(rowFn).join('') : (emptyHtml || '<tr><td colspan=\"6\">Không có dữ liệu</td></tr>'); }}\n" + JS_ALGO_HELPERS + "\n")
    text = replace_once(text, "      var debug = data.debug || {{}};\n", "      var debug = data.debug || {{}};\n      var algo = algoCompare();\n")
    text = text.replace('PPO Trading Dashboard | 2026-03-23-fix11', f'PPO Trading Dashboard | {FIX_TAG}')
    text = text.replace('Replay checkpoint | 2026-03-23-fix11', f'Replay checkpoint | {FIX_TAG}')
    text = text.replace('<span class="rl-chip">Số run train: ${esc(runs().length || 0)}</span><span class="rl-chip">Số run replay: ${esc(replayRuns().length || 0)}</span>', '<span class="rl-chip">Chế độ train: Best PPO run</span><span class="rl-chip">Compare DQN: ${esc(algo.status || "n/a")}</span>')
    text = text.replace('<label class="rl-label">Chọn run train</label>', '<label class="rl-label">Best PPO run</label>')
    text = text.replace('<select id="runSelect" class="rl-select">', '<select id="runSelect" class="rl-select" disabled>')
    text = text.replace('Dashboard này đọc trực tiếp train cache và replay cache, không phụ thuộc Angular bind. Nếu đã dựng replay cache thì combobox run replay và checkpoint sẽ có dữ liệu ngay.', 'Dashboard này đọc trực tiếp cache local nhưng giao diện luôn khóa vào best PPO run. Phần replay checkpoint chỉ bám đúng run đó; phần compare thuật toán sẽ tự bật khi repo có artifact DQN/DDQ thật.')
    text = replace_once(text, "        </section>\n        \n\n        <section class=\"rl-panel rl-terminal\">", "        </section>\n        ${algoCompareSection(algo)}\n\n        <section class=\"rl-panel rl-terminal\">")
    return text


def patch_replay_builder(text: str) -> str:
    return text.replace('REPLAY_RUN_LIMIT = to_int(z.input("replay_run_limit", "3"), 3)', 'REPLAY_RUN_LIMIT = to_int(z.input("replay_run_limit", "1"), 1)')


def patch_note_file(note_path: Path) -> None:
    if not note_path.exists():
        return
    notebook = json.loads(note_path.read_text(encoding="utf-8"))
    paragraphs = notebook.get("paragraphs") or []
    for paragraph in paragraphs:
        title = paragraph.get("title")
        if title == "Build Replay Cache":
            paragraph["text"] = patch_replay_builder(paragraph.get("text") or "")
        elif title == "HTML Dashboard":
            paragraph["text"] = patch_html_dashboard(paragraph.get("text") or "")
    note_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")


def metric_value(run: dict) -> float:
    summary = run.get("summary") or {}
    eval_payload = run.get("eval") or {}
    final_test = eval_payload.get("final_test") or {}
    for value in (summary.get("total_return"), final_test.get("total_return")):
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float("-inf")


def keep_best_payload(payload: dict) -> dict:
    payload = dict(payload or {})
    runs = [run for run in (payload.get("runs") or []) if isinstance(run, dict)]
    if not runs:
        payload["runs"] = []
        payload["runMap"] = {}
        payload["defaultRunId"] = None
        return payload
    best_run = max(runs, key=lambda item: (metric_value(item), int(item.get("total_episodes") or 0), str(item.get("run_id") or "")))
    payload["runs"] = [best_run]
    payload["runMap"] = {best_run.get("run_id"): best_run} if best_run.get("run_id") else {}
    payload["defaultRunId"] = best_run.get("run_id")
    return payload


def summarize_payload_run(run: dict | None) -> dict | None:
    if not isinstance(run, dict):
        return None
    summary = run.get("summary") or {}
    eval_payload = run.get("eval") or {}
    final_test = eval_payload.get("final_test") or {}
    total_return = summary.get("total_return")
    if total_return in (None, ""):
        total_return = final_test.get("total_return")
    sharpe_ratio = summary.get("sharpe_ratio")
    if sharpe_ratio in (None, ""):
        sharpe_ratio = final_test.get("sharpe_ratio")
    win_rate = summary.get("win_rate")
    try:
        total_return_pct = round(float(total_return) * 100.0, 4)
    except (TypeError, ValueError):
        total_return_pct = None
    try:
        sharpe_ratio = round(float(sharpe_ratio), 4)
    except (TypeError, ValueError):
        sharpe_ratio = None
    try:
        win_rate_pct = round(float(win_rate) * 100.0, 4)
    except (TypeError, ValueError):
        win_rate_pct = None
    return {
        "algo": "PPO",
        "agent": run.get("agent") or "PPO_LSTM",
        "run_id": run.get("run_id"),
        "label": run.get("label") or run.get("run_id"),
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "win_rate_pct": win_rate_pct,
    }


def summarize_summary(summary_path: Path) -> dict | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    agent = str(summary.get("agent") or "").upper()
    if "DDQ" not in agent and "DQN" not in agent:
        return None
    final_metrics = summary.get("final_metrics") or {}
    try:
        total_return_pct = round(float(final_metrics.get("total_return")) * 100.0, 4)
    except (TypeError, ValueError):
        total_return_pct = None
    try:
        sharpe_ratio = round(float(final_metrics.get("sharpe_ratio")), 4)
    except (TypeError, ValueError):
        sharpe_ratio = None
    try:
        win_rate_pct = round(float(final_metrics.get("win_rate")) * 100.0, 4)
    except (TypeError, ValueError):
        win_rate_pct = None
    return {
        "algo": "DQN",
        "agent": summary.get("agent"),
        "run_id": summary.get("run_id") or summary_path.parent.name,
        "label": summary.get("run_id") or summary_path.parent.name,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "win_rate_pct": win_rate_pct,
    }


def build_algo_compare_from_repo(project_root: Path, payload: dict) -> dict:
    ppo = summarize_payload_run(((payload.get("runs") or [None])[0]))
    results_root = project_root / "results" / "runs"
    best_dqn = None
    if results_root.exists():
        for summary_path in results_root.glob("*/summary.json"):
            item = summarize_summary(summary_path)
            if item is None:
                continue
            if (
                best_dqn is None
                or (item.get("total_return_pct") or float("-inf")) > (best_dqn.get("total_return_pct") or float("-inf"))
                or (
                    (item.get("total_return_pct") or float("-inf")) == (best_dqn.get("total_return_pct") or float("-inf"))
                    and (item.get("sharpe_ratio") or float("-inf")) > (best_dqn.get("sharpe_ratio") or float("-inf"))
                )
            ):
                best_dqn = item
    if not ppo:
        return {"status": "empty", "message": "Không có PPO run để hiển thị.", "ppo": None, "dqn": best_dqn}
    if not best_dqn:
        return {
            "status": "missing_dqn",
            "message": "Chưa tìm thấy artifact DQN/DDQ trong results/runs để so sánh. Dashboard hiện chỉ hiển thị best PPO run.",
            "ppo": ppo,
            "dqn": None,
            "delta_total_return_pct": None,
            "delta_sharpe_ratio": None,
        }
    return {
        "status": "ready",
        "message": "Đã tìm thấy best PPO run và best DQN/DDQ run để so sánh.",
        "ppo": ppo,
        "dqn": best_dqn,
        "delta_total_return_pct": round(float(ppo["total_return_pct"]) - float(best_dqn["total_return_pct"]), 4) if ppo.get("total_return_pct") is not None and best_dqn.get("total_return_pct") is not None else None,
        "delta_sharpe_ratio": round(float(ppo["sharpe_ratio"]) - float(best_dqn["sharpe_ratio"]), 4) if ppo.get("sharpe_ratio") is not None and best_dqn.get("sharpe_ratio") is not None else None,
    }


def filter_replay_payload(replay_payload: dict, best_run_id: str | None) -> dict:
    replay_payload = dict(replay_payload or {})
    runs = [run for run in (replay_payload.get("runs") or []) if isinstance(run, dict)]
    if not runs:
        replay_payload["runs"] = []
        replay_payload["runMap"] = {}
        replay_payload["defaultRunId"] = None
        return replay_payload
    filtered = [run for run in runs if run.get("run_id") == best_run_id] or runs[:1]
    replay_payload["runs"] = filtered
    replay_payload["runMap"] = {run.get("run_id"): run for run in filtered if run.get("run_id")}
    replay_payload["defaultRunId"] = (filtered[0] or {}).get("run_id") if filtered else None
    return replay_payload


def patch_cache_files() -> None:
    cache_dir = REPO_ROOT / ".zeppelin_cache"
    train_cache_path = cache_dir / "dashboard_train_payload.json"
    replay_cache_path = cache_dir / "dashboard_replay_payload.json"
    dashboard_cache_path = cache_dir / "dashboard_payload.json"

    if not train_cache_path.exists():
        return

    train_payload = keep_best_payload(json.loads(train_cache_path.read_text(encoding="utf-8")))
    train_payload["algoCompare"] = build_algo_compare_from_repo(REPO_ROOT, train_payload)
    train_payload.setdefault("debug", {})
    train_payload["debug"]["build_tag"] = FIX_TAG
    train_payload["debug"]["algo_compare_status"] = train_payload["algoCompare"]["status"]
    train_payload["debug"]["algo_compare_message"] = train_payload["algoCompare"]["message"]
    train_cache_path.write_text(json.dumps(train_payload, ensure_ascii=False), encoding="utf-8")

    replay_payload = {}
    if replay_cache_path.exists():
        replay_payload = filter_replay_payload(
            json.loads(replay_cache_path.read_text(encoding="utf-8")),
            train_payload.get("defaultRunId"),
        )
        replay_cache_path.write_text(json.dumps(replay_payload, ensure_ascii=False), encoding="utf-8")

    dashboard_payload = dict(train_payload)
    if replay_payload:
        dashboard_payload["checkpointReplay"] = replay_payload
    dashboard_payload["debug"] = dict(dashboard_payload.get("debug") or {})
    dashboard_payload["debug"]["replay_status"] = (replay_payload or {}).get("status")
    dashboard_payload["debug"]["replay_run_count"] = len((replay_payload or {}).get("runs") or [])
    dashboard_payload["debug"]["replay_run_ids"] = [item.get("run_id") for item in ((replay_payload or {}).get("runs") or [])]
    dashboard_cache_path.write_text(json.dumps(dashboard_payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    patch_note_file(LOCAL_NOTE)
    patch_note_file(RUNTIME_NOTE)
    patch_cache_files()
    print(json.dumps({"status": "ok", "fix_tag": FIX_TAG, "patched_notes": [str(path) for path in (LOCAL_NOTE, RUNTIME_NOTE) if path.exists()]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
