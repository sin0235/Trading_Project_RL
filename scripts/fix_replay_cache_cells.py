from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_NOTE = ROOT / "notebooks" / "project_RL_nhom_09 .zpln"
RUNTIME_NOTE = Path(r"D:\Programs\zeppelin-0.12.0-bin-all\docker-data\notebook\project_RL_nhom_09 _2MNYSRNK4.zpln")


BUILD_REPLAY_TEXT = r"""%spark.pyspark
import importlib
import json
import sys
from pathlib import Path

if "locate_dir" not in globals():
    raise RuntimeError("Hãy chạy paragraph 'Setup Base Payload' trước.")

from scripts.dashboard_paths import DashboardProjectPaths

PROJECT_ROOT = z.input("project_root", DEFAULT_PROJECT_ROOT)
ENABLE_CHECKPOINT_REPLAY = str(z.input("enable_checkpoint_replay", "true")).strip().lower() in ("true", "1", "yes", "y")
REPLAY_MONTHS = to_int(z.input("replay_months", "12"), 12)
REPLAY_WARMUP_MONTHS = to_int(z.input("replay_warmup_months", "4"), 4)
REPLAY_RUN_LIMIT = to_int(z.input("replay_run_limit", "1"), 1)
CHECKPOINT_SAMPLES = to_int(z.input("checkpoint_samples", "6"), 6)
VNSTOCK_SOURCE = z.input("vnstock_source", "VCI").strip() or "VCI"
REPLAY_END_DATE = str(z.input("replay_end_date", "2026-02-28") or "2026-02-28").strip() or "2026-02-28"
REFRESH_REPLAY_CACHE = str(z.input("refresh_replay_cache", "false")).strip().lower() in ("true", "1", "yes", "y")
def replay_needs_rebuild(payload):
    if not isinstance(payload, dict):
        return True
    status = str(payload.get("status") or "").strip().lower()
    if status in ("error", "empty", "pending"):
        return True
    if status == "disabled":
        return False
    return not bool(payload.get("runs") or [])


def replay_cache_matches_request(payload):
    if replay_needs_rebuild(payload):
        return False
    run_sources = {
        str(item.get("data_source") or "").strip().lower()
        for item in (payload.get("runs") or [])
        if isinstance(item, dict)
    }
    if VNSTOCK_SOURCE.strip().lower() != "local" and run_sources != {"vnstock"}:
        return False
    return (
        int(payload.get("recent_months") or 0) == max(REPLAY_MONTHS, 1)
        and int(payload.get("warmup_months") or 0) == max(REPLAY_WARMUP_MONTHS, 1)
        and int(payload.get("run_limit") or 0) == max(REPLAY_RUN_LIMIT, 1)
        and int(payload.get("checkpoint_samples") or 0) == max(CHECKPOINT_SAMPLES, 1)
        and str(payload.get("vnstock_source") or "").strip().upper() == VNSTOCK_SOURCE.strip().upper()
        and str(payload.get("end_date") or "").strip() == REPLAY_END_DATE
    )


project_root_dir = locate_dir(PROJECT_ROOT, None, search_glob=False)
if project_root_dir is None:
    raise FileNotFoundError("Không tìm được project_root để dựng replay cache.")
paths = DashboardProjectPaths.from_project_root(project_root_dir)
paths.cache_dir.mkdir(parents=True, exist_ok=True)
train_cache_path = paths.train_cache_path
replay_cache_path = paths.replay_cache_path
dashboard_cache_path = paths.dashboard_cache_path
if not train_cache_path.exists():
    raise FileNotFoundError("Chưa có train cache. Hãy chạy paragraph 'Setup Base Payload' trước.")
train_payload = load_json(train_cache_path)

replay_payload = {
    "status": "disabled",
    "message": "Đã tắt replay checkpoint trong form cấu hình.",
    "warnings": [],
    "runs": [],
    "runMap": {},
    "defaultRunId": None,
}
used_cache = False
if ENABLE_CHECKPOINT_REPLAY:
    cached_payload = load_json(replay_cache_path) if replay_cache_path.exists() else None
    if replay_cache_path.exists() and not REFRESH_REPLAY_CACHE and replay_cache_matches_request(cached_payload):
        replay_payload = cached_payload
        used_cache = True
    else:
        try:
            resolved_project_root = str(Path(train_payload["project"]["project_root"]).resolve())
            if resolved_project_root not in sys.path:
                sys.path.insert(0, resolved_project_root)
            import src.constants as constants_module
            import src.data.download_data as download_data_module
            import scripts.zeppelin_checkpoint_replay_helpers as replay_module
            importlib.reload(constants_module)
            importlib.reload(download_data_module)
            replay_module = importlib.reload(replay_module)
            build_checkpoint_replay_payload = replay_module.build_checkpoint_replay_payload
            replay_payload = build_checkpoint_replay_payload(
                project_root=resolved_project_root,
                results_dir=train_payload["project"]["results_dir"],
                data_dir=train_payload["project"]["data_dir"],
                recent_months=max(REPLAY_MONTHS, 1),
                warmup_months=max(REPLAY_WARMUP_MONTHS, 1),
                run_limit=max(REPLAY_RUN_LIMIT, 1),
                checkpoint_samples=max(CHECKPOINT_SAMPLES, 1),
                vnstock_source=VNSTOCK_SOURCE,
                end_date=REPLAY_END_DATE,
            )
        except BaseException as exc:
            replay_payload = {
                "status": "error",
                "message": str(exc),
                "warnings": [str(exc)],
                "runs": [],
                "runMap": {},
                "defaultRunId": None,
            }
        with replay_cache_path.open("w", encoding="utf-8") as handle:
            json.dump(replay_payload, handle, ensure_ascii=False)

dashboard_payload = dict(train_payload)
dashboard_payload["checkpointReplay"] = replay_payload
with dashboard_cache_path.open("w", encoding="utf-8") as handle:
    json.dump(dashboard_payload, handle, ensure_ascii=False)

print(json.dumps({
    "build_status": "ok",
    "replay_cache_path": str(replay_cache_path),
    "dashboard_cache_path": str(dashboard_cache_path),
    "replay_status": replay_payload.get("status"),
    "replay_message": replay_payload.get("message"),
    "replay_run_count": len(replay_payload.get("runs") or []),
    "used_cache": used_cache,
    "recent_months": replay_payload.get("recent_months"),
    "end_date": replay_payload.get("end_date"),
    "vnstock_source": replay_payload.get("vnstock_source"),
}, ensure_ascii=False, indent=2))
"""


BIND_TEXT = r"""%spark.pyspark
import importlib
import json
import sys
from pathlib import Path

from scripts.dashboard_paths import DashboardProjectPaths


def load_json(path):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def to_int(value, default):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def replay_needs_rebuild(payload):
    if not isinstance(payload, dict):
        return True
    status = str(payload.get("status") or "").strip().lower()
    if status in ("error", "empty", "pending"):
        return True
    if status == "disabled":
        return False
    return not bool(payload.get("runs") or [])


def replay_cache_matches_request(payload):
    if replay_needs_rebuild(payload):
        return False
    run_sources = {
        str(item.get("data_source") or "").strip().lower()
        for item in (payload.get("runs") or [])
        if isinstance(item, dict)
    }
    if vnstock_source.strip().lower() != "local" and run_sources != {"vnstock"}:
        return False
    return (
        int(payload.get("recent_months") or 0) == max(replay_months, 1)
        and int(payload.get("warmup_months") or 0) == max(replay_warmup_months, 1)
        and int(payload.get("run_limit") or 0) == max(replay_run_limit, 1)
        and int(payload.get("checkpoint_samples") or 0) == max(checkpoint_samples, 1)
        and str(payload.get("vnstock_source") or "").strip().upper() == vnstock_source.strip().upper()
        and str(payload.get("end_date") or "").strip() == replay_end_date
    )


project_root = str(z.input("project_root", "/workspace/project") or "/workspace/project").strip() or "/workspace/project"
enable_checkpoint_replay = str(z.input("enable_checkpoint_replay", "true")).strip().lower() in ("true", "1", "yes", "y")
replay_months = to_int(z.input("replay_months", "12"), 12)
replay_warmup_months = to_int(z.input("replay_warmup_months", "4"), 4)
replay_run_limit = to_int(z.input("replay_run_limit", "1"), 1)
checkpoint_samples = to_int(z.input("checkpoint_samples", "6"), 6)
vnstock_source = str(z.input("vnstock_source", "VCI") or "VCI").strip() or "VCI"
replay_end_date = str(z.input("replay_end_date", "2026-02-28") or "2026-02-28").strip() or "2026-02-28"
refresh_replay_cache = str(z.input("refresh_replay_cache", "false")).strip().lower() in ("true", "1", "yes", "y")

paths = DashboardProjectPaths.from_project_root(project_root)
paths.cache_dir.mkdir(parents=True, exist_ok=True)
train_path = paths.train_cache_path
replay_path = paths.replay_cache_path
dashboard_path = paths.dashboard_cache_path

train_payload = load_json(train_path)
if not train_payload:
    raise FileNotFoundError("Chưa có train cache. Hãy chạy 'Setup Base Payload' trước.")

replay_payload = load_json(replay_path)
used_cache = bool(replay_payload) and replay_cache_matches_request(replay_payload) and not refresh_replay_cache
if enable_checkpoint_replay and (refresh_replay_cache or not replay_cache_matches_request(replay_payload)):
    try:
        resolved_project_root = str(Path(train_payload["project"]["project_root"]).resolve())
        if resolved_project_root not in sys.path:
            sys.path.insert(0, resolved_project_root)
        import src.constants as constants_module
        import src.data.download_data as download_data_module
        import scripts.zeppelin_checkpoint_replay_helpers as replay_module
        importlib.reload(constants_module)
        importlib.reload(download_data_module)
        replay_module = importlib.reload(replay_module)
        replay_payload = replay_module.build_checkpoint_replay_payload(
            project_root=resolved_project_root,
            results_dir=train_payload["project"]["results_dir"],
            data_dir=train_payload["project"]["data_dir"],
            recent_months=max(replay_months, 1),
            warmup_months=max(replay_warmup_months, 1),
            run_limit=max(replay_run_limit, 1),
            checkpoint_samples=max(checkpoint_samples, 1),
            vnstock_source=vnstock_source,
            end_date=replay_end_date,
        )
        used_cache = False
    except BaseException as exc:
        replay_payload = {
            "status": "error",
            "message": str(exc),
            "warnings": [str(exc)],
            "runs": [],
            "runMap": {},
            "defaultRunId": None,
        }
    replay_path.write_text(json.dumps(replay_payload, ensure_ascii=False), encoding="utf-8")
elif not enable_checkpoint_replay:
    replay_payload = {
        "status": "disabled",
        "message": "Đã tắt replay checkpoint trong form cấu hình.",
        "warnings": [],
        "runs": [],
        "runMap": {},
        "defaultRunId": None,
    }

payload = dict(train_payload)
if isinstance(replay_payload, dict):
    payload["checkpointReplay"] = {
        "status": replay_payload.get("status") or ("ready" if replay_payload.get("runs") else "pending"),
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
    "replay_payload": str(replay_path),
    "used_cache": used_cache,
    "has_replay": bool((payload.get("checkpointReplay") or {}).get("runs")),
    "replay_status": (payload.get("checkpointReplay") or {}).get("status"),
    "run_count": len(payload.get("runs") or []),
    "replay_run_count": len((payload.get("checkpointReplay") or {}).get("runs") or []),
    "recent_months": replay_payload.get("recent_months") if isinstance(replay_payload, dict) else None,
    "end_date": replay_payload.get("end_date") if isinstance(replay_payload, dict) else None,
    "vnstock_source": replay_payload.get("vnstock_source") if isinstance(replay_payload, dict) else None,
}, ensure_ascii=False, indent=2))
"""


def patch_note(path: Path) -> None:
    note = json.loads(path.read_text(encoding="utf-8-sig"))
    for paragraph in note.get("paragraphs") or []:
        title = str(paragraph.get("title") or "").strip()
        if title == "Build Replay Cache":
            paragraph["text"] = BUILD_REPLAY_TEXT
        elif title == "Bind Dashboard Data":
            paragraph["text"] = BIND_TEXT
    path.write_text(json.dumps(note, ensure_ascii=False, separators=(",", ":")), encoding="utf-8-sig")


def main() -> None:
    for target in (LOCAL_NOTE, RUNTIME_NOTE):
        patch_note(target)
        print(f"patched {target}")


if __name__ == "__main__":
    main()
