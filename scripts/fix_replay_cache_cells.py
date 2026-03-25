from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_NOTE = ROOT / "notebooks" / "project_RL_nhom_09 .zpln"
RUNTIME_NOTE_DIR = Path(r"D:\Programs\zeppelin-0.12.0-bin-all\docker-data\notebook")
RUNTIME_NOTE_GLOB = "project_RL_nhom_09*.zpln"


BUILD_REPLAY_TEXT = r"""%spark.pyspark
import importlib
import json
import sys
from pathlib import Path

if "locate_dir" not in globals():
    raise RuntimeError("Hãy chạy paragraph 'Setup Base Payload' trước.")

PROJECT_ROOT = z.input("project_root", DEFAULT_PROJECT_ROOT)
_bootstrap_project_root = str(PROJECT_ROOT or DEFAULT_PROJECT_ROOT).strip() or DEFAULT_PROJECT_ROOT
for _candidate in (Path(_bootstrap_project_root).expanduser(), Path(_bootstrap_project_root).expanduser().parent):
    try:
        _resolved = str(_candidate.resolve())
    except Exception:
        _resolved = str(_candidate)
    if _resolved and _resolved not in sys.path:
        sys.path.insert(0, _resolved)

import scripts.dashboard_paths as dashboard_paths_module
dashboard_paths_module = importlib.reload(dashboard_paths_module)
DashboardProjectPaths = dashboard_paths_module.DashboardProjectPaths
REPLAY_CHECKPOINT_SAMPLES_DEFAULT = dashboard_paths_module.REPLAY_CHECKPOINT_SAMPLES_DEFAULT
REPLAY_END_DATE_DEFAULT = dashboard_paths_module.REPLAY_END_DATE_DEFAULT
REPLAY_PAYLOAD_SCHEMA_VERSION = dashboard_paths_module.REPLAY_PAYLOAD_SCHEMA_VERSION
REPLAY_RECENT_MONTHS_DEFAULT = dashboard_paths_module.REPLAY_RECENT_MONTHS_DEFAULT
REPLAY_RUN_LIMIT_DEFAULT = dashboard_paths_module.REPLAY_RUN_LIMIT_DEFAULT
REPLAY_VNSTOCK_SOURCE_DEFAULT = dashboard_paths_module.REPLAY_VNSTOCK_SOURCE_DEFAULT
REPLAY_WARMUP_MONTHS_DEFAULT = dashboard_paths_module.REPLAY_WARMUP_MONTHS_DEFAULT

ENABLE_CHECKPOINT_REPLAY = str(z.input("enable_checkpoint_replay", "true")).strip().lower() in ("true", "1", "yes", "y")
REPLAY_MONTHS = REPLAY_RECENT_MONTHS_DEFAULT
REPLAY_WARMUP_MONTHS = REPLAY_WARMUP_MONTHS_DEFAULT
REPLAY_RUN_LIMIT = REPLAY_RUN_LIMIT_DEFAULT
CHECKPOINT_SAMPLES = REPLAY_CHECKPOINT_SAMPLES_DEFAULT
VNSTOCK_SOURCE = REPLAY_VNSTOCK_SOURCE_DEFAULT
REPLAY_END_DATE = REPLAY_END_DATE_DEFAULT
REFRESH_REPLAY_CACHE = str(z.input("refresh_replay_cache", "false")).strip().lower() in ("true", "1", "yes", "y")
REPLAY_CACHE_SCHEMA_VERSION = REPLAY_PAYLOAD_SCHEMA_VERSION
def replay_needs_rebuild(payload):
    if not isinstance(payload, dict):
        return True
    if int(payload.get("schema_version") or 0) != REPLAY_CACHE_SCHEMA_VERSION:
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
            import src.training.PPO as ppo_module
            import src.training.DDQ as ddq_module
            import scripts.dashboard_paths as dashboard_paths_module
            import scripts.zeppelin_checkpoint_replay_helpers as replay_module
            importlib.reload(constants_module)
            importlib.reload(download_data_module)
            importlib.reload(ppo_module)
            importlib.reload(ddq_module)
            importlib.reload(dashboard_paths_module)
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

DEFAULT_PROJECT_ROOT_FALLBACK = "/workspace/project"
project_root = str(z.input("project_root", DEFAULT_PROJECT_ROOT_FALLBACK) or DEFAULT_PROJECT_ROOT_FALLBACK).strip() or DEFAULT_PROJECT_ROOT_FALLBACK
for _candidate in (Path(project_root).expanduser(), Path(project_root).expanduser().parent):
    try:
        _resolved = str(_candidate.resolve())
    except Exception:
        _resolved = str(_candidate)
    if _resolved and _resolved not in sys.path:
        sys.path.insert(0, _resolved)

import scripts.dashboard_paths as dashboard_paths_module
dashboard_paths_module = importlib.reload(dashboard_paths_module)
DashboardProjectPaths = dashboard_paths_module.DashboardProjectPaths
REPLAY_CHECKPOINT_SAMPLES_DEFAULT = dashboard_paths_module.REPLAY_CHECKPOINT_SAMPLES_DEFAULT
REPLAY_END_DATE_DEFAULT = dashboard_paths_module.REPLAY_END_DATE_DEFAULT
REPLAY_PAYLOAD_SCHEMA_VERSION = dashboard_paths_module.REPLAY_PAYLOAD_SCHEMA_VERSION
REPLAY_RECENT_MONTHS_DEFAULT = dashboard_paths_module.REPLAY_RECENT_MONTHS_DEFAULT
REPLAY_RUN_LIMIT_DEFAULT = dashboard_paths_module.REPLAY_RUN_LIMIT_DEFAULT
REPLAY_VNSTOCK_SOURCE_DEFAULT = dashboard_paths_module.REPLAY_VNSTOCK_SOURCE_DEFAULT
REPLAY_WARMUP_MONTHS_DEFAULT = dashboard_paths_module.REPLAY_WARMUP_MONTHS_DEFAULT

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
    if int(payload.get("schema_version") or 0) != REPLAY_CACHE_SCHEMA_VERSION:
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


enable_checkpoint_replay = str(z.input("enable_checkpoint_replay", "true")).strip().lower() in ("true", "1", "yes", "y")
replay_months = REPLAY_RECENT_MONTHS_DEFAULT
replay_warmup_months = REPLAY_WARMUP_MONTHS_DEFAULT
replay_run_limit = REPLAY_RUN_LIMIT_DEFAULT
checkpoint_samples = REPLAY_CHECKPOINT_SAMPLES_DEFAULT
vnstock_source = REPLAY_VNSTOCK_SOURCE_DEFAULT
replay_end_date = REPLAY_END_DATE_DEFAULT
refresh_replay_cache = str(z.input("refresh_replay_cache", "false")).strip().lower() in ("true", "1", "yes", "y")
REPLAY_CACHE_SCHEMA_VERSION = REPLAY_PAYLOAD_SCHEMA_VERSION

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
        import src.training.PPO as ppo_module
        import src.training.DDQ as ddq_module
        import scripts.dashboard_paths as dashboard_paths_module
        import scripts.zeppelin_checkpoint_replay_helpers as replay_module
        importlib.reload(constants_module)
        importlib.reload(download_data_module)
        importlib.reload(ppo_module)
        importlib.reload(ddq_module)
        importlib.reload(dashboard_paths_module)
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
            paragraph["results"] = {"code": "SUCCESS", "msg": []}
            paragraph["status"] = "READY"
        elif title == "Bind Dashboard Data":
            paragraph["text"] = BIND_TEXT
            paragraph["results"] = {"code": "SUCCESS", "msg": []}
            paragraph["status"] = "READY"
        elif title == "HTML Dashboard":
            paragraph["results"] = {"code": "SUCCESS", "msg": []}
            paragraph["status"] = "READY"
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
