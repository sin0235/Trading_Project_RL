from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Đặt None để helper tự chọn best PPO run theo score.
FIXED_PPO_REPLAY_RUN_ID: str | None = "ppo_20260324_044152"
FIXED_PPO_REPLAY_DEFAULT_CHECKPOINT_ID: str | None = "final_model"

# Replay defaults: mọi helper/cell dashboard phải dùng chung bộ này.
REPLAY_RECENT_MONTHS_DEFAULT = 12
REPLAY_WARMUP_MONTHS_DEFAULT = 4
REPLAY_RUN_LIMIT_DEFAULT = 1
REPLAY_CHECKPOINT_SAMPLES_DEFAULT = 6
REPLAY_VNSTOCK_SOURCE_DEFAULT = "VCI"
REPLAY_END_DATE_DEFAULT = "2026-02-25"
REPLAY_MAX_FRAMES_DEFAULT = 420
REPLAY_PAYLOAD_SCHEMA_VERSION = 2

# Thư mục cố định cho line compare DDQ trên dashboard.
# Hiện dashboard lấy line so sánh từ:
#   results/compare/BranchingDDQ_LSTM/BranchingDDQ.pt
DDQ_COMPARE_DIRNAME = "BranchingDDQ_LSTM"
DDQ_COMPARE_LABEL = "Branching DDQ"


@dataclass(frozen=True)
class DashboardProjectPaths:
    project_root: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "DashboardProjectPaths":
        return cls(Path(project_root).resolve())

    @property
    def cache_dir(self) -> Path:
        return self.project_root / ".zeppelin_cache"

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    @property
    def processed_data_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def processed_v2_root(self) -> Path:
        return self.data_root / "processed_v2"

    @property
    def replay_fallback_root(self) -> Path:
        return self.data_root / "replay_fallback"

    @property
    def train_cache_path(self) -> Path:
        return self.cache_dir / "dashboard_train_payload.json"

    @property
    def replay_cache_path(self) -> Path:
        return self.cache_dir / "dashboard_replay_payload.json"

    @property
    def dashboard_cache_path(self) -> Path:
        return self.cache_dir / "dashboard_payload.json"

    @property
    def results_root(self) -> Path:
        return self.project_root / "results"

    @property
    def runs_root(self) -> Path:
        return self.results_root / "runs"

    @property
    def compare_root(self) -> Path:
        return self.results_root / "compare"

    @property
    def ddq_compare_root(self) -> Path:
        return self.compare_root / DDQ_COMPARE_DIRNAME

    @staticmethod
    def _dedupe_paths(paths: list[Path]) -> list[Path]:
        deduped: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path.resolve())
        return deduped

    @property
    def ddq_compare_artifact_roots(self) -> list[Path]:
        roots: list[Path] = [self.ddq_compare_root]
        if self.ddq_compare_root.exists():
            nested_runs = sorted(
                [
                    path
                    for path in self.ddq_compare_root.iterdir()
                    if path.is_dir() and path.name.lower() != "checkpoints"
                ],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            roots.extend(nested_runs)
        return self._dedupe_paths(roots)

    @property
    def ddq_checkpoint_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        for root in self.ddq_compare_artifact_roots:
            candidates.extend(
                [
                    root / "BranchingDDQ.pt",
                    root / "checkpoints" / "best_model.pt",
                    root / "best_model.pt",
                    root / "checkpoints" / "final_model.pt",
                    root / "final_model.pt",
                ]
            )
        return self._dedupe_paths(candidates)

    @property
    def ddq_config_json_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        for root in self.ddq_compare_artifact_roots:
            candidates.extend(
                [
                    root / "config_active.json",
                    root / "config.json",
                    root / "run_config.json",
                    root / "summary.json",
                ]
            )
        return self._dedupe_paths(candidates)
