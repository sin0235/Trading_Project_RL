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
REPLAY_PAYLOAD_SCHEMA_VERSION = 3

# Compare artifacts cho đường policy cố định trên dashboard.
# Ưu tiên DDQ mới, fallback sang Branching DDQ cũ nếu không có.
COMPARE_ARTIFACT_PRIORITY: tuple[tuple[str, str], ...] = (
    ("DDQ", "DDQ"),
    ("BranchingDDQ_LSTM", "Branching DDQ"),
)
DDQ_COMPARE_LABEL = "DDQ"


def compare_label_for_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    parts = {part.lower() for part in resolved.parts}
    for dirname, label in COMPARE_ARTIFACT_PRIORITY:
        if dirname.lower() in parts:
            return label
    return DDQ_COMPARE_LABEL


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
        return self.compare_root / COMPARE_ARTIFACT_PRIORITY[0][0]

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
        roots: list[Path] = []
        for dirname, _label in COMPARE_ARTIFACT_PRIORITY:
            compare_root = self.compare_root / dirname
            roots.append(compare_root)
            if compare_root.exists():
                nested_runs = sorted(
                    [
                        path
                        for path in compare_root.iterdir()
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
            if root.exists():
                candidates.extend(sorted(root.glob("best_model*.pt")))
                candidates.extend(sorted((root / "checkpoints").glob("best_model*.pt")))
                candidates.extend(sorted(root.glob("final_model*.pt")))
                candidates.extend(sorted((root / "checkpoints").glob("final_model*.pt")))
                candidates.extend(sorted(root.glob("checkpoint_*.pt")))
                candidates.extend(sorted((root / "checkpoints").glob("checkpoint_*.pt")))
                candidates.extend(sorted(root.glob("*.pt")))
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
