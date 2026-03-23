import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTE_PATHS = [
    REPO_ROOT / "notebooks" / "RL_PROJECT.zpln",
    Path(r"D:\Programs\zeppelin-0.12.0-bin-all\docker-data\notebook\RL_PROJECT_2MM3B3U1P.zpln"),
]


REPLACEMENTS = [
    (
        "PPO Trading Dashboard | 2026-03-23-fix12",
        "PPO Trading Dashboard",
    ),
    (
        "Replay checkpoint | 2026-03-23-fix12",
        "Replay checkpoint",
    ),
    (
        "Dashboard này đọc trực tiếp cache local nhưng giao diện luôn khóa vào best PPO run. Phần replay checkpoint chỉ bám đúng run đó; phần compare thuật toán sẽ tự bật khi repo có artifact DQN/DDQ thật.",
        "Theo dõi kết quả huấn luyện và mô phỏng giao dịch theo ngày trên dữ liệu gần nhất.",
    ),
    (
        '<span class="rl-chip">Run mặc định: ${esc(debug.default_run_id || run.run_id)}</span><span class="rl-chip">Build payload: ${esc(debug.build_tag || \'n/a\')}</span><span class="rl-chip">Replay: ${esc(replay.status || \'n/a\')}</span><span class="rl-chip">Nguồn replay: ${esc(replayRun.data_source_label || \'vnstock\')}</span><span class="rl-chip">Chế độ train: Best PPO run</span><span class="rl-chip">Compare DQN: ${esc(algo.status || "n/a")}</span>',
        '<span class="rl-chip">Run: ${esc(run.run_id || debug.default_run_id || "N/A")}</span><span class="rl-chip">Nguồn replay: ${esc(replayRun.data_source_label || "vnstock")}</span>',
    ),
    (
        '<label class="rl-label">Best PPO run</label>',
        '<label class="rl-label">Run huấn luyện</label>',
    ),
    (
        "return '<section class=\"rl-panel\"><div class=\"rl-kicker\">PPO vs DQN/DDQ</div><h3>So sánh best run theo thuật toán</h3><p>'+esc((algo && algo.message) || 'Chưa có artifact DQN/DDQ để so sánh.')+'</p><span class=\"rl-alert\">'+esc((algo && algo.message) || 'Chưa có artifact DQN/DDQ trong results/runs.')+'</span></section>';",
        "return '<section class=\"rl-panel\"><div class=\"rl-kicker\">PPO vs DQN/DDQ</div><h3>So sánh theo thuật toán</h3><p>Chưa có dữ liệu DQN/DDQ.</p></section>';",
    ),
    (
        "return '<section class=\"rl-panel\"><div class=\"rl-kicker\">PPO vs DQN/DDQ</div><h3>So sánh best run theo thuật toán</h3><p>Dashboard đang khóa vào best PPO run. Nếu trong repo có artifact DDQ/DQN, phần này sẽ đặt best DQN/DDQ cạnh best PPO để so nhanh hiệu quả và rủi ro.</p><div class=\"rl-compare-grid\">' +",
        "return '<section class=\"rl-panel\"><div class=\"rl-kicker\">PPO vs DQN/DDQ</div><h3>So sánh theo thuật toán</h3><div class=\"rl-compare-grid\">' +",
    ),
    (
        "<span>Best PPO run</span>",
        "<span>PPO</span>",
    ),
    (
        "<span>Best DQN/DDQ run</span>",
        "<span>DQN/DDQ</span>",
    ),
]


def patch_note(path: Path) -> bool:
    if not path.exists():
        return False
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for paragraph in notebook.get("paragraphs") or []:
        if paragraph.get("title") != "HTML Dashboard":
            continue
        text = paragraph.get("text") or ""
        original = text
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        if text != original:
            paragraph["text"] = text
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    return changed


def main() -> None:
    result = {}
    for note_path in NOTE_PATHS:
        result[str(note_path)] = patch_note(note_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
