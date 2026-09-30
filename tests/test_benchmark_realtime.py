import subprocess
from pathlib import Path

from scripts import benchmark_realtime


def test_default_variants_are_orthogonal_and_exclude_rejected_profiles() -> None:
    assert benchmark_realtime.DEFAULT_VARIANTS == (
        "baseline",
        "fp32-batch2",
        "fp16",
        "fp16-batch2",
    )


def test_modal_subprocess_forces_utf8(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **options):
        captured["command"] = command
        captured["options"] = options
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr(benchmark_realtime.subprocess, "run", fake_run)

    result = benchmark_realtime.run_variant("baseline", "video.mp4", 5.0, "test")

    options = captured["options"]
    assert isinstance(options, dict)
    environment = options["env"]
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert environment["PYTHONUTF8"] == "1"
    assert "--cpu=2.0" in captured["command"]
    assert result["exit_code"] == 1


def test_summary_path_skips_every_incompatible_existing_version(
    tmp_path: Path,
) -> None:
    base = tmp_path / "summary.csv"
    base.write_text("old,header\n", encoding="utf-8")
    (tmp_path / "summary_v2.csv").write_text("also,old\n", encoding="utf-8")

    resolved = benchmark_realtime._compatible_summary_path(base, ["new", "header"])

    assert resolved == tmp_path / "summary_v3.csv"
