from pathlib import Path

from typer.testing import CliRunner

from chunklab.cli.main import app

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_example_runs_offline(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(ROOT / "examples" / "quickstart")
    r = CliRunner().invoke(app, ["run", "exp.yaml", "--out", str(tmp_path / "r.json"), "--quiet"])
    assert r.exit_code == 0, r.output
    assert "markdown(chunk_size=512)|fake|k=3|hybrid=False" in r.stdout
    assert (tmp_path / "r.json").exists()
