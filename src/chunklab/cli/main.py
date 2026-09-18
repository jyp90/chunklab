from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

import chunklab
from chunklab.core.questions import build_llm, generate_questions, save_questions
from chunklab.core.runner import (
    ExperimentConfig,
    RunResult,
    check_regression,
    check_thresholds,
    format_table,
    parse_thresholds,
    run_experiment,
)
from chunklab.core.text import load_documents

app = typer.Typer(help="Benchmark your chunking before you ship it.", no_args_is_help=True)


@app.command()
def version() -> None:
    typer.echo(f"chunklab {chunklab.__version__}")


@app.command()
def run(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Experiment YAML")],
    out: Annotated[Path | None, typer.Option("--out", help="Result JSON path")] = None,
    baseline: Annotated[Path | None, typer.Option("--baseline", exists=True)] = None,
    max_drop: Annotated[float, typer.Option("--max-drop")] = 0.05,
    fail_below: Annotated[
        list[str] | None, typer.Option("--fail-below", help="METRIC=VALUE, repeatable")
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
) -> None:
    cfg = ExperimentConfig.from_yaml(config)
    base_dir = config.resolve().parent
    out = out or base_dir / "result.json"
    thresholds = parse_thresholds(fail_below or [])

    def progress(combo_id: str) -> None:
        if not quiet:
            typer.echo(f"  done: {combo_id}", err=True)

    result = run_experiment(cfg, base_dir, progress=progress)
    result.to_json(out)
    typer.echo(format_table(result))
    typer.echo(f"\nsaved: {out}")

    try:
        violations = check_thresholds(result, thresholds)
    except ValueError as e:
        typer.echo(f"\nERROR: {e}", err=True)
        raise typer.Exit(code=2) from None
    if baseline is not None:
        violations += check_regression(result, RunResult.from_json(baseline), max_drop)
    if violations:
        typer.echo("\nFAILED:", err=True)
        for v in violations:
            typer.echo(f"  - {v}", err=True)
        raise typer.Exit(code=1)
    typer.echo("\nOK")


@app.command("generate-questions")
def generate_questions_cmd(
    paths: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out")],
    llm: Annotated[
        str, typer.Option("--llm", help="fake | openai[:model] | gemini[:model]")
    ] = "openai",
    per_doc: Annotated[int, typer.Option("--per-doc")] = 10,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    min_len: Annotated[int, typer.Option("--min-len")] = 200,
    max_len: Annotated[int, typer.Option("--max-len")] = 600,
) -> None:
    docs = load_documents(paths)
    questions = generate_questions(
        docs, build_llm(llm), per_doc=per_doc, seed=seed, min_len=min_len, max_len=max_len
    )
    save_questions(questions, out)
    typer.echo(f"generated {len(questions)} questions from {len(docs)} document(s) -> {out}")


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
