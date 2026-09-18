from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypeVar, cast

import typer
import yaml

import chunklab
from chunklab.core.embedders import MissingApiKeyError
from chunklab.core.questions import build_llm, generate_questions, save_questions
from chunklab.core.runner import (
    ExperimentConfig,
    RunResult,
    check_regression,
    check_thresholds,
    format_table,
    parse_thresholds,
    run_experiment,
    unmatched_combos,
)
from chunklab.core.text import load_documents

app = typer.Typer(
    help="Benchmark your chunking before you ship it.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

# Exit 2 means "you asked for something impossible": bad config, bad flag, missing
# file, missing credentials. Exit 1 is reserved for a run that completed but failed
# a threshold or regression check. Anything else is a bug and keeps its traceback.
_USER_ERRORS = (
    MissingApiKeyError,
    ValueError,
    KeyError,
    FileNotFoundError,
    ImportError,
    yaml.YAMLError,
)

_F = TypeVar("_F", bound=Callable[..., None])


def _clean(e: Exception) -> str:
    # str(KeyError("x")) is repr'd as "'x'"; every other exception reads fine.
    if isinstance(e, KeyError) and e.args:
        return str(e.args[0])
    return str(e)


def _user_errors(fn: _F) -> _F:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _USER_ERRORS as e:
            typer.echo(f"error: {_clean(e)}", err=True)
            raise typer.Exit(code=2) from None

    return cast(_F, wrapper)


@app.command()
def version() -> None:
    typer.echo(f"chunklab {chunklab.__version__}")


@app.command()
@_user_errors
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

    result = run_experiment(
        cfg,
        base_dir,
        progress=progress,
        warn=lambda m: typer.echo(f"  warning: {m}", err=True),
    )
    result.to_json(out)
    typer.echo(format_table(result))
    misses = sum(c.embed_misses for c in result.combos)
    lookups = sum(c.embed_lookups for c in result.combos)
    typer.echo(f"\nembeddings: {misses} computed, {lookups - misses} served from cache")
    typer.echo(f"saved: {out}")

    violations = check_thresholds(result, thresholds)
    if baseline is not None:
        base = RunResult.from_json(baseline)
        missing = unmatched_combos(result, base)
        if missing:
            shown = ", ".join(missing[:3]) + ("…" if len(missing) > 3 else "")
            typer.echo(f"note: {len(missing)} combo(s) not in baseline, skipped: {shown}", err=True)
        violations += check_regression(result, base, max_drop)
    if violations:
        typer.echo("\nFAILED:", err=True)
        for v in violations:
            typer.echo(f"  - {v}", err=True)
        raise typer.Exit(code=1)
    typer.echo("\nOK")


@app.command("generate-questions")
@_user_errors
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
        docs,
        build_llm(llm),
        per_doc=per_doc,
        seed=seed,
        min_len=min_len,
        max_len=max_len,
        warn=lambda m: typer.echo(f"warning: {m}", err=True),
    )
    if not questions:
        typer.echo("error: no questions generated", err=True)
        raise typer.Exit(code=2)
    save_questions(questions, out)
    typer.echo(f"generated {len(questions)} questions from {len(docs)} document(s) -> {out}")


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
