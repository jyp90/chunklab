from __future__ import annotations

import functools
import json
import sys
import threading
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypeVar, cast

import pydantic
import typer
import uvicorn
import yaml

import chunklab
from chunklab.core.embedders import MissingApiKeyError
from chunklab.core.questions import build_llm, generate_questions, save_questions
from chunklab.core.runner import (
    FRAMEWORKS,
    ExperimentConfig,
    RunResult,
    check_regression,
    check_thresholds,
    format_table,
    parse_thresholds,
    run_experiment,
    unmatched_combos,
)
from chunklab.core.runner import recommend as recommend_combo
from chunklab.core.runner import snippet as build_snippet
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
    if isinstance(e, pydantic.ValidationError):
        return "; ".join(
            f"{'.'.join(map(str, err['loc']))}: {err['msg'].removeprefix('Value error, ')}"
            for err in e.errors()
        )
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


@app.command(
    help=(
        "Run the chunking × embedding × retrieval matrix from a YAML config and print the "
        "metrics table."
    )
)
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
    json_out: Annotated[
        bool, typer.Option("--json", help="Print the result JSON on stdout and nothing else")
    ] = False,
) -> None:
    cfg = ExperimentConfig.from_yaml(config)
    base_dir = config.resolve().parent
    out = out or base_dir / "result.json"
    questions_path = base_dir / cfg.questions
    if out.resolve() in {config.resolve(), questions_path.resolve()}:
        typer.echo(
            f"error: --out must not overwrite the config or questions file ({out})", err=True
        )
        raise typer.Exit(code=2)
    if not questions_path.exists():
        typer.echo(f"error: questions file not found: {questions_path}", err=True)
        raise typer.Exit(code=2)
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
    if json_out:
        typer.echo(json.dumps(result.to_dict()))
    misses = sum(c.embed_misses for c in result.combos)
    lookups = sum(c.embed_lookups for c in result.combos)
    # In --json mode stdout carries the JSON document alone; the human report goes to stderr.
    typer.echo(format_table(result), err=json_out)
    typer.echo(
        f"\nembeddings: {misses} computed, {lookups - misses} served from cache", err=json_out
    )
    typer.echo(f"saved: {out}", err=json_out)

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
    typer.echo("\nOK", err=json_out)


@app.command(
    "generate-questions",
    help="Sample passages from documents and ask an LLM to write one test question per passage.",
)
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


@app.command(help="Pick the best combo from a result: best hit@k, then precision, then size.")
@_user_errors
def recommend(
    result: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Result JSON")],
    tolerance: Annotated[float, typer.Option("--tolerance")] = 0.05,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    rec = recommend_combo(RunResult.from_json(result), tolerance)
    if rec is None:
        typer.echo(f"error: no usable combo in {result}", err=True)
        raise typer.Exit(code=2)
    if json_out:
        typer.echo(json.dumps({"combo_id": rec.combo_id, "reason": rec.reason}))
    else:
        typer.echo(f"{rec.combo_id} - {rec.reason}")


@app.command(help="Print copy-paste code for one combo in the framework of your choice.")
@_user_errors
def snippet(
    result: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Result JSON")],
    combo: Annotated[str, typer.Option("--combo", help="combo_id from the result")],
    framework: Annotated[
        str, typer.Option("--framework", help=" | ".join(FRAMEWORKS))
    ] = "langchain",
) -> None:
    run_result = RunResult.from_json(result)
    match = next((c for c in run_result.combos if c.combo_id == combo), None)
    if match is None:
        raise ValueError(f"combo '{combo}' not found in {result}")
    typer.echo(build_snippet(match, framework))


@app.command(help="Start the local web UI (documents, questions, experiments, results).")
def ui(
    port: Annotated[int, typer.Option("--port")] = 7860,
    workspace: Annotated[Path, typer.Option("--workspace", help="Project directory")] = Path("."),
    no_browser: Annotated[bool, typer.Option("--no-browser")] = False,
) -> None:
    from chunklab.server.app import create_app  # lazy: fastapi import cost only when needed

    web_app = create_app(workspace.resolve())
    url = f"http://127.0.0.1:{port}/"
    if not no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    typer.echo(f"chunklab ui -> {url}  (workspace: {web_app.state.workspace})", err=True)
    uvicorn.run(web_app, host="127.0.0.1", port=port, log_level="warning")


@app.command(help="Print the chunklab version.")
def version() -> None:
    typer.echo(f"chunklab {chunklab.__version__}")


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
