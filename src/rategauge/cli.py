"""Command-line entry points: golden | ingest | smoke | extract | batch | grade."""

import argparse
import logging
from pathlib import Path

from rategauge.goldenset import cbpol
from rategauge.sources import common, ecb, fed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rategauge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    golden = subparsers.add_parser("golden", help="build the CBPOL-derived golden set")
    golden.add_argument("--out", type=Path, default=Path("data/golden"))

    ingest = subparsers.add_parser("ingest", help="enumerate and cache Fed/ECB documents")
    ingest.add_argument("--cache", type=Path, default=Path("data/cache"))
    ingest.add_argument("--catalog", type=Path, default=Path("data/catalog/documents.csv"))
    ingest.add_argument("--bank", choices=["FED", "ECB", "all"], default="all")

    ingest_traps = subparsers.add_parser(
        "ingest-traps", help="enumerate and cache the trap set (documents with no decision)"
    )
    ingest_traps.add_argument("--cache", type=Path, default=Path("data/cache"))
    ingest_traps.add_argument("--catalog", type=Path, default=Path("data/catalog/traps.csv"))

    subparsers.add_parser("smoke", help="verify API keys with one tiny call per provider")

    extract = subparsers.add_parser("extract", help="run schema-constrained extraction")
    extract.add_argument("--model", required=True, help="model key from configs/models.yaml")
    extract.add_argument("--prompt", default="v001", help="prompt version (extract/prompts/)")
    group = extract.add_mutually_exclusive_group(required=True)
    group.add_argument("--docs", help="comma-separated doc_ids")
    group.add_argument("--dev-set", action="store_true", help="run the 12-doc dev subset")

    batch = subparsers.add_parser("batch", help="batch extraction at 50%% token cost")
    batch_sub = batch.add_subparsers(dest="batch_command", required=True)
    submit = batch_sub.add_parser("submit", help="submit one model x prompt batch")
    submit.add_argument("--model", required=True, help="model key from configs/models.yaml")
    submit.add_argument("--prompt", default="v001", help="prompt version (extract/prompts/)")
    submit_group = submit.add_mutually_exclusive_group(required=True)
    submit_group.add_argument("--docs", help="comma-separated doc_ids")
    submit_group.add_argument("--dev-set", action="store_true", help="run the 12-doc dev subset")
    submit_group.add_argument("--all", action="store_true", help="every document in the catalog")
    submit_group.add_argument(
        "--trap-set", action="store_true", help="every document in the trap catalog"
    )
    submit.add_argument(
        "--force",
        action="store_true",
        help="submit even if an uncollected batch for the same model+prompt exists",
    )
    batch_sub.add_parser("status", help="refresh and print batch statuses")
    collect = batch_sub.add_parser("collect", help="collect finished batches into artifacts")
    collect.add_argument("--batch-id", help="one specific batch id (default: all pending)")

    grade = subparsers.add_parser("grade", help="grade artifacts against the golden set")
    grade.add_argument("--models", required=True, help="comma-separated model keys")
    grade.add_argument("--prompt", default="v001", help="prompt version")

    serve = subparsers.add_parser("serve", help="run the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "golden":
        run_golden(args.out)
    elif args.command == "ingest":
        run_ingest(args.cache, args.catalog, args.bank)
    elif args.command == "ingest-traps":
        run_ingest_traps(args.cache, args.catalog)
    elif args.command == "smoke":
        run_smoke()
    elif args.command == "extract":
        run_extract(args.model, args.prompt, args.docs, args.dev_set)
    elif args.command == "batch":
        if args.batch_command == "submit":
            run_batch_submit(
                args.model,
                args.prompt,
                args.docs,
                args.dev_set,
                args.all,
                args.trap_set,
                args.force,
            )
        elif args.batch_command == "status":
            run_batch_status()
        elif args.batch_command == "collect":
            run_batch_collect(args.batch_id)
    elif args.command == "grade":
        run_grade(args.models.split(","), args.prompt)
    elif args.command == "serve":
        run_serve(args.host, args.port)


def run_golden(out_dir: Path) -> None:
    frame = cbpol.build_golden_set(out_dir)
    golden = frame[~frame["excluded"]]
    for ref_area in cbpol.REF_AREAS:
        subset = golden[golden["ref_area"] == ref_area]
        print(
            f"{ref_area}: {len(subset)} golden events "
            f"({subset['effective_date'].min()} .. {subset['effective_date'].max()})"
        )


def run_ingest(cache_dir: Path, catalog_path: Path, bank: str) -> None:
    refs: list[common.DocumentRef] = []
    if bank in ("FED", "all"):
        refs.extend(fed.enumerate_statements())
    if bank in ("ECB", "all"):
        refs.extend(ecb.enumerate_decisions())
    if bank != "all" and catalog_path.exists():
        # A filtered run must not evict the other bank from the shared catalog.
        kept = [ref for ref in common.read_catalog(catalog_path) if ref.bank != bank]
        refs = kept + refs
    _write_and_fetch(refs, catalog_path, cache_dir)


def run_ingest_traps(cache_dir: Path, catalog_path: Path) -> None:
    refs = list(fed.enumerate_minutes()) + list(ecb.enumerate_non_decisions())
    _write_and_fetch(refs, catalog_path, cache_dir)


def _write_and_fetch(
    refs: list[common.DocumentRef], catalog_path: Path, cache_dir: Path
) -> None:
    common.write_catalog(refs, catalog_path)
    documents = common.fetch_documents(refs, common.DocumentCache(cache_dir))
    missing = sorted(ref.doc_id for ref in refs if ref.doc_id not in documents)
    print(f"{len(refs)} documents enumerated; {len(documents)} cached; {len(missing)} failed")
    for doc_id in missing:
        print(f"  MISSING: {doc_id}")
    if missing:
        raise SystemExit(1)  # let automation distinguish partial from complete ingests


SMOKE_DOCUMENT = (
    "Source: Federal Reserve (FOMC) press release\n\n<document>\nThe Committee decided to "
    "maintain the target range for the federal funds rate at 4-1/4 to 4-1/2 percent.\n</document>"
)


def run_smoke() -> None:
    """One tiny schema-constrained call per provider to verify keys and plumbing."""
    from rategauge.config import load_credentials, load_models
    from rategauge.extract.clients import CLIENT_BUILDERS, EXTRACTORS
    from rategauge.extract.runner import load_prompt
    from rategauge.schema import RateDecision

    load_credentials()
    prompt = load_prompt("v001")
    models = load_models()
    by_provider = {model.provider: model for model in models.values()}  # cheapest last wins
    for provider in sorted(by_provider):
        model = min(
            (m for m in models.values() if m.provider == provider),
            key=lambda m: m.input_usd_per_mtok,
        )
        raw = EXTRACTORS[provider](
            CLIENT_BUILDERS[provider](), model.model_id, prompt, SMOKE_DOCUMENT
        )
        record = RateDecision.model_validate_json(raw.payload)
        cost = model.cost_usd(raw.input_tokens, raw.output_tokens)
        print(
            f"{provider:<10} {model.model_id:<28} OK  action={record.action} "
            f"range={record.target_range_lower_pct}-{record.target_range_upper_pct} "
            f"tokens={raw.input_tokens}+{raw.output_tokens} cost=${cost:.5f} "
            f"latency={raw.latency_ms}ms"
        )


def run_extract(model_key: str, prompt_version: str, docs: str | None, dev_set: bool) -> None:
    from rategauge.extract.runner import DEV_SET, run_extraction

    if dev_set:
        doc_ids = DEV_SET
    else:
        doc_ids = tuple(part.strip() for part in docs.split(",") if part.strip())
    rows = run_extraction(model_key, prompt_version, doc_ids)
    ok = sum(1 for row in rows if row["ok"])
    cost = sum(row["cost_usd"] for row in rows)
    print(f"{len(rows)} documents, {ok} valid records, {len(rows) - ok} failures, ${cost:.4f}")


# Rough per-document output size for pre-submit cost estimates (dev-set average
# was ~200; padded up so the printed estimate errs on the expensive side).
EST_OUTPUT_TOKENS_PER_DOC = 300


def run_batch_submit(
    model_key: str,
    prompt_version: str,
    docs: str | None,
    dev_set: bool,
    all_docs: bool,
    trap_set: bool,
    force: bool,
) -> None:
    from rategauge import corpus
    from rategauge.config import load_models
    from rategauge.extract.batch import submit_batch
    from rategauge.extract.runner import DEV_SET

    if dev_set:
        doc_ids = DEV_SET
    elif all_docs:
        doc_ids = tuple(ref.doc_id for ref in common.read_catalog(corpus.CATALOG_PATH))
    elif trap_set:
        doc_ids = tuple(ref.doc_id for ref in common.read_catalog(corpus.TRAPS_CATALOG_PATH))
    else:
        doc_ids = tuple(part.strip() for part in docs.split(",") if part.strip())
    state = submit_batch(model_key, prompt_version, doc_ids, force=force)
    model = load_models()[model_key]
    estimated_cost = model.cost_usd(
        state["estimated_input_tokens"], EST_OUTPUT_TOKENS_PER_DOC * len(doc_ids), batch=True
    )
    print(
        f"submitted {state['batch_id']} ({model_key}, {len(doc_ids)} documents); "
        f"~{state['estimated_input_tokens']:,} input tokens, "
        f"rough batch cost ~${estimated_cost:.2f}"
    )


def run_batch_status() -> None:
    from rategauge.extract import batch

    paths = batch.list_states()
    if not paths:
        print("no batches submitted")
        return
    for path in paths:
        state = batch.refresh_status(path)
        status = "collected" if state["collected"] else state["status"]
        print(
            f"{state['batch_id']:<42} {state['model_key']:<18} {state['prompt_version']:<6} "
            f"{len(state['documents']):>4} docs  {status}"
        )


def run_batch_collect(batch_id: str | None) -> None:
    from rategauge.extract import batch

    paths = batch.list_states()
    if batch_id:
        paths = [path for path in paths if path.stem == batch_id]
        if not paths:
            raise SystemExit(f"no state file for batch {batch_id}")
    pending, errors = 0, 0
    for path in paths:
        state = batch.load_state(path)
        if state["collected"]:
            if batch_id:
                print(f"{state['batch_id']} already collected")
            continue
        try:
            rows = batch.collect_batch(path)
        except batch.BatchNotReady as not_ready:
            print(f"not ready: {not_ready}")
            pending += 1
            continue
        except Exception as error:  # one broken batch must not block the rest
            print(f"ERROR {path.stem}: {error}")
            errors += 1
            continue
        ok = sum(1 for row in rows if row["ok"])
        state = batch.load_state(path)
        missing = len(state["missing_doc_ids"])
        cost = sum(row["cost_usd"] for row in rows)
        print(
            f"{state['batch_id']} ({state['model_key']}): {len(rows)} results, {ok} valid, "
            f"{len(rows) - ok} failures, {missing} missing, ${cost:.4f}"
        )
    if pending:
        print(f"{pending} batch(es) still processing - rerun `rategauge batch collect` later")
    if errors:
        raise SystemExit(1)


def run_grade(model_keys: list[str], prompt_version: str) -> None:
    import json

    from rategauge.evalsuite import grader, metrics
    from rategauge.schema import SCHEMA_VERSION

    golden = grader.GoldenSeries.load_all()
    scorecards_dir = Path("eval/scorecards")
    graded_dir = Path("eval/graded")
    summaries, graded_by_model = [], {}
    for model_key in (key.strip() for key in model_keys if key.strip()):
        stem = f"{model_key}__{prompt_version}__{SCHEMA_VERSION}"
        artifact_rows = grader.load_artifact(Path("eval/runs") / f"{stem}.jsonl")
        graded = grader.grade_rows(artifact_rows, golden)
        graded_by_model[model_key] = graded
        summary = metrics.summarize(graded, artifact_rows)
        summaries.append(summary)

        graded_dir.mkdir(parents=True, exist_ok=True)
        with (graded_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as handle:
            for row in graded:
                handle.write(json.dumps(row) + "\n")
        scorecards_dir.mkdir(parents=True, exist_ok=True)
        (scorecards_dir / f"{stem}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    print(metrics.league_table(summaries))
    if len(graded_by_model) > 1:
        print("\nPairwise McNemar (action correctness):")
        for test in metrics.pairwise_mcnemar(graded_by_model):
            print(
                f"  {test['model_a']} vs {test['model_b']}: "
                f"only-a={test['only_a_correct']} only-b={test['only_b_correct']} "
                f"p={test['p_value']} (n={test['n_paired']})"
            )


def run_serve(host: str, port: int) -> None:
    import uvicorn

    uvicorn.run("rategauge.serve.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
