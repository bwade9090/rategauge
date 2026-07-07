"""Command-line entry points: golden | ingest | smoke | extract."""

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

    subparsers.add_parser("smoke", help="verify API keys with one tiny call per provider")

    extract = subparsers.add_parser("extract", help="run schema-constrained extraction")
    extract.add_argument("--model", required=True, help="model key from configs/models.yaml")
    extract.add_argument("--prompt", default="v001", help="prompt version (extract/prompts/)")
    group = extract.add_mutually_exclusive_group(required=True)
    group.add_argument("--docs", help="comma-separated doc_ids")
    group.add_argument("--dev-set", action="store_true", help="run the 12-doc dev subset")

    grade = subparsers.add_parser("grade", help="grade artifacts against the golden set")
    grade.add_argument("--models", required=True, help="comma-separated model keys")
    grade.add_argument("--prompt", default="v001", help="prompt version")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "golden":
        run_golden(args.out)
    elif args.command == "ingest":
        run_ingest(args.cache, args.catalog, args.bank)
    elif args.command == "smoke":
        run_smoke()
    elif args.command == "extract":
        run_extract(args.model, args.prompt, args.docs, args.dev_set)
    elif args.command == "grade":
        run_grade(args.models.split(","), args.prompt)


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


if __name__ == "__main__":
    main()
