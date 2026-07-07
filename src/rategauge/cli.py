"""Command-line entry points: ``rategauge golden`` and ``rategauge ingest``."""

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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "golden":
        run_golden(args.out)
    elif args.command == "ingest":
        run_ingest(args.cache, args.catalog, args.bank)


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


if __name__ == "__main__":
    main()
