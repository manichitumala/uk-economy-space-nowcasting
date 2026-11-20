"""End-to-end pipeline: ingestion -> QC & aggregation -> features -> backtests -> analysis.

Usage:
    python run_pipeline.py                 # everything (downloads ~28 GB, keeps ~0.5 GB)
    python run_pipeline.py --from features # rerun from a later stage using cached data
"""
from __future__ import annotations

import argparse
import logging
import runpy

STAGES = [
    ("ingest", ["src.ingestion.ons", "src.ingestion.boundaries", "src.ingestion.satellite"]),
    ("aggregate", ["src.processing.regional_aggregation"]),
    ("features", ["src.features.build_features"]),
    ("train", ["src.models.train", "src.models.predict"]),
    ("analysis", ["src.models.analysis"]),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="ingest", choices=[s for s, _ in STAGES])
    args = ap.parse_args()
    names = [s for s, _ in STAGES]
    for stage, modules in STAGES[names.index(args.start):]:
        for m in modules:
            logging.info("=== %s: %s", stage, m)
            runpy.run_module(m, run_name="__main__")


if __name__ == "__main__":
    main()
