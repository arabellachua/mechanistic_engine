"""Refresh PubMed only, using all mapped pairs in a saved raw trial corpus.

Existing datasets/models remain unchanged. Resume by repeating the exact command.
"""

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from pubmed_client import query_pubmed_pair
from run_pipeline import filter_trials, normalize_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-trials", type=Path, default=Path("mechanistic_engine_output/data/raw_trials.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", default=date.today().isoformat())
    args = parser.parse_args()
    date.fromisoformat(args.as_of)
    raw = args.raw_trials.read_bytes()
    root = Path(__file__).resolve().parent
    identity = {"input_sha256": hashlib.sha256(raw).hexdigest(), "as_of": args.as_of,
                "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ["pubmed_client.py", "run_pipeline.py", "drug_target_db.py"]}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = args.output_dir / "input_manifest.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Source, code or date changed; choose a new output directory.")
    elif any(args.output_dir.iterdir()):
        raise ValueError("Output directory contains unrelated files; choose a new directory.")
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")

    studies = json.loads(raw)
    trials = normalize_trials(filter_trials(studies))
    mapped = trials.loc[~trials.primary_target.str.casefold().isin(["unknown", "unmapped", ""])]
    pairs = sorted(set(zip(mapped.primary_target, mapped.disease)))
    journal = args.output_dir / "pair_attempts.jsonl"
    results = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            results[(row["primary_target"], row["disease"])] = row
    failures = 0
    interrupted = False
    try:
        with journal.open("a") as handle:
            for i, (symbol, disease) in enumerate(pairs, 1):
                if results.get((symbol, disease), {}).get("pubmed_status") == "complete":
                    continue
                row = {"primary_target": symbol, "disease": disease,
                       "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                       **query_pubmed_pair(symbol, disease, as_of=args.as_of)}
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                results[(symbol, disease)] = row
                failures = failures + 1 if row["pubmed_status"] == "failed" else 0
                print(f"[{i}/{len(pairs)}] {symbol} / {disease}: {row['pubmed_status']}", flush=True)
                if failures >= 3:
                    print("Three consecutive complete failures; stopping. Rerun to resume.", flush=True)
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; saving completed progress for resume.", flush=True)
    finally:
        rows = [results[p] for p in pairs if p in results]
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output_dir / "pubmed_pair_features.csv", index=False)
        complete = sum(row["pubmed_status"] == "complete" for row in rows)
        summary = {"raw_trials": len(studies), "filtered_trials": len(trials),
                   "mapped_trials": len(mapped), "unknown_target_trials": len(trials)-len(mapped),
                   "unique_pairs": len(pairs), "attempted_pairs": len(rows), "complete_pairs": complete,
                   "incomplete_or_unattempted_pairs": len(pairs)-complete,
                   "as_of": args.as_of, "interrupted": interrupted,
                   "existing_model_outputs_modified": False}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if complete == len(pairs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
