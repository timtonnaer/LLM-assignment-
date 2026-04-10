"""
Script 03b: Collect Batch Results
Polls an existing Anthropic batch and saves results when done.
Usage:
  python scripts/03b_collect_batch.py --batch-id msgbatch_01Y7ngkvowMPxFDaF8GZT4PT
"""

import json
import time
import logging
import argparse
import re
import pandas as pd
from pathlib import Path

OUTPUTS_DIR    = Path("/Users/timtonnaer/risk_project/outputs")
CLASSIFIED_DIR = Path("/Users/timtonnaer/risk_project/outputs/classified")
LOG_FILE       = CLASSIFIED_DIR / "classification_log.jsonl"
DIFF_INDEX_CSV = OUTPUTS_DIR / "all_diff_index.csv"
BATCH_SIZE     = 40
POLL_INTERVAL  = 30

VALID_RISK_TYPES = {"operational", "cyber", "regulatory", "supply_chain",
                    "financing", "macroeconomic", "other"}
VALID_NATURES    = {"new_risk", "expanded_existing", "boilerplate"}
VALID_STYLES     = {"concrete", "vague"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_BOILERPLATE_RE = re.compile(
    r"there can be no assurance|no assurance can be given|forward.looking statements?"
    r"|we cannot (assure|guarantee|predict|be certain)|in the ordinary course of business"
    r"|from time to time|^\s*item\s+\d|risk factors? (are|include|may)"
    r"|described (elsewhere|above|below) in this|incorporated by reference"
    r"|for (further|more|additional) information|see (also|note|part|item)\s+\d"
    r"|may\s+(differ|vary)\s+materially|we (are|have been) subject to"
    r"|there (is|are) no guarantee|we make no (representation|warranty|assurance)"
    r"|actual results (could|may|might|will) differ|we undertake no obligation"
    r"|including but not limited to|among other things|without limitation"
    r"|the following risk factors|you should carefully consider",
    re.IGNORECASE,
)
BOILERPLATE_LABEL = {"risk_type": "other", "nature": "boilerplate", "style": "vague"}
MIN_WORDS = 15


def normalize(item):
    return {
        "risk_type": item.get("risk_type") if item.get("risk_type") in VALID_RISK_TYPES else "other",
        "nature":    item.get("nature")    if item.get("nature")    in VALID_NATURES    else "boilerplate",
        "style":     item.get("style")     if item.get("style")     in VALID_STYLES     else "vague",
    }


def unknown():
    return {"risk_type": "unknown", "nature": "unknown", "style": "unknown"}


def parse_response(raw, n):
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            padded = result[:n] + [{}] * max(0, n - len(result))
            return [normalize(item) for item in padded]
    except json.JSONDecodeError:
        pass
    return [unknown()] * n


def prefilter(raw_sentences):
    prefiltered, to_classify = [], []
    for s in raw_sentences:
        if len(s.split()) < MIN_WORDS:
            continue
        if _BOILERPLATE_RE.search(s):
            prefiltered.append({"sentence": s, **BOILERPLATE_LABEL})
        else:
            to_classify.append(s)
    return prefiltered, to_classify


def output_path(row):
    return CLASSIFIED_DIR / f"{row['cik']}_{row['year_old']}_{row['year_new']}_classified.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True, help="Anthropic batch ID to collect")
    args = parser.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    # Poll until complete
    while True:
        try:
            status = client.messages.batches.retrieve(args.batch_id)
            counts = status.request_counts
            logger.info(f"Status: {status.processing_status} | "
                        f"succeeded={counts.succeeded} processing={counts.processing} "
                        f"errored={counts.errored}")
            if status.processing_status == "ended":
                break
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.warning(f"Poll error: {e} — retrying in 30s...")
            time.sleep(POLL_INTERVAL)

    logger.info("Batch complete. Loading diff index...")
    index = pd.read_csv(DIFF_INDEX_CSV)

    # Rebuild diff_meta from index
    diff_meta = {}
    for _, row in index.iterrows():
        out = output_path(row)
        if out.exists():
            continue
        try:
            with open(row["filepath"]) as f:
                diff = json.load(f)
        except Exception:
            continue
        raw_sentences = diff.get("added", [])
        if len(raw_sentences) > 500:
            with open(out, "w") as f:
                json.dump([], f)
            continue
        prefiltered, to_classify = prefilter(raw_sentences)
        prefix = f"{row['cik']}_{row['year_old']}_{row['year_new']}"
        chunks = []
        for chunk_idx, i in enumerate(range(0, len(to_classify), BATCH_SIZE)):
            chunks.append((chunk_idx, to_classify[i:i + BATCH_SIZE]))
        diff_meta[prefix] = {"row": row, "prefiltered": prefiltered, "chunks": chunks}

    logger.info(f"Collecting results for {len(diff_meta)} diffs...")
    chunk_results = {}
    error_count = 0
    for result in client.messages.batches.results(args.batch_id):
        if result.result.type == "succeeded":
            raw = result.result.message.content[0].text
            prefix, chunk_idx_str = result.custom_id.rsplit("__", 1)
            chunk_idx = int(chunk_idx_str)
            if prefix in diff_meta and chunk_idx < len(diff_meta[prefix]["chunks"]):
                chunk = diff_meta[prefix]["chunks"][chunk_idx][1]
                chunk_results[result.custom_id] = parse_response(raw, len(chunk))
        else:
            error_count += 1
            prefix, chunk_idx_str = result.custom_id.rsplit("__", 1)
            chunk_idx = int(chunk_idx_str)
            if prefix in diff_meta and chunk_idx < len(diff_meta[prefix]["chunks"]):
                chunk = diff_meta[prefix]["chunks"][chunk_idx][1]
                chunk_results[result.custom_id] = [unknown()] * len(chunk)

    if error_count:
        logger.warning(f"{error_count} requests errored.")

    total_sentences, total_diffs = 0, 0
    for prefix, meta in diff_meta.items():
        row = meta["row"]
        results = list(meta["prefiltered"])
        for chunk_idx, chunk in meta["chunks"]:
            custom_id = f"{prefix}__{chunk_idx}"
            classifications = chunk_results.get(custom_id, [unknown()] * len(chunk))
            for sentence, cls in zip(chunk, classifications):
                results.append({"sentence": sentence, **cls})
        out = output_path(row)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        total_sentences += len(results)
        total_diffs += 1

    logger.info(f"\nDone. Saved {total_diffs} diffs | {total_sentences} sentences total.")


if __name__ == "__main__":
    main()
