"""
Script 03c: Fix Unknown Classifications
Only resubmits sentences that received 'unknown' labels — not entire diffs.
This is ~75% cheaper than rerunning the full classification.

Usage:
  python scripts/03c_fix_unknowns.py
"""

import json
import time
import logging
import re
import pandas as pd
from pathlib import Path

CLASSIFIED_DIR = Path("/Users/timtonnaer/risk_project/outputs/classified")
BATCH_SIZE     = 60    # larger batches = less overhead
MAX_TOKENS     = 500   # unknown sentences are short; don't need 800
POLL_INTERVAL  = 30

VALID_RISK_TYPES = {"operational", "cyber", "regulatory", "supply_chain",
                    "financing", "macroeconomic", "other"}
VALID_NATURES    = {"new_risk", "expanded_existing", "boilerplate"}
VALID_STYLES     = {"concrete", "vague"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a financial disclosure analyst specializing in SEC 10-K risk factor classification. "
    "Respond only with valid JSON arrays, no prose."
)

PROMPT_TEMPLATE = """Classify each of the following newly added/modified risk factor texts from SEC 10-K filings.

For EACH item return one JSON object with these exact keys:
  "risk_type": one of operational|cyber|regulatory|supply_chain|financing|macroeconomic|other
  "nature":    one of new_risk|expanded_existing|boilerplate
  "style":     one of concrete|vague

Definitions:
- risk_type: primary category. Use "macroeconomic" for market/economic/inflation risks.
- nature: "new_risk" = risk not previously disclosed; "expanded_existing" = materially expanded prior risk; "boilerplate" = formulaic with minimal informational value.
- style: "concrete" = specific, measurable, named; "vague" = ambiguous, hedged, non-committal.

Respond with a JSON ARRAY of {n} objects in the same order as the inputs. Nothing else.

Texts:
{texts}"""


def normalize(item):
    return {
        "risk_type": item.get("risk_type") if item.get("risk_type") in VALID_RISK_TYPES else "other",
        "nature":    item.get("nature")    if item.get("nature")    in VALID_NATURES    else "boilerplate",
        "style":     item.get("style")     if item.get("style")     in VALID_STYLES     else "vague",
    }


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
    # fallback: try boilerplate for all
    return [{"risk_type": "other", "nature": "boilerplate", "style": "vague"}] * n


def build_prompt(sentences):
    texts = "\n".join(f"{i+1}. {s.replace(chr(34), chr(39))}" for i, s in enumerate(sentences))
    return PROMPT_TEMPLATE.format(n=len(sentences), texts=texts)


def main():
    import anthropic
    client = anthropic.Anthropic()

    # ── Step 1: find all sentences that need fixing ───────────────────────────
    files_to_fix = {}   # filepath → list of (index_in_file, sentence)

    for f in sorted(CLASSIFIED_DIR.glob("*_classified.json")):
        with open(f) as fp:
            data = json.load(fp)
        if not data:
            continue
        unknowns = [(i, d["sentence"]) for i, d in enumerate(data)
                    if d.get("risk_type") == "unknown"]
        if unknowns:
            files_to_fix[f] = unknowns

    total_unknown = sum(len(v) for v in files_to_fix.values())
    logger.info(f"Files with unknowns: {len(files_to_fix)}")
    logger.info(f"Total unknown sentences to fix: {total_unknown}")

    if not files_to_fix:
        logger.info("Nothing to fix!")
        return

    # ── Step 2: build batch requests ─────────────────────────────────────────
    pending = []
    request_map = {}   # custom_id → (filepath, [(index, sentence), ...])

    for filepath, unknowns in files_to_fix.items():
        stem = filepath.stem.replace("_classified", "")
        for chunk_idx, i in enumerate(range(0, len(unknowns), BATCH_SIZE)):
            chunk = unknowns[i:i + BATCH_SIZE]
            sentences = [s for _, s in chunk]
            custom_id = f"{stem}__{chunk_idx}"
            request_map[custom_id] = (filepath, chunk)
            pending.append({
                "custom_id": custom_id,
                "params": {
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": MAX_TOKENS,
                    "system": [{
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    "messages": [{"role": "user", "content": build_prompt(sentences)}],
                }
            })

    logger.info(f"Submitting {len(pending)} requests to Batch API...")
    batch = client.messages.batches.create(requests=pending)
    batch_id = batch.id
    logger.info(f"Batch ID: {batch_id}")

    # ── Step 3: poll ──────────────────────────────────────────────────────────
    while True:
        try:
            status = client.messages.batches.retrieve(batch_id)
            counts = status.request_counts
            logger.info(f"Status: {status.processing_status} | "
                        f"succeeded={counts.succeeded} processing={counts.processing} "
                        f"errored={counts.errored}")
            if status.processing_status == "ended":
                break
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.warning(f"Poll error: {e} — retrying...")
            time.sleep(POLL_INTERVAL)

    # ── Step 4: collect and patch files ──────────────────────────────────────
    logger.info("Collecting results and patching files...")

    # Group results by filepath
    patches = {}   # filepath → {index: new_classification}
    for result in client.messages.batches.results(batch_id):
        filepath, chunk = request_map[result.custom_id]
        sentences = [s for _, s in chunk]
        if result.result.type == "succeeded":
            classifications = parse_response(
                result.result.message.content[0].text, len(sentences))
        else:
            classifications = [{"risk_type": "other", "nature": "boilerplate", "style": "vague"}] * len(sentences)

        if filepath not in patches:
            patches[filepath] = {}
        for (orig_idx, _), cls in zip(chunk, classifications):
            patches[filepath][orig_idx] = cls

    # Apply patches
    fixed_files, fixed_sentences = 0, 0
    for filepath, patch in patches.items():
        with open(filepath) as fp:
            data = json.load(fp)
        for idx, cls in patch.items():
            data[idx].update(cls)
            fixed_sentences += 1
        with open(filepath, "w") as fp:
            json.dump(data, fp, indent=2)
        fixed_files += 1

    logger.info(f"\nDone. Fixed {fixed_sentences} sentences across {fixed_files} files.")

    # ── Step 5: report remaining unknowns ────────────────────────────────────
    still_unknown = 0
    for f in CLASSIFIED_DIR.glob("*_classified.json"):
        with open(f) as fp:
            data = json.load(fp)
        still_unknown += sum(1 for d in data if d.get("risk_type") == "unknown")
    logger.info(f"Remaining unknowns after fix: {still_unknown}")


if __name__ == "__main__":
    main()
