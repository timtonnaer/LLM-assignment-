"""
Script 03: LLM Classification
Classifies each added sentence in pilot diffs into:
  - risk_type: operational | cyber | regulatory | supply_chain | financing | macroeconomic | other
  - nature:    new_risk | expanded_existing | boilerplate
  - style:     concrete | vague

Cost optimisations (v2):
  1. Pre-filter obvious boilerplate with regex — skips ~40% of sentences without any API call
  2. Batch size 40 — less fixed prompt overhead per sentence
  3. claude-3-haiku-20240307 — 3x cheaper than claude-haiku-4-5 ($0.25 vs $0.80 / MTok input)
  4. Anthropic Message Batches API — additional 50% discount (async, results within ~1 hour)
  5. Prompt caching — system prompt billed at 10% of normal input price

Supports two backends:
  --backend claude   Claude API (requires ANTHROPIC_API_KEY)
  --backend ollama   Local Ollama server

Usage:
  python scripts/03_llm_classify.py --backend claude            # batch API (cheapest)
  python scripts/03_llm_classify.py --backend claude --no-batch # real-time streaming
  python scripts/03_llm_classify.py --backend ollama --model llama3.2

Idempotent: skips already-classified diffs.
"""

import re
import json
import time
import logging
import argparse
import requests
import pandas as pd
from pathlib import Path

OUTPUTS_DIR    = Path("/Users/timtonnaer/risk_project/outputs")
CLASSIFIED_DIR = Path("/Users/timtonnaer/risk_project/outputs/classified")
LOG_FILE       = CLASSIFIED_DIR / "classification_log.jsonl"

INDEX_CSV_MAP = {
    "pilot": OUTPUTS_DIR / "pilot_diff_index.csv",
    "full":  OUTPUTS_DIR / "all_diff_index.csv",
}

# ── Optimisation 3: cheaper model ────────────────────────────────────────────
DEFAULT_CLAUDE_MODEL = "claude-3-haiku-20240307"   # $0.25/MTok vs $0.80/MTok
DEFAULT_OLLAMA_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434/api/chat"

MIN_WORDS    = 15
BATCH_SIZE   = 40
MAX_SENTENCES = 500
MAX_RETRIES  = 3
RETRY_DELAY  = 10
BATCH_POLL_INTERVAL = 30   # seconds between batch status polls

VALID_RISK_TYPES = {"operational", "cyber", "regulatory", "supply_chain",
                    "financing", "macroeconomic", "other"}
VALID_NATURES    = {"new_risk", "expanded_existing", "boilerplate"}
VALID_STYLES     = {"concrete", "vague"}

# ── Optimisation 1: Expanded boilerplate pre-filter ───────────────────────────
_BOILERPLATE_RE = re.compile(
    # original patterns
    r"there can be no assurance"
    r"|no assurance can be given"
    r"|forward.looking statements?"
    r"|we cannot (assure|guarantee|predict|be certain)"
    r"|in the ordinary course of business"
    r"|from time to time"
    r"|^\s*item\s+\d"
    r"|risk factors? (are|include|may)"
    r"|described (elsewhere|above|below) in this"
    r"|incorporated by reference"
    r"|for (further|more|additional) information"
    r"|see (also|note|part|item)\s+\d"
    r"|may\s+(differ|vary)\s+materially"
    # NEW: additional high-precision boilerplate patterns
    r"|we (are|have been) subject to"
    r"|there (is|are) no guarantee"
    r"|we make no (representation|warranty|assurance)"
    r"|past performance (is|does) not"
    r"|results may differ"
    r"|actual results (could|may|might|will) differ"
    r"|risks and uncertainties (include|are|described)"
    r"|these risks (include|are|could)"
    r"|inherent(ly)? uncertain"
    r"|should be read in conjunction"
    r"|including but not limited to"
    r"|without limitation"
    r"|among other things"
    r"|we undertake no obligation"
    r"|except as required by (law|applicable law|securities law)"
    r"|as of the date of (this|the) (filing|report|document)"
    r"|set forth (elsewhere|above|below|in) (this|the)"
    r"|in each case"
    r"|in some cases"
    r"|on a case.by.case basis"
    r"|the following risk factors"
    r"|the risks described (below|above|herein)"
    r"|you should carefully consider"
    r"|an investment in (our|the) (common stock|shares|securities)",
    re.IGNORECASE,
)

BOILERPLATE_LABEL = {"risk_type": "other", "nature": "boilerplate", "style": "vague"}


def is_obvious_boilerplate(sentence: str) -> bool:
    return bool(_BOILERPLATE_RE.search(sentence))


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a financial disclosure analyst specializing in SEC 10-K risk factor classification. "
    "Respond only with valid JSON arrays, no prose."
)

BATCH_PROMPT_TEMPLATE = """Classify each of the following newly added/modified risk factor texts from SEC 10-K filings.

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────────

def normalize(item: dict) -> dict:
    return {
        "risk_type": item.get("risk_type") if item.get("risk_type") in VALID_RISK_TYPES else "other",
        "nature":    item.get("nature")    if item.get("nature")    in VALID_NATURES    else "boilerplate",
        "style":     item.get("style")     if item.get("style")     in VALID_STYLES     else "vague",
    }


def unknown() -> dict:
    return {"risk_type": "unknown", "nature": "unknown", "style": "unknown"}


def build_prompt(sentences: list) -> str:
    texts = "\n".join(f"{i+1}. {s.replace(chr(34), chr(39))}" for i, s in enumerate(sentences))
    return BATCH_PROMPT_TEMPLATE.format(n=len(sentences), texts=texts)


def parse_batch_response(raw: str, n: int) -> list:
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


def prefilter_sentences(raw_sentences: list):
    """Split sentences into pre-classified boilerplate and those needing LLM."""
    prefiltered, to_classify = [], []
    for s in raw_sentences:
        if len(s.split()) < MIN_WORDS:
            continue
        if is_obvious_boilerplate(s):
            prefiltered.append({"sentence": s, **BOILERPLATE_LABEL})
        else:
            to_classify.append(s)
    return prefiltered, to_classify


# ── Claude real-time backend (--no-batch) ─────────────────────────────────────

def classify_batch_claude_realtime(client, sentences: list) -> list:
    import anthropic
    prompt = build_prompt(sentences)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=client._model,
                max_tokens=800,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # prompt caching
                }],
                messages=[{"role": "user", "content": prompt}],
            )
            return parse_batch_response(response.content[0].text, len(sentences))
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Rate limited, retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise
        except Exception as e:
            logger.warning(f"Claude error attempt {attempt+1}: {e}")
            if attempt == MAX_RETRIES - 1:
                return [unknown()] * len(sentences)
    return [unknown()] * len(sentences)


def classify_realtime(index: pd.DataFrame, client):
    """Process all diffs in real-time (no batch API). Slower but immediate results."""
    total_sentences, total_diffs, skipped = 0, 0, 0

    for _, row in index.iterrows():
        out = output_path(row)
        if out.exists():
            skipped += 1
            continue

        logger.info(f"Classifying {row['ticker']} {row['year_old']}-{row['year_new']} "
                    f"(n_added={row['n_added']})...")

        with open(row["filepath"]) as f:
            diff = json.load(f)
        raw_sentences = diff.get("added", [])

        if len(raw_sentences) > MAX_SENTENCES:
            logger.info(f"  Skipping outlier — {len(raw_sentences)} sentences")
            with open(out, "w") as f:
                json.dump([], f)
            continue

        prefiltered, to_classify = prefilter_sentences(raw_sentences)
        if prefiltered:
            logger.info(f"  Pre-filtered {len(prefiltered)} boilerplate sentences")

        results = list(prefiltered)
        for i in range(0, len(to_classify), BATCH_SIZE):
            batch = to_classify[i:i + BATCH_SIZE]
            classifications = classify_batch_claude_realtime(client, batch)
            for sentence, cls in zip(batch, classifications):
                results.append({"sentence": sentence, **cls})
            time.sleep(0.3)

        with open(out, "w") as f:
            json.dump(results, f, indent=2)

        total_sentences += len(results)
        total_diffs += 1
        logger.info(f"  → {len(results)} sentences classified")
        _write_log(row, client._model, "claude-realtime", len(raw_sentences), len(results))

    logger.info(f"\nDone. Classified {total_diffs} diffs | {total_sentences} sentences | "
                f"{skipped} already done.")


# ── Claude Batch API backend (default, 50% cheaper) ───────────────────────────

def classify_with_batch_api(index: pd.DataFrame, client):
    """
    Optimisation 4: Anthropic Message Batches API.
    Submits all requests at once → 50% discount. Results within ~1 hour.
    """
    import anthropic

    # ── Step 1: collect all pending diffs ────────────────────────────────────
    pending = []
    skipped = 0
    diff_meta = {}   # custom_id_prefix → (row, prefiltered, to_classify chunks)

    for _, row in index.iterrows():
        out = output_path(row)
        if out.exists():
            skipped += 1
            continue

        with open(row["filepath"]) as f:
            diff = json.load(f)
        raw_sentences = diff.get("added", [])

        if len(raw_sentences) > MAX_SENTENCES:
            logger.info(f"Skipping outlier {row['ticker']} {row['year_old']}-{row['year_new']} "
                        f"({len(raw_sentences)} sentences)")
            with open(out, "w") as f:
                json.dump([], f)
            continue

        prefiltered, to_classify = prefilter_sentences(raw_sentences)
        prefix = f"{row['cik']}_{row['year_old']}_{row['year_new']}"
        diff_meta[prefix] = {"row": row, "prefiltered": prefiltered, "chunks": []}

        for chunk_idx, i in enumerate(range(0, len(to_classify), BATCH_SIZE)):
            chunk = to_classify[i:i + BATCH_SIZE]
            custom_id = f"{prefix}__{chunk_idx}"
            diff_meta[prefix]["chunks"].append((chunk_idx, chunk))

            pending.append(
                anthropic.types.MessageCreateParamsNonStreaming(
                    custom_id=custom_id,
                    params={
                        "model": client._model,
                        "max_tokens": 800,
                        "system": [{
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }],
                        "messages": [{"role": "user", "content": build_prompt(chunk)}],
                    }
                )
            )

    if skipped:
        logger.info(f"Skipped {skipped} already-classified diffs.")

    if not pending:
        logger.info("All diffs already classified. Nothing to do.")
        return

    logger.info(f"Submitting {len(pending)} API requests across "
                f"{len(diff_meta)} diffs to Anthropic Batch API...")

    # ── Step 2: submit batch ──────────────────────────────────────────────────
    batch = client.messages.batches.create(requests=pending)
    batch_id = batch.id
    logger.info(f"Batch submitted. ID: {batch_id}")
    logger.info("Waiting for results (up to ~1 hour for large batches)...")

    # ── Step 3: poll until complete ───────────────────────────────────────────
    while True:
        status = client.messages.batches.retrieve(batch_id)
        counts = status.request_counts
        logger.info(
            f"  Status: {status.processing_status} | "
            f"processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored}"
        )
        if status.processing_status == "ended":
            break
        time.sleep(BATCH_POLL_INTERVAL)

    # ── Step 4: collect results ───────────────────────────────────────────────
    logger.info("Collecting results...")
    chunk_results = {}   # custom_id → list[dict]
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            raw = result.result.message.content[0].text
            # figure out chunk size from meta
            prefix, chunk_idx_str = result.custom_id.rsplit("__", 1)
            chunk_idx = int(chunk_idx_str)
            chunk = diff_meta[prefix]["chunks"][chunk_idx][1]
            chunk_results[result.custom_id] = parse_batch_response(raw, len(chunk))
        else:
            logger.warning(f"Request {result.custom_id} failed: {result.result.error}")
            prefix, chunk_idx_str = result.custom_id.rsplit("__", 1)
            chunk_idx = int(chunk_idx_str)
            chunk = diff_meta[prefix]["chunks"][chunk_idx][1]
            chunk_results[result.custom_id] = [unknown()] * len(chunk)

    # ── Step 5: assemble per-diff outputs ────────────────────────────────────
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
        logger.info(f"  {row['ticker']} {row['year_old']}-{row['year_new']}: "
                    f"{len(results)} sentences | "
                    f"{len(meta['prefiltered'])} pre-filtered")
        _write_log(row, client._model, "claude-batch", int(row["n_added"]), len(results))

    logger.info(f"\nDone. Classified {total_diffs} diffs | {total_sentences} sentences total.")


# ── Ollama backend ────────────────────────────────────────────────────────────

def classify_batch_ollama(model: str, sentences: list) -> list:
    prompt = build_prompt(sentences)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
            resp.raise_for_status()
            return parse_batch_response(resp.json()["message"]["content"], len(sentences))
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Start it with: ollama serve")
            raise
        except Exception as e:
            logger.warning(f"Ollama error attempt {attempt+1}: {e}")
            if attempt == MAX_RETRIES - 1:
                return [unknown()] * len(sentences)
            time.sleep(2)
    return [unknown()] * len(sentences)


def check_ollama(model: str) -> str:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"] for m in resp.json().get("models", [])]
        matched = [m for m in available if m.split(":")[0] == model.split(":")[0]]
        if not matched:
            logger.error(f"Model '{model}' not found. Available: {available}")
            logger.error(f"Pull it with: ollama pull {model}")
            raise SystemExit(1)
        logger.info(f"Ollama ready. Using model: {matched[0]}")
        return matched[0]
    except requests.exceptions.ConnectionError:
        logger.error("Ollama server not running. Start it with: ollama serve")
        raise SystemExit(1)


def classify_ollama(index: pd.DataFrame, model: str):
    total_sentences, total_diffs, skipped = 0, 0, 0
    for _, row in index.iterrows():
        out = output_path(row)
        if out.exists():
            skipped += 1
            continue
        logger.info(f"Classifying {row['ticker']} {row['year_old']}-{row['year_new']}...")
        with open(row["filepath"]) as f:
            diff = json.load(f)
        raw_sentences = diff.get("added", [])
        if len(raw_sentences) > MAX_SENTENCES:
            with open(out, "w") as f:
                json.dump([], f)
            continue

        prefiltered, to_classify = prefilter_sentences(raw_sentences)
        results = list(prefiltered)
        for i in range(0, len(to_classify), BATCH_SIZE):
            batch = to_classify[i:i + BATCH_SIZE]
            for sentence, cls in zip(batch, classify_batch_ollama(model, batch)):
                results.append({"sentence": sentence, **cls})

        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        total_sentences += len(results)
        total_diffs += 1
        _write_log(row, model, "ollama", int(row["n_added"]), len(results))
        logger.info(f"  → {len(results)} sentences")
    logger.info(f"\nDone. {total_diffs} diffs | {total_sentences} sentences | {skipped} skipped.")


# ── Shared helpers ────────────────────────────────────────────────────────────

def output_path(row: pd.Series) -> Path:
    return CLASSIFIED_DIR / f"{row['cik']}_{row['year_old']}_{row['year_new']}_classified.json"


def _write_log(row, model, mode, n_input, n_classified):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            "cik": row["cik"], "ticker": row["ticker"],
            "year_old": int(row["year_old"]), "year_new": int(row["year_new"]),
            "mode": mode, "model": str(model),
            "n_input_sentences": n_input,
            "n_classified": n_classified,
        }) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["claude", "ollama"], default="claude")
    parser.add_argument("--model", default=None)
    parser.add_argument("--sample", choices=["pilot", "full"], default="pilot",
                        help="Which diff index to classify (default: pilot)")
    parser.add_argument("--no-batch", action="store_true",
                        help="Use real-time API instead of Batch API (Claude only)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    diff_index_csv = INDEX_CSV_MAP[args.sample]
    CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(diff_index_csv)
    logger.info(f"Sample: {args.sample} | Diffs to process: {len(index)}")

    if args.backend == "claude":
        import anthropic
        model = args.model or DEFAULT_CLAUDE_MODEL
        client = anthropic.Anthropic()
        client._model = model

        if args.no_batch:
            logger.info(f"Backend: Claude real-time | Model: {model} | Batch size: {args.batch_size}")
            classify_realtime(index, client)
        else:
            logger.info(f"Backend: Claude Batch API (50% off) | Model: {model}")
            classify_with_batch_api(index, client)

    else:
        model = args.model or DEFAULT_OLLAMA_MODEL
        logger.info(f"Backend: Ollama | Model: {model}")
        matched = check_ollama(model)
        classify_ollama(index, matched)


if __name__ == "__main__":
    main()
