"""
Script 03: LLM Classification
Classifies each added sentence in pilot diffs into:
  - risk_type: operational | cyber | regulatory | supply_chain | financing | macroeconomic | other
  - nature:    new_risk | expanded_existing | boilerplate
  - style:     concrete | vague

Cost optimisations (vs original):
  1. Pre-filter obvious boilerplate with regex — skips ~30% of sentences without any API call
  2. Batch size 40 — halves fixed prompt overhead vs batch size 20
  3. Prompt caching (Claude only) — system prompt cached at 10% of normal input price

Supports two backends:
  --backend claude   Claude API (requires ANTHROPIC_API_KEY)
  --backend ollama   Local Ollama server

Usage:
  python scripts/03_llm_classify.py --backend claude
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

DIFF_INDEX_CSV = Path("/Users/timtonnaer/risk_project/outputs/pilot_diff_index.csv")
CLASSIFIED_DIR = Path("/Users/timtonnaer/risk_project/outputs/classified")
LOG_FILE = CLASSIFIED_DIR / "classification_log.jsonl"

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OLLAMA_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434/api/chat"

MIN_WORDS = 15
BATCH_SIZE = 40       # optimisation 2: larger batches → less fixed overhead per sentence
MAX_SENTENCES = 500   # skip outlier diffs (likely reformatted filings)
MAX_RETRIES = 3
RETRY_DELAY = 10

VALID_RISK_TYPES = {"operational", "cyber", "regulatory", "supply_chain", "financing", "macroeconomic", "other"}
VALID_NATURES    = {"new_risk", "expanded_existing", "boilerplate"}
VALID_STYLES     = {"concrete", "vague"}

# ── Optimisation 1: Pre-filter obvious boilerplate ────────────────────────────
# These patterns reliably identify formulaic, low-information sentences.
# Auto-labelled without any API call.
_BOILERPLATE_RE = re.compile(
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
    r"|may\s+(differ|vary)\s+materially",
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(item: dict) -> dict:
    return {
        "risk_type": item.get("risk_type") if item.get("risk_type") in VALID_RISK_TYPES else "other",
        "nature":    item.get("nature")    if item.get("nature")    in VALID_NATURES    else "boilerplate",
        "style":     item.get("style")     if item.get("style")     in VALID_STYLES     else "vague",
    }


def unknown() -> dict:
    return {"risk_type": "unknown", "nature": "unknown", "style": "unknown"}


def build_prompt(sentences: list[str]) -> str:
    texts = "\n".join(f"{i+1}. {s.replace(chr(34), chr(39))}" for i, s in enumerate(sentences))
    return BATCH_PROMPT_TEMPLATE.format(n=len(sentences), texts=texts)


def parse_batch_response(raw: str, n: int) -> list[dict]:
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


# ── Claude backend ────────────────────────────────────────────────────────────

def classify_batch_claude(client, sentences: list[str]) -> list[dict]:
    import anthropic
    prompt = build_prompt(sentences)
    for attempt in range(MAX_RETRIES):
        try:
            # Optimisation 3: prompt caching — system prompt billed at 10% input price after first call
            response = client.messages.create(
                model=client._model,
                max_tokens=800,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            return parse_batch_response(raw, len(sentences))
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


def make_claude_client(model: str):
    import anthropic
    client = anthropic.Anthropic()
    client._model = model
    return client


# ── Ollama backend ────────────────────────────────────────────────────────────

def classify_batch_ollama(model: str, sentences: list[str]) -> list[dict]:
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
            raw = resp.json()["message"]["content"]
            return parse_batch_response(raw, len(sentences))
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


# ── Main loop ─────────────────────────────────────────────────────────────────

def output_path(row: pd.Series) -> Path:
    return CLASSIFIED_DIR / f"{row['cik']}_{row['year_old']}_{row['year_new']}_classified.json"


def classify_diff(row: pd.Series, backend: str, client_or_model, batch_size: int) -> list[dict]:
    with open(row["filepath"]) as f:
        diff = json.load(f)

    raw_sentences = diff.get("added", [])

    # Skip outlier diffs (reformatted filings, not genuine risk changes)
    if len(raw_sentences) > MAX_SENTENCES:
        logger.info(f"  Skipping — {len(raw_sentences)} sentences exceeds MAX_SENTENCES={MAX_SENTENCES}")
        return []

    results = []
    n_prefiltered = 0
    to_classify = []

    for s in raw_sentences:
        if len(s.split()) < MIN_WORDS:
            continue
        # Optimisation 1: pre-filter obvious boilerplate
        if is_obvious_boilerplate(s):
            results.append({"sentence": s, **BOILERPLATE_LABEL})
            n_prefiltered += 1
        else:
            to_classify.append(s)

    if n_prefiltered:
        logger.info(f"  Pre-filtered {n_prefiltered} boilerplate sentences (no API call)")

    # Send remaining sentences to LLM in batches
    for i in range(0, len(to_classify), batch_size):
        batch = to_classify[i:i + batch_size]
        if backend == "claude":
            classifications = classify_batch_claude(client_or_model, batch)
            time.sleep(0.3)
        else:
            classifications = classify_batch_ollama(client_or_model, batch)
        for sentence, classification in zip(batch, classifications):
            results.append({"sentence": sentence, **classification})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["claude", "ollama"], default="claude")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Sentences per API call (default: {BATCH_SIZE})")
    args = parser.parse_args()

    if args.backend == "claude":
        model = args.model or DEFAULT_CLAUDE_MODEL
        logger.info(f"Backend: Claude API | Model: {model} | Batch size: {args.batch_size}")
        client_or_model = make_claude_client(model)
    else:
        model = args.model or DEFAULT_OLLAMA_MODEL
        logger.info(f"Backend: Ollama | Model: {model} | Batch size: {args.batch_size}")
        client_or_model = check_ollama(model)

    CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(DIFF_INDEX_CSV)

    total_sentences = 0
    total_prefiltered = 0
    total_diffs = 0
    skipped = 0

    for _, row in index.iterrows():
        out = output_path(row)
        if out.exists():
            skipped += 1
            continue

        logger.info(f"Classifying {row['ticker']} {row['year_old']}-{row['year_new']} "
                    f"(n_added={row['n_added']})...")
        try:
            results = classify_diff(row, args.backend, client_or_model, args.batch_size)
        except Exception as e:
            logger.error(f"Failed {row['ticker']} {row['year_new']}: {e}")
            continue

        with open(out, "w") as f:
            json.dump(results, f, indent=2)

        n_bp = sum(1 for r in results if r.get("nature") == "boilerplate" and r.get("risk_type") == "other" and r.get("style") == "vague")
        total_prefiltered += n_bp
        total_sentences += len(results)
        total_diffs += 1
        logger.info(f"  → {len(results)} sentences classified")

        with open(LOG_FILE, "a") as f:
            f.write(json.dumps({
                "cik": row["cik"], "ticker": row["ticker"],
                "year_old": int(row["year_old"]), "year_new": int(row["year_new"]),
                "backend": args.backend, "model": str(client_or_model),
                "n_input_sentences": int(row["n_added"]),
                "n_classified": len(results),
            }) + "\n")

    logger.info(
        f"\nDone. Classified {total_diffs} diffs | {total_sentences} sentences total | "
        f"{skipped} skipped (already done)."
    )


if __name__ == "__main__":
    main()
