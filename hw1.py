#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


EXTRACT_SYSTEM = """You are a meticulous receipt reader for Hong Kong supermarket receipts.
Read the receipt image carefully and return ONLY a JSON object (no prose, no code fences) with:
{{
  "items": [{{"name": str, "price": number}}],
  "discounts": [{{"name": str, "amount": number}}],
  "subtotal_before_rounding": number,
  "rounding": number,
  "final_paid": number
}}
Rules:
- "items": every purchased line at its ORIGINAL positive price (quantity x unit price when quantity > 1). Never include discount lines here.
- "discounts": every discount line as a POSITIVE number: promotions, coupons, member, app, packaging-damage, percentage discounts. Do not include rounding.
- "subtotal_before_rounding": the amount after discounts and before rounding, as printed.
- "rounding": signed rounding adjustment (0 if none).
- "final_paid": the final amount actually paid after rounding (total/cash/card amount).
- Use plain numbers in HKD, no currency symbols. Do not skip or invent lines."""

EXTRACT_HUMAN = "Extract the receipt data as JSON.{hint}"


def build_chain() -> Any:
    """Prompt | vision LLM | JSON parser: extract structured data from one receipt."""
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACT_SYSTEM),
            (
                "human",
                [
                    {"type": "image_url", "image_url": {"url": "{image_url}"}},
                    {"type": "text", "text": EXTRACT_HUMAN},
                ],
            ),
        ]
    )
    llm = ChatDeepSeek(model="deepseek-v4-flash-vision-exp", temperature=0, max_retries=3)
    return prompt | llm | JsonOutputParser()


def _dec(value: Any) -> Decimal:
    try:
        return Decimal(str(value).replace(",", "").replace("HK$", "").replace("$", "").strip())
    except InvalidOperation:
        return Decimal("0")


def _summarize(data: Any) -> dict[str, Decimal] | None:
    """Turn extracted JSON into totals; None if malformed."""
    if not isinstance(data, dict):
        return None
    try:
        gross = sum((_dec(i.get("price")) for i in data.get("items", [])), Decimal("0"))
        disc = sum((abs(_dec(d.get("amount"))) for d in data.get("discounts", [])), Decimal("0"))
        subtotal = _dec(data.get("subtotal_before_rounding"))
        rounding = _dec(data.get("rounding"))
        paid = _dec(data.get("final_paid"))
    except (AttributeError, TypeError):
        return None
    return {"gross": gross, "disc": disc, "subtotal": subtotal, "rounding": rounding, "paid": paid}


def _consistent(t: dict[str, Decimal] | None) -> bool:
    """Reflection check: items - discounts = subtotal, and subtotal + rounding = paid."""
    tol = Decimal("0.06")
    if t is None or t["paid"] <= 0:
        return False
    return abs(t["gross"] - t["disc"] - t["subtotal"]) <= tol and abs(t["subtotal"] + t["rounding"] - t["paid"]) <= tol


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Extract every receipt in parallel, re-check inconsistent ones, then sum in code."""
    inputs = [{"image_url": image_data_url(p), "hint": ""} for p in images]
    raw = chain.batch(inputs, config={"max_concurrency": 4}, return_exceptions=True)
    totals = [_summarize(r) for r in raw]

    for _ in range(2):
        bad = [i for i, t in enumerate(totals) if not _consistent(t)]
        if not bad:
            break
        retry_inputs = [
            {
                "image_url": inputs[i]["image_url"],
                "hint": (
                    "\n\nYour previous reading did not add up. Re-read every line carefully: "
                    "sum(items) - sum(discounts) must equal subtotal_before_rounding, and "
                    "subtotal_before_rounding + rounding must equal final_paid. "
                    "Check for missed lines, misread digits, and discounts wrongly listed as items."
                ),
            }
            for i in bad
        ]
        retried = chain.batch(retry_inputs, config={"max_concurrency": 4}, return_exceptions=True)
        for i, r in zip(bad, retried):
            t = _summarize(r)
            if t is not None and (_consistent(t) or totals[i] is None):
                totals[i] = t

    if any(t is None for t in totals):
        errors = [r for r in raw if isinstance(r, Exception)]
        raise errors[0] if errors else RuntimeError("could not read some receipts")
    valid = [t for t in totals if t is not None]
    total_paid = sum((t["paid"] for t in valid), Decimal("0"))
    total_gross = sum((t["gross"] for t in valid), Decimal("0"))
    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_gross:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
