# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution:

### Chain design

```
                 ┌─────────────────────────┐
 receipt.jpg ──▶ │ ChatPromptTemplate       │
 (per receipt)   │  system: extraction      │
                 │  rules + few-shot format │
                 │  human: image + text     │
                 └────────────┬─────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ ChatDeepSeek             │
                 │ deepseek-v4-flash-       │
                 │ vision-exp (temp=0)      │
                 └────────────┬─────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ JsonOutputParser         │
                 │ -> {items, discounts,    │
                 │     subtotal, rounding,  │
                 │     final_paid}          │
                 └────────────┬─────────────┘
                              ▼
      all receipts ──▶ chain.batch() (parallel calls)
                              ▼
                 ┌─────────────────────────┐
                 │ reflection check (code) │
                 │ items - discounts       │
                 │   == subtotal ?         │
                 │ subtotal + rounding     │
                 │   == final_paid ?       │
                 └──────┬───────────┬──────┘
                    consistent   inconsistent
                        │              │
                        │      retry same receipt
                        │      with a corrective hint
                        │      (up to 2 attempts)
                        ▼              ▼
                 ┌─────────────────────────┐
                 │ sum in Python (Decimal) │
                 │ Q1 = Σ final_paid       │
                 │ Q2 = Σ (items -         │
                 │        discounts) +     │
                 │        Σ discounts      │
                 │      = Σ items          │
                 └────────────┬─────────────┘
                              ▼
                 {"HK$...": Q1, "HK$...": Q2}
```

### Description

`build_chain()` wires a single reusable pipeline — a multimodal `ChatPromptTemplate`
(system instructions + one human message carrying the receipt image and a text
instruction), the vision-capable `deepseek-v4-flash-vision-exp` model at
`temperature=0`, and a `JsonOutputParser` — that turns one receipt image into a
structured JSON object of its items, discounts, subtotal, rounding, and final
paid amount. `answer_queries()` runs this chain over every receipt in the folder
with `chain.batch()` for parallel extraction, then applies a reflection step in
plain Python: each receipt's numbers must satisfy `items - discounts ==
subtotal` and `subtotal + rounding == final_paid` within a small tolerance; any
receipt that fails this check is re-sent through the chain with a corrective
hint describing exactly which line items commonly get missed or misread, up to
two retries. Once every receipt passes (or retries are exhausted), the two
final answers are computed deterministically in code rather than asking the
model to do arithmetic: Query 1 sums each receipt's `final_paid` (the actual
amount after rounding), and Query 2 sums each receipt's pre-discount item total
(`items`, i.e. `subtotal + discounts`), so rounding is correctly excluded from
Query 2 while it is correctly included in Query 1. Keeping the summation out of
the LLM and doing the reconciliation check + targeted retries in code makes the
result robust to occasional misreads on any single receipt and keeps each
response formatted as exactly one `HK$` amount.

