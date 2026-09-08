# Phase 1 — Decisions Log

Date: 2026-08-28 (investigation) · 2026-09-02 (execution)

## STEP 0 — Investigation result: the clean dataset does not exist

The Phase 1 brief (GROUND TRUTH item 6) states that "a separately-built, cleanly
labeled dataset already exists in a folder in the repo." **I searched for it and it
is not there.** Reporting this rather than proceeding on a false premise, because
Step 0's decision branch depends entirely on it.

### What I searched

| Location | Result |
|---|---|
| `dataset/` subdirectories | Only `lines/`, `raw_images/`, `raw_pdfs/`, `rejected/` — no `clean/`, `verified/`, `gold/`, `annotations/` |
| All `.jsonl`/`.csv`/`.tsv`/`.parquet` in repo | 12 files, all the known-broken pipeline outputs (`dataset.jsonl`, `lines.jsonl`, `train/val/test.jsonl`, error logs) |
| `trocr_training.zip` (174 MB) | Byte-identical copy of the broken set: `dataset/train.jsonl`, `dataset/val.jsonl`, 17,192 files in `dataset/lines/` |
| `exam-ocr.zip` (6.5 GB, 55,828 entries) | Full older snapshot of *this* repo (`.venv/lib` 36,784 · `dataset/lines` 17,192 · `dataset/raw_images` 1,052). No new dataset |
| `.cache/huggingface/hub` | Model weights only: `surya_det3`, `surya_layout3`, `surya_order`, `surya_rec2`, `xlm-roberta-base` |
| git branches / stashes / tags | Only `master`; no stashes, no tags |
| `~/Desktop/ocr_check/` | 5 crops **copied out of `dataset/lines/`** — all `aligned=False`, no labels. Eyeballing artifacts, not a dataset |
| `~/Downloads/.../farsi/` | 10 digit PNGs for a Hopfield-network assignment. Unrelated |
| `~/Documents`, `~/Desktop`, `~/Downloads`, `~/OneDrive`, 5 Google Drive mounts, iCloud | No Persian OCR / HTR / handwriting dataset |
| Shell history, agent transcripts (`.gemini`, `.cursor`, `.qodo`, `.trae`) | No reference to a clean-dataset path |

Verification of the 5 Desktop crops (all in `lines.jsonl`, all `aligned=False`):

| crop | label in JSONL |
|---|---|
| `c931871ef2ea` | `\| مانده بستانکار \| مانده بدهکار \| \|` (markdown table row) |
| `b286d12ea2fe` | `ق - ۱` |
| `c24b457fef08` | `تاریخ ۱ آبر ۱ ۹۷` |
| `05ea3f9df9cd` | `موجودی کالای شعبه ۸۵۰/۰۰۰` |
| `98ef5708ead1` | `(موجودی کالا) مانده ← ۲۵٪ / ۰۰۰،۶۴ مابقی ..... ۴ ملیان` |

**Verdict: NO.** No cleanly labeled Persian handwriting line dataset exists on this
machine. If one exists elsewhere (another machine, a Colab Drive folder, a HF repo),
point me at it and I will re-run Step 0 against it — that would save the entire
re-labeling cost below.

## Decision: fallback path, staged, cheapest-first

Taking the brief's fallback branch, with one change: **do not open with 22k paid VLM
calls.** Staged instead, so each stage is justified by the previous one's measured result.

1. **Re-detect lines with Surya** (free, local). `lines.jsonl` stores only
   `min(len(bboxes), len(label_lines))` boxes per page — 17,186 of ~34,536 detected.
   Re-running detection recovers the full set with no API cost.
2. **Geometry-filter *before* any API call** — this is the main cost lever. Measured on
   the 17,186 stored bboxes: **64.2% survive** (`h<24px` −15.3%, `aspect>16` −17.1%,
   portrait −3.4%). Applied to ~34,536 boxes → **~22,000 crops to label**.
3. **Try `surya_rec2` first** (free, already in `.cache/huggingface/hub`). Persian is in
   its supported set. If corpus CER against the gold set (step 5) is usable, it labels
   the bulk at zero cost. Measure on ~100 crops before committing.
4. **VLM only for what Surya can't read** — one call per crop, alignment by construction.
5. **Hand-verified gold set is mandatory and cannot be skipped.** Every auto-labeling
   path — VLM or Surya — produces *pseudo-labels*, and CER against pseudo-labels measures
   agreement, not accuracy. 200–300 lines transcribed by a human is the only trustworthy
   CER source. This is the single highest-value 2 hours in the whole project.

## Cost estimate (gate — no paid call runs without explicit approval)

~22,000 crops · mean **70 image tokens** each (measured: `w*h/750`, median 18, p95 353)
· ~120 prompt tokens · ~30 output tokens → **~4.2M input, ~0.66M output**.

| Model | Standard | Batch API (−50%) |
|---|---|---|
| Opus 4.8 ($5/$25 per MTok) | **$37** | $19 |
| Sonnet 4.6 ($3/$15) | **$22** | $11 |
| Haiku 4.5 ($1/$5) | **$7** | $4 |

Caveats: prompt caching does not help — the ~120-token system prompt is far below the
4,096-token minimum cacheable prefix on Opus. And `.env` currently points
`VLM_API_BASE` at `agentrouter.org/v1`, a third-party router whose pricing is not
Anthropic's; these figures assume the first-party API.

## Security

`dataset/label_errors.jsonl` (tracked in git) contains 188 copies of an HTTP 401
error message quoting the API key as `sk-BkU3M***…***YwQ6`.

Correction to an earlier reading of this: the key is **already masked by the
provider** — only the 5-char prefix and 4-char suffix are present, so this is not
a usable credential and no repo scrubbing is required (a redaction pass over the
file was a no-op; the file is unchanged from HEAD). Still worth rotating: the
prefix identifies which key was in use and narrows a brute-force search, and the
key was clearly live at the time. Rotation happens in the provider dashboard —
nothing in the repo can un-leak it.

Re-verified 2026-09-02, more precisely than before. The logged token is 52 chars
of which **39 are asterisks** — 8 visible prefix, 5 visible suffix. It maps to
`OPENROUTER_API_KEY` in `.env` (matched by prefix without printing either value).
A regex sweep for *unmasked* key shapes (`sk-…`, `sk-or-…`, `sk-ant-…`, `AQ.…`,
`AKIA…`) across every tracked file, every file in `logs/`, and every blob in git
history (1 commit) returns **nothing**. `.env` has never been committed and is
`.gitignore`d. Six live credentials sit in `.env` — rotate `OPENROUTER_API_KEY`
in the provider dashboard; the other five have not leaked anywhere in the repo.

`logs/auto_labeler.log` is `.gitignore`d and was never committed.

## Execution results (2026-09-02)

Stage 1 of the staged plan is done and stage 2 measured. Nothing was paid for.

### What ran

`python scripts/extract_and_align_lines.py --backend none` over all 1,008 manifest
pages, ~15 min on an M2 Pro (MPS). Detection recovered **33,648** boxes — the
legacy pipeline stored only 17,186 because it truncated to
`min(len(bboxes), len(label_lines))`.

| | |
|---|---|
| pages in manifest | 1,008 |
| byte-identical duplicates dropped | **63** (45 groups; 42 crawled-web) |
| pages scanned | 945 |
| boxes detected | 33,648 |
| boxes kept after geometry filter | **22,250** (66.1%) |
| crops written, rows emitted | 22,250 / 22,250 (0 orphans, 0 failures) |

Geometry drops: height <24px 5,283 (15.7%) · aspect >16:1 4,780 (14.2%) ·
portrait 1,308 (3.9%) · zero-area 27 (0.08%). Keep rate splits hard by page
resolution — 71.0% on the 835 high-res PDF renders, 15.2% on the 110 crawled
low-res pages, as predicted.

Split: 121 documents → train 16,509 / val 2,870 / test 2,871 lines
(74.2/12.9/12.9%), **0 shared documents, 0 shared pages, 0 shared crops** between
every split pair.

### Two problems found during execution, both fixed

**Duplicate page images could defeat the leakage check.** 45 groups of
byte-identical files exist in `dataset/raw_images` (63 redundant copies). Because
`derive_source_document` gives each UUID page its own document id, identical
pixels under two filenames become two *different* documents that the bin-packer
is free to put in different splits — and the leakage report would still read
zero, correctly, because the document ids differ. Fixed with a SHA-256 dedup pass
before detection (`dedupe_pages`). This is the one leakage vector a
document-grouped split does not close by itself.

**Crop filenames were `uuid4()`.** Every re-run wrote a fresh copy of every crop
and orphaned the previous run's files. Now content-addressed on
`sha1(source_image|bbox)`, so extraction is idempotent and `dataset/lines_v2/`
always matches the manifest.

### Stage 3 measured: surya_rec2 is not usable as a labeler

Free local recognition on 306 high-res crops across 12 documents: 168 accepted,
138 rejected by `is_vlm_artifact` (105 with no Persian letter at all). Eyeballing
the accepted 168 against the crops, the handwritten lines are mostly wrong —
`هاد و وروز کان و مورم و و و منز کے خان حال الام` for a line of Persian
handwriting — while printed text comes out clean. That is the same
printed-text bias the 597 `aligned=True` legacy survivors had. **surya_rec2 is
rejected as a bulk labeling backend.** It is not a CER baseline either: with no
gold set there is nothing to score it against.

So `dataset/lines_v2.jsonl` currently holds 22,250 correctly-cropped,
correctly-grouped, **unlabeled** rows. `text` is empty in every row and the split
files inherit that — `split_leakage_report.json` records
`"unlabeled_rows": 22250`, and the script prints it. This is a crop manifest, not
trainable data.

### Not done, and blocking Phase 2

1. **Hand-verify 200–300 lines into `dataset/gold.jsonl`.** Unchanged from the
   original plan and still the highest-value work available. Without it there is
   no trustworthy CER, so no way to accept or reject any labeling backend —
   including a paid one.
2. **Then label the corpus.** Cost re-derived on the actual 22,250 crops rather
   than projected from the 17,186 stored bboxes: 4.06M input tokens (1.16M of
   them image tokens), 0.78M output.

   | Model | Standard | Batch API (−50%) |
   |---|---|---|
   | Opus 4.8 | **$39.75** | $19.88 |
   | Sonnet 4.6 | **$23.85** | $11.93 |
   | Haiku 4.5 | **$7.95** | $3.98 |

   Close to the earlier projection ($37 / $22 / $7). No paid call has been made;
   `--backend vlm` refuses to run without `--approve-api-spend`.

### Crop quality caveat for Phase 2

Surya's boxes hug the ink: on a 200-crop sample, ink touches the bottom edge in
32.5% of crops and the top edge in 19.5%. Descenders and dots are being clipped.
Padding by 15% of line height halves it (bottom 21.5%, top 13.0%); 25% starts
pulling in the neighbouring line. Worth a `--pad` flag before training rather
than after, but it changes crop geometry, so it belongs in one deliberate re-run
rather than a patch mid-labeling.

Visual-hash check for repeated page furniture: 22,227 distinct hashes across
22,250 crops. The only repeats are 5 watermark bands in
`جزوه_ریاضی_پیش__نیکوکار_mathfree`, all inside one document. No cross-document
duplication.

---

# Phase 1B — Padding, Gold Set, Backend Bake-off

Date: 2026-09-02

## Step 1 — Re-crop with padding → `lines_v3`

`PAD_FRACTION = 0.075` per side (15% total), vertical only, clamped to the page.
Applied to the **crop**, never to the geometry filter — so the set of accepted
boxes is byte-identical to v2 (33,648 detected → 22,250 kept, same drop counts)
and the two runs are directly comparable. Rows now carry both `bbox` (detected,
what the filter judged) and `crop_bbox` (what was written to disk).

Full re-run from scratch: dedup 1,008 → 945 pages, 22,250 crops into
`dataset/lines_v3/`, 0 failures, 0 orphans. Split re-run on v3:
train 16,509 / val 2,870 / test 2,871, **0 shared documents, 0 shared pages,
0 shared crops** — `dataset/{train,val,test}_v3.jsonl`. The superseded `*_v2.jsonl`
split files were deleted; `lines_v2.jsonl` and `dataset/lines_v2/` stay as
reference.

### Why 0.075 and not the 0.15/side the brief suggested

Measured the clipped-ink distribution rather than picking a number. Contiguous
ink outside a detected box (gap of 2 clear rows ends the run) is **bimodal**:
median 0.16–0.23 of box height in the common case — a descender or a dot — but
the tail runs past 1.0 box height, and that tail is *the neighbouring line's
strokes*, which padding should not chase. Meanwhile 11.4% of boxes already
overlap a vertically-adjacent box, and padding closes the gap to a neighbour in
14.6% of cases at 0.05/side, **16.6% at 0.075**, 22.2% at 0.15. 0.075 recovers
descenders without importing the next line.

### Verified improvement, paired on the same box

Comparing v2 and v3 crops of the **same** detected box (n=300, page-level Otsu
threshold so the criterion doesn't shift as the crop grows):

| | |
|---|---|
| clipped in v2, clean in v3 (fixed) | 41 (13.7%) |
| clipped in both (padding insufficient) | 95 (31.7%) |
| clean in v2, clipped in v3 (worse) | 11 (3.7%) |
| clean in both | 153 (51.0%) |
| **net** | **+10.0 pp** |

Aggregate on the same sample: top 26.3% → 20.7%, bottom 32.3% → 26.7%, either
42.3% → **34.7%**.

Two honest caveats. First, "drop sharply" did not happen — 31.7% of crops are
still clipped, because that residual is the bimodal tail where the ink outside
the box belongs to the adjacent line. Chasing it with more padding imports the
neighbour, which is worse for training than a clipped descender. Second, my
Phase 1 numbers (32.5% bottom / 19.5% top) used a **crop-relative** ink
threshold, which grows more sensitive as the crop grows and therefore understated
the improvement. All Phase 1B figures use a fixed page-level Otsu threshold.

Padding costs 15% more image tokens (1.16M → 1.34M), which is $0.08 on Haiku
batch — irrelevant next to the crop-quality gain.

## Step 2 — Gold-set labeling script

`scripts/label_gold_interactive.py`. Samples 240 crops from **`test_v3.jsonl`
only**, restricted to documents with ≥50 lines — test has 60 groups but 56 are
single-page crawled web images contributing 1–3 lines each, so an even draw
across all 60 would spend the typing budget on web screenshots rather than exam
handwriting. Result: 60 lines each from the 4 substantial test documents,
interleaved so fatigue spreads evenly.

Each entry is appended and `flush()`ed immediately, so an interrupt costs at most
the line being typed. `--resume` (default) skips already-labeled crops; `s` skips
without recording, `u` undoes the previous entry, `q` saves and quits, empty input
records "illegible" as an empty label which the scorer skips. Typing something
that trips `is_vlm_artifact` (a markdown table, an emoji) asks for confirmation
rather than silently accepting a label the ingest path would reject.

25 tests cover the sampling invariants and the interactive paths end-to-end.
**This is the human's job and has not been run** — it is the remaining blocker.

## Step 3 — Backend capability check: all four models FAILED the gate

`scripts/check_vlm_backends.py`. Each model gets a **control image** containing a
freshly generated 6-character code (25^6 ≈ 244M, absent from every prompt).
Returning the code is the only accepted proof that pixels arrived. This exists
because the dangerous failure is not an error — it is a call that succeeds, drops
the image, and returns plausible Persian anyway. That is how the original
pipeline broke.

| model | image? | verdict | what came back |
|---|---|---|---|
| `claude-opus-4-8` | NO | **unreachable** | HTTP 402 "Budget pool quota has been exhausted" |
| `claude-opus-5` | NO | **unreachable** | HTTP 402, same |
| `glm-5.3` | NO | **no vision** | HTTP 400 `type 参数非法，取值范围 ['text']` |
| `deepseek-v4-flash` | NO | **no vision** | HTTP 400 `This model does not support image` |

**Usable for the bake-off: 0 of 4.** Step 4 cannot run, so Step 5 cannot either.

The unreachable/no-vision split is load-bearing, and the check establishes it
with a follow-up text-only call:

* `glm-5.3` and `deepseek-v4-flash` **answered a text-only call** and rejected the
  image. That is a capability fact — substitution is the right response.
* The two Claude IDs **failed the text-only call too, with the same 402**. Their
  vision support is therefore **untested, not absent**. Both IDs exist in the
  router's catalog (`/v1/models` lists exactly 5: `claude-opus-4-8`,
  `claude-opus-5`, `glm-5.3`, `deepseek-v4-flash`, `gpt-5.6-sol`) — this is a
  billing wall, not a bad model ID.

### Substitutions the brief asked about — both unavailable

Per the brief's instruction to report rather than guess, I probed the two named
substitutes and the router's remaining model:

| candidate | result |
|---|---|
| `glm-5.3-flash` | HTTP 503 — no channel available for this model under group `default` |
| `deepseek-v4-flash-vision-exp` | HTTP 503 — same |
| `gpt-5.6-sol` | HTTP 402 — same quota wall as the Claude IDs |

So neither proposed substitute is reachable on this endpoint either. **There is
no vision-capable backend available through `agentrouter.org` right now.**

### What needs a decision

Two independent choices, and I am not guessing at either:

1. **Fix access or change provider.** The 402 is an AgentRouter budget-pool limit
   (`.env` → `VLM_API_BASE=https://agentrouter.org/v1`). Either top up / raise the
   pool there, or point `.env` at the first-party Anthropic API, where
   `claude-opus-4-8` and the Batch API both exist and the cost table below applies
   as written. The first-party route also removes the pricing caveat — AgentRouter
   is a third-party router whose rates are not Anthropic's.
2. **Which model.** With the Claude IDs untested I cannot recommend one on
   measured CER. The defensible starting point is Haiku 4.5 on the Batch API
   ($4.06 for the whole corpus) with a bake-off against Sonnet/Opus on the gold
   60 — but that ordering only becomes real once the gold set exists.

Cost re-derived on the **padded** v3 crops (4.23M input, 1.34M image tokens,
0.78M output), first-party Anthropic pricing:

| Model | Standard | Batch API (−50%) |
|---|---|---|
| Opus 4.8 | $40.62 | **$20.31** |
| Sonnet 4.6 | $24.37 | **$12.19** |
| Haiku 4.5 | $8.12 | **$4.06** |

**No paid call has been made.** The capability check spent 8 tiny calls, all of
which were rejected before billing. `--backend vlm` still refuses without
`--approve-api-spend`.

## Steps 4 and 5 — blocked, not skipped

Step 4 (bake-off) needs both a gold set to score against and at least one backend
that passed Step 3. Neither exists. Step 5 (full-corpus labeling) needs Step 4's
measured rate and the human's backend choice. Running either now would mean
inventing numbers, which is the failure this phase was built to prevent.
