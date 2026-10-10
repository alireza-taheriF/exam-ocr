#!/usr/bin/env python3
"""Drive the student editor end to end with a real browser and save screenshots.

upload → wait for lines → edit one line → confirm → /admin, for each page given.
Run against a server started with a throwaway DATA_DIR so demo edits never reach
the training export; leave consent unchecked for the same reason.

    .venv-qwen/bin/python scripts/e2e_playwright.py --out dataset/reports/e2e \
        dataset/raw_images/<page>.png [...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def run_page(page, base: str, image: Path, out: Path, idx: int) -> dict:
    page.goto(base + "/")
    page.set_input_files("#file", str(image))
    page.fill("#course", "e2e-demo")
    t0 = time.perf_counter()
    page.click("#uploadBtn")
    page.wait_for_selector("#editor:not(.hidden)", timeout=30 * 60 * 1000)
    seconds = time.perf_counter() - t0
    page.wait_for_timeout(1500)  # let crop images load
    page.screenshot(path=str(out / f"{idx}_editor.png"), full_page=True)

    doc_id = page.evaluate("state.doc.id")
    n_text = page.locator("#lines textarea").count()
    n_low = page.locator(".line.low").count()

    edited = None
    if n_text:
        ta = page.locator("#lines textarea").first
        before = ta.input_value()
        ta.click()
        ta.fill(before + " (ویرایش آزمایشی)")
        ta.press("Tab")  # blur → save
        page.wait_for_selector("#lines .line .save:has-text('ذخیره شد')", timeout=15000)
        edited = {"before": before, "after": ta.input_value()}

    page.once("dialog", lambda d: d.accept())
    page.click("#confirmBtn")
    page.wait_for_selector("#confirmBtn:has-text('تأیید شده')", timeout=15000)
    page.screenshot(path=str(out / f"{idx}_confirmed.png"), full_page=False)
    return {"image": str(image), "doc_id": doc_id, "process_seconds": round(seconds, 1),
            "text_lines": n_text, "low_confidence_lines": n_low, "edit": edited}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception:  # bundled browser missing (offline machine): use installed Chrome
            browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1100, "height": 1400}, locale="fa-IR")
        for i, image in enumerate(args.images, 1):
            r = run_page(page, args.base, image, args.out, i)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            results.append(r)
        page.goto(args.base + "/admin")
        page.wait_for_timeout(2000)
        page.screenshot(path=str(args.out / "admin.png"), full_page=True)
        browser.close()

    (args.out / "e2e_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
