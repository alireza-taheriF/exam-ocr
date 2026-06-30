#!/usr/bin/env python3
"""
Collect Persian handwritten exam/note images from search engines.

Uses Playwright to discover image URLs and downloads them with retries.
Images are saved as UUID filenames under dataset/raw_images/.

Usage:
    pip install playwright requests Pillow
    playwright install chromium
    python crawler.py
    python crawler.py --max-per-keyword 30 --engines google,bing
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus, urlparse

import requests
from PIL import Image
from playwright.sync_api import Browser, Page, TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

logger = logging.getLogger("crawler")

DEFAULT_KEYWORDS = (
    "جزوه دست نویس فارسی",
    "برگه امتحانی دست نویس",
)

DEFAULT_ENGINES = ("google", "bing")
DEFAULT_OUTPUT_DIR = Path("dataset/raw_images")
MANIFEST_PATH = Path("dataset/manifest.jsonl")

MIN_WIDTH = 320
MIN_HEIGHT = 320
MIN_FILE_BYTES = 12_000
MAX_BYTES = 25 * 1024 * 1024

PAGE_TIMEOUT_MS = 45_000
NAVIGATION_RETRIES = 3
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT_SEC = 25
SCROLL_PAUSE_SEC = 1.2
MAX_SCROLLS = 12

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
SKIP_HOST_FRAGMENTS = (
    "gstatic.com/images/branding",
    "google.com/images/branding",
    "bing.com/rms",
    "favicon",
    "logo",
    "sprite",
    "icon",
)


@dataclass
class CrawlStats:
    discovered: int = 0
    downloaded: int = 0
    skipped_small: int = 0
    skipped_invalid: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class ImageRecord:
    file_id: str
    path: str
    source_url: str
    keyword: str
    engine: str
    width: int
    height: int
    bytes: int


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _is_likely_thumbnail_url(url: str) -> bool:
    lowered = url.lower()
    if any(fragment in lowered for fragment in SKIP_HOST_FRAGMENTS):
        return True
    # Common Google/Bing thumbnail size hints in URL.
    thumb_hints = ("=s32", "=s64", "=s128", "w=32", "w=64", "h=32", "h=64", "thumbnail")
    return any(hint in lowered for hint in thumb_hints)


def _guess_extension(url: str, content_type: str | None) -> str:
    path = urlparse(url).path.lower()
    for ext in IMAGE_EXTENSIONS:
        if path.endswith(ext):
            return ext if ext != ".jpeg" else ".jpg"

    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        mapping = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
        }
        if ct in mapping:
            return mapping[ct]
    return ".jpg"


def _validate_image_bytes(data: bytes) -> tuple[bool, int, int, str | None]:
    if len(data) < MIN_FILE_BYTES:
        return False, 0, 0, f"file too small ({len(data)} bytes)"

    if len(data) > MAX_BYTES:
        return False, 0, 0, f"file too large ({len(data)} bytes)"

    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
        with Image.open(BytesIO(data)) as img:
            width, height = img.size
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return False, width, height, f"dimensions {width}x{height} below minimum"
            return True, width, height, None
    except Exception as exc:
        return False, 0, 0, f"invalid image: {exc}"


def _retry(action, *, retries: int, label: str, pause_sec: float = 1.5):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return action()
        except PlaywrightTimeout as exc:
            last_exc = exc
            logger.warning("%s timeout (attempt %d/%d)", label, attempt, retries)
        except Exception as exc:
            last_exc = exc
            logger.warning("%s failed (attempt %d/%d): %s", label, attempt, retries, exc)
        if attempt < retries:
            time.sleep(pause_sec * attempt)
    raise RuntimeError(f"{label} failed after {retries} attempts") from last_exc


def _new_page(browser: Browser) -> Page:
    context = browser.new_context(
        user_agent=USER_AGENT,
        locale="fa-IR",
        viewport={"width": 1440, "height": 900},
    )
    page = context.new_page()
    page.set_default_timeout(PAGE_TIMEOUT_MS)
    return page


def _scroll_results(page: Page, scrolls: int) -> None:
    for i in range(scrolls):
        page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight * 0.9))")
        time.sleep(SCROLL_PAUSE_SEC)
        if i % 3 == 2:
            page.keyboard.press("End")
            time.sleep(SCROLL_PAUSE_SEC)


def _extract_google_image_urls(page: Page) -> list[str]:
  """Collect original/large image URLs from Google Images results."""
  urls: set[str] = set()

  # data-ou holds the original image URL on many result tiles.
  for attr in ("data-ou", "data-oh", "data-ow"):
    elements = page.query_selector_all(f"[{attr}]")
    for el in elements:
      value = el.get_attribute(attr)
      if value and value.startswith("http"):
        urls.add(value)

  # Thumbnail / cached URLs from img tags (filtered later by size).
  for img in page.query_selector_all("img[src]"):
    src = img.get_attribute("src") or ""
    if src.startswith("http") and not _is_likely_thumbnail_url(src):
      urls.add(src)

  # JSON-ish inline payloads occasionally contain image URLs.
  html = page.content()
  for match in re.findall(r'"(https?://[^"\\]+?\.(?:jpg|jpeg|png|webp))"', html, flags=re.I):
    if not _is_likely_thumbnail_url(match):
      urls.add(match.replace("\\u003d", "="))

  return list(urls)


def _extract_bing_image_urls(page: Page) -> list[str]:
  """Collect image URLs from Bing Images (prefers full murl over thumbnail)."""
  urls: set[str] = set()

  for anchor in page.query_selector_all("a.iusc"):
    meta = anchor.get_attribute("m")
    if not meta:
      continue
    try:
      payload = json.loads(meta)
    except json.JSONDecodeError:
      continue
    murl = payload.get("murl") or payload.get("turl")
    if murl and murl.startswith("http"):
      urls.add(murl)

  for img in page.query_selector_all("img.mimg, img[src]"):
    src = img.get_attribute("src") or ""
    if src.startswith("http") and "bing.com" not in src:
      if not _is_likely_thumbnail_url(src):
        urls.add(src)

  return list(urls)


def _search_url(engine: str, keyword: str) -> str:
  q = quote_plus(keyword)
  if engine == "google":
    return f"https://www.google.com/search?q={q}&tbm=isch&hl=fa"
  if engine == "bing":
    return f"https://www.bing.com/images/search?q={q}&form=HDRSC3&first=1"
  raise ValueError(f"Unknown engine: {engine}")


def _dismiss_consent_if_present(page: Page) -> None:
  selectors = [
    "button:has-text('Accept all')",
    "button:has-text('I agree')",
    "button:has-text('قبول همه')",
    "button:has-text('پذیرفتن همه')",
    "#L2AGLb",
    "#bnp_btn_accept",
  ]
  for selector in selectors:
    try:
      btn = page.locator(selector).first
      if btn.is_visible(timeout=1500):
        btn.click(timeout=3000)
        time.sleep(0.5)
        return
    except Exception:
      continue


def collect_urls(
    page: Page,
    *,
    engine: str,
    keyword: str,
    max_urls: int,
    scrolls: int,
) -> list[str]:
  url = _search_url(engine, keyword)
  logger.info("[%s] searching: %s", engine, keyword)

  def navigate():
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_load_state("networkidle", timeout=PAGE_TIMEOUT_MS)

  _retry(navigate, retries=NAVIGATION_RETRIES, label=f"{engine} navigation")
  _dismiss_consent_if_present(page)
  time.sleep(1.0)
  _scroll_results(page, scrolls)

  if engine == "google":
    found = _extract_google_image_urls(page)
  elif engine == "bing":
    found = _extract_bing_image_urls(page)
  else:
    found = []

  # Prefer unique, non-obvious-thumbnail URLs; cap count.
  cleaned: list[str] = []
  seen: set[str] = set()
  for item in found:
    if item in seen or _is_likely_thumbnail_url(item):
      continue
    seen.add(item)
    cleaned.append(item)
    if len(cleaned) >= max_urls:
      break

  logger.info("[%s] discovered %d candidate URLs for '%s'", engine, len(cleaned), keyword)
  return cleaned


def _download_bytes(url: str, session: requests.Session) -> bytes:
  def do_download() -> bytes:
    response = session.get(url, timeout=DOWNLOAD_TIMEOUT_SEC, allow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if content_type and not content_type.startswith("image/"):
      raise ValueError(f"unexpected content-type: {content_type}")
    return response.content

  return _retry(do_download, retries=DOWNLOAD_RETRIES, label=f"download {url[:80]}")


def save_image(
    data: bytes,
    *,
    output_dir: Path,
    source_url: str,
    keyword: str,
    engine: str,
    stats: CrawlStats,
) -> ImageRecord | None:
  ok, width, height, reason = _validate_image_bytes(data)
  if not ok:
    if reason and "dimensions" in reason or "too small" in (reason or ""):
      stats.skipped_small += 1
    else:
      stats.skipped_invalid += 1
    logger.debug("skip %s: %s", source_url[:80], reason)
    return None

  file_id = str(uuid.uuid4())
  ext = _guess_extension(source_url, None)
  out_path = output_dir / f"{file_id}{ext}"
  out_path.write_bytes(data)

  stats.downloaded += 1
  record = ImageRecord(
    file_id=file_id,
    path=str(out_path),
    source_url=source_url,
    keyword=keyword,
    engine=engine,
    width=width,
    height=height,
    bytes=len(data),
  )
  logger.info("saved %s (%dx%d, %d KB)", out_path.name, width, height, len(data) // 1024)
  return record


def append_manifest(record: ImageRecord, manifest_path: Path) -> None:
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  with manifest_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")


def crawl(
    *,
    keywords: Iterable[str],
    engines: Iterable[str],
    output_dir: Path,
    manifest_path: Path,
    max_per_keyword: int,
    scrolls: int,
) -> CrawlStats:
  output_dir.mkdir(parents=True, exist_ok=True)
  stats = CrawlStats()
  session = requests.Session()
  session.headers.update({"User-Agent": USER_AGENT})

  with sync_playwright() as playwright:
    try:
      browser = playwright.chromium.launch(headless=True)
    except Exception as exc:
      if "Executable doesn't exist" in str(exc):
        raise RuntimeError(
          "Playwright browser not installed. Run: playwright install chromium"
        ) from exc
      raise
    try:
      page = _new_page(browser)
      for keyword in keywords:
        for engine in engines:
          try:
            urls = collect_urls(
              page,
              engine=engine,
              keyword=keyword,
              max_urls=max_per_keyword * 3,
              scrolls=scrolls,
            )
          except Exception as exc:
            msg = f"{engine}/{keyword}: discovery failed: {exc}"
            logger.error(msg)
            stats.errors.append(msg)
            continue

          stats.discovered += len(urls)
          saved_for_pair = 0

          for url in urls:
            if saved_for_pair >= max_per_keyword:
              break
            try:
              data = _download_bytes(url, session)
              record = save_image(
                data,
                output_dir=output_dir,
                source_url=url,
                keyword=keyword,
                engine=engine,
                stats=stats,
              )
              if record:
                append_manifest(record, manifest_path)
                saved_for_pair += 1
            except Exception as exc:
              stats.failed += 1
              logger.debug("download failed for %s: %s", url[:80], exc)

          logger.info(
            "[%s] '%s': saved %d/%d",
            engine,
            keyword,
            saved_for_pair,
            max_per_keyword,
          )
    finally:
      browser.close()

  return stats


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Crawl Persian handwritten exam images.")
  parser.add_argument(
    "--target",
    type=int,
    default=None,
    help="Stop when total images in output-dir reaches this count (auto-adjusts max-per-keyword).",
  )
  parser.add_argument(
    "--keywords",
    nargs="+",
    default=list(DEFAULT_KEYWORDS),
    help="Search keywords (default: Persian handwritten notes/exams).",
  )
  parser.add_argument(
    "--engines",
    default=",".join(DEFAULT_ENGINES),
    help="Comma-separated engines: google,bing (default: both).",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Directory for downloaded images.",
  )
  parser.add_argument(
    "--manifest",
    type=Path,
    default=MANIFEST_PATH,
    help="JSONL manifest path for saved image metadata.",
  )
  parser.add_argument(
    "--max-per-keyword",
    type=int,
    default=40,
    help="Max images to save per keyword per engine.",
  )
  parser.add_argument(
    "--scrolls",
    type=int,
    default=MAX_SCROLLS,
    help="Number of scroll iterations per search page.",
  )
  parser.add_argument(
    "--min-width",
    type=int,
    default=MIN_WIDTH,
    help="Minimum image width in pixels.",
  )
  parser.add_argument(
    "--min-height",
    type=int,
    default=MIN_HEIGHT,
    help="Minimum image height in pixels.",
  )
  parser.add_argument("-v", "--verbose", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  _setup_logging(args.verbose)

  global MIN_WIDTH, MIN_HEIGHT
  MIN_WIDTH = args.min_width
  MIN_HEIGHT = args.min_height

  engines = [e.strip().lower() for e in args.engines.split(",") if e.strip()]
  invalid = [e for e in engines if e not in DEFAULT_ENGINES]
  if invalid:
    logger.error("Unknown engines: %s (use google,bing)", ", ".join(invalid))
    return 1

  max_per_keyword = args.max_per_keyword
  if args.target is not None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = sum(
      1
      for p in args.output_dir.iterdir()
      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    needed = max(0, args.target - existing)
    pairs = max(1, len(args.keywords) * len(engines))
    max_per_keyword = max(1, (needed + pairs - 1) // pairs)
    logger.info(
      "Target=%d existing=%d needed=%d → max_per_keyword=%d",
      args.target,
      existing,
      needed,
      max_per_keyword,
    )
    if needed == 0:
      logger.info("Already at or above target (%d). Nothing to crawl.", args.target)
      return 0

  logger.info(
    "Starting crawl | keywords=%d engines=%s output=%s",
    len(args.keywords),
    engines,
    args.output_dir,
  )

  stats = crawl(
    keywords=args.keywords,
    engines=engines,
    output_dir=args.output_dir,
    manifest_path=args.manifest,
    max_per_keyword=args.max_per_keyword,
    scrolls=args.scrolls,
  )

  logger.info(
    "Done | discovered=%d downloaded=%d skipped_small=%d skipped_invalid=%d failed=%d",
    stats.discovered,
    stats.downloaded,
    stats.skipped_small,
    stats.skipped_invalid,
    stats.failed,
  )
  if stats.errors:
    logger.warning("Errors (%d):", len(stats.errors))
    for err in stats.errors:
      logger.warning("  - %s", err)

  return 0 if stats.downloaded > 0 or not stats.errors else 1


if __name__ == "__main__":
  raise SystemExit(main())
