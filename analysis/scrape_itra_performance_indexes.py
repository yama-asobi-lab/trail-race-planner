"""
Scrape ITRA performance indexes for all TOR330 2025 finishers.

Steps:
1. Fetch the ITRA race results page for TOR330 2025 to extract RunnerSpace links
   (which embed each runner's numeric ITRA ID).
2. For each runner profile, fetch the page and parse the ITRA Performance Index.
3. Save the result as analysis/data/tor330_2025_itra_indexes.json.

Usage:
    python analysis/scrape_itra_performance_indexes.py
    python analysis/scrape_itra_performance_indexes.py --delay 2 --cooldown 30 --max-retries 5
    python analysis/scrape_itra_performance_indexes.py --resume --delay 2 --cooldown 60 --max-retries 3
    python analysis/scrape_itra_performance_indexes.py --resume --limit 50
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

RACE_RESULTS_URL = (
    "https://itra.run/Races/RaceResults/" "TOR330...Tor.des.G%C3%A9ants%C2%AE/2025/98185"
)
RUNNER_BASE_URL = "https://itra.run/RunnerSpace/"

OUTPUT_PATH = Path("analysis/data/tor330_2025_itra_indexes.json")

ITRA_HOME = "https://itra.run"

BROWSER_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
]

HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;" "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

BLOCK_PAGE_MARKERS = (
    "checking your browser before accessing",
    "just a moment",
    "verify you are human",
    "too many requests",
    "access denied",
    "cf-challenge",
    "captcha",
    "rate limit",
    "security check",
    "please wait while",
)


def build_headers() -> dict[str, str]:
    headers = dict(HEADERS)
    headers["User-Agent"] = random.choice(BROWSER_UAS)
    headers["Referer"] = ITRA_HOME + "/"
    headers["Origin"] = ITRA_HOME
    return headers


def warm_up_session(session: requests.Session, delay: float = 3.0) -> None:
    """Visit the ITRA homepage to establish cookies before hitting profile pages."""
    session.cookies.clear()
    session.headers.clear()
    session.headers.update(build_headers())
    print(f"Warming up session via {ITRA_HOME} …")
    try:
        resp = session.get(ITRA_HOME, timeout=30)
        print(
            f"  Homepage: HTTP {resp.status_code}, {len(resp.text)} chars, "
            f"{len(session.cookies)} cookies set"
        )
    except Exception as exc:
        print(f"  Warm-up failed (non-fatal): {exc}")
    time.sleep(delay)


def is_rate_limited_response(resp: requests.Response, context: str = "") -> bool:
    """Detect anti-bot blocks, CAPTCHA pages, and throttling notices.

    We intentionally avoid matching generic CDN references such as "cloudflare.com" because
    those are common in normal HTML and would create false positives on healthy pages.
    """
    status = resp.status_code
    if status in {429, 403, 503}:
        return True

    text = (resp.text or "").lower()
    if len(text.strip()) < 100:
        return True

    if any(marker in text for marker in BLOCK_PAGE_MARKERS):
        return True

    return False


def fetch_runner_links(session: requests.Session) -> list[dict]:
    """Parse the race results page and return a list of runner dicts with ITRA IDs."""
    print(f"Fetching race results: {RACE_RESULTS_URL}")
    resp = session.get(RACE_RESULTS_URL, timeout=30)
    if is_rate_limited_response(resp, RACE_RESULTS_URL):
        raise RateLimitedError(f"Rate-limited/blocked response for {RACE_RESULTS_URL}")
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    runners: list[dict] = []
    seen_ids: set[str] = set()

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        # Match /RunnerSpace/Name.Surname/123456
        m = re.search(r"/RunnerSpace/([^/?#]+)/(\d+)", href)
        if not m:
            continue

        slug = m.group(1)  # e.g. "RICHARD.Victor"
        itra_id = m.group(2)  # e.g. "207848"

        if itra_id in seen_ids:
            continue
        seen_ids.add(itra_id)

        name_text = a.get_text(strip=True)

        runners.append(
            {
                "itra_id": itra_id,
                "slug": slug,
                "name_from_link": name_text,
                "profile_url": f"{RUNNER_BASE_URL}{slug}/{itra_id}",
            }
        )

    print(f"Found {len(runners)} unique runner profile links.")
    return runners


class RateLimitedError(Exception):
    """Raised when the server returns an empty body (throttled)."""


def fetch_performance_index(session: requests.Session, profile_url: str) -> int | None:
    """Fetch a runner's profile page and return their ITRA Performance Index (int).

    Raises RateLimitedError if the server responds with an anti-bot block or other throttling page.
    Returns None if the page loads but no valid index is found (runner has no index).
    """
    resp = session.get(profile_url, timeout=30)
    if resp.status_code == 404:
        return None

    if is_rate_limited_response(resp, profile_url):
        raise RateLimitedError(
            f"Rate-limited/blocked response (status={resp.status_code}) for {profile_url}"
        )

    soup = BeautifulSoup(resp.text, "html.parser")

    # Strategy 1: look for the numeric index adjacent to "ITRA Performance Index" text
    for tag in soup.find_all(string=re.compile(r"ITRA Performance Index", re.I)):
        parent = tag.parent
        for node in [parent] + list(parent.parents)[:3]:
            text = node.get_text(" ", strip=True)
            m = re.search(r"\b(\d{3,4})\b", text)
            if m:
                val = int(m.group(1))
                if 100 <= val <= 1000:
                    return val

    # Strategy 2: scan whole page for "Performance Index" followed by a 3-4 digit number
    full_text = soup.get_text(" ")
    m = re.search(r"ITRA Performance Index\s*[^\d]*(\d{3,4})", full_text, re.I)
    if m:
        val = int(m.group(1))
        if 100 <= val <= 1000:
            return val

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape ITRA performance indexes for TOR330 2025 finishers"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between profile requests (default: 1.0)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=120.0,
        help="Base seconds to wait after a rate-limit or anti-bot block before retrying",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Maximum number of retries after a rate-limit/blocked page",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume: skip runners already present in the output JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only fetch the first N runners (for testing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if resuming
    existing: dict[str, dict] = {}
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Only keep runners where we got a definitive answer (not a rate-limit None)
        # We distinguish: runners saved with retry_needed=True should be re-fetched
        for r in data.get("runners", []):
            if not r.get("retry_needed"):
                existing[r["itra_id"]] = r
        print(f"Resuming: {len(existing)} runners already fetched (skipping retry_needed entries).")

    session = requests.Session()
    session.headers.update(build_headers())
    warm_up_session(session, delay=3.0)

    for attempt in range(args.max_retries):
        try:
            runners = fetch_runner_links(session)
            break
        except RateLimitedError:
            wait = min(args.cooldown * (2**attempt) + random.uniform(0.0, 5.0), 600.0)
            print(
                f"Results page blocked — waiting {wait:.0f}s before retrying "
                f"(attempt {attempt + 1}/{args.max_retries}) …"
            )
            time.sleep(wait)
            session.close()
            session = requests.Session()
            session.headers.update(build_headers())
            warm_up_session(session, delay=max(5.0, args.cooldown / 2))
    else:
        raise RuntimeError("Could not fetch race results after repeated anti-bot blocks.")

    if args.limit:
        runners = runners[: args.limit]

    results: list[dict] = list(existing.values())
    fetched_ids = set(existing.keys())
    errors: list[str] = []

    to_fetch = [r for r in runners if r["itra_id"] not in fetched_ids]
    print(f"Fetching performance indexes for {len(to_fetch)} runners …")

    for i, runner in enumerate(to_fetch, start=1):
        idx: int | None = None
        retry_needed = False

        for attempt in range(args.max_retries):
            try:
                idx = fetch_performance_index(session, runner["profile_url"])
                if idx is not None:
                    retry_needed = False
                    break

                if attempt < args.max_retries - 1:
                    wait = min(args.cooldown * (2**attempt) + random.uniform(0.0, 5.0), 600.0)
                    print(
                        f"  [{i}/{len(to_fetch)}] No index found on this page — waiting {wait:.0f}s "
                        f"before retrying (attempt {attempt + 1}/{args.max_retries}) …"
                    )
                    time.sleep(wait)
                    retry_needed = True
                    session.close()
                    session = requests.Session()
                    session.headers.update(build_headers())
                    warm_up_session(session, delay=max(5.0, args.cooldown / 2))
                    continue
                retry_needed = True
                break
            except RateLimitedError:
                wait = min(args.cooldown * (2**attempt) + random.uniform(0.0, 5.0), 600.0)
                print(
                    f"  [{i}/{len(to_fetch)}] Rate-limited/blocked — waiting {wait:.0f}s "
                    f"before retrying (attempt {attempt + 1}/{args.max_retries}) …"
                )
                time.sleep(wait)
                session.close()
                session = requests.Session()
                session.headers.update(build_headers())
                warm_up_session(session, delay=max(5.0, args.cooldown / 2))
                retry_needed = True
            except Exception as exc:
                print(f"  [{i}/{len(to_fetch)}] ERROR {runner['profile_url']}: {exc}")
                retry_needed = False
                break

        if idx is None and retry_needed:
            print(
                f"  [{i}/{len(to_fetch)}] Still no index after {args.max_retries} attempts; "
                "leaving this runner for a later retry."
            )

        runner["itra_performance_index"] = idx
        runner["retry_needed"] = retry_needed
        results.append(runner)
        status = f"{idx}" if idx is not None else ("RATE_LIMITED" if retry_needed else "not found")
        print(f"  [{i}/{len(to_fetch)}] {runner['name_from_link']:30s}  index={status}")

        # Save incrementally so progress is not lost on interruption
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "race": "TOR330 2025",
                    "source": RACE_RESULTS_URL,
                    "runners": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        time.sleep(args.delay)

    print(f"\nDone. {len(results)} runners saved to {output_path}")
    if errors:
        print(f"Errors on {len(errors)} URLs:")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
