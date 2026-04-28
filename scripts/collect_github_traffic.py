from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OWNER = os.environ["OWNER"]
REPO = os.environ["REPO"]
TOKEN = os.environ["GITHUB_TOKEN_FOR_TRAFFIC"]

BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}"
OUT_DIR = Path("analytics/github-traffic")
RAW_DIR = OUT_DIR / "raw"


def github_get(path: str) -> dict | list:
    request = Request(
        f"{BASE_URL}{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {TOKEN}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "libreyolo-traffic-collector",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"ERROR {e.code} on GET {path}: {body}", file=sys.stderr)
        raise
    except URLError as e:
        print(f"ERROR network failure on GET {path}: {e}", file=sys.stderr)
        raise


def upsert_csv(path: Path, key_fields: list[str], rows: list[dict]) -> None:
    existing: dict[tuple, dict] = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[tuple(row[k] for k in key_fields)] = row

    for row in rows:
        key = tuple(str(row[k]) for k in key_fields)
        existing[key] = {k: str(v) for k, v in row.items()}

    all_rows = list(existing.values())
    all_fields = sorted({f for row in all_rows for f in row.keys()})
    preferred = list(dict.fromkeys(key_fields + [
        "count", "uniques", "referrer", "path", "title",
        "snapshot_date", "collected_at",
    ]))
    fieldnames = [f for f in preferred if f in all_fields] + [
        f for f in all_fields if f not in preferred
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            sorted(all_rows, key=lambda r: tuple(r.get(k, "") for k in key_fields))
        )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    collected_at = datetime.now(timezone.utc).isoformat()
    snapshot_date = collected_at[:10]

    views_day = github_get("/traffic/views?per=day")
    views_week = github_get("/traffic/views?per=week")
    clones_day = github_get("/traffic/clones?per=day")
    clones_week = github_get("/traffic/clones?per=week")
    referrers = github_get("/traffic/popular/referrers")
    paths = github_get("/traffic/popular/paths")

    raw_snapshot = {
        "owner": OWNER,
        "repo": REPO,
        "collected_at": collected_at,
        "views_day": views_day,
        "views_week": views_week,
        "clones_day": clones_day,
        "clones_week": clones_week,
        "referrers": referrers,
        "paths": paths,
    }
    (RAW_DIR / f"{snapshot_date}.json").write_text(
        json.dumps(raw_snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )

    daily_rows = []
    for item in views_day.get("views", []):
        daily_rows.append({
            "date": item["timestamp"][:10], "metric": "views",
            "count": item["count"], "uniques": item["uniques"],
            "collected_at": collected_at,
        })
    for item in clones_day.get("clones", []):
        daily_rows.append({
            "date": item["timestamp"][:10], "metric": "clones",
            "count": item["count"], "uniques": item["uniques"],
            "collected_at": collected_at,
        })
    upsert_csv(OUT_DIR / "daily.csv", ["date", "metric"], daily_rows)

    weekly_rows = []
    for item in views_week.get("views", []):
        weekly_rows.append({
            "week_start": item["timestamp"][:10], "metric": "views",
            "count": item["count"], "uniques": item["uniques"],
            "collected_at": collected_at,
        })
    for item in clones_week.get("clones", []):
        weekly_rows.append({
            "week_start": item["timestamp"][:10], "metric": "clones",
            "count": item["count"], "uniques": item["uniques"],
            "collected_at": collected_at,
        })
    upsert_csv(OUT_DIR / "weekly.csv", ["week_start", "metric"], weekly_rows)

    upsert_csv(
        OUT_DIR / "referrers.csv",
        ["snapshot_date", "referrer"],
        [{
            "snapshot_date": snapshot_date, "referrer": item["referrer"],
            "count": item["count"], "uniques": item["uniques"],
            "collected_at": collected_at,
        } for item in referrers],
    )

    upsert_csv(
        OUT_DIR / "paths.csv",
        ["snapshot_date", "path"],
        [{
            "snapshot_date": snapshot_date, "path": item["path"],
            "title": item.get("title", ""),
            "count": item["count"], "uniques": item["uniques"],
            "collected_at": collected_at,
        } for item in paths],
    )


if __name__ == "__main__":
    main()
