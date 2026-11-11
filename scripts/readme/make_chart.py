"""Draw the README accuracy vs latency chart.

The numbers come from Vision Analysis (https://www.visionanalysis.org), the
LibreYOLO benchmark site, file ``generated/verified-results.v1.json`` in
https://github.com/LibreYOLO/vision-analysis. Only verified submissions are
used, with no edits: COCO val2017 mAP50-95 on the 500-image subset and
end-to-end p50 latency (preprocess, inference, postprocess, batch 1), all
measured on one device, an NVIDIA Jetson Orin Nano Super 8GB running
TensorRT FP16.

    python scripts/readme/make_chart.py            # redraw from the CSV
    python scripts/readme/make_chart.py --refresh  # pull fresh numbers first

Writes ``docs/readme/chart-coco-latency-{light,dark}.svg``. Standard library
only.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CSV_PATH = HERE / "coco_latency_jetson_orin.csv"
OUT_DIR = ROOT / "docs" / "readme"
SOURCE_URL = (
    "https://raw.githubusercontent.com/LibreYOLO/vision-analysis/main/"
    "generated/verified-results.v1.json"
)

HARDWARE = "jetson_orin"
FORMAT = "tensorrt"
PRECISION = "fp16"

# Family label -> model ids in size order. The line joins them in this order.
FAMILIES = {
    "YOLOv9": ["yolov9t", "yolov9s", "yolov9m", "yolov9c"],
    "RF-DETR": ["rfdetr-n", "rfdetr-s", "rfdetr-m", "rfdetr-l"],
    "D-FINE": ["dfine-n", "dfine-s", "dfine-m", "dfine-l", "dfine-x"],
    "DEIM": ["deim-n", "deim-s", "deim-m", "deim-l", "deim-x"],
    "RT-DETRv2": [
        "rtdetrv2-r18",
        "rtdetrv2-r34",
        "rtdetrv2-r50m",
        "rtdetrv2-r50",
        "rtdetrv2-r101",
    ],
}

# Categorical slots, fixed order, light and dark steps of the same hues.
COLORS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#c98500", "#d55181"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}
INK = {
    "light": {
        "text": "#1f2328",
        "muted": "#59636e",
        "grid": "#d1d9e0",
        "ring": "#ffffff",
    },
    "dark": {
        "text": "#f0f6fc",
        "muted": "#9198a1",
        "grid": "#3d444d",
        "ring": "#0d1117",
    },
}

# Where each end label sits, relative to the last point (dx, dy in px).
LABEL_OFFSET = {
    "YOLOv9": (-10, -6),
    "RF-DETR": (10, 4),
    "D-FINE": (10, -2),
    "DEIM": (10, 8),
    "RT-DETRv2": (10, 12),
}


def refresh() -> None:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:
        results = json.load(resp)["results"]
    wanted = {mid: fam for fam, ids in FAMILIES.items() for mid in ids}
    rows = []
    for r in results:
        if (
            r["hardware"].get("id") == HARDWARE
            and r["runtime"]["format"] == FORMAT
            and r["runtime"]["precision"] == PRECISION
            and r["dataset"]["id"] == "coco2017"
            and r["model"]["id"] in wanted
        ):
            rows.append(
                {
                    "family": wanted[r["model"]["id"]],
                    "model": r["model"]["id"],
                    "input_size": r["model"]["input_size"],
                    "map50_95": round(r["accuracy"]["mAP_50_95"], 4),
                    "latency_p50_ms": r["timing"]["total_ms"]["p50"],
                    "eval_images": r["dataset"]["num_images"],
                    "source_file": r["source_file"],
                }
            )
    found = {row["model"] for row in rows}
    missing = sorted(set(wanted) - found)
    if missing:
        raise SystemExit(f"missing from Vision Analysis: {missing}")
    if len(rows) != len(found):
        raise SystemExit("duplicate submissions for one model; pick one by hand")
    order = [mid for ids in FAMILIES.values() for mid in ids]
    rows.sort(key=lambda row: order.index(row["model"]))
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {CSV_PATH.relative_to(ROOT)} ({len(rows)} models)")


def load() -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {fam: [] for fam in FAMILIES}
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            series[row["family"]].append(
                (float(row["latency_p50_ms"]), float(row["map50_95"]) * 100)
            )
    return series


def render(series: dict[str, list[tuple[float, float]]], mode: str) -> str:
    ink = INK[mode]
    width, height = 820, 470
    left, right, top, bottom = 64, width - 104, 24, height - 64
    x_min, x_max, y_min, y_max = 20.0, 120.0, 40.0, 64.0

    def x(ms: float) -> float:
        return left + (ms - x_min) / (x_max - x_min) * (right - left)

    def y(ap: float) -> float:
        return bottom - (ap - y_min) / (y_max - y_min) * (bottom - top)

    font = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"
    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" font-family="{font}" font-size="13" '
            'role="img" aria-label="COCO mAP against latency on Jetson Orin Nano Super, TensorRT FP16">'
        )
    ]
    for ms in range(int(x_min), int(x_max) + 1, 20):
        out.append(
            f'<line x1="{x(ms):.1f}" y1="{top}" x2="{x(ms):.1f}" y2="{bottom}" stroke="{ink["grid"]}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x(ms):.1f}" y="{bottom + 20}" fill="{ink["muted"]}" text-anchor="middle">{ms}</text>'
        )
    for ap in range(int(y_min), int(y_max) + 1, 4):
        out.append(
            f'<line x1="{left}" y1="{y(ap):.1f}" x2="{right}" y2="{y(ap):.1f}" stroke="{ink["grid"]}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{left - 10}" y="{y(ap) + 4:.1f}" fill="{ink["muted"]}" text-anchor="end">{ap}</text>'
        )
    out.append(
        f'<text x="{(left + right) / 2:.1f}" y="{height - 18}" fill="{ink["text"]}" text-anchor="middle">'
        "Latency per image, ms (lower is faster)</text>"
    )
    out.append(
        f'<text transform="translate(18 {(top + bottom) / 2:.1f}) rotate(-90)" fill="{ink["text"]}" '
        'text-anchor="middle">COCO mAP 50-95</text>'
    )
    for color, (fam, pts) in zip(COLORS[mode], series.items()):
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x(a):.1f},{y(b):.1f}"
            for i, (a, b) in enumerate(pts)
        )
        out.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
        )
        for a, b in pts:
            out.append(
                f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="4.5" fill="{color}" '
                f'stroke="{ink["ring"]}" stroke-width="2"><title>{fam}: {b:.1f} mAP, {a:.1f} ms</title></circle>'
            )
        lx, ly = pts[-1]
        dx, dy = LABEL_OFFSET[fam]
        anchor = "end" if dx < 0 else "start"
        out.append(
            f'<text x="{x(lx) + dx:.1f}" y="{y(ly) + dy:.1f}" fill="{ink["text"]}" '
            f'font-weight="600" text-anchor="{anchor}">{fam}</text>'
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--refresh", action="store_true", help="re-read numbers from Vision Analysis"
    )
    args = parser.parse_args()
    if args.refresh or not CSV_PATH.exists():
        refresh()
    series = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        path = OUT_DIR / f"chart-coco-latency-{mode}.svg"
        path.write_text(render(series, mode))
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
