#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PIO Rajac — Agricultural parcels Sentinel-2 monitoring.

Builds monthly JSON files for each agricultural parcel in PIO Rajac.
Statistics per parcel + aggregated per culture (Пшеница, Кромпир, Пашњак, Ливада).

Each parcel typically has 20-75 pixels at 10m resolution, so we use:
- ee.Image.reduceRegion() for statistics (no FeatureCollection limits)
- ee.Image.sample() per parcel for pixel-level visualization

Outputs:
- public/results/index.json
- public/results/YYYY-MM.json   (monthly composites with per-parcel + per-culture)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import ee

ROOT = Path(__file__).resolve().parent
PARCELS = ROOT / "public" / "boundaries" / "pio-rajac-agro-parcels.geojson"
RESULTS = ROOT / "public" / "results"

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_SCALE_M = 10.0  # Native Sentinel-2 resolution for B2, B3, B4, B8
SCL_BAD = [1, 3, 8, 9, 10, 11]

INDICES: List[Dict[str, Any]] = [
    {"id": "ndvi", "label": "NDVI", "name_sr": "Виталност културе",
     "expr": "(B8 - B4) / (B8 + B4)", "min": -0.2, "max": 0.95, "palette": "vegetation",
     "desc": "Виталност и густина биомaсе. Кoрисно код cвих усева."},
    {"id": "evi", "label": "EVI", "name_sr": "Побољшани вeгeтациoни индeкс",
     "expr": "2.5 * ((B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0))", "min": -0.2, "max": 1.0, "palette": "vegetation",
     "desc": "Густи покривач. Бoљи од NDVI за зреле усеве."},
    {"id": "ndre", "label": "NDRE", "name_sr": "Хлорофил (Red Edge)",
     "expr": "(B8 - B5) / (B8 + B5)", "min": -0.1, "max": 0.6, "palette": "chlorophyll",
     "desc": "Концeнтpaциja хлорофила. Рани показатeљ стрeсa код Кромпира и Пшеницe."},
    {"id": "ndmi", "label": "NDMI", "name_sr": "Влажност биомаce",
     "expr": "(B8 - B11) / (B8 + B11)", "min": -0.4, "max": 0.6, "palette": "moisture",
     "desc": "Caдржaj воде у листу. Кључно за прaћeњe нaводнjaвaња и сyшног стрeсa."},
    {"id": "ndwi", "label": "NDWI", "name_sr": "Водeне површинe",
     "expr": "(B3 - B8) / (B3 + B8)", "min": -0.5, "max": 0.6, "palette": "water",
     "desc": "Стajaћа вoдa нa парцeли (плaвљeње, локвe)."},
    {"id": "nbr", "label": "NBR", "name_sr": "Опoжарена пoврeшинa",
     "expr": "(B8 - B12) / (B8 + B12)", "min": -0.5, "max": 0.9, "palette": "burn",
     "desc": "Дeтeкциja oпaжeних делoвa парцeла (захтев бeзбeдности)."},
    {"id": "bsi", "label": "BSI", "name_sr": "Голo тлo (после жетвe)",
     "expr": "((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))", "min": -0.5, "max": 0.5, "palette": "bare",
     "desc": "Гoло тлo — отривa жeтвy, орaње, ерoзиjy."},
]


@dataclass
class Period:
    id: str
    label: str
    start: str
    end: str


def log(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    tmp.replace(path)


def month_after(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def previous_month_start(d: date) -> date:
    first = date(d.year, d.month, 1)
    prev_last = first - timedelta(days=1)
    return date(prev_last.year, prev_last.month, 1)


def period_from_month_start(start: date) -> Period:
    end = month_after(start)
    return Period(
        id=f"{start.year}-{start.month:02d}",
        label=f"{start.month:02d}/{start.year}",
        start=start.isoformat(),
        end=end.isoformat(),
    )


def candidate_months_since(start_year: int, start_month: int) -> List[Period]:
    """Yield months from (start_year, start_month) through the last complete month."""
    today = date.today()
    last_complete = previous_month_start(today)
    cur = date(start_year, start_month, 1)
    out = []
    while cur <= last_complete:
        out.append(period_from_month_start(cur))
        cur = month_after(cur)
    return out


def init_ee() -> None:
    project = os.environ.get("GEE_PROJECT", "deft-epigram-414409").strip()
    secret = os.environ.get("GEE_SERVICE_ACCOUNT_JSON", "").strip()
    if not secret:
        raise RuntimeError("Nedostaje GitHub secret GEE_SERVICE_ACCOUNT_JSON.")
    try:
        key = json.loads(secret)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GEE_SERVICE_ACCOUNT_JSON nije validan JSON.") from exc
    email = key.get("client_email")
    if not email:
        raise RuntimeError("Service-account JSON nema client_email.")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        json.dump(key, tmp)
        tmp_path = tmp.name
    ee.Initialize(ee.ServiceAccountCredentials(email, key_file=tmp_path), project=project)
    log(f"Earth Engine initialized for project: {project}; SA: {email}")


def safe_round(v: Any, ndigits: int = 4) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, ndigits)
    except Exception:
        return None


def mask_s2_scl(img: ee.Image) -> ee.Image:
    scl = img.select("SCL")
    mask = ee.Image.constant(1)
    for cls in SCL_BAD:
        mask = mask.And(scl.neq(cls))
    return img.updateMask(mask)


def s2_collection_for_period(period: Period, region: ee.Geometry, max_cloud_pct: int = 80) -> ee.ImageCollection:
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterDate(period.start, period.end)
        .filterBounds(region)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(mask_s2_scl)
    )


def build_index_image(collection: ee.ImageCollection) -> ee.Image:
    needed = ["B2", "B3", "B4", "B5", "B8", "B11", "B12"]
    composite = collection.select(needed).median().multiply(0.0001).rename(needed)
    bands = []
    for idx in INDICES:
        band = composite.expression(idx["expr"], {b: composite.select(b) for b in needed}).rename(idx["id"])
        bands.append(band)
    return ee.Image.cat(bands)


def lonlat_bbox_from_center(lon: float, lat: float, scale_m: float) -> List[float]:
    dlat = scale_m / 111_320.0
    dlon = scale_m / (111_320.0 * max(0.15, math.cos(math.radians(lat))))
    return [round(lon - dlon/2, 6), round(lat - dlat/2, 6),
            round(lon + dlon/2, 6), round(lat + dlat/2, 6)]


def feature_to_ee_geometry(feature: Dict[str, Any]) -> ee.Geometry:
    geom = feature["geometry"]
    if geom["type"] == "Polygon":
        return ee.Geometry.Polygon(geom["coordinates"])
    if geom["type"] == "MultiPolygon":
        return ee.Geometry.MultiPolygon(geom["coordinates"])
    raise ValueError(f"Unsupported geometry: {geom['type']}")


def reduce_parcel_stats(image: ee.Image, region: ee.Geometry, scale_m: float) -> Dict[str, Dict[str, Any]]:
    """Per-index statistics for a single parcel."""
    band_ids = [idx["id"] for idx in INDICES]
    reducer = (
        ee.Reducer.minMax()
        .combine(ee.Reducer.mean(), sharedInputs=True)
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.percentile([10, 90]), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )
    info = image.select(band_ids).reduceRegion(
        reducer=reducer, geometry=region, scale=scale_m,
        maxPixels=10_000, bestEffort=True, tileScale=2,
    ).getInfo() or {}
    out = {}
    for bid in band_ids:
        out[bid] = {
            "min": safe_round(info.get(f"{bid}_min")),
            "max": safe_round(info.get(f"{bid}_max")),
            "mean": safe_round(info.get(f"{bid}_mean")),
            "median": safe_round(info.get(f"{bid}_median")),
            "p10": safe_round(info.get(f"{bid}_p10")),
            "p90": safe_round(info.get(f"{bid}_p90")),
            "stdDev": safe_round(info.get(f"{bid}_stdDev")),
            "count": int(info.get(f"{bid}_count") or 0),
        }
    return out


def sample_parcel_pixels(image: ee.Image, region: ee.Geometry, scale_m: float) -> List[Dict[str, Any]]:
    """Sample individual pixels in a parcel for visualization."""
    band_ids = [idx["id"] for idx in INDICES]
    fc = image.select(band_ids).sample(region=region, scale=scale_m, geometries=True, tileScale=2)
    data = fc.getInfo() or {}
    features = data.get("features", []) if isinstance(data, dict) else []
    out = []
    for i, feat in enumerate(features, 1):
        props = feat.get("properties") or {}
        coords = (feat.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        values = {}
        any_valid = False
        for band in band_ids:
            v = props.get(band)
            r = safe_round(v, 4) if v is not None else None
            values[band] = r
            if r is not None:
                any_valid = True
        if not any_valid:
            continue
        out.append({
            "id": f"px-{i:04d}",
            "lon": round(lon, 6),
            "lat": round(lat, 6),
            "bbox": lonlat_bbox_from_center(lon, lat, scale_m),
            "v": values,
        })
    return out


def aggregate_by_culture(parcel_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-parcel stats by culture name (Пшеница, Кромпир, Пашњак, Ливада)."""
    by_culture = defaultdict(lambda: {"parcels": [], "area_ha": 0.0, "pixel_count": 0})
    for p in parcel_results:
        culture = p["culture"]
        by_culture[culture]["parcels"].append(p["id"])
        by_culture[culture]["area_ha"] += p["area_ha"]

    out = {}
    for culture, info in by_culture.items():
        agg = {"area_ha": round(info["area_ha"], 2),
               "parcels_count": len(info["parcels"]),
               "parcel_ids": info["parcels"],
               "stats": {}}
        # Pool stats per index, weighting by pixel count (count from NDVI)
        for idx in INDICES:
            bid = idx["id"]
            vals_count = []
            for p in parcel_results:
                if p["culture"] != culture:
                    continue
                st = p["stats"].get(bid, {})
                if st.get("mean") is None or st.get("count", 0) == 0:
                    continue
                vals_count.append((st["mean"], st["count"]))
            if vals_count:
                total_w = sum(c for _, c in vals_count)
                if total_w > 0:
                    weighted_mean = sum(v*c for v, c in vals_count) / total_w
                    agg["stats"][bid] = {
                        "mean": round(weighted_mean, 4),
                        "pixel_count": total_w,
                    }
        out[culture] = agg
    return out


def build_period_result(period: Period, parcels_fc: Dict[str, Any], scale_m: float) -> Optional[Dict[str, Any]]:
    """Process one period (month) for all parcels."""
    log(f"Processing {period.id} at {scale_m}m...")

    # Combined region for image building (union of all parcels' bboxes)
    all_coords = []
    for f in parcels_fc["features"]:
        g = f["geometry"]
        if g["type"] == "Polygon":
            all_coords.extend(g["coordinates"][0])
        elif g["type"] == "MultiPolygon":
            for poly in g["coordinates"]:
                all_coords.extend(poly[0])
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    region = ee.Geometry.Rectangle(bbox)

    coll = s2_collection_for_period(period, region)
    n = int(coll.size().getInfo() or 0)
    if n == 0:
        log(f"  No Sentinel-2 images for {period.id}")
        return None

    image = build_index_image(coll)

    # Per-parcel processing
    parcel_results = []
    for i, feat in enumerate(parcels_fc["features"], 1):
        props = feat.get("properties") or {}
        culture = props.get("culture") or props.get("name") or f"Parcela-{i}"
        area_ha = props.get("area_ha") or 0
        try:
            geom = feature_to_ee_geometry(feat)
        except Exception as exc:
            log(f"  WARN parcel {i} ({culture}): geometry error: {exc}")
            continue
        try:
            stats = reduce_parcel_stats(image, geom, scale_m)
            pixels = sample_parcel_pixels(image, geom, scale_m)
        except Exception as exc:
            log(f"  WARN parcel {i} ({culture}): {exc}")
            continue
        ndvi_count = stats.get("ndvi", {}).get("count", 0)
        if ndvi_count == 0:
            log(f"  parcel {i} ({culture}): 0 valid pixels (clouded)")
        parcel_results.append({
            "id": f"P{i:02d}",
            "name": props.get("name", culture),
            "culture": culture,
            "area_ha": float(area_ha),
            "stats": stats,
            "pixels": pixels,
        })
        log(f"  OK parcel {i} ({culture}): {len(pixels)} pixels, NDVI={stats.get('ndvi',{}).get('mean')}")

    by_culture = aggregate_by_culture(parcel_results)

    return {
        "ok": True,
        "meta": {
            "id": period.id,
            "label": period.label,
            "kind": "monthly",
            "source": "Copernicus Sentinel-2 SR Harmonized",
            "dataset": S2_COLLECTION,
            "scale_m": scale_m,
            "date_start": period.start,
            "date_end": period.end,
            "image_count": n,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "indices": [{"id": i["id"], "label": i["label"], "name_sr": i["name_sr"], "desc": i["desc"],
                         "min": i["min"], "max": i["max"], "palette": i["palette"]} for i in INDICES],
        },
        "parcels": parcel_results,
        "by_culture": by_culture,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2026)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE_M)
    args = parser.parse_args()

    if not PARCELS.exists():
        raise RuntimeError(f"Nedostaje: {PARCELS}")
    parcels_fc = load_json(PARCELS)
    n_parcels = len(parcels_fc["features"])
    total_ha = sum(f["properties"].get("area_ha", 0) for f in parcels_fc["features"])
    log(f"Loaded {n_parcels} parcels, total {total_ha:.2f} ha")

    RESULTS.mkdir(parents=True, exist_ok=True)
    init_ee()

    periods = candidate_months_since(args.start_year, args.start_month)
    log(f"Will process {len(periods)} months from {args.start_year}-{args.start_month:02d}")

    index_periods = []
    for period in periods:
        try:
            result = build_period_result(period, parcels_fc, args.scale)
        except Exception as exc:
            log(f"ERROR month {period.id}: {type(exc).__name__}: {exc}")
            continue
        if not result:
            continue
        write_json(RESULTS / f"{period.id}.json", result)
        index_periods.append({
            "id": period.id, "label": period.label, "kind": "monthly",
            "image_count": result["meta"]["image_count"], "scale_m": args.scale,
        })

    index = {
        "ok": True,
        "label": "PIO Rajac — пољопривредне парцеле",
        "latest": index_periods[0]["id"] if index_periods else None,
        "periods": list(reversed(index_periods)),
        "indices": [{"id": i["id"], "label": i["label"], "name_sr": i["name_sr"], "desc": i["desc"],
                     "min": i["min"], "max": i["max"], "palette": i["palette"]} for i in INDICES],
        "cultures": sorted({f["properties"].get("culture", "?") for f in parcels_fc["features"]}),
        "parcels": [{"id": f"P{i+1:02d}", "name": f["properties"].get("name"),
                     "culture": f["properties"].get("culture"),
                     "area_ha": f["properties"].get("area_ha")}
                    for i, f in enumerate(parcels_fc["features"])],
        "total_area_ha": round(total_ha, 2),
        "scale_m": args.scale,
        "source": "Copernicus Sentinel-2 SR Harmonized",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "coverage": {
            "count": len(index_periods),
            "first": index_periods[-1]["id"] if index_periods else None,
            "last": index_periods[0]["id"] if index_periods else None,
        },
        "note": "Месечни Sentinel-2 мониторинг пољопривредних парцела ПИО „Рајац\" на 10 m. Теренска провера + сателитски подаци.",
    }
    write_json(RESULTS / "index.json", index)
    log(f"DONE: {len(index_periods)} periods written")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
