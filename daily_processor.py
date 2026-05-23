#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PIO Rajac — Daily Sentinel-2 monitoring + trends + alerts engine.

For each cloud-free Sentinel-2 acquisition in the last LOOKBACK_DAYS days,
compute per-parcel statistics for all 7 indices and save:

  public/daily/YYYY-MM-DD.json   (one per acquisition)
  public/daily/index.json        (manifest of all daily snapshots)
  public/trends/current-month.json
  public/alerts/active.json
  public/alerts/history.jsonl

Alarms are detected by comparing the latest acquisition vs. a baseline
(N days ago, same parcel). Thresholds are loaded from
public/alerts/thresholds.json if it exists, else defaults are used.

Designed to run daily via GitHub Actions cron (idempotent — skip days
already processed by checking output file existence).
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
from typing import Any, Dict, List, Optional, Tuple

import ee

ROOT = Path(__file__).resolve().parent
PARCELS = ROOT / "public" / "boundaries" / "pio-rajac-agro-parcels.geojson"
DAILY_DIR = ROOT / "public" / "daily"
TRENDS_DIR = ROOT / "public" / "trends"
ALERTS_DIR = ROOT / "public" / "alerts"

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_SCALE_M = 10.0
SCL_BAD = [1, 3, 8, 9, 10, 11]

LOOKBACK_DAYS = 10          # how many days back to scan for acquisitions
CLOUD_OK_PCT_OVER_PARCELS = 25.0   # max valid-cloud-over-parcels %
ACQUISITION_MIN_VALID_PCT = 65.0   # min valid-pixel coverage % over parcels

INDICES: List[Dict[str, Any]] = [
    {"id": "ndvi", "label": "NDVI", "name_sr": "Виталност културе",
     "expr": "(B8 - B4) / (B8 + B4)", "min": -0.2, "max": 0.95, "palette": "vegetation",
     "desc": "Виталност и густина биомaсе."},
    {"id": "evi", "label": "EVI", "name_sr": "Побољшани вeгeтациoни индeкс",
     "expr": "2.5 * ((B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0))", "min": -0.2, "max": 1.0, "palette": "vegetation",
     "desc": "Бoљи од NDVI за зреле усеве."},
    {"id": "ndre", "label": "NDRE", "name_sr": "Хлорофил (Red Edge)",
     "expr": "(B8 - B5) / (B8 + B5)", "min": -0.1, "max": 0.6, "palette": "chlorophyll",
     "desc": "Рани показатељ стреса хлорофила."},
    {"id": "ndmi", "label": "NDMI", "name_sr": "Влажност биомаce",
     "expr": "(B8 - B11) / (B8 + B11)", "min": -0.4, "max": 0.6, "palette": "moisture",
     "desc": "Caдржaj воде у листу."},
    {"id": "ndwi", "label": "NDWI", "name_sr": "Водeне површинe",
     "expr": "(B3 - B8) / (B3 + B8)", "min": -0.5, "max": 0.6, "palette": "water",
     "desc": "Стajaћа вoдa нa парцeли."},
    {"id": "nbr", "label": "NBR", "name_sr": "Опoжарена пoврeшинa",
     "expr": "(B8 - B12) / (B8 + B12)", "min": -0.5, "max": 0.9, "palette": "burn",
     "desc": "Дeтeкциja oпaжeних делoвa."},
    {"id": "bsi", "label": "BSI", "name_sr": "Голo тлo",
     "expr": "((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))", "min": -0.5, "max": 0.5, "palette": "bare",
     "desc": "Гoло тлo — отривa жeтвy, орaње, ерoзиjy."},
]

# ── Alert defaults (used when public/alerts/thresholds.json missing) ──
DEFAULT_THRESHOLDS = {
    "drought_stress": {
        "label": "Сушни стрес",
        "level": "warning",
        "color": "#f6c95f",
        "icon": "💧",
        "logic": "OR",
        "conditions": [
            {"index": "ndmi", "delta_days": 7, "drop_at_least": 0.08},
            {"index": "ndre", "delta_days": 7, "drop_at_least": 0.05}
        ]
    },
    "vitality_drop": {
        "label": "Пад виталности",
        "level": "warning",
        "color": "#f6c95f",
        "icon": "🌿",
        "logic": "AND",
        "conditions": [
            {"index": "ndvi", "delta_days": 7, "drop_at_least": 0.08}
        ]
    },
    "crisis_stress": {
        "label": "Кризни стрес",
        "level": "critical",
        "color": "#ee6a55",
        "icon": "🔥",
        "logic": "AND",
        "conditions": [
            {"index": "ndvi", "delta_days": 7, "drop_at_least": 0.15},
            {"index": "ndmi", "delta_days": 7, "drop_at_least": 0.10}
        ]
    },
    "fire_or_harvest": {
        "label": "Могући пожар или нагла жетва",
        "level": "critical",
        "color": "#ee6a55",
        "icon": "🚒",
        "logic": "AND",
        "conditions": [
            {"index": "nbr", "delta_days": 5, "drop_at_least": 0.20}
        ]
    },
    "bare_soil": {
        "label": "Голо тло / могућа ерозија",
        "level": "warning",
        "color": "#f6a763",
        "icon": "🟫",
        "logic": "OR",
        "conditions": [
            {"index": "bsi", "absolute_at_least": 0.20},
            {"index": "bsi", "delta_days": 7, "gain_at_least": 0.15}
        ]
    },
    "recovery": {
        "label": "Опоравак",
        "level": "info",
        "color": "#7cc26b",
        "icon": "🌱",
        "logic": "AND",
        "conditions": [
            {"index": "ndvi", "delta_days": 7, "gain_at_least": 0.05}
        ]
    }
}


@dataclass
class Acquisition:
    iso_date: str       # "2026-05-22"
    id: str             # "2026-05-22"
    image_ids: List[str]  # ee Image IDs for that date


def log(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    tmp.replace(path)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def mask_s2_scl(img: ee.Image) -> ee.Image:
    scl = img.select("SCL")
    mask = ee.Image.constant(1)
    for cls in SCL_BAD:
        mask = mask.And(scl.neq(cls))
    return img.updateMask(mask)


def feature_to_ee_geometry(feature: Dict[str, Any]) -> ee.Geometry:
    geom = feature["geometry"]
    if geom["type"] == "Polygon":
        return ee.Geometry.Polygon(geom["coordinates"])
    if geom["type"] == "MultiPolygon":
        return ee.Geometry.MultiPolygon(geom["coordinates"])
    raise ValueError(f"Unsupported geometry: {geom['type']}")


def parcels_union_region(parcels_fc: Dict[str, Any]) -> ee.Geometry:
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
    return ee.Geometry.Rectangle([min(lons), min(lats), max(lons), max(lats)])


def s2_collection_for_range(start: date, end: date, region: ee.Geometry) -> ee.ImageCollection:
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterDate(start.isoformat(), end.isoformat())
        .filterBounds(region)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))
    )


def build_index_image_single(img: ee.Image) -> ee.Image:
    needed = ["B2", "B3", "B4", "B5", "B8", "B11", "B12"]
    composite = img.select(needed).multiply(0.0001).rename(needed)
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


def reduce_parcel_stats_single(image: ee.Image, region: ee.Geometry, scale_m: float) -> Dict[str, Dict[str, Any]]:
    """Per-index stats for one parcel on a single-image basis."""
    band_ids = [idx["id"] for idx in INDICES]
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), sharedInputs=True)
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
            "mean": safe_round(info.get(f"{bid}_mean")),
            "median": safe_round(info.get(f"{bid}_median")),
            "stdDev": safe_round(info.get(f"{bid}_stdDev")),
            "count": int(info.get(f"{bid}_count") or 0),
        }
    return out


def sample_parcel_pixels_single(image: ee.Image, region: ee.Geometry, scale_m: float) -> List[Dict[str, Any]]:
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


def list_acquisitions_for_lookback(region: ee.Geometry, lookback_days: int) -> List[Acquisition]:
    today = date.today()
    start = today - timedelta(days=lookback_days)
    end = today + timedelta(days=1)
    coll = s2_collection_for_range(start, end, region)
    info = coll.aggregate_array("system:id").getInfo() or []
    time_info = coll.aggregate_array("system:time_start").getInfo() or []
    by_date: Dict[str, List[str]] = defaultdict(list)
    for img_id, t_ms in zip(info, time_info):
        try:
            d = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc).date()
        except Exception:
            continue
        by_date[d.isoformat()].append(img_id)
    out = []
    for iso in sorted(by_date.keys()):
        out.append(Acquisition(iso_date=iso, id=iso, image_ids=by_date[iso]))
    return out


def evaluate_acquisition(acq: Acquisition, parcels_fc: Dict[str, Any], region: ee.Geometry,
                         scale_m: float) -> Optional[Dict[str, Any]]:
    """Process one acquisition date: build composite of that day's images, compute per-parcel stats."""
    coll = ee.ImageCollection(S2_COLLECTION).filter(
        ee.Filter.inList("system:id", acq.image_ids)
    ).map(mask_s2_scl)
    n = int(coll.size().getInfo() or 0)
    if n == 0:
        return None

    # Build composite (median if multiple, otherwise single) over that day
    image = build_index_image_single(coll.median())

    # Quick coverage check over the union region using NDVI band
    ndvi = image.select("ndvi")
    cov = ndvi.reduceRegion(
        reducer=ee.Reducer.count().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=region, scale=scale_m, maxPixels=10_000, bestEffort=True,
    ).getInfo() or {}
    valid_count = int(cov.get("ndvi_count") or 0)
    # Estimate total possible pixel count
    total_area_m2 = region.area().getInfo()
    total_possible = total_area_m2 / (scale_m * scale_m)
    valid_pct = (valid_count / total_possible * 100.0) if total_possible > 0 else 0.0

    parcel_results = []
    for i, feat in enumerate(parcels_fc["features"], 1):
        props = feat.get("properties") or {}
        culture = props.get("culture") or props.get("name") or f"Parcela-{i}"
        area_ha = props.get("area_ha") or 0
        try:
            geom = feature_to_ee_geometry(feat)
        except Exception as exc:
            log(f"  WARN parcel {i}: geometry error: {exc}")
            continue
        try:
            stats = reduce_parcel_stats_single(image, geom, scale_m)
            pixels = sample_parcel_pixels_single(image, geom, scale_m)
        except Exception as exc:
            log(f"  WARN parcel {i}: {exc}")
            continue
        ndvi_count = stats.get("ndvi", {}).get("count", 0)
        parcel_results.append({
            "id": f"P{i:02d}",
            "name": props.get("name", culture),
            "culture": culture,
            "area_ha": float(area_ha),
            "valid_pixels": ndvi_count,
            "stats": stats,
            "pixels": pixels,
        })
        log(f"  parcel {i} ({culture}): {ndvi_count} valid px, NDVI={stats.get('ndvi',{}).get('mean')}")

    return {
        "ok": True,
        "meta": {
            "id": acq.id,
            "date": acq.iso_date,
            "kind": "daily",
            "source": "Copernicus Sentinel-2 SR Harmonized",
            "image_count": n,
            "image_ids": acq.image_ids,
            "valid_coverage_pct": round(valid_pct, 1),
            "scale_m": scale_m,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "indices": [{"id": i["id"], "label": i["label"], "name_sr": i["name_sr"], "desc": i["desc"],
                         "min": i["min"], "max": i["max"], "palette": i["palette"]} for i in INDICES],
        },
        "parcels": parcel_results,
    }


# ──────────────────────── Trends & Alerts engine ────────────────────────

def load_all_daily_jsons() -> List[Dict[str, Any]]:
    """Return list of daily JSON dicts, sorted ASC by date."""
    if not DAILY_DIR.exists():
        return []
    out = []
    for p in sorted(DAILY_DIR.glob("*.json")):
        if p.name == "index.json":
            continue
        try:
            j = load_json(p)
            if j and isinstance(j, dict):
                out.append(j)
        except Exception:
            continue
    return out


def find_baseline(daily_records: List[Dict[str, Any]], target_date: date, delta_days: int,
                  parcel_id: str, index_id: str) -> Optional[float]:
    """Find a parcel's mean value as close to (target_date - delta_days) as possible.
    Looks in window [target - delta - tolerance, target - delta + tolerance]
    with tolerance growing if needed."""
    desired = target_date - timedelta(days=delta_days)
    candidates = []
    for rec in daily_records:
        try:
            rec_date = date.fromisoformat(rec["meta"]["date"])
        except Exception:
            continue
        if rec_date >= target_date:
            continue
        for p in rec.get("parcels", []):
            if p.get("id") != parcel_id:
                continue
            v = (p.get("stats") or {}).get(index_id, {}).get("mean")
            if v is None:
                continue
            candidates.append((abs((rec_date - desired).days), v, rec_date))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return float(candidates[0][1])


def compute_alerts_for_parcel(daily_records: List[Dict[str, Any]], target_record: Dict[str, Any],
                              parcel: Dict[str, Any], thresholds: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate all alert types for one parcel on the target acquisition."""
    try:
        target_date = date.fromisoformat(target_record["meta"]["date"])
    except Exception:
        return []
    parcel_id = parcel["id"]
    cur_stats = parcel.get("stats") or {}
    triggered = []
    for alert_key, cfg in thresholds.items():
        logic = (cfg.get("logic") or "AND").upper()
        condition_results = []
        evidence = []
        for cond in cfg.get("conditions", []):
            idx_id = cond.get("index")
            cur_v = (cur_stats.get(idx_id) or {}).get("mean")
            cond_ok = False
            ev = {"index": idx_id, "current": cur_v}
            if cur_v is None:
                condition_results.append(False)
                evidence.append({**ev, "reason": "no_current_value"})
                continue
            if "absolute_at_least" in cond:
                cond_ok = cur_v >= cond["absolute_at_least"]
                ev["threshold"] = f">={cond['absolute_at_least']}"
            elif "drop_at_least" in cond:
                dd = int(cond.get("delta_days", 7))
                base = find_baseline(daily_records, target_date, dd, parcel_id, idx_id)
                if base is None:
                    condition_results.append(False)
                    evidence.append({**ev, "reason": f"no_baseline_{dd}d"})
                    continue
                delta = cur_v - base
                cond_ok = delta <= -float(cond["drop_at_least"])
                ev.update({"baseline": base, "delta": round(delta, 4),
                           "threshold": f"drop>={cond['drop_at_least']} over {dd}d"})
            elif "gain_at_least" in cond:
                dd = int(cond.get("delta_days", 7))
                base = find_baseline(daily_records, target_date, dd, parcel_id, idx_id)
                if base is None:
                    condition_results.append(False)
                    evidence.append({**ev, "reason": f"no_baseline_{dd}d"})
                    continue
                delta = cur_v - base
                cond_ok = delta >= float(cond["gain_at_least"])
                ev.update({"baseline": base, "delta": round(delta, 4),
                           "threshold": f"gain>={cond['gain_at_least']} over {dd}d"})
            condition_results.append(cond_ok)
            evidence.append(ev)
        if not condition_results:
            continue
        fired = all(condition_results) if logic == "AND" else any(condition_results)
        if fired:
            triggered.append({
                "type": alert_key,
                "label": cfg.get("label", alert_key),
                "level": cfg.get("level", "warning"),
                "color": cfg.get("color", "#f6c95f"),
                "icon": cfg.get("icon", "⚠️"),
                "evidence": evidence,
                "logic": logic,
            })
    return triggered


def compute_trends_for_parcel(daily_records: List[Dict[str, Any]], parcel_id: str,
                              window_days: int = 30) -> Dict[str, Any]:
    """Per-index history (last `window_days`) for a parcel."""
    cutoff = date.today() - timedelta(days=window_days)
    band_ids = [i["id"] for i in INDICES]
    series: Dict[str, List[Dict[str, Any]]] = {b: [] for b in band_ids}
    for rec in daily_records:
        try:
            rec_date = date.fromisoformat(rec["meta"]["date"])
        except Exception:
            continue
        if rec_date < cutoff:
            continue
        for p in rec.get("parcels", []):
            if p.get("id") != parcel_id:
                continue
            for bid in band_ids:
                v = (p.get("stats") or {}).get(bid, {}).get("mean")
                if v is None:
                    continue
                series[bid].append({"date": rec_date.isoformat(), "v": v})
    return series


def build_trends_and_alerts(parcels_fc: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Read all daily JSONs, build current-month trends and active alerts."""
    daily_records = load_all_daily_jsons()
    if not daily_records:
        return ({"ok": True, "trends": {}, "note": "Нема дневних снимака."},
                {"ok": True, "active": [], "generated_at": datetime.now(timezone.utc).isoformat()})

    # Load configurable thresholds
    th_file = ALERTS_DIR / "thresholds.json"
    thresholds = load_json(th_file, DEFAULT_THRESHOLDS) or DEFAULT_THRESHOLDS

    latest = daily_records[-1]
    today = date.today()
    cur_month = today.strftime("%Y-%m")

    # Trends per parcel (per index, last 30 days)
    trends = {}
    for i, feat in enumerate(parcels_fc["features"], 1):
        pid = f"P{i:02d}"
        trends[pid] = compute_trends_for_parcel(daily_records, pid, window_days=30)

    # Alerts on latest acquisition only (avoid duplicates from older days)
    active = []
    for p in latest.get("parcels", []):
        pid = p["id"]
        alerts = compute_alerts_for_parcel(daily_records, latest, p, thresholds)
        for a in alerts:
            active.append({
                "parcel_id": pid,
                "parcel_name": p.get("name"),
                "culture": p.get("culture"),
                "area_ha": p.get("area_ha"),
                "acquisition": latest["meta"]["date"],
                **a,
            })

    trends_out = {
        "ok": True,
        "month": cur_month,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "trends": trends,
        "latest_acquisition": latest["meta"]["date"],
        "acquisitions_count": len(daily_records),
    }
    alerts_out = {
        "ok": True,
        "active": active,
        "count": len(active),
        "by_level": {
            "critical": sum(1 for a in active if a["level"] == "critical"),
            "warning":  sum(1 for a in active if a["level"] == "warning"),
            "info":     sum(1 for a in active if a["level"] == "info"),
        },
        "latest_acquisition": latest["meta"]["date"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    return trends_out, alerts_out


def update_alert_history(alerts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Append new alerts to history.jsonl. Returns list of *new* alerts (not seen before for that parcel+type+acquisition)."""
    hist_path = ALERTS_DIR / "history.jsonl"
    seen: set = set()
    if hist_path.exists():
        with hist_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                key = (r.get("parcel_id"), r.get("type"), r.get("acquisition"))
                seen.add(key)
    new_alerts = []
    for a in alerts.get("active", []):
        key = (a.get("parcel_id"), a.get("type"), a.get("acquisition"))
        if key in seen:
            continue
        record = {**a, "first_seen_at": datetime.now(timezone.utc).isoformat()}
        append_jsonl(hist_path, record)
        new_alerts.append(record)
    return new_alerts


# ──────────────────────── Daily index manifest ────────────────────────

def update_daily_index(parcels_fc: Dict[str, Any]) -> None:
    items = []
    if DAILY_DIR.exists():
        for p in sorted(DAILY_DIR.glob("*.json")):
            if p.name == "index.json":
                continue
            try:
                j = load_json(p)
                m = (j or {}).get("meta") or {}
                items.append({
                    "id": m.get("id"),
                    "date": m.get("date"),
                    "image_count": m.get("image_count"),
                    "valid_coverage_pct": m.get("valid_coverage_pct"),
                    "kind": "daily",
                })
            except Exception:
                continue
    total_ha = sum(f["properties"].get("area_ha", 0) for f in parcels_fc["features"])
    idx = {
        "ok": True,
        "label": "PIO Rajac — пољопривредне парцеле (дневни)",
        "kind": "daily",
        "latest": items[-1]["id"] if items else None,
        "acquisitions": list(reversed(items)),
        "indices": [{"id": i["id"], "label": i["label"], "name_sr": i["name_sr"], "desc": i["desc"],
                     "min": i["min"], "max": i["max"], "palette": i["palette"]} for i in INDICES],
        "parcels": [{"id": f"P{i+1:02d}", "name": f["properties"].get("name"),
                     "culture": f["properties"].get("culture"),
                     "area_ha": f["properties"].get("area_ha")}
                    for i, f in enumerate(parcels_fc["features"])],
        "total_area_ha": round(total_ha, 2),
        "source": "Copernicus Sentinel-2 SR Harmonized",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    write_json(DAILY_DIR / "index.json", idx)


def write_default_thresholds_if_missing() -> None:
    th_path = ALERTS_DIR / "thresholds.json"
    if not th_path.exists():
        write_json(th_path, DEFAULT_THRESHOLDS)
        log(f"Wrote default thresholds → {th_path}")


# ──────────────────────── Main ────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS,
                        help="How many days back to scan for acquisitions (default 10).")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE_M)
    parser.add_argument("--force", action="store_true",
                        help="Reprocess dates even if their JSON already exists.")
    args = parser.parse_args()

    if not PARCELS.exists():
        raise RuntimeError(f"Nedostaje: {PARCELS}")
    parcels_fc = load_json(PARCELS)
    n_parcels = len(parcels_fc["features"])
    log(f"Loaded {n_parcels} parcels")

    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    TRENDS_DIR.mkdir(parents=True, exist_ok=True)
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    write_default_thresholds_if_missing()

    init_ee()
    region = parcels_union_region(parcels_fc)

    acquisitions = list_acquisitions_for_lookback(region, args.lookback_days)
    log(f"Found {len(acquisitions)} acquisitions in last {args.lookback_days} days")

    processed = 0
    skipped = 0
    for acq in acquisitions:
        out_path = DAILY_DIR / f"{acq.id}.json"
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        log(f"Processing {acq.id} ({len(acq.image_ids)} S2 image(s))...")
        try:
            result = evaluate_acquisition(acq, parcels_fc, region, args.scale)
        except Exception as exc:
            log(f"ERROR {acq.id}: {type(exc).__name__}: {exc}")
            continue
        if not result:
            log(f"  no usable data on {acq.id}")
            continue
        valid_pct = (result.get("meta") or {}).get("valid_coverage_pct", 0)
        if valid_pct < ACQUISITION_MIN_VALID_PCT:
            log(f"  rejected {acq.id}: valid coverage {valid_pct}% < {ACQUISITION_MIN_VALID_PCT}% (too cloudy)")
            continue
        write_json(out_path, result)
        processed += 1
        log(f"  OK wrote {out_path.name} (coverage {valid_pct}%)")

    log(f"Processed: {processed} new, skipped: {skipped} existing")

    # Build trends + alerts
    trends, alerts = build_trends_and_alerts(parcels_fc)
    write_json(TRENDS_DIR / "current-month.json", trends)
    write_json(ALERTS_DIR / "active.json", alerts)
    new_alerts = update_alert_history(alerts)
    log(f"Active alerts: {alerts['count']} ({alerts['by_level']['critical']} critical, "
        f"{alerts['by_level']['warning']} warning, {alerts['by_level']['info']} info); "
        f"new this run: {len(new_alerts)}")

    # Update manifest
    update_daily_index(parcels_fc)

    # Emit new alerts for email notifier (GitHub Actions consumes this artifact)
    new_alerts_path = ALERTS_DIR / "new-this-run.json"
    write_json(new_alerts_path, {"ok": True, "new_alerts": new_alerts,
                                 "generated_at": datetime.now(timezone.utc).isoformat()})

    log("DONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
