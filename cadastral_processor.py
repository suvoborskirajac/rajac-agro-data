#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PIO Rajac — Sentinel-2 monitoring for ALL cadastral parcels inside PIO Rajac.

The processor reads the prepared cadastral parcel GeoJSON and computes DIRECT
numeric statistics from COPERNICUS/S2_SR_HARMONIZED reflectance in Google Earth
Engine. It does not reconstruct NDVI values from PNG colours.

For each accepted acquisition date it calculates, per parcel:
  * NDVI, EVI, NDRE, NDMI, NDWI, NBR, BSI: mean, median, stdDev, valid count
  * valid Sentinel-2 coverage inside the parcel
  * NDVI class percentages (5 classes used by the owner page)
  * centroid/direction of every NDVI class within the parcel

Output is organised per parcel for fast use by /za-vlasnike/satelit/:
  public/cadastral/index.json
  public/cadastral/audit.json
  public/cadastral/parcels/<cadastral-code>/<parcel_id>.json

The script is idempotent. Existing acquisition dates are skipped unless --force
is supplied. It supports both 2026 backfill and daily lookback operation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ee

ROOT = Path(__file__).resolve().parent
PARCELS_PATH = ROOT / "public" / "boundaries" / "pio-rajac-cadastral-parcels.geojson"
OUT_ROOT = ROOT / "public" / "cadastral"
PARCEL_OUT = OUT_ROOT / "parcels"
INDEX_PATH = OUT_ROOT / "index.json"
AUDIT_PATH = OUT_ROOT / "audit.json"

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
SCL_BAD = [1, 3, 8, 9, 10, 11]
DEFAULT_SCALE_M = 10.0
DEFAULT_BATCH_SIZE = 750
MIN_BATCH_SIZE = 40
MIN_GOOD_PARCELS_PCT = 60.0
MIN_PARCEL_VALID_COVERAGE_PCT = 50.0
MAX_METADATA_CLOUD = 80
GETINFO_RETRIES = 4

INDICES: List[Dict[str, Any]] = [
    {"id":"ndvi","label":"NDVI","name_sr":"Вегетациона активност / зелена маса","expr":"(B8-B4)/(B8+B4)"},
    {"id":"evi","label":"EVI","name_sr":"Побољшани вегетациони индекс","expr":"2.5*((B8-B4)/(B8+6.0*B4-7.5*B2+1.0))"},
    {"id":"ndre","label":"NDRE","name_sr":"Хлорофил (Red Edge)","expr":"(B8-B5)/(B8+B5)"},
    {"id":"ndmi","label":"NDMI","name_sr":"Влажност вегетације","expr":"(B8-B11)/(B8+B11)"},
    {"id":"ndwi","label":"NDWI","name_sr":"Вода / влажне површине","expr":"(B3-B8)/(B3+B8)"},
    {"id":"nbr","label":"NBR","name_sr":"Пожар / нагла промена биомасе","expr":"(B8-B12)/(B8+B12)"},
    {"id":"bsi","label":"BSI","name_sr":"Голо земљиште","expr":"((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))"},
]

NDVI_CLASSES: List[Tuple[str, str, float, Optional[float]]] = [
    ("dark_green", "Веома јака зелена маса", 0.75, None),
    ("green", "Добра зелена маса", 0.55, 0.75),
    ("light_green", "Умерена зелена маса", 0.35, 0.55),
    ("yellow", "Слаба / жућкаста вегетација", 0.10, 0.35),
    ("brown", "Врло слаба или без активне вегетације", -999.0, 0.10),
]

@dataclass(frozen=True)
class Acquisition:
    iso_date: str
    image_ids: Tuple[str, ...]


def log(msg: str) -> None:
    print(msg, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def write_json(path: Path, data: Any, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        else:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    tmp.replace(path)


def safe_float(v: Any, digits: int = 4) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
        if not math.isfinite(x):
            return None
        return round(x, digits)
    except Exception:
        return None


def init_ee() -> None:
    project = os.environ.get("GEE_PROJECT", "deft-epigram-414409").strip()
    secret = os.environ.get("GEE_SERVICE_ACCOUNT_JSON", "").strip()
    if not secret:
        raise RuntimeError("Nedostaje GitHub secret GEE_SERVICE_ACCOUNT_JSON")
    try:
        key = json.loads(secret)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GEE_SERVICE_ACCOUNT_JSON nije validan JSON") from exc
    email = key.get("client_email")
    if not email:
        raise RuntimeError("Service-account JSON nema client_email")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        json.dump(key, tmp)
        key_path = tmp.name
    try:
        ee.Initialize(ee.ServiceAccountCredentials(email, key_file=key_path), project=project)
    finally:
        try:
            os.unlink(key_path)
        except OSError:
            pass
    log(f"Earth Engine initialized: project={project}; account={email}")


def feature_to_ee_geometry(feature: Dict[str, Any]) -> ee.Geometry:
    geom = feature["geometry"]
    gt = geom.get("type")
    if gt == "Polygon":
        return ee.Geometry.Polygon(geom["coordinates"], proj="EPSG:4326", geodesic=False)
    if gt == "MultiPolygon":
        return ee.Geometry.MultiPolygon(geom["coordinates"], proj="EPSG:4326", geodesic=False)
    raise ValueError(f"Unsupported geometry: {gt}")


def mask_s2_scl(img: ee.Image) -> ee.Image:
    scl = img.select("SCL")
    mask = ee.Image.constant(1)
    for cls in SCL_BAD:
        mask = mask.And(scl.neq(cls))
    return img.updateMask(mask)


def build_index_image(img: ee.Image) -> ee.Image:
    needed = ["B2", "B3", "B4", "B5", "B8", "B11", "B12"]
    refl = img.select(needed).multiply(0.0001).rename(needed)
    bands = []
    env = {b: refl.select(b) for b in needed}
    for idx in INDICES:
        bands.append(refl.expression(idx["expr"], env).rename(idx["id"]))
    return ee.Image.cat(bands).toFloat()


def build_aux_image(index_img: ee.Image) -> ee.Image:
    ndvi = index_img.select("ndvi")
    valid = ndvi.multiply(0).add(1).unmask(0).rename("valid_frac")
    ll = ee.Image.pixelLonLat()
    bands: List[ee.Image] = [valid]
    for cid, _label, lo, hi in NDVI_CLASSES:
        cond = ndvi.gte(lo) if hi is None else ndvi.gte(lo).And(ndvi.lt(hi))
        class_area = cond.unmask(0).rename(f"{cid}_frac")
        class_lon = ll.select("longitude").updateMask(cond).rename(f"{cid}_lon")
        class_lat = ll.select("latitude").updateMask(cond).rename(f"{cid}_lat")
        bands.extend([class_area, class_lon, class_lat])
    return ee.Image.cat(bands).toFloat()


def collection_region(parcels: Sequence[Dict[str, Any]]) -> ee.Geometry:
    minx = min(float(f["properties"]["bbox"][0]) for f in parcels)
    miny = min(float(f["properties"]["bbox"][1]) for f in parcels)
    maxx = max(float(f["properties"]["bbox"][2]) for f in parcels)
    maxy = max(float(f["properties"]["bbox"][3]) for f in parcels)
    return ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:4326", geodesic=False)


def list_acquisitions(start: date, end_inclusive: date, region: ee.Geometry) -> List[Acquisition]:
    end_exclusive = end_inclusive + timedelta(days=1)
    coll = (ee.ImageCollection(S2_COLLECTION)
            .filterDate(start.isoformat(), end_exclusive.isoformat())
            .filterBounds(region)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", MAX_METADATA_CLOUD)))
    ids = retry_getinfo(coll.aggregate_array("system:id"), "scene ids") or []
    times = retry_getinfo(coll.aggregate_array("system:time_start"), "scene dates") or []
    by_date: Dict[str, List[str]] = defaultdict(list)
    for img_id, t_ms in zip(ids, times):
        try:
            d = datetime.fromtimestamp(float(t_ms) / 1000.0, tz=timezone.utc).date().isoformat()
        except Exception:
            continue
        by_date[d].append(str(img_id))
    return [Acquisition(d, tuple(sorted(v))) for d, v in sorted(by_date.items())]


def retry_getinfo(obj: Any, label: str) -> Any:
    last = None
    for attempt in range(1, GETINFO_RETRIES + 1):
        try:
            return obj.getInfo()
        except Exception as exc:  # network/quota/backend errors
            last = exc
            wait = min(30, 2 ** attempt)
            log(f"WARN getInfo {label} attempt {attempt}/{GETINFO_RETRIES}: {exc}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Earth Engine getInfo failed for {label}: {last}")


def make_ee_fc(features: Sequence[Dict[str, Any]]) -> ee.FeatureCollection:
    ee_features = []
    for ft in features:
        pid = str(ft["properties"]["parcela_id"])
        ee_features.append(ee.Feature(feature_to_ee_geometry(ft), {"pid": pid}))
    return ee.FeatureCollection(ee_features)


def reduce_batch(index_img: ee.Image, aux_img: ee.Image, features: Sequence[Dict[str, Any]],
                 scale_m: float, depth: int = 0) -> Dict[str, Dict[str, Any]]:
    """Batch reduceRegions with automatic split fallback for large/complex batches."""
    if not features:
        return {}
    try:
        fc = make_ee_fc(features)
        stats_reducer = (ee.Reducer.mean()
                         .combine(ee.Reducer.median(), sharedInputs=True)
                         .combine(ee.Reducer.stdDev(), sharedInputs=True)
                         .combine(ee.Reducer.count(), sharedInputs=True))
        stats_fc = index_img.reduceRegions(
            collection=fc, reducer=stats_reducer, scale=scale_m, tileScale=4,
        )
        aux_fc = aux_img.reduceRegions(
            collection=fc, reducer=ee.Reducer.mean(), scale=scale_m, tileScale=4,
        )
        stats_info = retry_getinfo(stats_fc, f"stats batch n={len(features)}") or {}
        aux_info = retry_getinfo(aux_fc, f"aux batch n={len(features)}") or {}
        stats_map = {str((x.get("properties") or {}).get("pid")): (x.get("properties") or {})
                     for x in stats_info.get("features", [])}
        aux_map = {str((x.get("properties") or {}).get("pid")): (x.get("properties") or {})
                   for x in aux_info.get("features", [])}
        out: Dict[str, Dict[str, Any]] = {}
        for ft in features:
            pid = str(ft["properties"]["parcela_id"])
            out[pid] = {"stats": stats_map.get(pid, {}), "aux": aux_map.get(pid, {})}
        return out
    except Exception as exc:
        if len(features) <= MIN_BATCH_SIZE:
            raise
        half = len(features) // 2
        log(f"WARN batch n={len(features)} failed ({exc}); splitting to {half}+{len(features)-half}")
        left = reduce_batch(index_img, aux_img, features[:half], scale_m, depth + 1)
        right = reduce_batch(index_img, aux_img, features[half:], scale_m, depth + 1)
        left.update(right)
        return left


def direction_code(lon: Optional[float], lat: Optional[float], bbox: Sequence[float]) -> Optional[str]:
    if lon is None or lat is None or len(bbox) != 4:
        return None
    minx, miny, maxx, maxy = map(float, bbox)
    dxden = max((maxx - minx) / 2.0, 1e-9)
    dyden = max((maxy - miny) / 2.0, 1e-9)
    dx = (lon - (minx + maxx) / 2.0) / dxden
    dy = (lat - (miny + maxy) / 2.0) / dyden
    t = 0.25
    h = "" if abs(dx) < t else ("E" if dx > 0 else "W")
    v = "" if abs(dy) < t else ("N" if dy > 0 else "S")
    if not h and not v:
        return "C"
    return v + h


def parse_parcel_result(ft: Dict[str, Any], raw: Dict[str, Any], acq: Acquisition,
                        scale_m: float) -> Dict[str, Any]:
    props = ft["properties"]
    stats_props = raw.get("stats") or {}
    aux = raw.get("aux") or {}
    stats: Dict[str, Any] = {}
    for idx in INDICES:
        bid = idx["id"]
        stats[bid] = {
            "mean": safe_float(stats_props.get(f"{bid}_mean")),
            "median": safe_float(stats_props.get(f"{bid}_median")),
            "stdDev": safe_float(stats_props.get(f"{bid}_stdDev")),
            "count": int(stats_props.get(f"{bid}_count") or 0),
        }
    valid_frac = safe_float(aux.get("valid_frac_mean", aux.get("valid_frac")), 6)
    coverage = round(max(0.0, min(100.0, (valid_frac or 0.0) * 100.0)), 1)
    classes: Dict[str, Any] = {}
    class_locations: Dict[str, Any] = {}
    for cid, label, _lo, _hi in NDVI_CLASSES:
        total_frac = safe_float(aux.get(f"{cid}_frac_mean", aux.get(f"{cid}_frac")), 7) or 0.0
        valid_pct = 0.0
        if valid_frac and valid_frac > 0:
            valid_pct = max(0.0, min(100.0, total_frac / valid_frac * 100.0))
        lon = safe_float(aux.get(f"{cid}_lon_mean", aux.get(f"{cid}_lon")), 7)
        lat = safe_float(aux.get(f"{cid}_lat_mean", aux.get(f"{cid}_lat")), 7)
        classes[cid] = round(valid_pct, 1)
        class_locations[cid] = {
            "label": label,
            "direction": direction_code(lon, lat, props.get("bbox") or []),
            "centroid": [lon, lat] if lon is not None and lat is not None else None,
        }
    return {
        "date": acq.iso_date,
        "image_count": len(acq.image_ids),
        "image_ids": list(acq.image_ids),
        "scale_m": scale_m,
        "valid_coverage_pct": coverage,
        "valid_pixels": stats["ndvi"]["count"],
        "stats": stats,
        "ndvi_classes_pct": classes,
        "ndvi_class_location": class_locations,
    }


def load_parcel_series(parcels: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ft in parcels:
        meta = ft["properties"]
        pid = str(meta["parcela_id"])
        ko_code = str(meta.get("kat_opst_1") or "KO")
        path = PARCEL_OUT / ko_code / f"{pid}.json"
        old = load_json(path, None)
        series = []
        if isinstance(old, dict) and isinstance(old.get("series"), list):
            series = old["series"]
        out[pid] = {"parcel": meta, "series": series, "_path": path}
    return out


def merge_record(parcel_obj: Dict[str, Any], record: Dict[str, Any]) -> None:
    d = record["date"]
    series = [r for r in parcel_obj.get("series", []) if r.get("date") != d]
    series.append(record)
    series.sort(key=lambda r: r.get("date", ""))
    parcel_obj["series"] = series


def save_all_parcels(series_map: Dict[str, Dict[str, Any]]) -> None:
    for i, (pid, obj) in enumerate(series_map.items(), 1):
        path: Path = obj["_path"]
        payload = {
            "ok": True,
            "source": "Google Earth Engine · COPERNICUS/S2_SR_HARMONIZED",
            "numeric_values": "direct Sentinel-2 index statistics; not reconstructed from PNG colours",
            "parcel": obj["parcel"],
            "series": obj["series"],
            "updated_at": utc_now(),
        }
        write_json(path, payload, pretty=False)
        if i % 250 == 0:
            log(f"saved parcel files: {i}/{len(series_map)}")


def build_catalog(parcels: Sequence[Dict[str, Any]], series_map: Dict[str, Dict[str, Any]],
                  accepted: List[Dict[str, Any]], rejected: List[Dict[str, Any]], scale_m: float) -> Dict[str, Any]:
    parcel_catalog = []
    for ft in parcels:
        p = ft["properties"]
        pid = str(p["parcela_id"])
        ko_code = str(p.get("kat_opst_1") or "KO")
        parcel_catalog.append({
            "parcela_id": pid,
            "brparcele": p.get("brparcele"),
            "kat_opstin": p.get("kat_opstin"),
            "Kultura": p.get("Kultura"),
            "Vrstazemljista": p.get("Vrstazemljista"),
            "area_total_ha": p.get("area_total_ha"),
            "area_pio_ha": p.get("area_pio_ha"),
            "pio_share_pct": p.get("pio_share_pct"),
            "path": f"parcels/{ko_code}/{pid}.json",
            "records": len(series_map[pid].get("series", [])),
        })
    dates = sorted({a["date"] for a in accepted})
    return {
        "ok": True,
        "label": "PIO Rajac — све катастарске парцеле · Sentinel-2",
        "source": "Google Earth Engine · COPERNICUS/S2_SR_HARMONIZED",
        "method": "direct numerical index calculation from Sentinel-2 reflectance + SCL cloud mask",
        "numeric_values_reconstructed_from_png": False,
        "scale_m": scale_m,
        "parcel_count": len(parcels),
        "latest": dates[-1] if dates else None,
        "acquisition_count": len(dates),
        "acquisitions": sorted(accepted, key=lambda x: x["date"]),
        "rejected_acquisitions": sorted(rejected, key=lambda x: x["date"])[-100:],
        "indices": [{"id": x["id"], "label": x["label"], "name_sr": x["name_sr"]} for x in INDICES],
        "ndvi_classes": [
            {"id":cid,"label":label,"min":lo if lo>-900 else -1.0,"max":hi if hi is not None else 1.0}
            for cid,label,lo,hi in NDVI_CLASSES
        ],
        "parcels": parcel_catalog,
        "updated_at": utc_now(),
    }


def build_audit(parcels: Sequence[Dict[str, Any]], series_map: Dict[str, Dict[str, Any]], catalog: Dict[str, Any]) -> Dict[str, Any]:
    counts = [len(series_map[str(f["properties"]["parcela_id"])].get("series", [])) for f in parcels]
    latest_dates = []
    coverage_vals = []
    for obj in series_map.values():
        s = obj.get("series") or []
        if s:
            latest_dates.append(s[-1].get("date"))
            coverage_vals.extend([float(r.get("valid_coverage_pct") or 0.0) for r in s])
    return {
        "ok": True,
        "parcel_count": len(parcels),
        "records_total": sum(counts),
        "records_per_parcel_min": min(counts) if counts else 0,
        "records_per_parcel_max": max(counts) if counts else 0,
        "records_per_parcel_avg": round(sum(counts)/len(counts), 2) if counts else 0,
        "acquisition_count": catalog.get("acquisition_count", 0),
        "latest": catalog.get("latest"),
        "coverage_avg_pct": round(sum(coverage_vals)/len(coverage_vals), 1) if coverage_vals else None,
        "numeric_values_reconstructed_from_png": False,
        "generated_at": utc_now(),
    }


def process_acquisition(acq: Acquisition, parcels: Sequence[Dict[str, Any]], scale_m: float,
                        batch_size: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    log(f"SCENE {acq.iso_date}: {len(acq.image_ids)} Sentinel-2 image(s)")
    coll = ee.ImageCollection(S2_COLLECTION).filter(ee.Filter.inList("system:id", list(acq.image_ids))).map(mask_s2_scl)
    n = int(retry_getinfo(coll.size(), f"collection size {acq.iso_date}") or 0)
    if n <= 0:
        raise RuntimeError(f"No images for acquisition {acq.iso_date}")
    # same-day median joins adjacent tiles when needed while preserving masked pixels
    source = coll.median()
    index_img = build_index_image(source)
    aux_img = build_aux_image(index_img)
    raw_all: Dict[str, Dict[str, Any]] = {}
    for start in range(0, len(parcels), batch_size):
        part = parcels[start:start+batch_size]
        log(f"  batch {start+1}-{start+len(part)} / {len(parcels)}")
        raw_all.update(reduce_batch(index_img, aux_img, part, scale_m))
    records: Dict[str, Dict[str, Any]] = {}
    good = 0
    coverages: List[float] = []
    for ft in parcels:
        pid = str(ft["properties"]["parcela_id"])
        rec = parse_parcel_result(ft, raw_all.get(pid, {}), acq, scale_m)
        records[pid] = rec
        cov = float(rec.get("valid_coverage_pct") or 0.0)
        coverages.append(cov)
        if cov >= MIN_PARCEL_VALID_COVERAGE_PCT and int(rec.get("valid_pixels") or 0) >= 1:
            good += 1
    good_pct = round(good / len(parcels) * 100.0, 1) if parcels else 0.0
    mean_cov = round(sum(coverages)/len(coverages), 1) if coverages else 0.0
    scene_meta = {
        "date": acq.iso_date,
        "image_count": len(acq.image_ids),
        "image_ids": list(acq.image_ids),
        "parcels_total": len(parcels),
        "parcels_good_coverage": good,
        "good_coverage_pct": good_pct,
        "mean_valid_coverage_pct": mean_cov,
        "accepted": good_pct >= MIN_GOOD_PARCELS_PCT,
    }
    log(f"  coverage: good parcels={good}/{len(parcels)} ({good_pct}%), mean valid={mean_cov}%")
    return records, scene_meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", help="YYYY-MM-DD, inclusive")
    ap.add_argument("--end-date", help="YYYY-MM-DD, inclusive; default today")
    ap.add_argument("--lookback-days", type=int, default=10)
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE_M)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-scenes", type=int, default=0, help="0 = all")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not PARCELS_PATH.is_file():
        raise SystemExit(f"Missing parcel file: {PARCELS_PATH}")
    fc = load_json(PARCELS_PATH, {})
    parcels = fc.get("features") or []
    if not parcels:
        raise SystemExit("Prepared cadastral parcel GeoJSON has no features")
    if len(parcels) < 2000:
        log(f"WARN: expected ~2299 cadastral parcels, got {len(parcels)}")
    start = date.fromisoformat(args.start_date) if args.start_date else date.today() - timedelta(days=max(1,args.lookback_days))
    end = date.fromisoformat(args.end_date) if args.end_date else date.today()
    if end < start:
        raise SystemExit("end-date is before start-date")
    batch_size = max(MIN_BATCH_SIZE, int(args.batch_size))

    init_ee()
    region = collection_region(parcels)
    acquisitions = list_acquisitions(start, end, region)
    if args.max_scenes > 0:
        acquisitions = acquisitions[:args.max_scenes]
    log(f"Prepared parcels: {len(parcels)}")
    log(f"Candidate acquisition dates {start}..{end}: {len(acquisitions)}")

    old_catalog = load_json(INDEX_PATH, {}) or {}
    accepted_map = {x.get("date"): x for x in old_catalog.get("acquisitions", []) if x.get("date")}
    rejected_map = {x.get("date"): x for x in old_catalog.get("rejected_acquisitions", []) if x.get("date")}
    already = set(accepted_map)
    series_map = load_parcel_series(parcels)

    changed = False
    for acq in acquisitions:
        if acq.iso_date in already and not args.force:
            log(f"SKIP {acq.iso_date}: already processed")
            continue
        records, scene_meta = process_acquisition(acq, parcels, args.scale, batch_size)
        if not scene_meta["accepted"]:
            log(f"REJECT {acq.iso_date}: usable parcel coverage below {MIN_GOOD_PARCELS_PCT}%")
            rejected_map[acq.iso_date] = scene_meta
            accepted_map.pop(acq.iso_date, None)
            continue
        for pid, rec in records.items():
            merge_record(series_map[pid], rec)
        accepted_map[acq.iso_date] = scene_meta
        rejected_map.pop(acq.iso_date, None)
        changed = True
        log(f"ACCEPT {acq.iso_date}")

    if changed or not INDEX_PATH.exists() or args.force:
        save_all_parcels(series_map)
    catalog = build_catalog(
        parcels, series_map,
        list(accepted_map.values()), list(rejected_map.values()), args.scale,
    )
    write_json(INDEX_PATH, catalog, pretty=False)
    write_json(AUDIT_PATH, build_audit(parcels, series_map, catalog), pretty=True)
    log(f"DONE: accepted acquisitions={catalog['acquisition_count']}; parcels={catalog['parcel_count']}; latest={catalog['latest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
