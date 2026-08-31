#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PIO Rajac — monthly long-term Sentinel-2 indices for all cadastral parcels.

This is a separate historical pipeline. It DOES NOT modify the existing daily
cadastral processor or public/cadastral outputs.

Input:
  public/boundaries/pio-rajac-cadastral-parcels.geojson

Output (compatible in shape with the PHP parcel long-term builder):
  public/cadastral-longterm/stats/<YEAR>/NDVI_monthly.json
  public/cadastral-longterm/stats/<YEAR>/NDMI_monthly.json
  public/cadastral-longterm/stats/<YEAR>/RECI_monthly.json
  public/cadastral-longterm/catalog.json
  public/cadastral-longterm/audit.json

Formulas deliberately match piorajac.rs/kopindeksi/parcel_stats_builder.php:
  NDVI = (B8 - B4) / (B8 + B4)                  at 10 m
  NDMI = (B8A - B11) / (B8A + B11)              at 20 m
  RECI = (B8A / B5) - 1                          at 20 m

Data source:
  - 2015-06 through 2017-03: COPERNICUS/S2_HARMONIZED (L1C, QA60 mask)
  - 2017-04 onward: COPERNICUS/S2_SR_HARMONIZED (L2A, SCL mask)

The 2015/2016 values are therefore TOA/L1C and are explicitly marked as such;
2017+ uses surface reflectance where available. No value is fabricated for a
month with zero Sentinel-2 scenes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ee

ROOT = Path(__file__).resolve().parent
PARCELS_PATH = ROOT / "public" / "boundaries" / "pio-rajac-cadastral-parcels.geojson"
OUT_ROOT = ROOT / "public" / "cadastral-longterm"
STATS_ROOT = OUT_ROOT / "stats"
CATALOG_PATH = OUT_ROOT / "catalog.json"
AUDIT_PATH = OUT_ROOT / "audit.json"

SR_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
L1C_COLLECTION = "COPERNICUS/S2_HARMONIZED"
SR_START = date(2017, 4, 1)  # use full calendar months consistently
S2_START = date(2015, 6, 23)
SCL_BAD = [1, 3, 8, 9, 10, 11]
MAX_METADATA_CLOUD = 80
DEFAULT_BATCH_SIZE = 750
MIN_BATCH_SIZE = 40
GETINFO_RETRIES = 4

INDEX_META = {
    "NDVI": {"label": "NDVI", "desc": "вегетациона активност и зелена маса", "resolution": 10},
    "NDMI": {"label": "NDMI", "desc": "влажност вегетације и водни стрес", "resolution": 20},
    "RECI": {"label": "RECI", "desc": "хлорофил и јачина активне вегетације", "resolution": 20},
}


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
    key = json.loads(secret)
    email = key.get("client_email")
    if not email:
        raise RuntimeError("GEE_SERVICE_ACCOUNT_JSON nema client_email")
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
    log(f"Earth Engine initialized: project={project}")


def retry_getinfo(obj: Any, label: str) -> Any:
    last = None
    for attempt in range(1, GETINFO_RETRIES + 1):
        try:
            return obj.getInfo()
        except Exception as exc:
            last = exc
            wait = min(30, 2 ** attempt)
            log(f"WARN getInfo {label} {attempt}/{GETINFO_RETRIES}: {exc}; retry {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Earth Engine getInfo failed for {label}: {last}")


def feature_to_ee_geometry(feature: Dict[str, Any]) -> ee.Geometry:
    geom = feature["geometry"]
    gt = geom.get("type")
    if gt == "Polygon":
        return ee.Geometry.Polygon(geom["coordinates"], proj="EPSG:4326", geodesic=False)
    if gt == "MultiPolygon":
        return ee.Geometry.MultiPolygon(geom["coordinates"], proj="EPSG:4326", geodesic=False)
    raise ValueError(f"Unsupported geometry: {gt}")


def make_ee_fc(features: Sequence[Dict[str, Any]]) -> ee.FeatureCollection:
    return ee.FeatureCollection([
        ee.Feature(feature_to_ee_geometry(ft), {"pid": str(ft["properties"]["parcela_id"])})
        for ft in features
    ])


def collection_region(parcels: Sequence[Dict[str, Any]]) -> ee.Geometry:
    bboxes = [f.get("properties", {}).get("bbox") for f in parcels]
    bboxes = [b for b in bboxes if isinstance(b, list) and len(b) == 4]
    if bboxes:
        return ee.Geometry.Rectangle([
            min(float(b[0]) for b in bboxes), min(float(b[1]) for b in bboxes),
            max(float(b[2]) for b in bboxes), max(float(b[3]) for b in bboxes),
        ], proj="EPSG:4326", geodesic=False)
    # Fallback: union bounds from a small FeatureCollection.
    return make_ee_fc(parcels[:200]).geometry().bounds()


def month_bounds(year: int, month: int) -> Tuple[date, date]:
    start = date(year, month, 1)
    end = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    return start, end


def mask_sr(img: ee.Image) -> ee.Image:
    scl = img.select("SCL")
    mask = ee.Image.constant(1)
    for cls in SCL_BAD:
        mask = mask.And(scl.neq(cls))
    return img.updateMask(mask)


def mask_l1c(img: ee.Image) -> ee.Image:
    qa = img.select("QA60")
    cloud = 1 << 10
    cirrus = 1 << 11
    mask = qa.bitwiseAnd(cloud).eq(0).And(qa.bitwiseAnd(cirrus).eq(0))
    return img.updateMask(mask)


def monthly_source(year: int, month: int, region: ee.Geometry) -> Tuple[Optional[ee.Image], Dict[str, Any]]:
    start, end = month_bounds(year, month)
    if end <= S2_START:
        return None, {"scene_count": 0, "source": None, "level": None, "mask": None}
    if start >= SR_START:
        collection_id, level, mask_fn = SR_COLLECTION, "L2A/SR", mask_sr
    else:
        collection_id, level, mask_fn = L1C_COLLECTION, "L1C/TOA", mask_l1c
    coll = (ee.ImageCollection(collection_id)
            .filterDate(start.isoformat(), end.isoformat())
            .filterBounds(region)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", MAX_METADATA_CLOUD))
            .map(mask_fn))
    n = int(retry_getinfo(coll.size(), f"scene count {year}-{month:02d}") or 0)
    meta = {
        "scene_count": n,
        "source": collection_id,
        "level": level,
        "mask": "SCL classes 1,3,8,9,10,11" if level == "L2A/SR" else "QA60 cloud/cirrus bits 10/11",
    }
    if n <= 0:
        return None, meta
    # Median across all cloud-masked observations in the calendar month.
    return coll.median(), meta


def reflectance(source: ee.Image) -> ee.Image:
    bands = ["B4", "B5", "B8", "B8A", "B11"]
    return source.select(bands).multiply(0.0001).rename(bands)


def build_ndvi_stack(source: ee.Image) -> ee.Image:
    r = reflectance(source)
    b8, b4 = r.select("B8"), r.select("B4")
    ndvi = b8.subtract(b4).divide(b8.add(b4).add(1e-6)).rename("ndvi")
    valid = ndvi.mask().reduce(ee.Reducer.min()).unmask(0).rename("valid")
    return ee.Image.cat([ndvi, valid]).toFloat()


def build_rededge_stack(source: ee.Image) -> ee.Image:
    r = reflectance(source)
    b8a, b11, b5 = r.select("B8A"), r.select("B11"), r.select("B5")
    ndmi = b8a.subtract(b11).divide(b8a.add(b11).add(1e-6)).rename("ndmi")
    reci = b8a.divide(b5.add(1e-6)).subtract(1.0).rename("reci")
    valid = ndmi.mask().reduce(ee.Reducer.min()).And(reci.mask().reduce(ee.Reducer.min())).unmask(0).rename("valid")
    return ee.Image.cat([ndmi, reci, valid]).toFloat()


def combined_reducer() -> ee.Reducer:
    return (ee.Reducer.mean()
            .combine(ee.Reducer.median(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.minMax(), sharedInputs=True)
            .combine(ee.Reducer.percentile([25, 75], ["p25", "p75"]), sharedInputs=True)
            .combine(ee.Reducer.count(), sharedInputs=True))


def reduce_batch(stack: ee.Image, features: Sequence[Dict[str, Any]], scale: int, depth: int = 0) -> Dict[str, Dict[str, Any]]:
    if not features:
        return {}
    try:
        fc = make_ee_fc(features)
        info = retry_getinfo(stack.reduceRegions(
            collection=fc, reducer=combined_reducer(), scale=scale, tileScale=4
        ), f"reduce scale={scale} n={len(features)}") or {}
        return {
            str((x.get("properties") or {}).get("pid")): (x.get("properties") or {})
            for x in info.get("features", [])
        }
    except Exception as exc:
        if len(features) <= MIN_BATCH_SIZE:
            raise
        half = len(features) // 2
        log(f"WARN batch scale={scale} n={len(features)} failed ({exc}); split")
        left = reduce_batch(stack, features[:half], scale, depth + 1)
        right = reduce_batch(stack, features[half:], scale, depth + 1)
        left.update(right)
        return left


def parcel_area_ha(p: Dict[str, Any], pio: bool = True) -> float:
    candidates = (["area_pio_ha", "povrsina_ha"] if pio else ["area_total_ha", "povrsina_ha"])
    for k in candidates:
        try:
            v = float(p.get(k) or 0)
            if v > 0:
                return v
        except Exception:
            pass
    try:
        v = float(p.get("povrsina") or 0) / 10000.0
        if v > 0:
            return v
    except Exception:
        pass
    return 0.0


def expected_pixels(p: Dict[str, Any], resolution: int) -> int:
    area_m2 = parcel_area_ha(p, pio=True) * 10000.0
    return max(1, int(round(area_m2 / (resolution * resolution)))) if area_m2 > 0 else 0


def series_entry(month_key: str, start: date, end: date, props: Dict[str, Any], band: str,
                 raw: Dict[str, Any], resolution: int, valid_key: str = "valid") -> Dict[str, Any]:
    exp = expected_pixels(props, resolution)
    count = int(raw.get(f"{band}_count") or 0)
    valid_mean = safe_float(raw.get(f"{valid_key}_mean"), 6)
    valid_pct = round(max(0.0, min(100.0, (valid_mean or 0.0) * 100.0)), 2)
    masked = max(0, exp - min(count, exp)) if exp else None
    return {
        "month": month_key,
        "from": start.isoformat() + "T00:00:00Z",
        "to": end.isoformat() + "T00:00:00Z",
        "mean": safe_float(raw.get(f"{band}_mean")),
        "median": safe_float(raw.get(f"{band}_median")),
        "p25": safe_float(raw.get(f"{band}_p25")),
        "p75": safe_float(raw.get(f"{band}_p75")),
        "min": safe_float(raw.get(f"{band}_min")),
        "max": safe_float(raw.get(f"{band}_max")),
        "stdev": safe_float(raw.get(f"{band}_stdDev")),
        "sample_count": exp,
        "no_data_count": masked,
        "cloud_free_pct": valid_pct,
        "valid_pixel_count": count,
        "masked_geometry_pixel_count": masked,
        "geometry_pixel_count": exp,
        "expected_pixel_count_approx": exp,
        "resolution_m": resolution,
    }


def row_meta(ft: Dict[str, Any]) -> Dict[str, Any]:
    p = ft.get("properties") or {}
    total_ha = parcel_area_ha(p, pio=False)
    return {
        "parcela_id": str(p.get("parcela_id") or ""),
        "brparcele": p.get("brparcele") or p.get("Brojparcele") or "",
        "povrsina": int(round(total_ha * 10000.0)) if total_ha else int(p.get("povrsina") or 0),
        "povrsina_ha": round(total_ha, 4) if total_ha else safe_float(p.get("povrsina_ha"), 4) or 0,
        "status_par": p.get("status_par") or "",
        "kat_opstin": p.get("kat_opstin") or p.get("kat_opst_1") or "",
        "opstina_im": p.get("opstina_im") or "",
        "sentinel_reliability": p.get("sentinel_reliability") or "",
    }


def load_rows(year: int, index: str, parcels: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    path = STATS_ROOT / str(year) / f"{index}_monthly.json"
    old = load_json(path, {}) or {}
    by_pid = {str(r.get("parcela_id")): r for r in old.get("rows", []) if r.get("parcela_id")}
    for ft in parcels:
        meta = row_meta(ft)
        pid = meta["parcela_id"]
        if pid not in by_pid:
            meta["series"] = []
            by_pid[pid] = meta
        else:
            # Refresh metadata but preserve series.
            series = by_pid[pid].get("series") or []
            by_pid[pid].update(meta)
            by_pid[pid]["series"] = series
    return by_pid


def merge_month(row: Dict[str, Any], entry: Dict[str, Any]) -> None:
    mk = entry["month"]
    s = [x for x in (row.get("series") or []) if x.get("month") != mk]
    s.append(entry)
    s.sort(key=lambda x: x.get("month", ""))
    row["series"] = s


def save_bundle(year: int, index: str, rows: Dict[str, Dict[str, Any]], month_meta: Dict[str, Any]) -> Dict[str, Any]:
    cfg = INDEX_META[index]
    out_rows = list(rows.values())
    out_rows.sort(key=lambda r: (str(r.get("kat_opstin") or ""), str(r.get("brparcele") or "")))
    months = sorted({s.get("month") for r in out_rows for s in (r.get("series") or []) if s.get("month")})
    bundle = {
        "ok": True,
        "year": year,
        "index": index,
        "index_label": cfg["label"],
        "index_desc": cfg["desc"],
        "updated": utc_now(),
        "max_cloud": MAX_METADATA_CLOUD,
        "resolution_m": cfg["resolution"],
        "aggregation": "P1M",
        "months_present": months,
        "parcel_count_total": len(out_rows),
        "parcel_count_written": len(out_rows),
        "source": "Google Earth Engine · Sentinel-2 monthly median composite",
        "method_note": "2015-06..2017-03 L1C/TOA+QA60; 2017-04+ L2A/SR+SCL. Formula matches piorajac.rs parcel_stats_builder.php.",
        "month_sources": month_meta,
        "rows": out_rows,
    }
    path = STATS_ROOT / str(year) / f"{index}_monthly.json"
    write_json(path, bundle, pretty=False)
    return bundle


def update_catalog_and_audit() -> None:
    years: Dict[str, Any] = {}
    total_rows = 0
    for ydir in sorted([p for p in STATS_ROOT.glob("*") if p.is_dir()]):
        y = ydir.name
        yi = {}
        for idx in INDEX_META:
            p = ydir / f"{idx}_monthly.json"
            if not p.is_file():
                continue
            j = load_json(p, {}) or {}
            yi[idx] = {
                "months_present": j.get("months_present") or [],
                "parcel_count_written": j.get("parcel_count_written") or 0,
                "resolution_m": j.get("resolution_m"),
                "path": f"stats/{y}/{idx}_monthly.json",
            }
            total_rows += int(j.get("parcel_count_written") or 0)
        if yi:
            years[y] = yi
    catalog = {
        "ok": True,
        "updated": utc_now(),
        "parcel_source": "public/boundaries/pio-rajac-cadastral-parcels.geojson",
        "indices": list(INDEX_META),
        "years": years,
        "source_policy": {
            "2015-06_to_2017-03": L1C_COLLECTION,
            "2017-04_onward": SR_COLLECTION,
        },
    }
    write_json(CATALOG_PATH, catalog, pretty=True)
    audit = {
        "ok": True,
        "updated": utc_now(),
        "years_present": sorted(years),
        "index_year_files": sum(len(v) for v in years.values()),
        "row_sets_total": total_rows,
        "warning": "L1C/TOA (2015-2017-03) and L2A/SR (2017-04+) are not identical processing levels; source is recorded per month.",
    }
    write_json(AUDIT_PATH, audit, pretty=True)


def process_month(year: int, month: int, parcels: Sequence[Dict[str, Any]], region: ee.Geometry,
                  batch_size: int) -> Dict[str, Any]:
    start, end = month_bounds(year, month)
    source, src_meta = monthly_source(year, month, region)
    mk = f"{year}-{month:02d}"
    if source is None:
        log(f"{mk}: no scenes; skipped")
        return {mk: src_meta}

    log(f"{mk}: {src_meta['scene_count']} scene(s), {src_meta['level']}")
    ndvi_stack = build_ndvi_stack(source)
    red_stack = build_rededge_stack(source)
    raw10: Dict[str, Dict[str, Any]] = {}
    raw20: Dict[str, Dict[str, Any]] = {}
    for off in range(0, len(parcels), batch_size):
        part = parcels[off:off + batch_size]
        log(f"  NDVI 10m batch {off+1}-{off+len(part)}/{len(parcels)}")
        raw10.update(reduce_batch(ndvi_stack, part, 10))
        log(f"  NDMI+RECI 20m batch {off+1}-{off+len(part)}/{len(parcels)}")
        raw20.update(reduce_batch(red_stack, part, 20))

    rows_by_index = {idx: load_rows(year, idx, parcels) for idx in INDEX_META}
    for ft in parcels:
        p = ft.get("properties") or {}
        pid = str(p.get("parcela_id") or "")
        merge_month(rows_by_index["NDVI"][pid], series_entry(mk, start, end, p, "ndvi", raw10.get(pid, {}), 10))
        merge_month(rows_by_index["NDMI"][pid], series_entry(mk, start, end, p, "ndmi", raw20.get(pid, {}), 20))
        merge_month(rows_by_index["RECI"][pid], series_entry(mk, start, end, p, "reci", raw20.get(pid, {}), 20))

    for idx in INDEX_META:
        path = STATS_ROOT / str(year) / f"{idx}_monthly.json"
        old = load_json(path, {}) or {}
        mm = dict(old.get("month_sources") or {})
        mm[mk] = src_meta
        save_bundle(year, idx, rows_by_index[idx], mm)
    update_catalog_and_audit()
    return {mk: src_meta}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True, help="2015..current year")
    ap.add_argument("--month", type=int, default=0, help="1..12; 0 = all months in year")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--force", action="store_true", help="recompute month even if already present")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    current_year = date.today().year
    if args.year < 2015 or args.year > current_year:
        raise SystemExit(f"year must be 2015..{current_year}")
    if args.month < 0 or args.month > 12:
        raise SystemExit("month must be 0..12")
    if not PARCELS_PATH.is_file():
        raise SystemExit(f"Missing parcel file: {PARCELS_PATH}")
    fc = load_json(PARCELS_PATH, {}) or {}
    parcels = fc.get("features") or []
    ids = [str((f.get("properties") or {}).get("parcela_id") or "") for f in parcels]
    if len(parcels) != 2299:
        raise SystemExit(f"Expected 2299 parcels, got {len(parcels)}")
    if not all(ids) or len(set(ids)) != len(ids):
        raise SystemExit("parcela_id missing or not unique")

    init_ee()
    region = collection_region(parcels)
    batch_size = max(MIN_BATCH_SIZE, int(args.batch_size))
    months = [args.month] if args.month else list(range(1, 13))
    for m in months:
        # A month is complete only if all 3 bundles already advertise it.
        mk = f"{args.year}-{m:02d}"
        complete = True
        for idx in INDEX_META:
            old = load_json(STATS_ROOT / str(args.year) / f"{idx}_monthly.json", {}) or {}
            if mk not in (old.get("months_present") or []):
                complete = False
                break
        if complete and not args.force:
            log(f"SKIP {mk}: already present in NDVI+NDMI+RECI")
            continue
        process_month(args.year, m, parcels, region, batch_size)
    update_catalog_and_audit()
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
