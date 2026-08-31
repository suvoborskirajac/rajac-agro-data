#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIO Rajac — statistical cards for 2,299 cadastral parcels.

Inputs already present in the repository:
  public/cadastral-longterm/stats/<YEAR>/NDVI_monthly.json
  public/cadastral-longterm/stats/<YEAR>/NDMI_monthly.json
  public/cadastral-longterm/stats/<YEAR>/RECI_monthly.json
  public/boundaries/pio-rajac-cadastral-parcels.geojson

Independent drought context:
  Google Earth Engine · IDAHO_EPSCOR/TERRACLIMATE · PDSI
  Rajac-wide monthly mean, standardized within 2018-2025.

Outputs:
  public/cadastral-statistics/catalog.json
  public/cadastral-statistics/summary.json
  public/cadastral-statistics/climate_context.json
  public/cadastral-statistics/cards/<parcela_id>.json

Primary comparable Sentinel-2 trend period:
  2018-2025 (L2A/SR only)

Sensitivity period:
  2016-2025 (includes L1C/TOA before 2017-04)
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import time
from collections import defaultdict, Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ee

ROOT = Path(__file__).resolve().parent
STATS_ROOT = ROOT / "public" / "cadastral-longterm" / "stats"
PARCELS_PATH = ROOT / "public" / "boundaries" / "pio-rajac-cadastral-parcels.geojson"
OUT_ROOT = ROOT / "public" / "cadastral-statistics"
CARDS_ROOT = OUT_ROOT / "cards"

INDICES = ("NDVI", "NDMI", "RECI")
YEARS = range(2015, 2027)
PRIMARY_YEARS = range(2018, 2026)
SENSITIVITY_YEARS = range(2016, 2026)

SEASONS = {
    "spring": {"label": "Пролеће", "months": (3, 4, 5), "min_months": 3},
    "summer": {"label": "Лето", "months": (6, 7, 8), "min_months": 3},
    "autumn": {"label": "Јесен", "months": (9, 10, 11), "min_months": 3},
    "growing": {"label": "Вегетациони период", "months": (4, 5, 6, 7, 8, 9), "min_months": 5},
}

TERRACLIMATE = "IDAHO_EPSCOR/TERRACLIMATE"
GETINFO_RETRIES = 4


def log(msg: str) -> None:
    print(msg, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_json_pretty(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def rnd(v: Any, n: int = 4) -> Optional[float]:
    x = safe_float(v)
    return None if x is None else round(x, n)


def mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    a = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return statistics.mean(a) if a else None


def median(vals: Iterable[Optional[float]]) -> Optional[float]:
    a = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return statistics.median(a) if a else None


def stddev(vals: Iterable[Optional[float]]) -> Optional[float]:
    a = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return statistics.stdev(a) if len(a) >= 2 else None


def erf_approx(x: float) -> float:
    sign = -1.0 if x < 0 else 1.0
    ax = abs(x)
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t * math.exp(-ax*ax))
    return sign * y


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf_approx(x / math.sqrt(2.0)))


def mann_kendall_sen(points: Sequence[Tuple[int, float]]) -> Dict[str, Any]:
    pts = [(int(x), float(y)) for x, y in points if y is not None and math.isfinite(float(y))]
    pts.sort()
    n = len(pts)
    if n < 4:
        return {"n": n, "tau": None, "p": None, "sen_slope": None, "z": None, "class": "недовољно података"}

    s = 0
    slopes: List[float] = []
    ys = [p[1] for p in pts]
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = ys[j] - ys[i]
            s += 1 if d > 0 else (-1 if d < 0 else 0)
            dx = pts[j][0] - pts[i][0]
            if dx:
                slopes.append(d / dx)

    ties = Counter(round(y, 8) for y in ys)
    tie_adj = sum(t * (t - 1) * (2*t + 5) for t in ties.values() if t > 1)
    variance = (n*(n-1)*(2*n+5) - tie_adj) / 18.0
    z = 0.0
    if variance > 0:
        if s > 0:
            z = (s - 1) / math.sqrt(variance)
        elif s < 0:
            z = (s + 1) / math.sqrt(variance)
    p = 2.0 * (1.0 - normal_cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    slopes.sort()
    if not slopes:
        sen = None
    elif len(slopes) % 2:
        sen = slopes[len(slopes)//2]
    else:
        k = len(slopes)//2
        sen = (slopes[k-1] + slopes[k]) / 2.0

    if sen is None:
        cls = "недовољно података"
    elif p < 0.05:
        cls = "статистички значајан раст" if sen > 0 else ("статистички значајан пад" if sen < 0 else "стабилно")
    else:
        cls = "благ раст (није значајан)" if sen > 0 else ("благ пад (није значајан)" if sen < 0 else "без јасног тренда")

    return {
        "n": n,
        "tau": rnd(tau, 4),
        "p": rnd(p, 6),
        "sen_slope": rnd(sen, 6),
        "z": rnd(z, 4),
        "class": cls,
    }


def bh_qvalues(items: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    valid = [(key, float(p)) for key, p in items if p is not None and math.isfinite(float(p))]
    valid.sort(key=lambda x: x[1])
    m = len(valid)
    q: Dict[str, float] = {}
    prev = 1.0
    for rank in range(m, 0, -1):
        key, p = valid[rank - 1]
        val = min(prev, p * m / rank, 1.0)
        q[key] = val
        prev = val
    return q


def rankdata(vals: Sequence[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j+1]] == vals[order[i]]:
            j += 1
        r = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 4:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    dx = [a - mx for a in x]
    dy = [b - my for b in y]
    den = math.sqrt(sum(a*a for a in dx) * sum(b*b for b in dy))
    return None if den == 0 else sum(a*b for a, b in zip(dx, dy)) / den


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 4:
        return None
    return pearson(rankdata(list(x)), rankdata(list(y)))


def pettitt(values: Sequence[Tuple[str, float]]) -> Dict[str, Any]:
    vals = [(k, float(v)) for k, v in values if v is not None and math.isfinite(float(v))]
    n = len(vals)
    if n < 12:
        return {"n": n, "month": None, "p": None, "direction": None, "delta_before_after": None}

    ys = [v for _, v in vals]
    ranks = rankdata(ys)
    u = []
    for t in range(1, n + 1):
        u.append(2.0 * sum(ranks[:t]) - t * (n + 1))
    kidx = max(range(n), key=lambda i: abs(u[i]))
    K = abs(u[kidx])
    p = min(1.0, 2.0 * math.exp((-6.0 * K * K) / (n**3 + n**2)))
    before = mean(ys[:kidx+1])
    after = mean(ys[kidx+1:])
    delta = None if before is None or after is None else after - before
    direction = None if delta is None else ("раст" if delta > 0 else ("пад" if delta < 0 else "без промене"))
    return {
        "n": n,
        "month": vals[kidx][0],
        "p": rnd(p, 6),
        "direction": direction,
        "delta_before_after": rnd(delta, 4),
    }


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
    log(f"Earth Engine initialized: {project}")


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


def aoi_bbox() -> Tuple[ee.Geometry, Dict[str, Any]]:
    g = load_json(PARCELS_PATH)
    feats = g.get("features") or []
    if len(feats) != 2299:
        raise RuntimeError(f"Expected 2299 parcels, got {len(feats)}")
    xs, ys = [], []

    def walk(o: Any) -> None:
        if isinstance(o, list):
            if len(o) >= 2 and isinstance(o[0], (int, float)) and isinstance(o[1], (int, float)):
                xs.append(float(o[0])); ys.append(float(o[1]))
            else:
                for v in o: walk(v)

    for ft in feats:
        walk((ft.get("geometry") or {}).get("coordinates"))
    bbox = [min(xs), min(ys), max(xs), max(ys)]
    return ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False), {"bbox_wgs84": [rnd(x,7) for x in bbox]}


def load_climate_context() -> Dict[str, Any]:
    init_ee()
    region, meta = aoi_bbox()
    coll = (ee.ImageCollection(TERRACLIMATE)
            .filterDate("2015-01-01", "2026-01-01")
            .filterBounds(region)
            .select(["pdsi", "pr", "pet"]))
    n = int(retry_getinfo(coll.size(), "TerraClimate size") or 0)

    def item_to_feature(obj: Any) -> ee.Feature:
        img = ee.Image(obj)
        d = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5000,
            bestEffort=True,
            maxPixels=10000000,
        )
        return ee.Feature(None, d.set("system:time_start", img.get("system:time_start")))

    fc = ee.FeatureCollection(coll.toList(n).map(item_to_feature))
    info = retry_getinfo(fc, "TerraClimate monthly context") or {}
    rows = []
    for ft in info.get("features", []):
        p = ft.get("properties") or {}
        ms = p.get("system:time_start")
        if ms is None:
            continue
        dt = datetime.fromtimestamp(float(ms)/1000.0, tz=timezone.utc)
        rows.append({
            "month": dt.strftime("%Y-%m"),
            "pdsi_raw": rnd(p.get("pdsi"), 5),
            "precip_raw": rnd(p.get("pr"), 5),
            "pet_raw": rnd(p.get("pet"), 5),
        })
    rows.sort(key=lambda r: r["month"])

    base = [r["pdsi_raw"] for r in rows if "2018-01" <= r["month"] <= "2025-12" and r["pdsi_raw"] is not None]
    mu = mean(base)
    sd = stddev(base)
    for r in rows:
        if r["pdsi_raw"] is not None and mu is not None and sd not in (None, 0):
            r["pdsi_z_2018_2025"] = rnd((r["pdsi_raw"] - mu) / sd, 4)
        else:
            r["pdsi_z_2018_2025"] = None
        r["relative_drought"] = bool(r["pdsi_z_2018_2025"] is not None and r["pdsi_z_2018_2025"] <= -1.0)

    return {
        "ok": True,
        "updated": utc_now(),
        "source": TERRACLIMATE,
        "aoi": meta,
        "definition": "Rajac-wide monthly TerraClimate PDSI mean; standardized over 2018-2025. Relative drought month = PDSI z <= -1.",
        "warning": "This is an independent regional drought context, not parcel-level soil moisture.",
        "rows": rows,
    }


def load_monthly_database() -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, Any]]]], Dict[str, Dict[str, Any]]]:
    # data[index][pid][YYYY-MM] = compact monthly observation
    data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {idx: defaultdict(dict) for idx in INDICES}
    parcel_meta: Dict[str, Dict[str, Any]] = {}

    for year in YEARS:
        for idx in INDICES:
            path = STATS_ROOT / str(year) / f"{idx}_monthly.json"
            if not path.exists():
                continue
            log(f"Load {year} {idx}: {path}")
            bundle = load_json(path)
            for row in bundle.get("rows") or []:
                pid = str(row.get("parcela_id") or "")
                if not pid:
                    continue
                if pid not in parcel_meta:
                    parcel_meta[pid] = {
                        "parcela_id": pid,
                        "brparcele": row.get("brparcele"),
                        "povrsina_ha": row.get("povrsina_ha"),
                        "kat_opstin": row.get("kat_opstin"),
                        "status_par": row.get("status_par"),
                    }
                for s in row.get("series") or []:
                    key = str(s.get("month") or "")
                    if not key:
                        continue
                    data[idx][pid][key] = {
                        "mean": safe_float(s.get("mean")),
                        "median": safe_float(s.get("median")),
                        "valid_pct": safe_float(s.get("cloud_free_pct")),
                        "pixels": safe_float(s.get("expected_pixel_count_approx") or s.get("sample_count")),
                    }
    if len(parcel_meta) != 2299:
        raise RuntimeError(f"Expected 2299 parcel IDs in stats, got {len(parcel_meta)}")
    return data, parcel_meta


def year_aggregate(monthly: Dict[str, Dict[str, Any]], year: int) -> Dict[str, Any]:
    rows = [(k, v) for k, v in monthly.items() if k.startswith(f"{year:04d}-") and v.get("mean") is not None]
    vals = [v["mean"] for _, v in rows]
    return {
        "mean": rnd(mean(vals), 4),
        "median": rnd(median(vals), 4),
        "min": rnd(min(vals), 4) if vals else None,
        "max": rnd(max(vals), 4) if vals else None,
        "std": rnd(stddev(vals), 4),
        "months": len(vals),
        "complete_for_trend": len(vals) >= 10,
    }


def season_aggregate(monthly: Dict[str, Dict[str, Any]], year: int, season: str) -> Dict[str, Any]:
    spec = SEASONS[season]
    vals = []
    used = []
    for m in spec["months"]:
        key = f"{year:04d}-{m:02d}"
        v = monthly.get(key, {}).get("mean")
        if v is not None:
            vals.append(float(v)); used.append(key)
    return {
        "mean": rnd(mean(vals), 4),
        "median": rnd(median(vals), 4),
        "min": rnd(min(vals), 4) if vals else None,
        "max": rnd(max(vals), 4) if vals else None,
        "std": rnd(stddev(vals), 4),
        "months": len(vals),
        "expected_months": len(spec["months"]),
        "complete_for_trend": len(vals) >= spec["min_months"],
        "used": used,
    }


def monthly_anomalies(monthly: Dict[str, Dict[str, Any]], start_year: int = 2018, end_year: int = 2025) -> Dict[str, float]:
    climatology: Dict[int, List[float]] = defaultdict(list)
    for k, row in monthly.items():
        y, m = map(int, k.split("-"))
        v = row.get("mean")
        if start_year <= y <= end_year and v is not None:
            climatology[m].append(float(v))
    clim_mean = {m: mean(vals) for m, vals in climatology.items()}
    out = {}
    for k, row in monthly.items():
        y, m = map(int, k.split("-"))
        v = row.get("mean")
        c = clim_mean.get(m)
        if start_year <= y <= end_year and v is not None and c is not None:
            out[k] = float(v) - float(c)
    return out


def trend_for_aggregate(series: Dict[int, Dict[str, Any]], years: Iterable[int]) -> Dict[str, Any]:
    pts = []
    for y in years:
        r = series.get(y)
        if r and r.get("complete_for_trend") and r.get("mean") is not None:
            pts.append((y, float(r["mean"])))
    return mann_kendall_sen(pts)


def quality_for_index(monthly: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for k, r in monthly.items() if "2018-01" <= k <= "2025-12"]
    expected = 8 * 12
    valid = [r for r in rows if r.get("mean") is not None]
    completeness = len(valid) / expected if expected else 0
    valid_pct = mean([r.get("valid_pct") for r in valid])
    pixels = median([r.get("pixels") for r in valid])
    score = min(completeness, (valid_pct or 0) / 100.0)
    if score >= 0.90:
        grade = "A"
    elif score >= 0.75:
        grade = "B"
    elif score >= 0.50:
        grade = "C"
    else:
        grade = "D"
    return {
        "grade": grade,
        "completeness_pct": rnd(completeness * 100, 1),
        "mean_monthly_valid_coverage_pct": rnd(valid_pct, 1),
        "median_expected_pixels": rnd(pixels, 1),
        "small_parcel_warning": bool(pixels is not None and pixels < 4),
    }


def drought_response(ndvi_anom: Dict[str, float], ndmi_anom: Dict[str, float], climate_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    drought_months = sorted([
        r["month"] for r in climate_rows
        if "2018-01" <= r["month"] <= "2025-12" and r.get("relative_drought")
    ])
    ndvi_vals = [ndvi_anom[k] for k in drought_months if k in ndvi_anom]
    ndmi_vals = [ndmi_anom[k] for k in drought_months if k in ndmi_anom]

    # Drought episodes from consecutive calendar months.
    def month_index(k: str) -> int:
        y, m = map(int, k.split("-"))
        return y * 12 + (m - 1)

    episodes: List[List[str]] = []
    cur: List[str] = []
    prev = None
    for k in drought_months:
        mi = month_index(k)
        if prev is None or mi == prev + 1:
            cur.append(k)
        else:
            if cur: episodes.append(cur)
            cur = [k]
        prev = mi
    if cur: episodes.append(cur)

    recovery = []
    all_keys = sorted(ndvi_anom)
    key_to_i = {k: i for i, k in enumerate(all_keys)}
    for ep in episodes:
        end = ep[-1]
        if end not in key_to_i:
            continue
        i0 = key_to_i[end]
        found = None
        for j in range(i0 + 1, min(len(all_keys), i0 + 13)):
            k = all_keys[j]
            if ndvi_anom.get(k) is not None and ndvi_anom[k] >= 0:
                found = max(1, month_index(k) - month_index(end))
                break
        if found is not None:
            recovery.append(found)

    ndvi_mean = mean(ndvi_vals)
    ndmi_mean = mean(ndmi_vals)
    if ndvi_mean is None:
        cls = "недовољно података"
    elif ndvi_mean <= -0.08:
        cls = "висока осетљивост"
    elif ndvi_mean <= -0.04:
        cls = "умерена осетљивост"
    elif ndvi_mean < 0:
        cls = "блага осетљивост"
    else:
        cls = "без видљивог негативног NDVI одговора"

    return {
        "context": "TerraClimate PDSI z<=-1, Rajac-wide, 2018-2025",
        "drought_month_count": len(drought_months),
        "ndvi_mean_calendar_anomaly_during_drought": rnd(ndvi_mean, 4),
        "ndmi_mean_calendar_anomaly_during_drought": rnd(ndmi_mean, 4),
        "recovery_months_median": rnd(median(recovery), 1),
        "recovery_episode_count": len(recovery),
        "class": cls,
    }


def build_card(pid: str, meta: Dict[str, Any], db: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]], climate: Dict[str, Any]) -> Dict[str, Any]:
    card: Dict[str, Any] = {
        "ok": True,
        "updated": utc_now(),
        "parcela": meta,
        "analysis_period_primary": "2018-2025",
        "analysis_period_sensitivity": "2016-2025",
        "indices": {},
        "ndvi_ndmi": {},
        "drought": {},
        "quality": {},
        "change_point": {},
        "interpretation": "",
    }

    for idx in INDICES:
        monthly = db[idx][pid]
        annual = {y: year_aggregate(monthly, y) for y in YEARS}
        seasonal = {
            season: {y: season_aggregate(monthly, y, season) for y in YEARS}
            for season in SEASONS
        }
        card["indices"][idx] = {
            "annual": annual,
            "seasonal": seasonal,
            "trend": {
                "annual_2018_2025": trend_for_aggregate(annual, PRIMARY_YEARS),
                "annual_2016_2025": trend_for_aggregate(annual, SENSITIVITY_YEARS),
                **{
                    f"{season}_2018_2025": trend_for_aggregate(seasonal[season], PRIMARY_YEARS)
                    for season in ("spring", "summer", "growing")
                },
            },
        }
        card["quality"][idx] = quality_for_index(monthly)

        anom = monthly_anomalies(monthly)
        pett = pettitt(sorted(anom.items()))
        card["change_point"][idx] = {
            **pett,
            "series": "calendar-month-deseasonalized anomaly 2018-2025",
            "significant_0_05": bool(pett.get("p") is not None and pett["p"] < 0.05),
        }

    # NDVI-NDMI relation, monthly anomalies and summer annual means in L2A period.
    ndvi_anom = monthly_anomalies(db["NDVI"][pid])
    ndmi_anom = monthly_anomalies(db["NDMI"][pid])
    common = sorted(set(ndvi_anom) & set(ndmi_anom))
    rho_month = spearman([ndvi_anom[k] for k in common], [ndmi_anom[k] for k in common])

    ndvi_summer = card["indices"]["NDVI"]["seasonal"]["summer"]
    ndmi_summer = card["indices"]["NDMI"]["seasonal"]["summer"]
    years_common = [
        y for y in PRIMARY_YEARS
        if ndvi_summer[y].get("complete_for_trend") and ndmi_summer[y].get("complete_for_trend")
        and ndvi_summer[y].get("mean") is not None and ndmi_summer[y].get("mean") is not None
    ]
    rho_summer = spearman(
        [float(ndvi_summer[y]["mean"]) for y in years_common],
        [float(ndmi_summer[y]["mean"]) for y in years_common],
    )
    card["ndvi_ndmi"] = {
        "period": "2018-2025",
        "monthly_anomaly_spearman_rho": rnd(rho_month, 4),
        "monthly_common_n": len(common),
        "summer_annual_spearman_rho": rnd(rho_summer, 4),
        "summer_years_n": len(years_common),
    }

    card["drought"] = drought_response(ndvi_anom, ndmi_anom, climate["rows"])
    return card


def interpretation(card: Dict[str, Any]) -> str:
    ndvi = card["indices"]["NDVI"]["trend"]["annual_2018_2025"]
    ndmi = card["indices"]["NDMI"]["trend"]["annual_2018_2025"]
    reci = card["indices"]["RECI"]["trend"]["annual_2018_2025"]
    qn = ndvi.get("q")
    qm = ndmi.get("q")
    sn = ndvi.get("sen_slope")
    sm = ndmi.get("sen_slope")
    sr = reci.get("sen_slope")

    parts = []
    sig_ndvi = qn is not None and qn < 0.10 and sn is not None
    sig_ndmi = qm is not None and qm < 0.10 and sm is not None

    if sig_ndvi and sig_ndmi and sn > 0 and sm > 0:
        parts.append("NDVI и NDMI показују робустан позитиван тренд: раст вегетационе активности прати стабилан/побољшан водни статус.")
    elif sig_ndvi and sig_ndmi and sn > 0 and sm < 0:
        parts.append("NDVI расте, а NDMI опада: парцела је зеленија, али статистика указује на растући водни стрес.")
    elif sig_ndvi and sn > 0:
        parts.append("NDVI показује робустан раст, али NDMI нема једнако јасан статистички одговор.")
    elif sig_ndvi and sn < 0:
        parts.append("NDVI показује робустан пад вегетационе активности.")
    else:
        parts.append("За 2018–2025 нема довољно јаког FDR-коригованог доказа о дугорочном NDVI тренду.")

    drought = card.get("drought") or {}
    if drought.get("class") in ("висока осетљивост", "умерена осетљивост"):
        parts.append(f"Релативно сушни месеци показују {drought.get('class')} вегетације.")
    rec = drought.get("recovery_months_median")
    if rec is not None:
        parts.append(f"Медијално време NDVI опоравка после сушне епизоде је око {rec:g} месеци.")

    cp = card["change_point"]["NDVI"]
    if cp.get("significant_0_05"):
        parts.append(f"У NDVI аномалијама је детектована могућа нагла промена око {cp.get('month')} ({cp.get('direction')}).")

    if card["quality"]["NDVI"].get("small_parcel_warning") or card["quality"]["NDMI"].get("small_parcel_warning"):
        parts.append("Парцела је мала у односу на Sentinel пиксел; резултат тумачити са повећаним опрезом.")

    return " ".join(parts)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CARDS_ROOT.mkdir(parents=True, exist_ok=True)

    climate = load_climate_context()
    write_json_pretty(OUT_ROOT / "climate_context.json", climate)

    db, meta = load_monthly_database()
    pids = sorted(meta)

    log("Building parcel statistics...")
    cards: Dict[str, Dict[str, Any]] = {}
    p_groups: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for i, pid in enumerate(pids, 1):
        if i == 1 or i % 100 == 0:
            log(f"Cards {i}/{len(pids)}")
        card = build_card(pid, meta[pid], db, climate)
        cards[pid] = card
        for idx in INDICES:
            for trend_name, t in card["indices"][idx]["trend"].items():
                p = t.get("p")
                if p is not None:
                    p_groups[f"{idx}:{trend_name}"].append((pid, float(p)))

    log("Applying Benjamini-Hochberg FDR across 2,299 parcels...")
    q_maps = {group: bh_qvalues(items) for group, items in p_groups.items()}

    trend_class_counter: Dict[str, Counter] = {idx: Counter() for idx in INDICES}
    quality_counter: Counter = Counter()

    for i, pid in enumerate(pids, 1):
        card = cards[pid]
        for idx in INDICES:
            for trend_name, t in card["indices"][idx]["trend"].items():
                q = q_maps.get(f"{idx}:{trend_name}", {}).get(pid)
                t["q"] = rnd(q, 6)
                t["fdr_significant_q_0_10"] = bool(q is not None and q < 0.10)
            trend_class_counter[idx][card["indices"][idx]["trend"]["annual_2018_2025"].get("class") or "—"] += 1

        grades = [card["quality"][idx]["grade"] for idx in INDICES]
        overall = max(grades, key=lambda x: "ABCD".index(x))
        card["quality"]["overall_grade"] = overall
        quality_counter[overall] += 1
        card["interpretation"] = interpretation(card)
        write_json(CARDS_ROOT / f"{pid}.json", card)

    summary = {
        "ok": True,
        "updated": utc_now(),
        "parcel_count": len(pids),
        "primary_period": "2018-2025 L2A/SR only",
        "sensitivity_period": "2016-2025",
        "indices": list(INDICES),
        "seasonal_metrics": list(SEASONS),
        "tests": [
            "Mann-Kendall",
            "Sen slope",
            "Benjamini-Hochberg FDR q",
            "Spearman NDVI-NDMI",
            "Pettitt change-point",
            "TerraClimate PDSI drought-response proxy",
        ],
        "quality_grade_counts": dict(sorted(quality_counter.items())),
        "annual_2018_2025_trend_class_counts": {
            idx: dict(trend_class_counter[idx]) for idx in INDICES
        },
        "warning": "Drought context is Rajac-wide TerraClimate PDSI, not parcel-level soil moisture. L1C/L2A sensitivity is explicitly separated.",
    }
    catalog = {
        "ok": True,
        "updated": utc_now(),
        "parcel_count": len(pids),
        "card_path_template": "cards/{parcela_id}.json",
        "summary": "summary.json",
        "climate_context": "climate_context.json",
    }
    write_json_pretty(OUT_ROOT / "summary.json", summary)
    write_json_pretty(OUT_ROOT / "catalog.json", catalog)

    log("DONE")
    log(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
