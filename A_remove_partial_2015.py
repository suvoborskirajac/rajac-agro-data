#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove partial 2015 Sentinel-2 index data from ACTIVE Rajac analytics.

Backup is Git history. SOURCE_COMMIT records the exact commit that still
contains the original 2015 files.
"""
from __future__ import annotations
import json, os, shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"public"/"cadastral-longterm"
STATS=OUT/"stats"
CAT=OUT/"catalog.json"
AUD=OUT/"audit.json"
ARCH=OUT/"archive_2015_partial.json"
INDICES=("NDVI","NDMI","RECI")

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
def load(p:Path, default:Any=None):
    try:return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:return default
def write(p:Path,obj:Any):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    t.replace(p)

def main():
    source_sha=os.environ.get("SOURCE_COMMIT","").strip() or "see previous Git commit"
    y2015=STATS/"2015"
    originals=[]
    if y2015.is_dir():
        for p in sorted(y2015.glob("*_monthly.json")):
            j=load(p,{}) or {}
            originals.append({
                "path":str(p.relative_to(ROOT)),
                "index":j.get("index"),
                "months_present":j.get("months_present") or [],
                "parcel_count_written":j.get("parcel_count_written") or 0,
                "size_bytes":p.stat().st_size,
            })

    write(ARCH,{
        "ok":True,
        "archived_at":now(),
        "status":"2015 removed from active analytics; recoverable from Git history",
        "reason":"2015 Sentinel-2 coverage is partial",
        "source_commit_before_removal":source_sha,
        "original_files":originals,
        "active_period_after_removal":"2016-2026",
    })

    if y2015.is_dir():
        shutil.rmtree(y2015)

    years={}
    total_rows=0
    for ydir in sorted([p for p in STATS.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p:int(p.name)):
        y=int(ydir.name)
        if y<2016: continue
        yi={}
        for idx in INDICES:
            p=ydir/f"{idx}_monthly.json"
            if not p.is_file(): continue
            j=load(p,{}) or {}
            yi[idx]={
                "months_present":j.get("months_present") or [],
                "parcel_count_written":j.get("parcel_count_written") or 0,
                "resolution_m":j.get("resolution_m"),
                "path":f"stats/{y}/{idx}_monthly.json",
            }
            total_rows += int(j.get("parcel_count_written") or 0)
        if yi: years[str(y)]=yi

    catalog=load(CAT,{}) or {}
    catalog.update({
        "ok":True,"updated":now(),"indices":list(INDICES),"years":years,
        "active_period":"2016-2026",
        "excluded_years":{"2015":"partial Sentinel-2 year; backed up in Git history"},
        "source_policy":{
            "2016-01_to_2017-03":"COPERNICUS/S2_HARMONIZED",
            "2017-04_onward":"COPERNICUS/S2_SR_HARMONIZED",
        },
    })
    write(CAT,catalog)
    audit={
        "ok":True,"updated":now(),"years_present":sorted(years),
        "index_year_files":sum(len(v) for v in years.values()),
        "row_sets_total":total_rows,"active_period":"2016-2026",
        "excluded_years":["2015"],"archive_metadata":"archive_2015_partial.json",
        "warning":"2016-2017-03 L1C/TOA; 2017-04+ L2A/SR. Primary statistical trend remains 2018-2025 L2A-only."
    }
    write(AUD,audit)
    assert not y2015.exists()
    assert "2015" not in years
    assert audit["index_year_files"]==33, audit
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
