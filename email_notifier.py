#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Email notifier for new agro alerts.

Reads public/alerts/new-this-run.json (produced by daily_processor.py),
and if non-empty, sends a single summary email to recipients via SMTP.

Required environment variables (set as GitHub Actions secrets):
- SMTP_HOST       — e.g. smtp.gmail.com
- SMTP_PORT       — e.g. 587
- SMTP_USER       — sender mailbox
- SMTP_PASSWORD   — app-specific password or SMTP password
- ALERT_RECIPIENTS — comma-separated emails

Gracefully no-ops if any required env var is missing — script will print
a warning but exit 0 so the workflow doesn't fail.
"""
from __future__ import annotations

import json
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent
NEW_ALERTS_PATH = ROOT / "public" / "alerts" / "new-this-run.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_alerts() -> List[Dict[str, Any]]:
    if not NEW_ALERTS_PATH.exists():
        log(f"No alerts file at {NEW_ALERTS_PATH}")
        return []
    try:
        with NEW_ALERTS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        log(f"Could not read alerts: {exc}")
        return []
    return data.get("new_alerts") or []


def build_html(alerts: List[Dict[str, Any]]) -> str:
    if not alerts:
        return "<p>Нема нових аларма.</p>"
    rows = []
    for a in alerts:
        lvl = a.get("level", "warning")
        color = a.get("color", "#f6c95f")
        rows.append(f"""
        <tr>
            <td style="padding:8px;border-bottom:1px solid #ddd;background:{color}22;">
                <b>{a.get('icon','⚠️')} {a.get('label','')}</b><br>
                <small style="color:#555;">{lvl.upper()}</small>
            </td>
            <td style="padding:8px;border-bottom:1px solid #ddd;">
                <b>{a.get('parcel_id','?')}</b> &middot; {a.get('culture','?')}<br>
                <small>{a.get('area_ha','?')} ha</small>
            </td>
            <td style="padding:8px;border-bottom:1px solid #ddd;font-size:12px;">
                {format_evidence(a.get('evidence', []))}
            </td>
        </tr>
        """)
    acq = alerts[0].get("acquisition", "")
    return f"""
    <html><body style="font-family:Arial,sans-serif;color:#222;">
        <h2 style="color:#2d5e1f;">🌱 Нови алармi — PIO Rajac агро мониторинг</h2>
        <p>Сателит: <b>Sentinel-2</b> · Снимак: <b>{acq}</b></p>
        <table style="border-collapse:collapse;width:100%;max-width:720px;">
            <thead>
                <tr style="background:#e7f1d6;text-align:left;">
                    <th style="padding:8px;">Аларм</th>
                    <th style="padding:8px;">Парцeлa</th>
                    <th style="padding:8px;">Доказ</th>
                </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        <p style="margin-top:24px;font-size:12px;color:#666;">
            Aутоматска порука. Прикaз на саjту:
            <a href="https://piorajac.rs/agro-kopernikus/">piorajac.rs/agro-kopernikus</a>
        </p>
    </body></html>
    """


def format_evidence(evidence: List[Dict[str, Any]]) -> str:
    parts = []
    for ev in evidence:
        idx = ev.get("index", "?").upper()
        cur = ev.get("current")
        base = ev.get("baseline")
        delta = ev.get("delta")
        if base is not None and delta is not None:
            parts.append(f"{idx}: {cur} (Δ {delta:+.3f} од {base:.3f})")
        elif cur is not None:
            parts.append(f"{idx}: {cur}")
        else:
            parts.append(f"{idx}: —")
    return "<br>".join(parts)


def send_email(alerts: List[Dict[str, Any]]) -> int:
    host = os.environ.get("SMTP_HOST", "").strip()
    port = os.environ.get("SMTP_PORT", "587").strip()
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    recipients_raw = os.environ.get("ALERT_RECIPIENTS", "").strip()

    if not (host and user and password and recipients_raw):
        log("SMTP env vars not configured — skipping email send.")
        return 0

    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    if not recipients:
        log("No recipients configured — skipping.")
        return 0

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🌱 PIO Rajac — {len(alerts)} новi алармi"
    msg["From"] = user
    msg["To"] = ", ".join(recipients)
    msg["Date"] = formatdate(localtime=True)

    html_body = build_html(alerts)
    plain_body = f"Има {len(alerts)} нових аларма. Видети на https://piorajac.rs/agro-kopernikus"
    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(host, int(port), timeout=30) as srv:
            srv.starttls()
            srv.login(user, password)
            srv.sendmail(user, recipients, msg.as_string())
        log(f"Sent email to {len(recipients)} recipient(s)")
    except Exception as exc:
        log(f"SMTP send failed: {exc}")
        return 0  # do not fail the workflow on email errors
    return 0


def main() -> int:
    alerts = load_alerts()
    if not alerts:
        log("No new alerts — nothing to send.")
        return 0
    log(f"Sending email for {len(alerts)} new alert(s)")
    return send_email(alerts)


if __name__ == "__main__":
    sys.exit(main())
