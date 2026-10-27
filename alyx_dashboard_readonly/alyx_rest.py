"""Read-only Alyx REST access using the same login as the VFT GUI."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Allows the pure mapping/client tests to run before setup.
    def load_dotenv(_path):
        return False


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
DEFAULT_URL = os.getenv("ALYX_URL", "https://cazettes.iblcore.org").rstrip("/")
DEFAULT_LAB = os.getenv("ALYX_LAB", "cazetteslab")


def connect(base_url: str, username: str, password: str):
    from one.webclient import AlyxClient, RestSchemeOpenApi

    if not base_url.startswith(("http://", "https://")):
        raise ValueError("Alyx URL must start with http:// or https://")
    if not username or not password:
        raise ValueError("Alyx username and password are required")
    client = AlyxClient(base_url=base_url.rstrip("/"), silent=True)
    client.authenticate(
        username=username.strip(), password=password, cache_token=False, force=True
    )
    client.rest("users", "read", id=username.strip(), no_cache=True)
    # ONE normally caches the schema for a week. Refresh it so the dashboard sees
    # any endpoint IBL may add later, without persisting the supplied password.
    schema = client.get("/api/schema", clobber=True, expires=True)
    client._rest_schemes = RestSchemeOpenApi(schema)
    return client


def _record_id(record: dict) -> str:
    if record.get("id"):
        return str(record["id"])
    return str(record.get("url", "")).rstrip("/").rsplit("/", 1)[-1]


def _day(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _poop_actions(client, lab: str, subject: str | None = None) -> dict[tuple[str, str], list[dict]]:
    """Return poop actions when the optional companion endpoint is available."""
    if "other-actions" not in client.rest_schemes.endpoints:
        return {}
    filters = {"lab": lab}
    if subject is not None:
        filters["subject"] = subject
    actions = client.rest("other-actions", "list", no_cache=True, **filters)
    grouped = defaultdict(list)
    for action in actions:
        procedures = {str(value).strip().lower() for value in action.get("procedures", [])}
        if "poop sampling" not in procedures:
            continue
        key = (action.get("subject"), _day(action.get("start_time")))
        grouped[key].append({
            "id": action.get("id"),
            "start_time": action.get("start_time"),
            "narrative": action.get("narrative"),
            "json": action.get("json") or {},
        })
    return grouped


def load_raw_sessions(client, lab: str = DEFAULT_LAB, subject: str | None = None) -> list[dict]:
    """Read session details only; no create/update/delete REST action is used."""
    filters = {"lab": lab}
    if subject is not None:
        filters["subject"] = subject
    summaries = client.rest("sessions", "list", no_cache=True, **filters)
    poop_by_subject_day = _poop_actions(client, lab, subject)
    rows = []
    for summary in summaries:
        detail = client.rest(
            "sessions", "read", id=_record_id(summary), no_cache=True
        )
        if subject is not None and detail.get("subject") != subject:
            continue
        subject = detail.get("subject")
        start_time = detail.get("start_time")
        rows.append({
            "session_id": detail.get("id") or _record_id(detail),
            "mouse_id": subject,
            "start_time": start_time,
            "end_time": detail.get("end_time"),
            "session_number": detail.get("number"),
            "protocol": detail.get("task_protocol"),
            "session_type": detail.get("type"),
            "n_correct_trials": detail.get("n_correct_trials"),
            "n_trials": detail.get("n_trials"),
            "narrative": detail.get("narrative"),
            "session_json": detail.get("json") or {},
            "location": detail.get("location"),
            "experimentalists": detail.get("users") or [],
            "poop_actions": poop_by_subject_day.get((subject, _day(start_time)), []),
        })
    return rows
