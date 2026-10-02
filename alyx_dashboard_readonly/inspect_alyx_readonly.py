"""Create a credential-free structural report of the available Alyx session data."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from getpass import getpass
import json
from pathlib import Path

from alyx_rest import DEFAULT_LAB, DEFAULT_URL, connect, load_raw_sessions


def json_keys(value, prefix=""):
    keys = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.add(path)
            keys.update(json_keys(child, path))
    elif isinstance(value, list):
        for child in value:
            keys.update(json_keys(child, prefix))
    return keys


def serializable(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError


def main():
    base_url = input(f"Alyx URL [{DEFAULT_URL}]: ").strip() or DEFAULT_URL
    username = input("Alyx username: ").strip()
    password = getpass("Alyx password: ")
    client = connect(base_url, username, password)
    rows = load_raw_sessions(client, DEFAULT_LAB)
    report = {
        "session_count": len(rows),
        "mice": sorted({row["mouse_id"] for row in rows if row.get("mouse_id")}),
        "experimentalists": sorted({
            user for row in rows for user in (row.get("experimentalists") or [])
        }),
        "protocol_counts": dict(Counter(row.get("protocol") or "" for row in rows)),
        "location_counts": dict(Counter(row.get("location") or "" for row in rows)),
        "session_json_keys": sorted({
            key for row in rows for key in json_keys(row.get("session_json"))
        }),
        "poop_action_json_keys": sorted({
            key
            for row in rows
            for action in (row.get("poop_actions") or [])
            for key in json_keys(action.get("json"))
        }),
        "date_min": min((row["start_time"] for row in rows if row.get("start_time")), default=None),
        "date_max": max((row["start_time"] for row in rows if row.get("start_time")), default=None),
    }
    output = Path(__file__).with_name("schema_report.json")
    output.write_text(
        json.dumps(report, indent=2, default=serializable) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}")
    print(f"Sessions: {report['session_count']}; mice: {len(report['mice'])}")


if __name__ == "__main__":
    main()
