"""Flatten Alyx sessions into the user-facing VFT session table."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
import re


COLUMNS = [
    "mouse_id",
    "date",
    "protocol",
    "session_time",
    "experimentalist",
    "recording_type",
    "training_type",
    "session_probabilities",
    "version",
    "correct_bouts",
    "valid_bouts",
    "total_bouts",
    "manual_reward_ul",
    "reward_ul",
    "total_reward_ul",
    "licks",
    "rewarded_licks",
    "unrewarded_licks",
    "session_duration_min",
    "reward_rate_per_min",
    "poops_number",
    "tube",
    "headfix",
    "comfort_on_treadmill",
    "bout_rate_per_min",
    "session_comments",
    "rig_number",
    "session_id",
]

TRAINING_TYPES = {
    "1": "1st training",
    "1.0": "1st training",
    "2": "2nd training",
    "2.0": "2nd training",
    "3": "Task",
    "3.0": "Task",
    "4": "Task flexiflipping",
    "4.0": "Task flexiflipping",
    "1sttraining": "1st training",
    "2ndtraining": "2nd training",
    "task": "Task",
    "task_flexiflipping": "Task flexiflipping",
}

RECORDING_TYPES = {
    "0": "None",
    "0.0": "None",
    "1": "Neuropixel habituation",
    "1.0": "Neuropixel habituation",
    "2": "Neuropixel",
    "2.0": "Neuropixel",
    "3": "Optogenetics",
    "3.0": "Optogenetics",
}


def _dict(value):
    return value if isinstance(value, dict) else {}


def _first(*values):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _datetime(value):
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _metadata(payload):
    payload = _dict(payload)
    return _dict(payload.get("virtual_foraging"))


def _probabilities(payload, metadata):
    direct = _first(
        metadata.get("session_probabilities"),
        metadata.get("probas"),
        payload.get("session_probabilities"),
        payload.get("probas"),
        payload.get("Probas"),
    )
    if direct is not None:
        return str(direct)
    reward = _first(metadata.get("reward_probability"), metadata.get("reward_proba"))
    switch = _first(metadata.get("switch_probability"), metadata.get("switch_proba"))
    if reward is not None or switch is not None:
        return f"reward={reward}; switch={switch}"
    blocks = _dict(metadata.get("blocks"))
    rewards = blocks.get("reward_probabilities")
    switches = blocks.get("switch_probabilities")
    if rewards or switches:
        return f"reward={rewards}; switch={switches}"
    return None


def _poop_count(actions):
    total = 0.0
    found = False
    for action in actions if isinstance(actions, list) else []:
        payload = _dict(action.get("json"))
        value = _number(_first(payload.get("count"), payload.get("poop_number")))
        if value is None:
            narrative = str(action.get("narrative") or "")
            match = re.search(r"(?i)(\d+(?:\.\d+)?)\s+poop", narrative)
            value = _number(match.group(1)) if match else None
        if value is not None:
            total += value
            found = True
    return total if found else None


def session_to_row(raw: dict) -> dict:
    payload = _dict(raw.get("session_json"))
    metadata = _metadata(payload)
    trials = _dict(payload.get("trials"))
    water = _dict(payload.get("water_administration"))
    start = _datetime(raw.get("start_time"))
    end = _datetime(raw.get("end_time"))
    duration = None
    if start and end:
        duration = (end - start).total_seconds() / 60
        if duration < 0:
            duration = None
    duration = _number(_first(duration, payload.get("duration_minutes"), metadata.get("duration_minutes")))

    total_bouts = _number(_first(raw.get("n_trials"), trials.get("total")))
    valid_bouts = _number(trials.get("valid"))
    correct_bouts = _number(_first(raw.get("n_correct_trials"), trials.get("correct")))
    bout_rate = _number(_first(payload.get("bout_rate"), metadata.get("bout_rate")))
    if bout_rate is None and total_bouts is not None and duration and duration > 0:
        bout_rate = total_bouts / duration

    protocol = _first(raw.get("protocol"), raw.get("session_type"), "Unspecified")
    training_raw = metadata.get("training_type")
    training_type = _first(
        TRAINING_TYPES.get(str(training_raw).lower()), training_raw,
        TRAINING_TYPES.get(str(protocol).lower()),
    )
    recording_type = _first(metadata.get("recording_type"), metadata.get("rec_type"))
    if recording_type is not None:
        recording_type = RECORDING_TYPES.get(str(recording_type), str(recording_type))

    rig = _first(metadata.get("rig_number"), raw.get("location"), payload.get("rig_number"))
    experimenters = raw.get("experimentalists") or []
    if isinstance(experimenters, str):
        experimenters = [experimenters]

    lick_data = _dict(_first(payload.get("licks"), metadata.get("licks")))
    rewarded_licks = _number(_first(
        payload.get("rewarded_licks"), payload.get("rewardedLicks"),
        metadata.get("rewarded_licks"), metadata.get("rewardedLicks"),
        lick_data.get("rewarded"), lick_data.get("rewarded_licks"),
    ))
    unrewarded_licks = _number(_first(
        payload.get("unrewarded_licks"), payload.get("non_rewarded_licks"),
        payload.get("nonRewardedLicks"), metadata.get("unrewarded_licks"),
        metadata.get("non_rewarded_licks"), metadata.get("nonRewardedLicks"),
        lick_data.get("unrewarded"), lick_data.get("non_rewarded"),
    ))

    row = {
        "mouse_id": raw.get("mouse_id"),
        "date": start.date().isoformat() if start else None,
        "protocol": protocol,
        "session_time": start.time().replace(microsecond=0).isoformat() if start else None,
        "experimentalist": ", ".join(experimenters),
        "recording_type": recording_type,
        "training_type": training_type,
        "session_probabilities": _probabilities(payload, metadata),
        "version": _first(metadata.get("version"), raw.get("session_number")),
        "correct_bouts": correct_bouts,
        "valid_bouts": valid_bouts,
        "total_bouts": total_bouts,
        "manual_reward_ul": _number(water.get("manual")),
        "reward_ul": _number(water.get("reward")),
        "total_reward_ul": _number(water.get("total")),
        "licks": _number(_first(payload.get("licks"), payload.get("total_licks"), metadata.get("licks"))),
        "rewarded_licks": rewarded_licks,
        "unrewarded_licks": unrewarded_licks,
        "session_duration_min": duration,
        "reward_rate_per_min": _number(_first(payload.get("reward_rate"), metadata.get("reward_rate"))),
        "poops_number": _poop_count(raw.get("poop_actions")),
        "tube": _first(metadata.get("tube"), payload.get("tube")),
        "headfix": _first(metadata.get("headfix"), payload.get("headfix")),
        "comfort_on_treadmill": _first(
            metadata.get("comfort_on_treadmill"), metadata.get("confort_on_treadmill"),
            payload.get("comfort_on_treadmill"), payload.get("confort_on_treadmill"),
        ),
        "bout_rate_per_min": bout_rate,
        "session_comments": raw.get("narrative") or "",
        "rig_number": rig,
        "session_id": raw.get("session_id"),
    }
    return {column: row.get(column) for column in COLUMNS}


def normalize_sessions(raw_sessions: list[dict]) -> list[dict]:
    return [session_to_row(session) for session in raw_sessions]

