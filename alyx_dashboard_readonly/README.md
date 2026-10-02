# VFT dashboard handoff and REST starter

The main deliverable is `CODEX_HANDOFF.md`: paste it into Codex on the dashboard
computer. The remaining files are a tested REST/normalization starter that the
other Codex can inspect, adapt, or replace while building the dashboard. They
use the same Alyx username/password as the VFT GUI and never call create,
update, patch, or delete operations.

## Setup

```bash
cd alyx_dashboard_readonly
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env
```

The `.env` file contains only the Alyx URL and lab name. Do not put a username,
password, or token in it. Users enter their normal Alyx credentials on the
dashboard login page. Authentication tokens remain in that Streamlit browser
session and ONE is instructed not to cache the login token on disk.

Inspect the available data before changing mappings. This command prompts for
the same Alyx login:

```bash
python inspect_alyx_readonly.py
```

This writes `schema_report.json`, which contains no password and is ignored by
Git. Then start the dashboard:

```bash
streamlit run dashboard.py
```

Run the normalization tests with:

```bash
python -m unittest test_normalize.py
```

## Current data contract

The table has one Alyx `Session` per row. Core session, trial, water, user,
location, and narrative data are read from `actions_session` and its relations.
Poop count is joined from same-mouse, same-date `OtherAction` records carrying
the `Poop sampling` procedure only if IBL later exposes the optional
`other-actions` REST endpoint. On the current server, poop sampling remains
blank because those database records are not reachable through an existing REST
endpoint.

Values introduced by the current VFT GUI are expected under
`Session.json.virtual_foraging`. Historical Excel imports contain the core
trial/water statistics but do not necessarily contain recording type, training
type, probabilities, licks, tube score, headfix score, or treadmill comfort.
Missing values remain blank. `bout_rate_per_min` is derived as total bouts /
session duration when an explicit stored rate is unavailable. New GUI sessions
store both `licks` and `bout_rate` directly in `Session.json`.
