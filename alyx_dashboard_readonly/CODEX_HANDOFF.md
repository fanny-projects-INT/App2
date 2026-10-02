# Handoff prompt for the dashboard computer

Copy everything below into the first message to Codex on the dashboard computer.

---

We are building a read-only user dashboard for the Cazettes lab Virtual
Foraging Task. Work from the `alyx_dashboard_readonly` folder supplied with this
message. Use the existing Alyx REST API and do not modify Alyx.

Context:

- Alyx is hosted and administered by IBL. We can use its existing REST API with
  each user's normal Alyx username/password, but we do not have PostgreSQL or
  server deployment access.
- A separate VFT GUI now creates subjects and sessions through existing Alyx
  REST endpoints. New sessions store task metadata under
  `Session.json.virtual_foraging`, basic results under `Session.json.trials`,
  water/reward results under `Session.json.water_administration`, and free-text
  comments in `Session.narrative`. New GUI sessions also store `licks` and
  `bout_rate` directly in `Session.json`.
- Historical data were imported from Excel. Those sessions reliably contain
  dates, protocol, start/end time when available, users, location, correct/valid/
  total trials, water totals, reward rate, and narrative. Some requested fields
  were not imported and must remain null unless the real database contains an
  identified source.
- Poop samples are `OtherAction` records, but the deployed server does not
  expose `other-actions`. Leave poop count blank unless IBL adds that endpoint
  or another existing endpoint is found to expose the records.
- An older proof of concept exists in the Alyx repository under
  `alyx/virtual_foraging_reports`, but it cannot be deployed on IBL's server.
  Reuse its metric definitions if that repository is available locally.

Connection pattern already used successfully by the VFT GUI:

```python
from one.webclient import AlyxClient

client = AlyxClient(base_url="https://cazettes.iblcore.org", silent=True)
client.authenticate(
    username=username,
    password=password,
    cache_token=False,
    force=True,
)
client.rest("users", "read", id=username, no_cache=True)  # validates login

session_summaries = client.rest(
    "sessions", "list", lab="cazetteslab", no_cache=True
)
session = client.rest(
    "sessions", "read", id=session_summaries[0]["id"], no_cache=True
)
```

The `sessions/list` response is abbreviated. Always call `sessions/read` for
each selected session before extracting detailed fields. The dashboard login
must collect the normal Alyx username/password interactively. Use
`cache_token=False` and do not write the password or token to a file.

Where the information is stored in a detailed session response:

- mouse: `subject`
- date and session time: `start_time`
- duration: difference between `end_time` and `start_time`
- protocol: `task_protocol`
- version/session number: `number` (verify this convention against real data)
- experimentalists: `users`
- rig: `location`, with newer GUI fallback `json.virtual_foraging.rig_number`
- comments: `narrative`
- correct bouts: `n_correct_trials`, fallback `json.trials.correct`
- valid bouts: `json.trials.valid`
- total bouts: `n_trials`, fallback `json.trials.total`
- manual/reward/total water in µL: `json.water_administration.manual`,
  `.reward`, and `.total`
- reward rate: `json.reward_rate`
- newer-session licks and bout rate: `json.licks` and `json.bout_rate`
- recording/training/probability metadata from the current GUI:
  `json.virtual_foraging`
- Neuropixel details when present: `probe_insertion[*].json`

Source-code map if the three repositories are present on the other computer:

- `global_package/py_ui/alyx_integration.py`: production REST login and current
  session population contract (`connect`, `create_session`).
- `global_package/py_ui/VFT_gui_01.py`: fields stored in
  `json.virtual_foraging` (`_session_metadata`).
- `alyx/alyx/virtual_foraging_reports/services.py`: earlier read-only session
  metric definitions (`session_to_record`).
- `alyx_virtual_foraging_task/json_db/excel_to_json.py`: historical Excel-to-JSON
  mapping (`_parse_session`).
- `alyx_virtual_foraging_task/injection/inject_to_alyx.py`: historical JSON to
  Alyx model population (`_create_session`, `_create_poop_sample`).
- `alyx_virtual_foraging_task/exp_database/vf_all_mice.normalized.json`: normalized
  historical import source, if copied locally. Treat it as supporting evidence;
  Alyx remains the dashboard source.

Target dashboard:

- Multi-select filters for mouse and experimentalist.
- One session per table row.
- Columns, in order: date, protocol, session time, experimentalist, recording
  type, training type, session probabilities, version, correct bouts, valid
  bouts, total bouts, manual reward, reward, total reward, licks, session
  duration, reward rate, poop number, tube, headfix, comfort on treadmill, bout
  rate, session comments, and rig number. Keep mouse ID visible too when more
  than one mouse is selected.
- Allow downloading the filtered table as CSV.
- Show data completeness so users can distinguish zero from unavailable data.

Files already provided:

- `dashboard.py`: runnable Streamlit dashboard starter.
- `alyx_rest.py`: authentication plus list/read-only session retrieval using
  ONE's `AlyxClient`.
- `normalize.py`: tolerant mapping from Alyx records to the requested columns.
- `inspect_alyx_readonly.py`: produces a credential-free report of protocols,
  users, locations, and JSON key paths actually present in Alyx.
- `.env.example`, `requirements.txt`, tests, and setup instructions.

Your first steps:

1. Read all files in this folder and any local `AGENTS.md` instructions.
2. Create `.env` locally from `.env.example`; it contains only URL/lab defaults.
   Never put a username, password, or token in that file or in logs.
3. Log in through the dashboard with a normal Alyx account. Keep
   `cache_token=False`; the password must remain only in the login request and
   must never be persisted.
4. Run `python inspect_alyx_readonly.py` and examine `schema_report.json`.
5. Run `python -m unittest test_normalize.py`.
6. Start `streamlit run dashboard.py` and compare several sessions against Alyx
   or the original source data.
7. Adapt JSON-key aliases only when supported by observed records. Do not parse
   tube/headfix/comfort scores from narrative text unless the user explicitly
   approves a documented parsing rule.
8. Check whether `version` means Alyx `Session.number` for both historical and
   new records. If it does not, identify the real stored source and update the
   mapping.
9. Report which requested columns have complete, partial, or absent coverage,
   then propose the smallest ingestion changes needed for future sessions.

Important interpretation rules:

- Reward/manual/total values in session JSON are microlitres.
- Linked Alyx `WaterAdministration.water_administered` values are millilitres;
  do not mix their units without converting.
- Session duration is `(end_time - start_time)` in minutes.
- Reward rate is rewards per minute.
- Bout rate is bouts per minute and may be derived as total bouts / duration.
- Missing values stay null and are never converted to zero.
- Only use Alyx `list` and `read` REST actions. Do not call create, update,
  partial_update, or delete.

Complete the dashboard locally, validate it against real read-only records, and
leave clear launch instructions for experimentalists.

---
