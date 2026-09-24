# INVIBE behavior & electrophysiology dashboard

The Streamlit app reads the behavioral Feather database directly. Its
electrophysiology view discovers final IBL GUI channel locations for each mouse
and displays all selected sessions in an interactive 3D atlas-space view.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Default sources:

- behavior: `\\SynoINVIBE_Caze\INVIBE_team_Cazettes\data\database\full_db_all_rigs.feather`
- ephys/Brainreg root: `F:\coregistration_db`
- open field root: `F:\Behavior`

They can be overridden without changing the code:

```powershell
$env:APP_BEHAVIOR_DB = "D:\data\full_db_all_rigs.feather"
$env:APP_EPHYS_ROOT = "D:\ephys"
$env:APP_OPENFIELD_ROOT = "D:\behavior"
$env:APP_ATLAS_DIR = "C:\Users\me\.brainglobe\allen_mouse_25um_v1.2"
$env:OPENAI_API_KEY = "your-project-api-key"
# Optional server-wide override; otherwise the UI selects Sol or Terra
$env:OPENAI_CHAT_MODEL = "gpt-5.6-sol"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The Cazette Bot uses the OpenAI Responses API with validated read-only local
tools. Its general query engine can inspect the live schema, filter and
aggregate Behavior or Ephys data, compute descriptive statistics and
correlations, and render line, bar, scatter, histogram, density, box or violin
plots. `Advanced` uses GPT-5.6 Sol with high reasoning effort; `Fast` uses
GPT-5.6 Terra. It has no shell, filesystem-write, database-write, or source-code
editing tool. API responses are requested with `store=False`. If
`OPENAI_API_KEY` is not configured on the server, the key can be entered for
the current Streamlit session from the Cazette Bot tab.

For each mouse, fitted channel locations are discovered below:

`YYYY_MM_DD\alf\probeNN\channel_locations.json`

The IBL coordinates are converted from ML/AP/DV relative to Bregma into Allen
CCF space. Tracks from the same session use the same color, while the line style
distinguishes probes. Each track is rendered as a straight total-least-squares
3D fit inside the Allen mouse brain root mesh (`meshes\997.obj`).

The electrophysiology overview places a compact 3D atlas next to clickable
session cards. Each card summarizes insertion type, expected probe target,
good-unit count, and valid behavioral bouts. Clicking a card controls the
details shown below: `Overview`, `Activity`, or `Units & QC`.
